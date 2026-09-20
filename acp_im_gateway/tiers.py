"""Approval tiers: what may be approved silently, and what must always ask.

Two lists are evaluated on every ``session/request_permission``:

* ``always_ask`` — money or irreversible. This is a **code gate**, not a
  posture: it forces a tap even when the chat posture is ``auto``/``yolo``, and
  it wins over ``auto_allow``. A command that is both harmless and dangerous
  (``git status && rm -rf build``) is dangerous.
* ``auto_allow`` — harmless and read-only commands, approved without a tap so
  the phone is not asked to confirm ``git status`` for the tenth time.

Anything else follows the chat posture (``/aprobar`` per chat, defaulting to
``GATEWAY_APPROVAL_POSTURE`` / ``ask``).

Matching is case-insensitive and inspects the request's ``toolCall.kind``,
``toolCall.title``, the ``rawInput`` payload (a shell command string, generic
tool arguments) and the affected file locations. The two tiers match
*asymmetrically*, on purpose:

* ``always_ask`` matches broadly, anywhere in that haystack — a false positive
  only costs one tap;
* ``auto_allow`` only matches an affirmative *command head* (``git status``,
  ``pytest``, ``ls``), never a loose substring — a false positive there would be
  a silent approval, so ``ls`` must not silently bless ``false ls`` or a
  ``results`` search.

A compound line (``a && b | c``) is approved only when **every** segment is
harmless and read-only: any ``always_ask`` segment forces the tap no matter what
else is in the line. Interpreter paths are normalised (``./.venv/bin/python -m
pytest`` == ``python3 -m pytest`` == ``python -m pytest``) and a leading env
assignment or a harmless prefix (``cd``, ``export``, ``echo``, ``true``,
``time``, ``nice``) is skipped before the head is matched.

Every decision is logged at INFO with its reason, so the journal explains why
something was auto-approved or why it was forced to a tap.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .render import curl_creates_data, keyword_match, rm_forced_recursive, shell_commands

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids an import cycle)
    from .approvals import PermissionRequest

_logger = logging.getLogger("acp_im_gateway.tiers")

AUTO_ALLOW = "auto_allow"
ASK = "ask"

TIER_ALWAYS_ASK = "always_ask"
TIER_AUTO_ALLOW = "auto_allow"
TIER_POSTURE = "posture"

#: Chat postures (``/aprobar``). ``ask`` = tap for anything not in
#: ``auto_allow``; ``auto`` = silent for anything not in ``always_ask``. ``yolo``
#: is accepted as a synonym of ``auto`` for backwards compatibility.
POSTURE_ASK = "ask"
POSTURE_AUTO = "auto"
_WIDENING_POSTURES = ("auto", "yolo")
_POSTURE_NAMES = {
    "preguntar": POSTURE_ASK,
    "ask": POSTURE_ASK,
    "auto": POSTURE_AUTO,
    "yolo": POSTURE_AUTO,
    "": POSTURE_ASK,
}

#: Harmless and read-only. Approve silently, log at INFO.
DEFAULT_AUTO_ALLOW: tuple[str, ...] = (
    "git status",
    "git diff",
    "git log",
    "git show",
    "git branch",
    "git remote -v",
    "ls",
    "cat",
    "head",
    "tail",
    "grep",
    "rg",
    "find",
    "pwd",
    "wc",
    "pytest",
    "python -m pytest",
)

#: Money or irreversible. Must still require a tap, even in ``auto``/``yolo``.
DEFAULT_ALWAYS_ASK: tuple[str, ...] = (
    "fal.ai",
    "fal_client",
    "FAL_KEY",
    "elevenlabs",
    "openai",
    "anthropic",
    "stripe",
    "curl",
    "deploy",
    "rsync",
    "scp",
    "wp",
    "rm -rf",
    "git push",
    "git reset --hard",
    "docker push",
    "npm publish",
    "aws",
    "gcloud",
    "hetzner",
    "cloudpanel",
)

#: The ``curl`` entry only forces a tap when curl would *create* data.
CURL_PATTERN = "curl"
DATA_CREATING_LABEL = "(data-creating)"
#: ``rm -rf`` also covers its equivalent spellings (``rm -fr``, ``rm -r -f``, …).
RM_RECURSIVE_PATTERN = "rm -rf"
RM_RECURSIVE_LABEL = "(recursive force)"

#: Fragments that mean a command is *not* read-only, whatever its head is:
#: a redirection or a substitution can write or delete anything.
_WRITE_FRAGMENTS = (">", "<", "`", "$(", "${")
#: Flags that make an otherwise read-only command destructive.
_DESTRUCTIVE_TOKENS = (
    "-delete",
    "-exec",
    "-execdir",
    "-D",
    "--delete",
    "--force",
    "--hard",
    "--prune",
    "--remove",
    "--no-preserve-root",
    "--output",
)
#: Flags that are destructive only for particular commands: ``-d`` deletes a git
#: branch, but ``ls -d */``, ``git log -d`` and ``grep -d skip`` are read-only.
_DESTRUCTIVE_BY_SCOPE = {"-d": ("git branch", "git tag")}

#: Leading tokens that are harmless on their own: their arguments are paths or
#: values, never a command, so ``cd app`` cannot do anything a tap should gate.
_HARMLESS_PREFIXES = ("cd", "export", "echo", "true")
#: Prefixes that wrap *another* command and pass its argv through, so the real
#: head is what follows them: ``time pytest -q``, ``nice -n 5 pytest -q``.
_WRAPPER_PREFIXES = ("time", "nice")
#: Bound the prefix chain so a pathological line cannot spin.
_MAX_PREFIX_DEPTH = 8
#: ``python``, ``python3``, ``python3.12`` … all name the same interpreter.
_PYTHON_HEAD = re.compile(r"python(\d+(\.\d+)*)?$")

#: Tool kinds whose *arguments* may be shell commands (so an agent that models a
#: shell tool as a generic call is still covered by ``auto_allow``).
EXECUTE_KINDS = ("execute", "bash", "shell", "run", "terminal", "exec")

_COMMAND_KEYS = ("command", "cmd", "script", "commandLine", "command_line", "shell")
_PATH_KEYS = ("file_path", "filePath", "path", "paths", "files", "target")

# --------------------------------------------------------------------------- subject


@dataclass(frozen=True)
class TierSubject:
    """Everything a tier may look at, flattened for matching."""

    kind: str = ""
    title: str = ""
    commands: tuple[str, ...] = ()
    arguments: tuple[str, ...] = ()
    locations: tuple[str, ...] = ()
    payload: str = ""

    @property
    def haystack(self) -> str:
        return " ".join(
            part for part in (self.kind, self.title, *self.commands, *self.arguments, *self.locations, self.payload) if part
        )


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _flatten(value: Any, *, limit: int = 16) -> list[str]:
    out: list[str] = []
    stack: list[Any] = [value]
    while stack and len(out) < limit:
        item = stack.pop(0)
        if isinstance(item, str):
            if item.strip():
                out.append(item)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return out


def _commands_of(raw_input: Any) -> list[str]:
    if isinstance(raw_input, str):
        return [raw_input] if raw_input.strip() else []
    if not isinstance(raw_input, Mapping):
        return []
    commands: list[str] = []
    for key in _COMMAND_KEYS:
        value = raw_input.get(key)
        if isinstance(value, str) and value.strip():
            commands.append(value)
        elif isinstance(value, (list, tuple)):
            commands.extend(part for part in value if isinstance(part, str) and part.strip())
    return commands


def _arguments_of(raw_input: Any) -> list[str]:
    """Generic tool arguments: every other string in the payload."""
    if not isinstance(raw_input, Mapping):
        return _flatten(raw_input) if not isinstance(raw_input, str) else []
    out: list[str] = []
    for key, value in raw_input.items():
        if key in _COMMAND_KEYS:
            continue
        out.extend(_flatten(value))
    return out


def _locations_of(tool_call: Mapping[str, Any]) -> list[str]:
    paths: list[str] = []
    locations = tool_call.get("locations")
    if isinstance(locations, (list, tuple)):
        for location in locations:
            if isinstance(location, Mapping):
                path = location.get("path") or location.get("file")
                if isinstance(path, str) and path.strip():
                    paths.append(path)
            elif isinstance(location, str) and location.strip():
                paths.append(location)
    raw_input = tool_call.get("rawInput")
    if isinstance(raw_input, Mapping):
        for key in _PATH_KEYS:
            value = raw_input.get(key)
            if isinstance(value, str) and value.strip():
                paths.append(value)
            elif isinstance(value, (list, tuple)):
                paths.extend(part for part in value if isinstance(part, str) and part.strip())
    out: list[str] = []
    for path in paths:
        if path not in out:
            out.append(path)
    return out


def subject_from_tool_call(
    tool_call: Mapping[str, Any] | None,
    *,
    title: str = "",
    kind: str = "",
) -> TierSubject:
    """Build a :class:`TierSubject` from an ACP ``toolCall`` payload."""
    tool = dict(tool_call or {})
    raw_input = tool.get("rawInput")
    resolved_kind = str(tool.get("kind") or tool.get("toolKind") or kind or "")
    resolved_title = str(tool.get("title") or tool.get("name") or title or "")
    commands = _commands_of(raw_input)
    payload = _stringify(raw_input)
    if not commands and isinstance(raw_input, str):
        commands = [raw_input]
    return TierSubject(
        kind=resolved_kind,
        title=resolved_title,
        commands=tuple(commands),
        arguments=tuple(_arguments_of(raw_input)),
        locations=tuple(_locations_of(tool)),
        payload=payload,
    )


def subject_from_request(request: "PermissionRequest") -> TierSubject:
    """Build a :class:`TierSubject` from a parsed ``session/request_permission``."""
    return subject_from_tool_call(
        getattr(request, "tool_call", None),
        title=str(getattr(request, "title", "") or ""),
        kind=str(getattr(request, "kind", "") or ""),
    )


# --------------------------------------------------------------------------- decisions


@dataclass(frozen=True)
class ApprovalDecision:
    """What to do with one permission request, and why."""

    action: str = ASK
    reason: str = ""
    tier: str = TIER_POSTURE
    pattern: str | None = None

    @property
    def auto_approve(self) -> bool:
        return self.action == AUTO_ALLOW

    @property
    def forced(self) -> bool:
        return self.tier == TIER_ALWAYS_ASK


@dataclass(frozen=True)
class TierRules:
    """The two lists, as resolved for one chat."""

    auto_allow: tuple[str, ...] = DEFAULT_AUTO_ALLOW
    always_ask: tuple[str, ...] = DEFAULT_ALWAYS_ASK


def parse_tier_list(raw: Any, *, default: Sequence[str]) -> tuple[str, ...]:
    """Parse a tier list from a comma-separated string, a sequence, or ``None``.

    ``None`` means "not configured" -> the documented defaults. An empty string
    or an empty list means "this tier is off", which is honoured literally.
    """
    if raw is None:
        return tuple(default)
    if isinstance(raw, str):
        items: Iterable[Any] = raw.replace(";", ",").split(",") if raw.strip() else []
    else:
        items = raw
    out: list[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def normalize_posture(raw: Any) -> str:
    """``/aprobar`` value -> ``ask`` | ``auto``.

    ``preguntar``/``ask`` mean "tap for anything not in ``auto_allow``";
    ``auto``/``yolo`` mean "silent for anything not in ``always_ask``". Anything
    unrecognised falls back to ``ask``: the cautious direction.
    """
    return _POSTURE_NAMES.get(str(raw or "").strip().lower(), POSTURE_ASK)


# --------------------------------------------------------------------------- tiers


class ApprovalTiers:
    """Resolve a permission request to auto-approve or a tap, with a reason."""

    def __init__(
        self,
        *,
        auto_allow: Sequence[str] | None = None,
        always_ask: Sequence[str] | None = None,
        posture: str = "ask",
        overrides: Mapping[Any, Mapping[str, Any]] | None = None,
        postures: Mapping[Any, str] | None = None,
        log: logging.Logger | None = None,
    ) -> None:
        self.auto_allow = tuple(auto_allow) if auto_allow is not None else DEFAULT_AUTO_ALLOW
        self.always_ask = tuple(always_ask) if always_ask is not None else DEFAULT_ALWAYS_ASK
        self.posture = (posture or "ask").strip().lower()
        self.log = log or _logger
        self.overrides: dict[int, TierRules] = {}
        for chat_id, rules in (overrides or {}).items():
            try:
                key = int(chat_id)
            except (TypeError, ValueError):
                continue
            self.overrides[key] = _as_rules(rules, self.auto_allow, self.always_ask)
        #: Per-chat posture (``/aprobar``): ``ask`` | ``auto``. Absent -> the
        #: global default. Kept in sync with the persisted binding by the gateway.
        self._postures: dict[int, str] = {}
        for chat_id, value in (postures or {}).items():
            self.set_posture(chat_id, value)

    # ------------------------------------------------------------------ wiring

    @classmethod
    def from_config(cls, config: Any, *, log: logging.Logger | None = None) -> "ApprovalTiers":
        """Build from a :class:`acp_im_gateway.config.Config` (duck-typed)."""
        return cls(
            auto_allow=getattr(config, "auto_allow", None),
            always_ask=getattr(config, "always_ask", None),
            posture=getattr(config, "approval_posture", "ask") or "ask",
            overrides=getattr(config, "tier_overrides", None),
            log=log,
        )

    def rules_for(self, chat_id: Any) -> TierRules:
        try:
            key = int(chat_id)
        except (TypeError, ValueError):
            return TierRules(auto_allow=self.auto_allow, always_ask=self.always_ask)
        override = self.overrides.get(key)
        if override is not None:
            return override
        return TierRules(auto_allow=self.auto_allow, always_ask=self.always_ask)

    # ------------------------------------------------------------------ posture

    def posture_for(self, chat_id: Any) -> str:
        """The effective chat posture: the per-chat ``/aprobar`` value or the default."""
        try:
            key = int(chat_id)
        except (TypeError, ValueError):
            return self.posture
        return self._postures.get(key, self.posture)

    def set_posture(self, chat_id: Any, posture: str) -> str:
        """Record a chat's posture (``/aprobar``). Returns the normalised value.

        ``preguntar``/``ask`` -> ``ask``, ``auto``/``yolo`` -> ``auto``. Only the
        *gateway's* answer changes: the agent's own posture is never touched, so
        ``always_ask`` still sees every request (see
        ``Gateway._apply_approval_posture``).
        """
        value = normalize_posture(posture)
        try:
            key = int(chat_id)
        except (TypeError, ValueError):
            return value
        self._postures[key] = value
        return value

    def clear_posture(self, chat_id: Any) -> None:
        try:
            key = int(chat_id)
        except (TypeError, ValueError):
            return
        self._postures.pop(key, None)

    # ------------------------------------------------------------------ deciding

    def decide(self, chat_id: Any, subject: TierSubject) -> ApprovalDecision:
        """Return the tier decision for one request, and log it at INFO."""
        decision = self.evaluate(chat_id, subject)
        self.log.info(
            "approval tier: %s (%s) in chat %s -> %s: %s",
            subject.title or subject.kind or "tool call",
            subject.kind or "unknown kind",
            chat_id,
            decision.action,
            decision.reason,
        )
        return decision

    def evaluate(self, chat_id: Any, subject: TierSubject) -> ApprovalDecision:
        """Decide without logging (``decide`` adds the INFO line)."""
        rules = self.rules_for(chat_id)

        forced = self._always_ask_hit(rules, subject)
        if forced is not None:
            return ApprovalDecision(
                action=ASK,
                reason=f"always_ask gate: {forced!r} is money or irreversible, so a tap is required",
                tier=TIER_ALWAYS_ASK,
                pattern=forced,
            )

        allowed = self._auto_allow_hit(rules, subject)
        if allowed is not None:
            return ApprovalDecision(
                action=AUTO_ALLOW,
                reason=f"auto_allow: {allowed!r} is read-only",
                tier=TIER_AUTO_ALLOW,
                pattern=allowed,
            )

        posture = self.posture_for(chat_id)
        if posture in _WIDENING_POSTURES:
            return ApprovalDecision(
                action=AUTO_ALLOW,
                reason=f"no tier matched; posture {posture!r} approves without a tap",
                tier=TIER_POSTURE,
            )
        return ApprovalDecision(
            action=ASK,
            reason=f"no tier matched; posture {posture!r} asks",
            tier=TIER_POSTURE,
        )

    # ------------------------------------------------------------------ matching

    def _always_ask_hit(self, rules: TierRules, subject: TierSubject) -> str | None:
        haystack = subject.haystack
        for pattern in rules.always_ask:
            if pattern.strip().lower() == CURL_PATTERN:
                # `curl` gates only when it would *create* data (-d/--data/-F/
                # -X POST…). A bare `curl https://…` GET is not money either way:
                # it is not in auto_allow, so it still ends in a tap.
                if curl_creates_data(subject.commands):
                    return f"{CURL_PATTERN} {DATA_CREATING_LABEL}"
                continue
            if pattern.strip().lower() == RM_RECURSIVE_PATTERN:
                if rm_forced_recursive(subject.commands):
                    return f"{RM_RECURSIVE_PATTERN} {RM_RECURSIVE_LABEL}"
                if keyword_match(pattern, haystack):
                    return pattern
                continue
            if keyword_match(pattern, haystack):
                return pattern
        return None

    def _auto_allow_hit(self, rules: TierRules, subject: TierSubject) -> str | None:
        """Every simple command in a candidate must be harmless and read-only.

        ``git status && git diff`` is approved. ``git log | sh``,
        ``cat x > /etc/passwd``, ``ls $(rm -rf y)``, ``sudo cat /etc/shadow`` and
        ``find . -delete`` are not: a whitelist that only looked at the first word
        of a line would quietly bless the rest of it. ``cd app && pytest -q`` and
        ``./.venv/bin/python -m pytest`` *are* approved, because ``cd`` is a
        harmless prefix and the interpreter path is normalised to ``python``. A
        false negative here costs a tap; a false positive silences one.
        """
        patterns: list[tuple[list[str], str]] = [
            (tokens, pattern) for pattern in rules.auto_allow if (tokens := _tokens(pattern))
        ]
        if not patterns:
            return None
        candidates = list(subject.commands)
        if subject.kind.strip().lower() in EXECUTE_KINDS:
            candidates.extend(subject.arguments)
        for candidate in candidates:
            simples = shell_commands(candidate) or [candidate]
            hits: list[str] = []
            for simple in simples:
                hit = _harmless_pattern(simple, patterns)
                if hit is None:
                    hits = []
                    break
                hits.append(hit)
            if hits:
                return hits[0]
        return None


def _as_rules(raw: Any, auto_allow: Sequence[str], always_ask: Sequence[str]) -> TierRules:
    """Accept a ``TierRules`` or a plain ``{"auto_allow": …, "always_ask": …}`` mapping."""
    if isinstance(raw, TierRules):
        return raw
    mapping = raw if isinstance(raw, Mapping) else {}
    return TierRules(
        auto_allow=parse_tier_list(mapping.get("auto_allow"), default=auto_allow),
        always_ask=parse_tier_list(mapping.get("always_ask"), default=always_ask),
    )


def _is_read_only(command: str) -> bool:
    """False for anything a whitelist must not silently approve."""
    if any(fragment in command for fragment in _WRITE_FRAGMENTS):
        return False
    return _tokens_read_only(_split_command(command))


def _tokens_read_only(tokens: Sequence[str]) -> bool:
    """Scope-aware destructive-flag check on one command's argv.

    Run on the *effective* argv too (after ``time``/``nice`` are peeled), so
    ``time git branch -d main`` cannot hide ``git branch``'s scope rule.
    """
    scope = _scope(list(tokens))
    return not any(_is_destructive(token, scope) for token in tokens)


def _scope(tokens: list[str]) -> str:
    """``git branch`` for ``git branch -d`` — the command, without env assignments."""
    real = [
        os.path.basename(token)
        for token in tokens
        if not ("=" in token and not token.startswith("-"))
    ]
    return " ".join(real[:2])


def _is_destructive(token: str, scope: str = "") -> bool:
    """``--delete`` and ``--delete=…`` are the same flag; ``-d`` depends on the command."""
    name = token.partition("=")[0] if token.startswith("--") else token
    if name in _DESTRUCTIVE_TOKENS:
        return True
    scopes = _DESTRUCTIVE_BY_SCOPE.get(name)
    return bool(scopes) and scope in scopes


def _tokens(pattern: str) -> list[str]:
    text = (pattern or "").strip()
    if not text:
        return []
    try:
        import shlex

        return shlex.split(text)
    except ValueError:
        return text.split()


def _split_command(command: str) -> list[str]:
    try:
        import shlex

        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def _is_env_assignment(token: str) -> bool:
    """``FOO=1`` is an assignment; ``-x``/``/abs/path`` are not."""
    return "=" in token and not token.startswith("-") and not token.startswith("/")


def _effective_tokens(tokens: list[str]) -> tuple[list[str], str | None]:
    """Strip leading env assignments and harmless prefixes from one command.

    Returns ``(rest, harmless)``. ``harmless`` is the label of what swallowed the
    whole segment (``cd app`` -> ``"cd"``, ``FOO=1`` -> ``"FOO"``) when nothing
    but assignments/prefixes remained; a command wrapper (``time``, ``nice``) is
    peeled off and the *real* head is returned in ``rest`` so it can be matched.
    """
    rest = list(tokens)
    label: str | None = None
    for _ in range(_MAX_PREFIX_DEPTH):
        while rest and _is_env_assignment(rest[0]):
            label = label or rest[0].split("=", 1)[0]
            rest.pop(0)
        if not rest:
            return [], label or "no-op"
        head = os.path.basename(rest[0]).lower()
        if head in _HARMLESS_PREFIXES:
            return [], head
        if head not in _WRAPPER_PREFIXES:
            return rest, None
        label = head
        rest.pop(0)
        if head == "nice":
            # `nice -n 10` / `nice --adjustment=10` / `nice 10`: options and the
            # adjustment value, then the real command.
            while rest and (rest[0].startswith("-") or rest[0].isdigit()):
                rest.pop(0)
        else:  # time: flags only, then the real command
            while rest and rest[0].startswith("-"):
                rest.pop(0)
    return rest, None


def _normalise_head(tokens: Sequence[str], length: int) -> list[str]:
    """Lower-case head tokens with the interpreter path normalised.

    ``./.venv/bin/python3.12`` and ``python3`` both become ``python``, so the
    three spellings of the same pytest run match one ``auto_allow`` entry.
    """
    out: list[str] = []
    for index, token in enumerate(tokens[:length]):
        text = token.lower()
        if index == 0:
            text = os.path.basename(text)
            if _PYTHON_HEAD.match(text):
                text = "python"
        out.append(text)
    return out


def _harmless_pattern(simple: str, patterns: Sequence[tuple[list[str], str]]) -> str | None:
    """The ``auto_allow`` entry that blesses ``simple``, or None to keep asking.

    Read-only first: a redirect, a substitution or a destructive flag disqualifies
    the segment whatever its head is.
    """
    if not _is_read_only(simple):
        return None
    rest, harmless = _effective_tokens(_split_command(simple))
    if harmless is not None:
        return harmless
    # Re-check the peeled argv: a wrapper must not hide a scope-gated flag.
    if not _tokens_read_only(rest):
        return None
    for tokens, pattern in patterns:
        if _normalise_head(rest, len(tokens)) == _normalise_head(tokens, len(tokens)):
            return pattern
    return None


def decide_request(
    tiers: ApprovalTiers | None,
    chat_id: Any,
    request: "PermissionRequest",
) -> ApprovalDecision | None:
    """Convenience wrapper for the approval bridge; ``None`` means "just ask"."""
    if tiers is None:
        return None
    try:
        return tiers.decide(chat_id, subject_from_request(request))
    except Exception:  # pragma: no cover - defensive: never break a turn
        tiers.log.exception("could not evaluate the approval tiers; asking instead")
        return None


def shell_heads(subject: TierSubject) -> list[str]:
    """Every simple command a subject would run (for logs and tests)."""
    out: list[str] = []
    for command in subject.commands:
        out.extend(shell_commands(command))
    return out
