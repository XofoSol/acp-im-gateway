"""Approval tiers: what may be approved silently, and what must always ask.

Two lists are evaluated on every ``session/request_permission``:

* ``always_ask`` — money or irreversible. This is a **code gate**, not a
  posture: it forces a tap even when the chat posture is ``auto``/``yolo``, and
  it wins over ``auto_allow``. A command that is both harmless and dangerous
  (``git status && rm -rf build``) is dangerous.
* ``auto_allow`` — harmless and read-only commands, approved without a tap so
  the phone is not asked to confirm ``git status`` for the tenth time.

Anything else follows the chat posture (``GATEWAY_APPROVAL_POSTURE``, default
``ask``).

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

Every decision is logged at INFO with its reason, so the journal explains why
something was auto-approved or why it was forced to a tap.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from .render import curl_creates_data, keyword_match, rm_forced_recursive, shell_commands
from .render import command_head as _head_matches

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids an import cycle)
    from .approvals import PermissionRequest

_logger = logging.getLogger("acp_im_gateway.tiers")

AUTO_ALLOW = "auto_allow"
ASK = "ask"

TIER_ALWAYS_ASK = "always_ask"
TIER_AUTO_ALLOW = "auto_allow"
TIER_POSTURE = "posture"

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

        if self.posture in ("auto", "yolo"):
            return ApprovalDecision(
                action=AUTO_ALLOW,
                reason=f"no tier matched; posture {self.posture!r} approves without a tap",
                tier=TIER_POSTURE,
            )
        return ApprovalDecision(
            action=ASK,
            reason=f"no tier matched; posture {self.posture!r} asks",
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
        """Every simple command in a candidate must be on the list, and read-only.

        ``git status && git diff`` is approved. ``git log | sh``,
        ``cd app && git status``, ``cat x > /etc/passwd``, ``ls $(rm -rf y)``,
        ``sudo cat /etc/shadow`` and ``find . -delete`` are not: a whitelist that
        only looked at the first word of a line would quietly bless the rest of it.
        A false negative here costs a tap; a false positive silences one.
        """
        patterns: list[tuple[list[str], str]] = [(tokens, pattern) for pattern in rules.auto_allow if (tokens := _tokens(pattern))]
        if not patterns:
            return None
        candidates = list(subject.commands)
        if subject.kind.strip().lower() in EXECUTE_KINDS:
            candidates.extend(subject.arguments)
        for candidate in candidates:
            simples = shell_commands(candidate) or [candidate]
            hits: list[str] = []
            for simple in simples:
                if not _is_read_only(simple):
                    hits = []
                    break
                hit = next(
                    (pattern for tokens, pattern in patterns if _head_matches(simple, tokens)), None
                )
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
    try:
        import shlex

        tokens = shlex.split(command, posix=True)
    except ValueError:
        tokens = command.split()
    scope = _scope(tokens)
    return not any(_is_destructive(token, scope) for token in tokens)


def _scope(tokens: list[str]) -> str:
    """``git branch`` for ``git branch -d`` — the command, without env assignments."""
    import os

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
