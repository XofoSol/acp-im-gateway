"""Configuration loading: defaults <- optional TOML file <- environment <- CLI overrides.

Nothing personal lives in this file: every value is either a neutral default
(``~/Projects``, ``reasonix acp``) or comes from the environment / a TOML file the
operator owns. ``.env.example`` documents every variable.
"""

from __future__ import annotations

import os
import shlex
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .render import DEFAULT_OVERFLOW_CHARS, DEFAULT_TOOL_OUTPUT_LINES
from .tiers import (
    DEFAULT_ALWAYS_ASK,
    DEFAULT_AUTO_ALLOW,
    TierRules,
    parse_tier_list,
)

DEFAULT_AGENT_CMD = "reasonix acp"
DEFAULT_PROJECTS_ROOT = "~/Projects"
DEFAULT_AGENT_INDEX_DIR = "~/.reasonix/projects"
DEFAULT_STATE_FILE = "~/.local/state/acp-im-gateway/state.json"
DEFAULT_CONFIG_FILE = "~/.config/acp-im-gateway/config.toml"
DEFAULT_ENV_FILE = ".env"
DEFAULT_TELEGRAM_API_BASE = "https://api.telegram.org"

BUSY_MODES = ("steer", "queue")
#: Tool-approval postures an ACP agent commonly advertises. "" = leave it alone.
APPROVAL_POSTURES = ("ask", "auto", "yolo", "")

class ConfigError(Exception):
    """Raised when the configuration cannot be used as given."""


def expand_path(raw: str | os.PathLike[str], base: Path | None = None) -> Path:
    """Expand ``~`` and make ``raw`` absolute (relative paths resolve against ``base``)."""
    text = os.fspath(raw).strip()
    if not text:
        raise ConfigError("empty path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base or Path.cwd()) / path
    return Path(os.path.normpath(path))


def parse_id_list(raw: str | Iterable[Any] | None) -> frozenset[int]:
    """Parse ``"1, 2,3"`` / ``[1, "2"]`` into a set of ids. Bad entries raise."""
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        items: list[Any] = [chunk for chunk in raw.replace(";", ",").split(",")]
    else:
        items = list(raw)
    out: set[int] = set()
    for item in items:
        if isinstance(item, str):
            item = item.strip()
            if not item:
                continue
        try:
            out.add(int(item))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"not a numeric id: {item!r}") from exc
    return frozenset(out)


def parse_path_list(raw: str | Iterable[Any] | None, base: Path | None = None) -> tuple[Path, ...]:
    """Parse a comma-separated (or iterable) list of paths."""
    if raw is None:
        return ()
    if isinstance(raw, str):
        items: list[Any] = [chunk for chunk in raw.replace(";", ",").split(",") if chunk.strip()]
    else:
        items = list(raw)
    return tuple(expand_path(str(item), base) for item in items)


def split_command(raw: str | Iterable[str]) -> tuple[str, ...]:
    """Split an agent command string into argv (``reasonix acp`` -> ``("reasonix", "acp")``)."""
    if isinstance(raw, str):
        argv = tuple(shlex.split(raw))
    else:
        argv = tuple(str(part) for part in raw)
    if not argv:
        raise ConfigError("agent command is empty")
    return argv


def parse_bool(raw: Any, default: bool = False) -> bool:
    """Tolerant truthiness for env vars / TOML values."""
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off", ""):
        return False
    return default


def load_dotenv(path: Path) -> dict[str, str]:
    """Read a simple ``KEY=VALUE`` .env file. Missing file -> ``{}``.

    Handles comments, blank lines, optional ``export`` prefix and single/double
    quoted values. It is deliberately tiny: no variable interpolation, no magic.
    """
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def load_toml(path: Path) -> dict[str, Any]:
    """Read an optional TOML config file. Missing file -> ``{}``."""
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc


@dataclass(frozen=True)
class Config:
    """Effective gateway configuration."""

    # Telegram
    telegram_bot_token: str = ""
    telegram_api_base: str = DEFAULT_TELEGRAM_API_BASE
    allowed_user_ids: frozenset[int] = frozenset()
    allowed_chat_ids: frozenset[int] = frozenset()

    # Paths
    projects_root: Path = field(default_factory=lambda: Path(DEFAULT_PROJECTS_ROOT).expanduser())
    allowed_roots: tuple[Path, ...] = ()
    agent_index_dir: Path = field(default_factory=lambda: Path(DEFAULT_AGENT_INDEX_DIR).expanduser())
    state_file: Path = field(default_factory=lambda: Path(DEFAULT_STATE_FILE).expanduser())

    # Agent
    agent_cmd: tuple[str, ...] = ("reasonix", "acp")
    turn_timeout: float = 0.0
    restart_backoff_max: float = 30.0

    # Runtime
    edit_interval: float = 1.2
    send_interval: float = 1.0
    poll_timeout: int = 30
    busy_mode: str = "steer"
    #: Default *gateway* chat posture before a chat runs ``/aprobar`` (``ask`` or
    #: ``auto``). It never changes the agent's own ``tool_approval`` posture: the
    #: agent stays in ``ask`` so the ``always_ask`` money gate always sees the call.
    approval_posture: str = "ask"
    pairing_ttl: float = 900.0
    approval_timeout: float = 300.0
    discovery_depth: int = 1
    log_level: str = "INFO"
    dry_run: bool = False

    # Rendering (Part A of the v1.1 spec: the chat must read like the CLI)
    tool_output_lines: int = DEFAULT_TOOL_OUTPUT_LINES
    overflow_chars: int = DEFAULT_OVERFLOW_CHARS
    show_thinking: bool = True
    heartbeat_seconds: float = 60.0

    # Approval tiers (Part B): global defaults, overridable per chat.
    auto_allow: tuple[str, ...] = DEFAULT_AUTO_ALLOW
    always_ask: tuple[str, ...] = DEFAULT_ALWAYS_ASK
    tier_overrides: Mapping[int, TierRules] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers

    def resolved_roots(self) -> tuple[Path, ...]:
        """Roots a chat may bind inside: ``ALLOWED_ROOTS`` or ``PROJECTS_ROOT``."""
        return tuple(self.allowed_roots) or (self.projects_root,)

    def safe_table(self) -> list[tuple[str, str]]:
        """Human-readable effective config, with the bot token redacted."""
        token = "set" if self.telegram_bot_token else "missing"
        return [
            ("telegram_bot_token", token),
            ("telegram_api_base", self.telegram_api_base),
            ("allowed_user_ids", ",".join(str(i) for i in sorted(self.allowed_user_ids)) or "(none)"),
            ("allowed_chat_ids", ",".join(str(i) for i in sorted(self.allowed_chat_ids)) or "(DMs only)"),
            ("projects_root", str(self.projects_root)),
            ("allowed_roots", ",".join(str(p) for p in self.resolved_roots())),
            ("agent_index_dir", str(self.agent_index_dir)),
            ("state_file", str(self.state_file)),
            ("agent_cmd", " ".join(self.agent_cmd)),
            ("turn_timeout", str(self.turn_timeout)),
            ("edit_interval", str(self.edit_interval)),
            ("send_interval", str(self.send_interval)),
            ("poll_timeout", str(self.poll_timeout)),
            ("busy_mode", self.busy_mode),
            ("approval_posture", self.approval_posture or "(agent default)"),
            ("pairing_ttl", str(self.pairing_ttl)),
            ("approval_timeout", str(self.approval_timeout)),
            ("discovery_depth", str(self.discovery_depth)),
            ("log_level", self.log_level),
            ("dry_run", str(self.dry_run)),
            ("tool_output_lines", str(self.tool_output_lines)),
            ("overflow_chars", str(self.overflow_chars)),
            ("show_thinking", str(self.show_thinking)),
            ("heartbeat_seconds", str(self.heartbeat_seconds)),
            ("auto_allow", ",".join(self.auto_allow) or "(none)"),
            ("always_ask", ",".join(self.always_ask) or "(none)"),
            (
                "tier_overrides",
                ",".join(str(chat) for chat in sorted(self.tier_overrides)) or "(none)",
            ),
        ]

    def with_overrides(self, **overrides: Any) -> "Config":
        """Return a copy with ``None`` values ignored (handy for CLI flags)."""
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean) if clean else self

    def validate(self, *, require_token: bool = True) -> None:
        """Fail fast with an actionable message instead of misbehaving at runtime."""
        problems: list[str] = []
        if require_token and not self.telegram_bot_token:
            problems.append(
                "TELEGRAM_BOT_TOKEN is not set (create a bot with @BotFather, then put the "
                "token in .env or the environment)"
            )
        if self.busy_mode not in BUSY_MODES:
            problems.append(f"GATEWAY_BUSY_MODE must be one of {BUSY_MODES}, got {self.busy_mode!r}")
        if self.approval_posture not in APPROVAL_POSTURES:
            problems.append(
                f"GATEWAY_APPROVAL_POSTURE must be one of {APPROVAL_POSTURES}, "
                f"got {self.approval_posture!r}"
            )
        for name, value in (
            ("GATEWAY_EDIT_INTERVAL", self.edit_interval),
            ("GATEWAY_SEND_INTERVAL", self.send_interval),
            ("GATEWAY_PAIRING_TTL", self.pairing_ttl),
            ("GATEWAY_APPROVAL_TIMEOUT", self.approval_timeout),
        ):
            if value < 0:
                problems.append(f"{name} must be >= 0, got {value!r}")
        if self.poll_timeout < 1:
            problems.append(f"GATEWAY_POLL_TIMEOUT must be >= 1, got {self.poll_timeout!r}")
        if self.discovery_depth < 0:
            problems.append(f"GATEWAY_DISCOVERY_DEPTH must be >= 0, got {self.discovery_depth!r}")
        if self.tool_output_lines < 1:
            problems.append(
                f"GATEWAY_TOOL_OUTPUT_LINES must be >= 1, got {self.tool_output_lines!r}"
            )
        if self.overflow_chars < 512:
            problems.append(
                f"GATEWAY_OVERFLOW_CHARS must be >= 512 (and well under 4096), "
                f"got {self.overflow_chars!r}"
            )
        if self.heartbeat_seconds < 0:
            problems.append(
                f"GATEWAY_HEARTBEAT_SECONDS must be >= 0, got {self.heartbeat_seconds!r}"
            )
        if not self.agent_cmd:
            problems.append("REASONIX_ACP_CMD is empty")
        if problems:
            raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))

    # ------------------------------------------------------------------ factories

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        toml_path: Path | None = None,
        env_file: Path | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> "Config":
        """Build a config from defaults, an optional TOML file, then the environment.

        ``env`` defaults to ``os.environ`` merged under a ``.env`` file (real
        environment wins over the file, matching common practice).
        """
        environ: dict[str, str] = {}
        base_dir = Path.cwd()
        if env is None:
            if env_file is None:
                env_file = expand_path(os.environ.get("GATEWAY_ENV_FILE") or DEFAULT_ENV_FILE, base_dir)
            # A real environment variable always wins over the .env file.
            environ.update(load_dotenv(env_file))
            environ.update({key: value for key, value in os.environ.items()})
        else:
            environ.update({key: str(value) for key, value in env.items()})

        if toml_path is None:
            toml_path = expand_path(environ.get("GATEWAY_CONFIG") or DEFAULT_CONFIG_FILE, base_dir)
        data = load_toml(toml_path)

        tg = dict(data.get("telegram") or {})
        paths = dict(data.get("paths") or {})
        agent = dict(data.get("agent") or {})
        runtime = dict(data.get("runtime") or {})
        approvals = dict(data.get("approvals") or {})

        def raw(env_name: str, section: Mapping[str, Any], key: str) -> Any:
            """Environment value (if non-empty) else TOML value else None."""
            value = environ.get(env_name)
            if value is not None and value != "":
                return value
            return section.get(key)

        def text(env_name: str, section: Mapping[str, Any], key: str, default: str) -> str:
            value = raw(env_name, section, key)
            return default if value is None or value == "" else str(value)

        def number(env_name: str, section: Mapping[str, Any], key: str, default: float) -> float:
            value = raw(env_name, section, key)
            if value is None or value == "":
                return default
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{env_name} must be a number, got {value!r}") from exc

        def tier_list(env_name: str, key: str, default: tuple[str, ...]) -> tuple[str, ...]:
            """Tier lists are lists, not scalars: an *explicitly empty* env value
            means "turn this tier off", while an absent one means the defaults."""
            if env_name in environ:
                return parse_tier_list(environ.get(env_name), default=default)
            return parse_tier_list(approvals.get(key), default=default)

        cfg = cls(
            telegram_bot_token=text("TELEGRAM_BOT_TOKEN", tg, "bot_token", ""),
            telegram_api_base=text(
                "TELEGRAM_API_BASE", tg, "api_base", DEFAULT_TELEGRAM_API_BASE
            ).rstrip("/"),
            allowed_user_ids=parse_id_list(raw("ALLOWED_USER_IDS", tg, "allowed_user_ids")),
            allowed_chat_ids=parse_id_list(raw("ALLOWED_CHAT_IDS", tg, "allowed_chat_ids")),
            projects_root=expand_path(
                text("PROJECTS_ROOT", paths, "projects_root", DEFAULT_PROJECTS_ROOT), base_dir
            ),
            allowed_roots=parse_path_list(raw("ALLOWED_ROOTS", paths, "allowed_roots"), base_dir),
            agent_index_dir=expand_path(
                text("REASONIX_PROJECTS_DIR", paths, "agent_index", DEFAULT_AGENT_INDEX_DIR), base_dir
            ),
            state_file=expand_path(
                text("GATEWAY_STATE_FILE", paths, "state_file", DEFAULT_STATE_FILE), base_dir
            ),
            agent_cmd=split_command(
                text("REASONIX_ACP_CMD", agent, "command", DEFAULT_AGENT_CMD)
            ),
            turn_timeout=number("GATEWAY_TURN_TIMEOUT", agent, "turn_timeout", 0.0),
            restart_backoff_max=number("GATEWAY_RESTART_BACKOFF_MAX", agent, "restart_backoff_max", 30.0),
            edit_interval=number("GATEWAY_EDIT_INTERVAL", runtime, "edit_interval", 1.2),
            send_interval=number("GATEWAY_SEND_INTERVAL", runtime, "send_interval", 1.0),
            poll_timeout=int(number("GATEWAY_POLL_TIMEOUT", runtime, "poll_timeout", 30)),
            busy_mode=text("GATEWAY_BUSY_MODE", runtime, "busy_mode", "steer").strip().lower(),
            approval_posture=text(
                "GATEWAY_APPROVAL_POSTURE", runtime, "approval_posture", "ask"
            ).strip().lower(),
            pairing_ttl=number("GATEWAY_PAIRING_TTL", runtime, "pairing_ttl", 900.0),
            approval_timeout=number("GATEWAY_APPROVAL_TIMEOUT", runtime, "approval_timeout", 300.0),
            discovery_depth=int(number("GATEWAY_DISCOVERY_DEPTH", runtime, "discovery_depth", 1)),
            log_level=text("GATEWAY_LOG_LEVEL", runtime, "log_level", "INFO").upper(),
            dry_run=parse_bool(raw("GATEWAY_DRY_RUN", runtime, "dry_run"), False),
            tool_output_lines=int(
                number(
                    "GATEWAY_TOOL_OUTPUT_LINES",
                    runtime,
                    "tool_output_lines",
                    DEFAULT_TOOL_OUTPUT_LINES,
                )
            ),
            overflow_chars=int(
                number("GATEWAY_OVERFLOW_CHARS", runtime, "overflow_chars", DEFAULT_OVERFLOW_CHARS)
            ),
            show_thinking=parse_bool(
                raw("GATEWAY_SHOW_THINKING", runtime, "show_thinking"),
                True,
            ),
            heartbeat_seconds=number(
                "GATEWAY_HEARTBEAT_SECONDS", runtime, "heartbeat_seconds", 60.0
            ),
            auto_allow=tier_list("GATEWAY_AUTO_ALLOW", "auto_allow", DEFAULT_AUTO_ALLOW),
            always_ask=tier_list("GATEWAY_ALWAYS_ASK", "always_ask", DEFAULT_ALWAYS_ASK),
            tier_overrides=_tier_overrides(approvals.get("chats")),
        )
        return cfg.with_overrides(**(dict(overrides) if overrides else {}))


def _tier_overrides(raw: Any) -> dict[int, TierRules]:
    """Per-chat tier overrides from ``[approvals.chats."<chat id>"]`` blocks."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[int, TierRules] = {}
    for chat_id, rules in raw.items():
        try:
            key = int(chat_id)
        except (TypeError, ValueError):
            raise ConfigError(f"[approvals.chats] key must be a numeric chat id, got {chat_id!r}")
        if not isinstance(rules, Mapping):
            raise ConfigError(f"[approvals.chats.{chat_id}] must be a table")
        out[key] = TierRules(
            auto_allow=parse_tier_list(rules.get("auto_allow"), default=DEFAULT_AUTO_ALLOW),
            always_ask=parse_tier_list(rules.get("always_ask"), default=DEFAULT_ALWAYS_ASK),
        )
    return out
