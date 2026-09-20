"""Bindings, project discovery, JSON state persistence and per-chat serialisation.

A *binding* maps a chat to a project root and (once known) an ACP session id, so
a gateway restart resumes where the user left off. The binding table, the Telegram
polling offset, the allowlist and pending pairing codes all live in one JSON state
file (``GATEWAY_STATE_FILE``, chmod 600).

Project candidates come from two generic sources, never a hand-written list:

1. ``PROJECTS_ROOT`` (default ``~/Projects``) — directories containing ``.git``;
2. the agent's own on-disk project index (default ``~/.reasonix/projects``) —
   directory names are an encoded path whose dashes decode back to ``/``.

The index source is a hint only: an entry that does not exist on disk is ignored.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .access import AccessPolicy
from .containment import resolve_path, resolve_root

STATE_VERSION = 1
SKIP_DIR_NAMES = frozenset(
    {
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".git",
        "dist",
        "build",
        "target",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".idea",
        ".vscode",
    }
)

_logger = logging.getLogger("acp_im_gateway.router")


class DiscoveryError(Exception):
    """A requested project name/path could not be resolved to one candidate."""


# --------------------------------------------------------------------------- state file


class StateStore:
    """Atomic, corrupt-file-tolerant JSON state persistence."""

    def __init__(self, path: str | os.PathLike[str], *, log: logging.Logger | None = None) -> None:
        self.path = Path(path).expanduser()
        self.log = log or _logger
        self._mtime: float | None = None

    def exists(self) -> bool:
        return self.path.is_file()

    def mtime(self) -> float | None:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    def load(self) -> dict[str, Any]:
        """Return the stored state, or ``{}`` when missing/unreadable/corrupt."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            self._mtime = None
            return {}
        except OSError as exc:
            self.log.warning("cannot read state file %s: %s", self.path, exc)
            return {}
        self._mtime = self.mtime()
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.log.error("state file %s is corrupt (%s); starting from empty state", self.path, exc)
            return {}
        if not isinstance(data, dict):
            self.log.error("state file %s does not contain an object; ignoring", self.path)
            return {}
        return data

    def save(self, state: Mapping[str, Any]) -> None:
        """Write atomically (tmp file + rename) with 0600 permissions."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(state, indent=2, sort_keys=True)
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)
        self._mtime = self.mtime()

    def changed_on_disk(self) -> bool:
        """True when another process (the CLI) rewrote the file since our last read."""
        current = self.mtime()
        return current is not None and current != self._mtime


# --------------------------------------------------------------------------- bindings


@dataclass
class Binding:
    """A chat's project + session, persisted so restarts can resume.

    ``approval`` holds the chat's **gateway** posture (``/aprobar``: ``ask`` or
    ``auto``), not the agent's; the agent is pinned to ``ask`` so the money gate
    always sees the request. ``None`` means the global default.
    """

    chat_id: int
    project_name: str
    project_root: str
    session_id: str | None = None
    model: str | None = None
    mode: str | None = None
    approval: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def touch(self) -> None:
        self.updated_at = time.time()

    @property
    def posture(self) -> str:
        """The chat posture: ``ask`` (default) or ``auto``."""
        return "auto" if str(self.approval or "").strip().lower() == "auto" else "ask"

    def short_session(self, length: int = 8) -> str:
        if not self.session_id:
            return "(none)"
        return self.session_id[:length]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "project_name": self.project_name,
            "project_root": self.project_root,
            "session_id": self.session_id,
            "model": self.model,
            "mode": self.mode,
            "approval": self.approval,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Binding":
        return cls(
            chat_id=int(data["chat_id"]),
            project_name=str(data.get("project_name") or Path(str(data["project_root"])).name),
            project_root=str(data["project_root"]),
            session_id=data.get("session_id"),
            model=data.get("model"),
            mode=data.get("mode"),
            approval=data.get("approval"),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
        )


# --------------------------------------------------------------------------- discovery


@dataclass(frozen=True)
class Project:
    """A discovered project candidate."""

    name: str
    path: Path
    source: str  # "projects_root" | "agent_index"

    @property
    def root(self) -> Path:
        """Fully resolved path (used for comparisons and containment checks)."""
        return resolve_path(self.path)

    def exists(self) -> bool:
        return self.path.is_dir()


def decode_index_name(name: str) -> Path:
    """Decode an agent index directory name back to the path it encodes.

    ``-home-dev-Projects-app`` -> ``/home/dev/Projects/app``. The encoding is
    lossy for real dashes in directory names, so callers must treat the result as
    a hint and verify it exists on disk.
    """
    text = name[1:] if name.startswith("-") else name
    return Path("/" + text.replace("-", "/"))


class ProjectDiscovery:
    """Find project candidates from the filesystem and the agent's project index."""

    def __init__(
        self,
        projects_root: str | os.PathLike[str],
        *,
        agent_index_dir: str | os.PathLike[str] | None = None,
        max_depth: int = 1,
        allowed_roots: Iterable[str | os.PathLike[str]] = (),
        skip_names: Iterable[str] = SKIP_DIR_NAMES,
        log: logging.Logger | None = None,
    ) -> None:
        self.projects_root = Path(projects_root).expanduser()
        self.agent_index_dir = None if agent_index_dir is None else Path(agent_index_dir).expanduser()
        self.max_depth = max(int(max_depth), 1)
        self.allowed_roots = tuple(Path(root).expanduser() for root in allowed_roots)
        self.skip_names = frozenset(skip_names)
        self.log = log or _logger

    # ------------------------------------------------------------------ sources

    def from_projects_root(self) -> list[Project]:
        """Directories under ``PROJECTS_ROOT`` that look like repositories."""
        found: list[Project] = []
        if not self.projects_root.is_dir():
            self.log.debug("projects root %s is not a directory", self.projects_root)
            return found
        self._scan(self.projects_root, self.max_depth, found)
        return found

    def _scan(self, base: Path, depth: int, found: list[Project]) -> None:
        """Collect repositories under ``base``; ``depth`` = levels still allowed below it."""
        try:
            entries = sorted(base.iterdir(), key=lambda entry: entry.name.lower())
        except OSError as exc:
            self.log.debug("cannot scan %s: %s", base, exc)
            return
        for entry in entries:
            if entry.name.startswith(".") or entry.name in self.skip_names:
                continue
            try:
                is_dir = entry.is_dir()
            except OSError:  # pragma: no cover - broken symlink / permission race
                continue
            if not is_dir:
                continue
            try:
                is_project = (entry / ".git").exists()
            except OSError:  # pragma: no cover
                is_project = False
            if is_project:
                found.append(Project(name=entry.name, path=entry, source="projects_root"))
                continue
            if depth > 1:
                self._scan(entry, depth - 1, found)

    def from_agent_index(self) -> list[Project]:
        """Hints from the agent's own project index; absent -> empty list."""
        found: list[Project] = []
        index_dir = self.agent_index_dir
        if index_dir is None or not index_dir.is_dir():
            self.log.debug("agent project index %s not present", index_dir)
            return found
        try:
            entries = sorted(index_dir.iterdir(), key=lambda entry: entry.name.lower())
        except OSError as exc:
            self.log.debug("cannot read agent project index %s: %s", index_dir, exc)
            return found
        for entry in entries:
            if not entry.is_dir():
                continue
            candidate = decode_index_name(entry.name)
            try:
                if not candidate.is_dir():
                    continue
            except OSError:  # pragma: no cover
                continue
            found.append(Project(name=candidate.name, path=candidate, source="agent_index"))
        return found

    def discover(self) -> list[Project]:
        """Both sources merged, de-duplicated by realpath, sorted by name."""
        merged: dict[str, Project] = {}
        for project in self.from_projects_root() + self.from_agent_index():
            key = str(project.root)
            existing = merged.get(key)
            if existing is None:
                merged[key] = project
            elif existing.source != "projects_root" and project.source == "projects_root":
                merged[key] = project
        return sorted(merged.values(), key=lambda project: (project.name.lower(), str(project.path)))

    # ------------------------------------------------------------------ resolving

    def find(self, name: str) -> list[Project]:
        """Candidates whose directory name matches ``name`` exactly (case-insensitive)."""
        wanted = name.strip().strip("/")
        if not wanted:
            return []
        matches = [
            project
            for project in self.discover()
            if project.name.lower() == wanted.lower() or str(project.path) == name
        ]
        if matches:
            return matches
        # Fall back to a suffix match so nested projects can be named by tail.
        return [project for project in self.discover() if str(project.path).endswith("/" + wanted)]

    def is_bindable(self, project: Project, roots: Sequence[Path] | None = None) -> bool:
        allowed = tuple(roots) if roots is not None else tuple(self.allowed_roots)
        if not allowed:
            return True
        real = project.root
        for root in allowed:
            real_root = resolve_path(root)
            try:
                if os.path.commonpath([str(real_root), str(real)]) == str(real_root):
                    return True
            except ValueError:  # pragma: no cover
                continue
        return False

    def resolve(self, target: str, path: str | os.PathLike[str] | None = None) -> Project:
        """Resolve ``/bind <name> [<path>]`` input to exactly one candidate.

        ``path`` (when given) wins and is expanded relative to ``PROJECTS_ROOT``.
        Otherwise the name must match exactly one *bindable* discovery; an unknown
        name is treated as a path, and an ambiguous name is an error listing the
        candidates.
        """
        if path:
            candidate = self._expand(path)
            return Project(name=target.strip() or candidate.name, path=candidate, source="bind")

        matches = self.find(target)
        if not matches:
            candidate = self._expand(target)
            if candidate.is_dir():
                return Project(name=candidate.name, path=candidate, source="bind")
            raise DiscoveryError(
                f"no project named {target!r} was discovered and {candidate} is not a directory"
            )
        bindable = [project for project in matches if self.is_bindable(project)]
        if len(bindable) == 1:
            return bindable[0]
        if len(matches) == 1:
            return matches[0]
        listing = "\n".join(f"  - {project.path}" for project in (bindable or matches))
        raise DiscoveryError(
            f"{target!r} matches several projects; pass an explicit path:\n{listing}"
        )

    def _expand(self, path: str | os.PathLike[str]) -> Path:
        text = os.path.expanduser(str(path).strip())
        expanded = Path(text)
        if not expanded.is_absolute():
            expanded = self.projects_root / expanded
        return Path(os.path.normpath(expanded))


# --------------------------------------------------------------------------- per-chat


@dataclass
class ChatRuntime:
    """Per-chat serialisation: one in-flight turn, an optional queue behind it."""

    chat_id: int
    lock: threading.RLock = field(default_factory=threading.RLock)
    busy: bool = False
    queue: list[str] = field(default_factory=list)
    last_activity: float = field(default_factory=time.time)
    pending_approvals: int = 0

    def enqueue(self, text: str) -> int:
        """Queue a message; returns its 1-based position."""
        self.queue.append(text)
        return len(self.queue)

    def dequeue(self) -> str | None:
        return self.queue.pop(0) if self.queue else None

    @property
    def queue_size(self) -> int:
        return len(self.queue)

    def state_label(self) -> str:
        if not self.busy:
            return "idle"
        return f"busy (queue: {self.queue_size})" if self.queue else "busy"


# --------------------------------------------------------------------------- router


class Router:
    """Owns bindings, discovery, per-chat runtime state and state persistence."""

    def __init__(
        self,
        *,
        store: StateStore,
        discovery: ProjectDiscovery,
        access: AccessPolicy,
        allowed_roots: Iterable[str | os.PathLike[str]],
        log: logging.Logger | None = None,
    ) -> None:
        self.store = store
        self.discovery = discovery
        self.access = access
        self.allowed_roots = tuple(Path(root).expanduser() for root in allowed_roots)
        self.log = log or _logger
        self.bindings: dict[int, Binding] = {}
        self.telegram_offset = 0
        self._runtimes: dict[int, ChatRuntime] = {}
        self._runtimes_lock = threading.Lock()

    # ------------------------------------------------------------------ persistence

    def load(self) -> None:
        """Load bindings/offset and merge in the persisted access state."""
        data = self.store.load()
        self.telegram_offset = int(data.get("telegram_offset") or 0)
        self.bindings.clear()
        for key, raw in (data.get("bindings") or {}).items():
            try:
                binding = Binding.from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                self.log.warning("ignoring malformed binding %r: %s", key, exc)
                continue
            self.bindings[int(key)] = binding
        self.access.merge(data.get("access"))
        self.log.info(
            "state loaded from %s: %d binding(s), offset %d",
            self.store.path,
            len(self.bindings),
            self.telegram_offset,
        )

    def snapshot(self, *, merge_disk: bool = False, adopt_bindings: bool = False) -> dict[str, Any]:
        """The state document, optionally read-modify-written against disk first.

        ``merge_disk`` always unions in the access section (so an approval made by
        the CLI is not lost). ``adopt_bindings`` additionally adopts bindings for
        chats this process has never seen: only :meth:`merge_external` does that,
        otherwise a local ``unbind`` would be undone by its own save.
        """
        if merge_disk:
            disk = self.store.load()
            self.access.merge(disk.get("access"))
            if adopt_bindings:
                for key, raw in (disk.get("bindings") or {}).items():
                    try:
                        chat_id = int(key)
                    except (TypeError, ValueError):
                        continue
                    if chat_id not in self.bindings:
                        try:
                            self.bindings[chat_id] = Binding.from_dict(raw)
                        except (KeyError, TypeError, ValueError):
                            continue
        return {
            "version": STATE_VERSION,
            "telegram_offset": self.telegram_offset,
            "bindings": {str(chat_id): binding.to_dict() for chat_id, binding in self.bindings.items()},
            "access": self.access.to_dict(),
        }

    def save(self) -> None:
        """Persist bindings, offset, allowlist and pairing codes atomically."""
        try:
            self.store.save(self.snapshot(merge_disk=True, adopt_bindings=False))
        except OSError as exc:
            self.log.error("cannot write state file %s: %s", self.store.path, exc)

    def merge_external(self) -> bool:
        """Pick up allowlist/pairing/binding changes written by the CLI.

        Returns True when something changed. Bindings for chats this process has
        never seen are adopted; in-memory bindings always win over disk.
        """
        if not self.store.changed_on_disk():
            return False
        data = self.store.load()
        changed = self.access.merge(data.get("access"))
        for key, raw in (data.get("bindings") or {}).items():
            try:
                chat_id = int(key)
                binding = Binding.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                continue
            if chat_id not in self.bindings:
                self.bindings[chat_id] = binding
                changed = True
        return changed

    # ------------------------------------------------------------------ bindings

    def get(self, chat_id: int) -> Binding | None:
        return self.bindings.get(int(chat_id))

    def bind(self, chat_id: int, target: str, path: str | os.PathLike[str] | None = None) -> Binding:
        """Bind a chat to a project, enforcing containment on the resolved path."""
        project = self.discovery.resolve(target, path)
        real = resolve_root(project.root, self.allowed_roots, what="project root")
        moment = time.time()
        binding = Binding(
            chat_id=int(chat_id),
            project_name=project.name,
            project_root=str(real),
            created_at=moment,
            updated_at=moment,
        )
        self.bindings[int(chat_id)] = binding
        self.save()
        self.log.info("chat %s bound to %s", chat_id, real)
        return binding

    def unbind(self, chat_id: int) -> bool:
        removed = self.bindings.pop(int(chat_id), None)
        if removed is not None:
            self.save()
            self.log.info("chat %s unbound from %s", chat_id, removed.project_root)
        return removed is not None

    def set_session(
        self,
        chat_id: int,
        session_id: str | None,
        *,
        model: str | None = None,
        mode: str | None = None,
        approval: str | None = None,
    ) -> Binding | None:
        binding = self.get(chat_id)
        if binding is None:
            return None
        binding.session_id = session_id
        if model is not None:
            binding.model = model
        if mode is not None:
            binding.mode = mode
        if approval is not None:
            binding.approval = approval
        binding.touch()
        self.save()
        return binding

    def set_approval(self, chat_id: int, posture: str | None) -> Binding | None:
        """Set the chat's gateway posture (``/aprobar``) and persist it."""
        binding = self.get(chat_id)
        if binding is None:
            return None
        binding.approval = posture
        binding.touch()
        self.save()
        self.log.info("chat %s approval posture set to %r", chat_id, posture)
        return binding

    def list_projects(self, chat_id: int | None = None) -> list[dict[str, Any]]:
        """Discovered projects with binding/bindability flags, for ``/projects``."""
        bound_chats = {
            int(chat): binding for chat, binding in self.bindings.items()
        }
        rows: list[dict[str, Any]] = []
        for project in self.discovery.discover():
            bound_chat_ids = [
                chat for chat, binding in bound_chats.items() if Path(binding.project_root) == project.root
            ]
            rows.append(
                {
                    "name": project.name,
                    "path": str(project.path),
                    "root": str(project.root),
                    "sources": project.source,
                    "bound_chat_ids": bound_chat_ids,
                    "bindable": self.discovery.is_bindable(project, self.allowed_roots),
                }
            )
        return rows

    def roots_label(self) -> str:
        return ", ".join(str(root) for root in self.allowed_roots)

    # ------------------------------------------------------------------ runtimes

    def runtime(self, chat_id: int) -> ChatRuntime:
        with self._runtimes_lock:
            runtime = self._runtimes.get(int(chat_id))
            if runtime is None:
                runtime = ChatRuntime(chat_id=int(chat_id))
                self._runtimes[int(chat_id)] = runtime
            return runtime

    def busy_chat_ids(self) -> list[int]:
        with self._runtimes_lock:
            return [chat_id for chat_id, runtime in self._runtimes.items() if runtime.busy]
