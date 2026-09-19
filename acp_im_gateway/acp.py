"""Minimal ACP (Agent Client Protocol) client: newline-delimited JSON-RPC 2.0 over stdio.

Surface built against ``reasonix acp`` v1.17.21 (and the ACP spec generally):

* ``initialize``                  -> protocol version, agent info, capabilities
* ``session/new {cwd, mcpServers}`` -> ``sessionId`` + ``configOptions``
* ``session/prompt {sessionId, prompt:[{type:"text", text}]}`` -> streams
  ``session/update`` notifications until it returns a stop reason
* ``session/cancel {sessionId}``  -> stop a turn
* ``session/list {cwd?}`` / ``session/load`` / ``session/close``
* ``session/request_permission``  -> agent *asks the client*; answered with the
  option id the agent advertised
* ``_reasonix.io/session/steer``   -> vendor mid-turn guidance, only when advertised

Robustness rules from the spec are implemented here: the agent command is
configurable, crashes respawn with backoff and fail in-flight turns cleanly,
unknown notifications are ignored, non-permission inbound requests are declined
(never hang), and stdout lines that are not JSON are logged and skipped.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from . import __version__

JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = 1
STEER_METHOD = "_reasonix.io/session/steer"

#: Default JSON-RPC error code for "this client does not implement that method".
METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603

_logger = logging.getLogger("acp_im_gateway.acp")


class _Sentinel:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self._name


#: Returned by an inbound-request handler that keeps the request open and answers later.
DEFER = _Sentinel("DEFER")
#: Returned by an inbound-request handler that wants a clean "method not found" reply.
DECLINE = _Sentinel("DECLINE")


class AcpError(Exception):
    """An ACP request failed (JSON-RPC error, transport error or bad payload)."""

    def __init__(self, message: str, *, code: int | None = None, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data


class AgentCrashed(AcpError):
    """The agent process died while requests were in flight."""


class AgentNotRunning(AcpError):
    """The agent process is not running (never started, or stopped)."""


class AcpTimeout(AcpError):
    """The agent did not answer within the configured timeout."""


# --------------------------------------------------------------------------- payloads


@dataclass
class AgentCapabilities:
    """Normalised view of the ``initialize`` result."""

    protocol_version: int | None = None
    agent_name: str | None = None
    agent_version: str | None = None
    load_session: bool = False
    session_capabilities: Mapping[str, Any] = field(default_factory=dict)
    prompt_capabilities: Mapping[str, Any] = field(default_factory=dict)
    steer_method: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def supports_steer(self) -> bool:
        return bool(self.steer_method)

    @property
    def supports_list(self) -> bool:
        return "list" in self.session_capabilities

    @property
    def supports_resume(self) -> bool:
        return "resume" in self.session_capabilities

    @property
    def supports_close(self) -> bool:
        return "close" in self.session_capabilities

    @property
    def supports_embedded_context(self) -> bool:
        return bool(self.prompt_capabilities.get("embeddedContext"))

    @property
    def supports_images(self) -> bool:
        return bool(self.prompt_capabilities.get("image"))

    @property
    def supports_audio(self) -> bool:
        return bool(self.prompt_capabilities.get("audio"))

    @classmethod
    def from_result(cls, result: Mapping[str, Any]) -> "AgentCapabilities":
        caps = dict(result.get("agentCapabilities") or {})
        info = dict(result.get("agentInfo") or {})
        return cls(
            protocol_version=result.get("protocolVersion"),
            agent_name=info.get("name"),
            agent_version=info.get("version"),
            load_session=bool(caps.get("loadSession")),
            session_capabilities=dict(caps.get("sessionCapabilities") or {}),
            prompt_capabilities=dict(caps.get("promptCapabilities") or {}),
            steer_method=_find_steer_method(caps),
            raw=dict(result),
        )


def _find_steer_method(capabilities: Mapping[str, Any]) -> str | None:
    """Locate the vendor steering method inside the agent's ``_meta`` block.

    Accepts both obvious shapes (a literal ``_reasonix.io/session/steer`` key, or
    ``_meta["reasonix.io"]["sessionSteer"]["method"]``) and, tolerantly, any nested
    ``sessionSteer``/``session_steer`` node.
    """

    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in ("sessionSteer", "session_steer", STEER_METHOD):
                    if isinstance(value, Mapping) and isinstance(value.get("method"), str):
                        found.append(value["method"])
                    elif isinstance(value, str):
                        found.append(value)
                    else:
                        found.append(STEER_METHOD)
                walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)
        elif isinstance(node, str) and node == STEER_METHOD:
            found.append(node)

    walk(capabilities)
    return found[0] if found else None


@dataclass
class SessionInfo:
    """One entry of ``session/list``."""

    session_id: str
    cwd: str | None = None
    updated_at: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "SessionInfo":
        return cls(
            session_id=str(raw.get("sessionId", "")),
            cwd=raw.get("cwd"),
            updated_at=raw.get("updatedAt"),
            raw=dict(raw),
        )


@dataclass
class NewSession:
    """Result of ``session/new`` (or ``session/load``)."""

    session_id: str
    config_options: list[dict[str, Any]] = field(default_factory=list)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def option(self, config_id: str) -> dict[str, Any] | None:
        for option in self.config_options:
            if option.get("id") == config_id:
                return option
        return None

    def option_value(self, config_id: str) -> Any:
        option = self.option(config_id)
        return None if option is None else option.get("currentValue")

    @property
    def model(self) -> Any:
        return self.option_value("model")

    @property
    def approval_posture(self) -> Any:
        return self.option_value("tool_approval")


@dataclass
class PromptResult:
    """Result of a completed ``session/prompt``."""

    session_id: str
    stop_reason: str | None
    updates: int = 0
    raw: Mapping[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- plumbing


class _PendingRequest:
    """An outbound request awaiting its JSON-RPC response."""

    __slots__ = ("id", "method", "event", "result", "error")

    def __init__(self, request_id: int, method: str) -> None:
        self.id = request_id
        self.method = method
        self.event = threading.Event()
        self.result: Mapping[str, Any] | None = None
        self.error: AcpError | None = None

    @property
    def done(self) -> bool:
        return self.event.is_set()

    def set_result(self, result: Mapping[str, Any] | None) -> None:
        self.result = result or {}
        self.event.set()

    def set_error(self, error: AcpError) -> None:
        self.error = error
        self.event.set()


class InboundRequest:
    """A request *from* the agent that the client must answer.

    ``session/request_permission`` is the important one: the approval bridge keeps
    the request open (handler returns :data:`DEFER`) and calls :meth:`respond`
    later, from the Telegram callback thread.
    """

    def __init__(self, client: "AcpClient", request_id: Any, method: str, params: Mapping[str, Any]) -> None:
        self._client = client
        self._lock = threading.Lock()
        self.id = request_id
        self.method = method
        self.params = dict(params)
        self.answered = False

    def _claim(self) -> bool:
        """Atomically take ownership of answering this request."""
        with self._lock:
            if self.answered:
                return False
            self.answered = True
            return True

    def respond(self, result: Mapping[str, Any] | None = None) -> bool:
        """Send a successful response. Safe to call once; later calls are no-ops."""
        if not self._claim():
            return False
        self._client._forget_inbound(self.id)
        return self._client._write(
            {"jsonrpc": JSONRPC_VERSION, "id": self.id, "result": dict(result or {})}
        )

    def fail(
        self,
        code: int = INTERNAL_ERROR,
        message: str = "client declined the request",
        data: Any = None,
    ) -> bool:
        """Send a JSON-RPC error response (a clean decline, never a hang)."""
        if not self._claim():
            return False
        self._client._forget_inbound(self.id)
        error: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            error["data"] = data
        return self._client._write({"jsonrpc": JSONRPC_VERSION, "id": self.id, "error": error})

    def abandon(self, reason: str = "agent restarted") -> None:
        """Mark answered without writing (the agent is gone)."""
        if not self._claim():
            return
        self._client._forget_inbound(self.id)
        _logger.debug("dropping inbound request %s (%s): %s", self.id, self.method, reason)


# --------------------------------------------------------------------------- client


class AcpClient:
    """A small, thread-safe ACP client speaking JSON-RPC 2.0 over stdio."""

    def __init__(
        self,
        command: Sequence[str] = ("reasonix", "acp"),
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        log: logging.Logger | None = None,
        request_timeout: float | None = 60.0,
        on_request: Callable[[InboundRequest], Any] | None = None,
        on_notification: Callable[[Mapping[str, Any]], None] | None = None,
        on_restart: Callable[[], None] | None = None,
        auto_respawn: bool = True,
        max_restart_attempts: int = 12,
        restart_backoff: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0),
        restart_backoff_max: float = 30.0,
        capabilities: AgentCapabilities | None = None,
    ) -> None:
        self.command = tuple(command)
        self.cwd = None if cwd is None else str(cwd)
        self.env = None if env is None else dict(env)
        self.log = log or _logger
        self.request_timeout = request_timeout
        self.on_request = on_request
        self.on_notification = on_notification
        self.on_restart = on_restart
        self.auto_respawn = auto_respawn
        self.max_restart_attempts = max_restart_attempts
        self.restart_backoff = tuple(restart_backoff)
        self.restart_backoff_max = restart_backoff_max
        self.capabilities = capabilities

        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._write_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._pending_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, _PendingRequest] = {}
        self._inbound: dict[Any, InboundRequest] = {}
        self._subscribers: dict[str, list[queue.Queue]] = {}
        self._stopping = False
        self._restart_attempts = 0
        self._pid: int | None = None
        self.stderr_tail: list[str] = []
        self.restarts = 0
        #: Every method name written to the agent (requests and notifications), in
        #: order. Lets tests prove, for example, that ``session/prompt`` was never
        #: sent to a real agent.
        self.sent_methods: list[str] = []

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> "AcpClient":
        """Spawn the agent and start reading. Idempotent while running."""
        with self._state_lock:
            if self._proc is not None and self._proc.poll() is None:
                return self
            self._stopping = False
            self._spawn_locked()
        return self

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def restart_attempts(self) -> int:
        return self._restart_attempts

    def stop(self, timeout: float = 5.0) -> None:
        """Terminate the agent, fail in-flight work, and join the reader threads."""
        with self._state_lock:
            self._stopping = True
            proc = self._proc
            self._proc = None
        if proc is not None:
            self._terminate(proc, timeout)
        self._fail_all(AgentNotRunning("agent stopped"))
        for request in list(self._inbound.values()):
            request.abandon("client stopped")
        current = threading.current_thread()
        for thread in (self._reader, self._stderr_reader):
            if thread is not None and thread is not current and thread.is_alive():
                thread.join(timeout=timeout)

    def __enter__(self) -> "AcpClient":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def _spawn_locked(self, *, start_stdout_reader: bool = True) -> None:
        env = dict(os.environ)
        if self.env:
            env.update(self.env)
        try:
            self._proc = subprocess.Popen(
                list(self.command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=env,
                start_new_session=os.name != "nt",
            )
        except FileNotFoundError as exc:
            self._proc = None
            raise AgentNotRunning(
                f"agent command not found: {' '.join(self.command)} ({exc})"
            ) from exc
        except OSError as exc:
            self._proc = None
            raise AgentNotRunning(f"could not start agent {' '.join(self.command)}: {exc}") from exc
        self._pid = self._proc.pid
        if start_stdout_reader:
            self._reader = threading.Thread(target=self._read_loop, name="acp-reader", daemon=True)
            self._reader.start()
        self._stderr_reader = threading.Thread(
            target=self._stderr_loop, args=(self._proc,), name="acp-stderr", daemon=True
        )
        self._stderr_reader.start()
        self.log.info("agent started: %s (pid %s)", " ".join(self.command), self._pid)

    def _terminate(self, proc: subprocess.Popen[str], timeout: float) -> None:
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self.log.warning("agent did not exit after %.1fs, killing", timeout)
                    proc.kill()
                    try:
                        proc.wait(timeout=2.0)
                    except subprocess.TimeoutExpired:  # pragma: no cover - very unusual
                        pass
        except OSError as exc:  # pragma: no cover - racing process teardown
            self.log.debug("error terminating agent: %s", exc)
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):  # pragma: no cover
                pass

    # ------------------------------------------------------------------ reading

    def _read_loop(self) -> None:
        while True:
            proc = self._proc
            if proc is None or proc.stdout is None:
                return
            try:
                line = proc.stdout.readline()
            except (ValueError, OSError):  # pragma: no cover - pipe closed underneath us
                line = ""
            if line:
                self._handle_line(line)
                continue

            # EOF: the agent exited. Fail in-flight work, then maybe respawn.
            self._on_agent_exit(proc)
            if self._stopping or not self.auto_respawn:
                return
            if not self._respawn_loop():
                return

    def _on_agent_exit(self, proc: subprocess.Popen[str]) -> None:
        code = proc.poll()
        with self._state_lock:
            if self._proc is proc:
                self._proc = None
            self._pid = None
        self._fail_all(AgentCrashed(f"agent exited (code {code}) while the request was in flight"))
        for request in list(self._inbound.values()):
            request.abandon(f"agent exited (code {code})")
        if self._stopping:
            self.log.info("agent stopped (code %s)", code)
        else:
            self.log.warning("agent exited (code %s)", code)

    def _respawn_loop(self) -> bool:
        while not self._stopping:
            delay = self._next_backoff()
            self.log.warning(
                "respawning agent in %.1fs (attempt %d/%d)",
                delay,
                self._restart_attempts,
                self.max_restart_attempts,
            )
            if not self._sleep(delay):
                return False
            self._restart_attempts += 1
            try:
                with self._state_lock:
                    # This thread is already reading stdout: it picks the new
                    # process up on the next loop iteration.
                    self._spawn_locked(start_stdout_reader=False)
            except AcpError as exc:
                self.log.error("respawn failed: %s", exc)
                if self._restart_attempts >= self.max_restart_attempts:
                    self.log.error("giving up after %d attempts", self._restart_attempts)
                    return False
                continue
            self.restarts += 1
            self._notify_restart()
            return True
        return False

    def _next_backoff(self) -> float:
        table = self.restart_backoff or (0.5,)
        index = min(max(self._restart_attempts - 1, 0), len(table) - 1)
        return float(min(table[index], self.restart_backoff_max))

    def _sleep(self, seconds: float) -> bool:
        """Sleep, waking early when :meth:`stop` is called. False when stopping."""
        deadline = time.monotonic() + max(seconds, 0.0)
        while not self._stopping:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(remaining, 0.1))
        return False

    def _notify_restart(self) -> None:
        if self.on_restart is None:
            return
        try:
            self.on_restart()
        except Exception:  # pragma: no cover - the gateway's own callback
            self.log.exception("on_restart callback failed")

    def _stderr_loop(self, proc: subprocess.Popen[str]) -> None:
        if proc.stderr is None:
            return
        for line in proc.stderr:
            text = line.rstrip("\n")
            if not text:
                continue
            self.stderr_tail.append(text)
            del self.stderr_tail[:-40]
            self.log.debug("agent stderr: %s", text)

    def _handle_line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            self.log.warning("skipping non-JSON line from agent: %.200s", text)
            return
        if not isinstance(message, Mapping):
            self.log.warning("skipping non-object JSON line from agent: %.200s", text)
            return

        if "id" in message and ("result" in message or "error" in message):
            self._resolve_pending(message)
            return
        if "id" in message and "method" in message:
            self._dispatch_inbound(message)
            return
        if "method" in message:
            self._dispatch_notification(message)
            return
        self.log.warning("ignoring unrecognised JSON-RPC message: %.200s", text)

    def _resolve_pending(self, message: Mapping[str, Any]) -> None:
        request_id = message.get("id")
        with self._pending_lock:
            pending = self._pending.pop(request_id, None)
        if pending is None:
            self.log.debug("response for unknown request id %r ignored", request_id)
            return
        error = message.get("error")
        if error:
            pending.set_error(
                AcpError(
                    str(error.get("message", "agent error")),
                    code=error.get("code"),
                    data=error.get("data"),
                )
            )
            return
        pending.set_result(message.get("result") or {})
        # A completed request proves the current process is healthy: reset backoff.
        self._restart_attempts = 0

    def _dispatch_notification(self, message: Mapping[str, Any]) -> None:
        method = message.get("method")
        params = message.get("params") or {}
        session_id = params.get("sessionId") if isinstance(params, Mapping) else None
        if method == "session/update" and session_id is not None:
            self._fan_out(str(session_id), message)
        else:
            # Unknown / unsupported notifications are ignored, by design.
            self.log.debug("ignoring notification %s", method)
        if self.on_notification is not None:
            try:
                self.on_notification(message)
            except Exception:  # pragma: no cover - observer callback
                self.log.exception("on_notification callback failed")

    def _dispatch_inbound(self, message: Mapping[str, Any]) -> None:
        request = InboundRequest(
            self,
            message.get("id"),
            str(message.get("method")),
            message.get("params") or {},
        )
        with self._state_lock:
            self._inbound[request.id] = request
        outcome: Any
        try:
            if self.on_request is None:
                outcome = self._default_inbound(request)
            else:
                outcome = self.on_request(request)
        except Exception as exc:  # a bug in the handler must not hang the agent
            self.log.exception("inbound request handler failed for %s", request.method)
            request.fail(INTERNAL_ERROR, f"client handler failed: {exc}")
            return

        if outcome is DEFER:
            return
        if outcome is DECLINE or outcome is None:
            request.fail(METHOD_NOT_FOUND, f"{request.method} is not supported by this client")
            return
        if isinstance(outcome, Mapping):
            request.respond(outcome)
            return
        request.fail(INTERNAL_ERROR, f"client handler returned {type(outcome).__name__}")

    def _default_inbound(self, request: InboundRequest) -> Any:
        """No handler configured: refuse permission requests, decline the rest."""
        if request.method == "session/request_permission":
            return {"outcome": "cancelled"}
        return DECLINE

    def _forget_inbound(self, request_id: Any) -> None:
        with self._state_lock:
            self._inbound.pop(request_id, None)

    # ------------------------------------------------------------------ writing

    def _write(self, message: Mapping[str, Any]) -> bool:
        payload = json.dumps(message, separators=(",", ":"), default=str) + "\n"
        method = message.get("method")
        if method is not None:
            self.sent_methods.append(str(method))
        with self._write_lock:
            proc = self._proc
            stdin = None if proc is None else proc.stdin
            if proc is None or proc.poll() is not None or stdin is None:
                self.log.warning("cannot write %s: agent is not running", message.get("method"))
                return False
            try:
                stdin.write(payload)
                stdin.flush()
            except (OSError, ValueError) as exc:
                self.log.warning("cannot write %s: %s", message.get("method"), exc)
                return False
        return True

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> bool:
        return self._write({"jsonrpc": JSONRPC_VERSION, "method": method, "params": dict(params or {})})

    def request(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Mapping[str, Any]:
        """Send a request and block for its result.

        ``timeout=None`` means "use the client default"; pass a float to override.
        """
        effective = self.request_timeout if timeout is None else timeout
        pending = self._register_pending(method)
        if not self._write(
            {
                "jsonrpc": JSONRPC_VERSION,
                "id": pending.id,
                "method": method,
                "params": dict(params or {}),
            }
        ):
            self._discard_pending(pending.id)
            raise AgentNotRunning(f"agent is not running (request {method})")
        if not pending.event.wait(effective):
            self._discard_pending(pending.id)
            raise AcpTimeout(f"{method} timed out after {effective}s", code=-32001)
        if pending.error is not None:
            raise pending.error
        return pending.result or {}

    def _register_pending(self, method: str) -> _PendingRequest:
        with self._pending_lock:
            request_id = self._next_id
            self._next_id += 1
            pending = _PendingRequest(request_id, method)
            self._pending[request_id] = pending
        return pending

    def _discard_pending(self, request_id: int) -> None:
        with self._pending_lock:
            self._pending.pop(request_id, None)

    def _fail_all(self, error: AcpError) -> None:
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        for request in pending:
            request.set_error(error)

    # ------------------------------------------------------------------ updates

    def subscribe(self, session_id: str) -> "queue.Queue[Mapping[str, Any]]":
        """Register a queue receiving ``session/update`` notifications for a session."""
        channel: queue.Queue[Mapping[str, Any]] = queue.Queue()
        with self._state_lock:
            self._subscribers.setdefault(session_id, []).append(channel)
        return channel

    def unsubscribe(self, session_id: str, channel: "queue.Queue[Mapping[str, Any]]") -> None:
        with self._state_lock:
            channels = self._subscribers.get(session_id)
            if not channels:
                return
            try:
                channels.remove(channel)
            except ValueError:
                pass
            if not channels:
                self._subscribers.pop(session_id, None)

    def _fan_out(self, session_id: str, message: Mapping[str, Any]) -> None:
        with self._state_lock:
            channels = list(self._subscribers.get(session_id, ()))
        if not channels:
            self.log.debug("session/update for unsubscribed session %s", session_id)
            return
        for channel in channels:
            channel.put(message)

    # ------------------------------------------------------------------ ACP API

    def initialize(
        self,
        *,
        client_name: str = "acp-im-gateway",
        client_version: str = __version__,
        capabilities: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> AgentCapabilities:
        """Perform the ACP handshake and remember the agent's capabilities."""
        params = {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": dict(
                capabilities
                or {
                    # The gateway implements neither filesystem nor terminal
                    # proxying, so it must not advertise them.
                    "fs": {"readTextFile": False, "writeTextFile": False},
                    "terminal": False,
                }
            ),
            "clientInfo": {"name": client_name, "version": client_version},
        }
        result = self.request("initialize", params, timeout=timeout)
        self.capabilities = AgentCapabilities.from_result(result)
        self.log.info(
            "agent ready: %s %s (protocol %s, steer=%s)",
            self.capabilities.agent_name,
            self.capabilities.agent_version,
            self.capabilities.protocol_version,
            self.capabilities.steer_method or "unsupported",
        )
        return self.capabilities

    def new_session(
        self,
        cwd: str | os.PathLike[str],
        *,
        mcp_servers: Sequence[Mapping[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> NewSession:
        result = self.request(
            "session/new",
            {"cwd": str(cwd), "mcpServers": [dict(server) for server in (mcp_servers or ())]},
            timeout=timeout,
        )
        session_id = result.get("sessionId")
        if not session_id:
            raise AcpError(f"session/new returned no sessionId: {dict(result)!r}")
        return NewSession(
            session_id=str(session_id),
            config_options=[dict(option) for option in result.get("configOptions") or ()],
            raw=dict(result),
        )

    def load_session(
        self,
        session_id: str,
        cwd: str | os.PathLike[str],
        *,
        mcp_servers: Sequence[Mapping[str, Any]] | None = None,
        timeout: float | None = None,
    ) -> NewSession:
        """Re-attach to an existing session (only when the agent advertises it)."""
        if self.capabilities is not None and not self.capabilities.load_session:
            raise AcpError("agent does not advertise loadSession")
        result = self.request(
            "session/load",
            {
                "sessionId": session_id,
                "cwd": str(cwd),
                "mcpServers": [dict(server) for server in (mcp_servers or ())],
            },
            timeout=timeout,
        )
        return NewSession(
            session_id=str(result.get("sessionId") or session_id),
            config_options=[dict(option) for option in result.get("configOptions") or ()],
            raw=dict(result),
        )

    def list_sessions(
        self, cwd: str | os.PathLike[str] | None = None, *, timeout: float | None = None
    ) -> list[SessionInfo]:
        if self.capabilities is not None and not self.capabilities.supports_list:
            raise AcpError("agent does not advertise session/list")
        params: dict[str, Any] = {}
        if cwd is not None:
            params["cwd"] = str(cwd)
        result = self.request("session/list", params, timeout=timeout)
        return [SessionInfo.from_raw(entry) for entry in result.get("sessions") or ()]

    def close_session(self, session_id: str, *, timeout: float | None = None) -> Mapping[str, Any]:
        return self.request("session/close", {"sessionId": session_id}, timeout=timeout)

    def cancel(self, session_id: str) -> bool:
        """Ask the agent to stop the current turn (a notification, per ACP)."""
        return self.notify("session/cancel", {"sessionId": session_id})

    def set_config_option(
        self, session_id: str, config_id: str, value: Any, *, timeout: float | None = None
    ) -> Mapping[str, Any]:
        return self.request(
            "session/set_config_option",
            {"sessionId": session_id, "configId": config_id, "value": value},
            timeout=timeout,
        )

    def steer(self, session_id: str, text: str, *, timeout: float | None = None) -> Mapping[str, Any]:
        """Send mid-turn guidance. Only valid when the agent advertises steering."""
        method = (self.capabilities.steer_method if self.capabilities else None) or STEER_METHOD
        if self.capabilities is not None and not self.capabilities.supports_steer:
            raise AcpError("agent does not advertise session steering")
        return self.request(
            method,
            {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
            timeout=timeout,
        )

    def prompt(
        self,
        session_id: str,
        text: str,
        *,
        on_update: Callable[[Mapping[str, Any]], None] | None = None,
        on_tick: Callable[[], None] | None = None,
        timeout: float | None = 0.0,
        tick: float = 0.25,
    ) -> PromptResult:
        """Run one turn. Blocks until the agent returns a stop reason.

        ``session/update`` notifications for ``session_id`` are handed to
        ``on_update`` as they arrive; ``on_tick`` is called on every idle poll so
        callers can flush coalesced output (streaming Telegram edits).

        ``timeout=0`` (the default) means no cap: a turn ends when the agent says
        so. Pass a positive value to bound it; ``/stop`` cancels regardless.
        """
        effective = None if timeout in (None, 0) else float(timeout)
        channel = self.subscribe(session_id)
        try:
            payload = {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": text}],
            }
            pending = self._register_pending("session/prompt")
            if not self._write(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "id": pending.id,
                    "method": "session/prompt",
                    "params": payload,
                }
            ):
                self._discard_pending(pending.id)
                raise AgentNotRunning("agent is not running (request session/prompt)")

            updates = 0
            idle = 0
            deadline = None if effective is None else time.monotonic() + effective
            while True:
                if deadline is not None and time.monotonic() > deadline:
                    self._discard_pending(pending.id)
                    raise AcpTimeout(
                        f"session/prompt exceeded {effective}s", code=-32001
                    )
                try:
                    notification = channel.get(timeout=tick)
                except queue.Empty:
                    if on_tick is not None:
                        try:
                            on_tick()
                        except Exception:  # pragma: no cover - caller's own callback
                            self.log.exception("prompt on_tick callback failed")
                    if pending.done:
                        idle += 1
                        if idle >= 2:  # ~0.5s grace for trailing notifications
                            break
                    continue
                idle = 0
                updates += 1
                if on_update is not None:
                    try:
                        on_update(notification)
                    except Exception:  # pragma: no cover - caller's own callback
                        self.log.exception("prompt on_update callback failed")

            if not pending.event.wait(1.0):  # pragma: no cover - pending always set here
                self._discard_pending(pending.id)
                raise AgentCrashed("agent disappeared before the turn finished")
            if pending.error is not None:
                raise pending.error
            result = pending.result or {}
            return PromptResult(
                session_id=session_id,
                stop_reason=result.get("stopReason"),
                updates=updates,
                raw=dict(result),
            )
        finally:
            self.unsubscribe(session_id, channel)
