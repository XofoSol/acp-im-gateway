"""Gateway orchestration: Telegram <-> router <-> ACP agent, with approvals.

The turn loop is deliberately thin. User text is forwarded **verbatim** to the
agent over ACP (no LLM in the middle, no rewriting), ``session/update``
notifications are rendered into one Telegram message that is edited in place, and
``session/request_permission`` goes through :mod:`acp_im_gateway.approvals`.

Per chat there is exactly one in-flight turn. A second message during a turn is
either steered into it (when the agent advertises session steering) or queued,
per ``GATEWAY_BUSY_MODE``.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .access import AccessPolicy
from .acp import AcpClient, AcpError, AgentCapabilities, AgentCrashed, AcpTimeout, SessionInfo
from .approvals import ApprovalBridge
from .config import Config
from .containment import ContainmentError
from .router import (
    Binding,
    DiscoveryError,
    ProjectDiscovery,
    Router,
    StateStore,
)
from .telegram import MessageStream, TelegramClient, TelegramError

_logger = logging.getLogger("acp_im_gateway.gateway")

BOT_COMMANDS: tuple[tuple[str, str], ...] = (
    ("projects", "List discovered projects and which one is bound"),
    ("bind", "Bind this chat to a project: /bind <name> [<path>]"),
    ("new", "Start a fresh session in this chat's project"),
    ("stop", "Cancel the running turn and drop queued messages"),
    ("status", "Show project, session, model, approval posture and queue"),
    ("unbind", "Forget the binding for this chat"),
    ("help", "Show this help"),
)

HELP_TEXT = """acp-im-gateway — drive an ACP coding agent from this chat.

Commands
  /projects            list discovered projects and which one is bound
  /bind <name> [<path>]  bind this chat to a project
  /new                 start a fresh session in this chat's project
  /stop                cancel the running turn (drops queued messages)
  /status              project, session, model, approval posture, queue
  /unbind              forget this chat's binding
  /help                this message

Anything else you type is sent verbatim to the agent, and its reply is streamed
into a single message that is edited in place. Approvals appear as buttons.

Direct messages only by default: group chats must be listed in ALLOWED_CHAT_IDS.
"""

_STOP_MARKS = {
    "end_turn": "✅ done",
    "cancelled": "⏹️ cancelled",
    "canceled": "⏹️ cancelled",
    "max_tokens": "⚠️ stopped: token limit",
    "max_turn_requests": "⚠️ stopped: request limit",
    "refusal": "🚫 refused by the agent",
}
_TOOL_ICONS = {
    "read": "📖",
    "edit": "✏️",
    "execute": "▶️",
    "think": "💭",
    "fetch": "🌐",
    "search": "🔎",
}


def _content_text(content: Any) -> str:
    """Extract text from an ACP content block (or a list of them)."""
    if isinstance(content, Mapping):
        if content.get("type") in (None, "text"):
            return str(content.get("text") or "")
        return ""
    if isinstance(content, (list, tuple)):
        return "".join(_content_text(item) for item in content)
    if isinstance(content, str):
        return content
    return ""


@dataclass
class TurnView:
    """Reduce a turn's ``session/update`` stream into one Telegram-friendly body.

    Segments keep the agent's ordering: text runs and tool-call lines appear in
    the order they happened. Tool lines are updated in place by tool-call id.
    """

    thought_chunks: int = 0
    tool_calls: int = 0
    placeholder: str = "💭 working…"
    _segments: list[str] = field(default_factory=list)
    _buffer: list[str] = field(default_factory=list)
    _tool_segments: dict[str, int] = field(default_factory=dict)
    _seen_tools: set[str] = field(default_factory=set)

    # ------------------------------------------------------------------ reducing

    def _flush_text(self) -> None:
        if self._buffer:
            text = "".join(self._buffer)
            self._buffer.clear()
            if text.strip():
                self._segments.append(text)

    def apply(self, notification: Mapping[str, Any]) -> None:
        params = notification.get("params") or {}
        update = params.get("update") or {}
        if not isinstance(update, Mapping):
            return
        kind = str(update.get("sessionUpdate") or "")

        if kind == "agent_message_chunk":
            self._buffer.append(_content_text(update.get("content")))
        elif kind == "agent_thought_chunk":
            self.thought_chunks += 1
        elif kind == "tool_call":
            self._apply_tool_call(update)
        elif kind == "tool_call_update":
            self._apply_tool_call(update, is_update=True)
        elif kind == "plan":
            self._apply_plan(update)
        # user_message_chunk / available_commands_update / current_mode_update /
        # config_option_update and anything unknown are intentionally ignored.

    def _apply_tool_call(self, update: Mapping[str, Any], *, is_update: bool = False) -> None:
        tool_id = str(update.get("toolCallId") or update.get("id") or f"tool-{self.tool_calls}")
        title = str(update.get("title") or "").strip()
        kind = str(update.get("kind") or "").strip()
        status = str(update.get("status") or "").strip()
        icon = _TOOL_ICONS.get(kind.lower(), "🔧")

        existing_index = self._tool_segments.get(tool_id)
        if existing_index is None and is_update and tool_id in self._seen_tools:
            return
        if existing_index is not None:
            previous = self._segments[existing_index]
            if not title:
                title = previous.split("[")[0].replace(icon, "").strip() or tool_id
            line = f"{icon} {title} [{status or 'pending'}]"
            if line != previous:
                self._segments[existing_index] = line
            return

        self._flush_text()
        if tool_id not in self._seen_tools:
            self.tool_calls += 1
            self._seen_tools.add(tool_id)
        line = f"{icon} {title or tool_id} [{status or 'pending'}]"
        self._tool_segments[tool_id] = len(self._segments)
        self._segments.append(line)

    def _apply_plan(self, update: Mapping[str, Any]) -> None:
        entries = update.get("entries") or update.get("plan") or []
        lines: list[str] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            status = str(entry.get("status") or entry.get("priority") or "")
            mark = "✅" if status in ("completed", "done") else "▫️"
            text = str(entry.get("content") or entry.get("text") or "").strip()
            if text:
                lines.append(f"{mark} {text}")
        if not lines:
            return
        self._flush_text()
        self._segments.append("📋 plan:\n" + "\n".join(lines))

    # ------------------------------------------------------------------ rendering

    def render(self) -> str:
        parts = [segment for segment in self._segments if segment.strip()]
        if self._buffer:
            tail = "".join(self._buffer)
            if tail.strip():
                parts.append(tail)
        body = "\n".join(parts).strip()
        if body:
            return body
        if self.thought_chunks:
            return f"💭 thinking… ({self.thought_chunks} chunk(s) of reasoning withheld)"
        return self.placeholder


class Gateway:
    """Long-polling Telegram gateway around one ACP agent process."""

    def __init__(
        self,
        config: Config,
        *,
        telegram: TelegramClient | None = None,
        acp: AcpClient | None = None,
        log: logging.Logger | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.log = log or _logger
        self.clock = clock
        self.telegram = telegram or TelegramClient(
            config.telegram_bot_token,
            api_base=config.telegram_api_base,
            dry_run=config.dry_run,
            send_interval=config.send_interval,
            log=self.log,
        )
        self.access = AccessPolicy.from_config(
            allowed_user_ids=config.allowed_user_ids,
            allowed_chat_ids=config.allowed_chat_ids,
            pairing_ttl=config.pairing_ttl,
        )
        self.store = StateStore(config.state_file, log=self.log)
        self.router = Router(
            store=self.store,
            discovery=ProjectDiscovery(
                config.projects_root,
                agent_index_dir=config.agent_index_dir,
                max_depth=config.discovery_depth,
                allowed_roots=config.resolved_roots(),
                log=self.log,
            ),
            access=self.access,
            allowed_roots=config.resolved_roots(),
            log=self.log,
        )
        self.approvals = ApprovalBridge(
            self.telegram,
            chat_for_session=self._chat_for_session,
            timeout=config.approval_timeout,
            log=self.log,
        )
        self.acp = acp or AcpClient(
            config.agent_cmd,
            log=self.log,
            on_request=self.approvals.handle_inbound,
            on_restart=self._on_agent_restart,
            restart_backoff_max=config.restart_backoff_max,
        )
        if acp is not None:
            # Injectable client (tests): still route approvals through the bridge.
            self.acp.on_request = self.approvals.handle_inbound
            self.acp.on_restart = self._on_agent_restart

        self.capabilities: AgentCapabilities | None = None
        self._live_sessions: set[str] = set()
        self._sessions_lock = threading.Lock()
        self._stop = threading.Event()
        self._started = False

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Load persisted state, start the agent and complete the ACP handshake."""
        if self._started:
            return
        self.router.load()
        self._started = True
        if not self.config.dry_run:
            self.telegram.set_my_commands(BOT_COMMANDS)
        try:
            self._ensure_agent()
        except AcpError as exc:
            self.log.error(
                "cannot talk ACP to the agent (%s). Check REASONIX_ACP_CMD (currently: %s) "
                "and that the agent is installed and configured.",
                exc,
                " ".join(self.config.agent_cmd),
            )
            raise

    def stop(self) -> None:
        self._stop.set()
        self.acp.stop()
        self.router.save()

    @property
    def running(self) -> bool:
        return not self._stop.is_set()

    def run(self, *, max_iterations: int | None = None) -> None:
        """Long-poll Telegram until stopped (or ``max_iterations`` polls, for tests)."""
        self.start()
        offset = self.router.telegram_offset
        iterations = 0
        while not self._stop.is_set():
            if max_iterations is not None and iterations >= max_iterations:
                break
            iterations += 1
            self.approvals.expire_due()
            self._merge_external_state()
            try:
                updates = self.telegram.get_updates(
                    offset if offset > 0 else None, timeout=self.config.poll_timeout
                )
            except TelegramError as exc:
                self.log.error("getUpdates failed: %s", exc)
                self._stop.wait(min(5.0, max(1.0, float(self.config.poll_timeout))))
                continue
            if not updates:
                continue
            for update in updates:
                try:
                    update_id = int(update.get("update_id", 0))
                except (TypeError, ValueError):
                    continue
                offset = max(offset, update_id + 1)
                try:
                    self.handle_update(update)
                except Exception:  # one bad update must not kill the gateway
                    self.log.exception("failed to handle update %s", update_id)
            self.router.telegram_offset = offset
            self.router.save()

    # ------------------------------------------------------------------ updates

    def handle_update(self, update: Mapping[str, Any]) -> None:
        if "message" in update:
            self.handle_message(update["message"])
        elif "callback_query" in update:
            self.handle_callback(update["callback_query"])
        else:
            self.log.debug("ignoring update keys: %s", sorted(update))

    def handle_message(self, message: Mapping[str, Any]) -> None:
        chat = dict(message.get("chat") or {})
        chat_id = chat.get("id")
        chat_type = chat.get("type", "private")
        user = dict(message.get("from") or {})
        user_id = user.get("id")
        if chat_id is None:
            return
        chat_id = int(chat_id)

        if not self.access.is_user_allowed(user_id):
            self._offer_pairing(user_id, chat_id, user, chat_type)
            return
        if not self.access.is_chat_allowed(chat_id, chat_type):
            self._reply(
                chat_id,
                "This group chat is not enabled. Add its id to ALLOWED_CHAT_IDS and restart the gateway.",
            )
            return

        text = message.get("text")
        if not text:
            self._reply(
                chat_id,
                "I can only handle text messages (image and audio attachments are out of scope for v1).",
            )
            return
        text = str(text).strip()
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(chat_id, int(user_id), text)
            return
        self._handle_prompt(chat_id, text)

    def handle_callback(self, callback_query: Mapping[str, Any]) -> None:
        user = dict(callback_query.get("from") or {})
        message = dict(callback_query.get("message") or {})
        chat = dict(message.get("chat") or {})
        chat_id = chat.get("id")
        authorized = self.access.authorize(
            user.get("id"), None if chat_id is None else int(chat_id), chat.get("type")
        )
        answer = self.approvals.handle_callback(callback_query, authorized=authorized)
        query_id = callback_query.get("id")
        if not query_id:
            return
        if answer:
            self.telegram.answer_callback_query(str(query_id), text=answer[:190])
        else:
            self.telegram.answer_callback_query(str(query_id))

    # ------------------------------------------------------------------ commands

    def _handle_command(self, chat_id: int, user_id: int, text: str) -> None:
        parts = text.split()
        command = parts[0][1:].split("@", 1)[0].strip().lower()
        args = parts[1:]
        handlers: dict[str, Callable[[int, list[str]], None]] = {
            "start": lambda cid, _args: self._reply(cid, HELP_TEXT),
            "help": lambda cid, _args: self._reply(cid, HELP_TEXT),
            "projects": lambda cid, _args: self._cmd_projects(cid),
            "bind": lambda cid, cmd_args: self._cmd_bind(cid, cmd_args),
            "new": lambda cid, _args: self._cmd_new(cid),
            "stop": lambda cid, _args: self._cmd_stop(cid),
            "status": lambda cid, _args: self._cmd_status(cid),
            "unbind": lambda cid, _args: self._cmd_unbind(cid),
        }
        handler = handlers.get(command)
        if handler is None:
            self._reply(chat_id, f"Unknown command /{command}. Try /help.")
            return
        try:
            handler(chat_id, args)
        except TelegramError as exc:
            self.log.error("telegram error while handling /%s: %s", command, exc)

    def _cmd_projects(self, chat_id: int) -> None:
        rows = self.router.list_projects(chat_id)
        if not rows:
            self._reply(
                chat_id,
                "No projects discovered.\n"
                f"Scanned: {self.config.projects_root} (depth {self.config.discovery_depth}) and "
                f"{self.config.agent_index_dir}.\n"
                "Put a repository in there, or bind an explicit path with /bind <path>.",
            )
            return
        binding = self.router.get(chat_id)
        lines = [f"Projects (allowed roots: {self.router.roots_label()})"]
        for row in rows:
            if binding is not None and row["root"] == binding.project_root:
                marker = "✅"
            elif row["bound_chat_ids"]:
                marker = "• "
            else:
                marker = "  "
            suffix = "" if row["bindable"] else "  (outside allowed roots — not bindable)"
            lines.append(f"{marker} {row['name']} — {row['path']}{suffix}")
        lines.append("")
        lines.append("Bind with /bind <name> (or /bind <name> <path>).")
        self._reply(chat_id, "\n".join(lines))

    def _cmd_bind(self, chat_id: int, args: Sequence[str]) -> None:
        if not args:
            self._reply(chat_id, "Usage: /bind <name> [<path>]  — see /projects.")
            return
        target = args[0]
        path = args[1] if len(args) > 1 else None
        try:
            binding = self.router.bind(chat_id, target, path)
        except ContainmentError as exc:
            self.log.warning("containment gate rejected bind for chat %s: %s", chat_id, exc)
            self._reply(chat_id, f"🚫 Refused: {exc}")
            return
        except DiscoveryError as exc:
            self._reply(chat_id, f"❓ {exc}")
            return
        self._reply(
            chat_id,
            f"✅ Bound to {binding.project_name}: {binding.project_root}\n"
            "Send /new for a fresh session, or just type your prompt.",
        )

    def _cmd_unbind(self, chat_id: int) -> None:
        if self.router.unbind(chat_id):
            self._reply(chat_id, "Unbound. Use /projects then /bind <name> to pick another.")
        else:
            self._reply(chat_id, "This chat was not bound to anything.")

    def _cmd_new(self, chat_id: int) -> None:
        binding = self.router.get(chat_id)
        if binding is None:
            self._reply(chat_id, self._no_binding_text())
            return
        runtime = self.router.runtime(chat_id)
        if runtime.busy:
            self._reply(chat_id, "A turn is running. Send /stop first, then /new.")
            return
        old_session = binding.session_id
        self.router.set_session(chat_id, None)
        self._forget_live(old_session)
        try:
            session_id = self._ensure_session(chat_id, binding, force_new=True)
        except AcpError as exc:
            self._reply(chat_id, f"⚠️ Could not start a session: {exc}")
            return
        self._reply(
            chat_id,
            f"🆕 Fresh session {session_id[:8]} in {binding.project_root}.\n"
            f"model {binding.model or '?'} · approval {binding.approval or '?'}",
        )

    def _cmd_stop(self, chat_id: int) -> None:
        binding = self.router.get(chat_id)
        runtime = self.router.runtime(chat_id)
        dropped = 0
        with runtime.lock:
            dropped = len(runtime.queue)
            runtime.queue.clear()
        if not runtime.busy or binding is None or not binding.session_id:
            self._reply(chat_id, "Nothing is running." + self._dropped_note(dropped))
            return
        self.acp.cancel(binding.session_id)
        self._reply(chat_id, "⏹️ Cancelling the running turn." + self._dropped_note(dropped))

    def _dropped_note(self, dropped: int) -> str:
        if not dropped:
            return ""
        return f" Dropped {dropped} queued message(s)."

    def _cmd_status(self, chat_id: int) -> None:
        binding = self.router.get(chat_id)
        runtime = self.router.runtime(chat_id)
        caps = self.capabilities
        lines = [
            "📊 status",
            f"chat: {chat_id}",
        ]
        if binding is None:
            lines.append("project: (not bound) — see /projects")
        else:
            lines.append(f"project: {binding.project_name} — {binding.project_root}")
            live = binding.session_id in self._live_sessions if binding.session_id else False
            lines.append(
                f"session: {binding.short_session()} "
                f"({'live in this process' if live else 'will resume on the next message' if binding.session_id else 'none yet'})"
            )
            lines.append(f"model: {binding.model or '?'} · mode: {binding.mode or '?'}")
            lines.append(f"approval posture: {binding.approval or 'ask'}")
        lines.append(f"agent: {'running' if self.acp.running else 'stopped'} (pid {self.acp.pid})")
        if caps is not None:
            lines.append(
                f"capabilities: {caps.agent_name} {caps.agent_version} · "
                f"steer {caps.steer_method or 'unsupported'} · "
                f"list {'yes' if caps.supports_list else 'no'} · "
                f"load {'yes' if caps.load_session else 'no'}"
            )
        lines.append(
            f"state: {runtime.state_label()} · queued: {runtime.queue_size} · "
            f"approvals open: {self.approvals.open_count(chat_id)} · "
            f"busy mode: {self.config.busy_mode}"
        )
        self._reply(chat_id, "\n".join(lines))

    def _no_binding_text(self) -> str:
        return (
            "This chat is not bound to a project yet.\n"
            "Send /projects to see what is available, then /bind <name>."
        )

    # ------------------------------------------------------------------ prompting

    def _handle_prompt(self, chat_id: int, text: str) -> None:
        """Route a message either into the running turn or into a new one.

        No Bot API call happens while the chat lock is held, so /stop never waits
        on a slow network round trip.
        """
        binding = self.router.get(chat_id)
        if binding is None:
            self._reply(chat_id, self._no_binding_text())
            return

        runtime = self.router.runtime(chat_id)
        steer = False
        position = 0
        start_turn = False
        with runtime.lock:
            runtime.last_activity = self.clock()
            if runtime.busy:
                if self._can_steer(binding):
                    steer = True
                else:
                    position = runtime.enqueue(text)
            else:
                runtime.busy = True
                start_turn = True

        if steer:
            if self._steer(chat_id, binding, text):
                return
            with runtime.lock:
                position = runtime.enqueue(text)

        if position:
            self._reply(
                chat_id,
                f"⏳ A turn is already running; queued as #{position}. "
                "It will run as soon as the current turn ends (/stop drops the queue).",
            )
            return

        if start_turn:
            worker = threading.Thread(
                target=self._turn_loop, args=(chat_id, text), name=f"turn-{chat_id}", daemon=True
            )
            worker.start()

    def _can_steer(self, binding: Binding) -> bool:
        """Steering needs the config, an advertised capability and a live session."""
        caps = self.capabilities
        return (
            self.config.busy_mode == "steer"
            and caps is not None
            and caps.supports_steer
            and binding.session_id is not None
            and binding.session_id in self._live_sessions
        )

    def _steer(self, chat_id: int, binding: Binding, text: str) -> bool:
        """Send mid-turn guidance. False when that failed (the caller queues instead)."""
        try:
            self.acp.steer(binding.session_id or "", text)
        except AcpError as exc:
            self.log.warning("steer failed (%s); queueing instead", exc)
            return False
        self._reply(chat_id, "↪️ Steered the running turn.")
        return True

    def _turn_loop(self, chat_id: int, text: str) -> None:
        """Run turns for one chat, draining the queue, then mark the chat idle.

        The normal exit clears ``busy`` *inside* the same critical section as the
        dequeue, and returns from there: a message arriving at that instant starts
        its own turn instead of queueing behind a turn that has already finished.
        Only the abnormal exits (the binding disappeared, or a crash) still own the
        chat, and those are the paths that clean up.
        """
        runtime = self.router.runtime(chat_id)
        try:
            while True:
                binding = self.router.get(chat_id)
                if binding is None:
                    break
                self._execute_turn(chat_id, binding, text)
                with runtime.lock:
                    queued = runtime.dequeue()
                    if queued is None:
                        runtime.busy = False
                        runtime.last_activity = self.clock()
                        return
                    text = queued
        except Exception:  # pragma: no cover - defensive: keep the gateway alive
            self.log.exception("turn loop for chat %s failed", chat_id)

        with runtime.lock:
            runtime.busy = False
            runtime.last_activity = self.clock()
            dropped = len(runtime.queue)
            runtime.queue.clear()
        if dropped:
            self.log.warning(
                "dropped %d queued message(s) for chat %s after a failure", dropped, chat_id
            )
            self._reply(
                chat_id,
                f"⚠️ Dropped {dropped} queued message(s) after an internal error. "
                "Please send them again.",
            )

    def _execute_turn(self, chat_id: int, binding: Binding, text: str) -> None:
        view = TurnView()
        stream = MessageStream(
            self.telegram,
            chat_id,
            edit_interval=self.config.edit_interval,
            log=self.log,
        )
        if not self._stream_push(stream, view.render()):
            # Telegram is unreachable: do not spend a turn nobody would ever see.
            self.log.error("cannot reach Telegram for chat %s; skipping this turn", chat_id)
            self._reply(
                chat_id,
                "⚠️ Cannot reach Telegram right now, so this turn was not started. "
                "Please send the message again.",
            )
            return
        self.telegram.send_chat_action(chat_id, "typing")

        error_text: str | None = None
        stop_reason: str | None = None
        try:
            try:
                session_id = self._ensure_session(chat_id, binding)
            except AcpError as exc:
                if binding.session_id:
                    self._forget_live(binding.session_id)
                error_text = f"⚠️ agent error: {exc}"
            else:

                def on_update(notification: Mapping[str, Any]) -> None:
                    nonlocal error_text
                    try:
                        view.apply(notification)
                    except Exception:  # pragma: no cover - defensive: keep the turn alive
                        self.log.exception("failed to render a session/update")
                        error_text = "⚠️ failed to render an agent update"
                        return
                    self._stream_push(stream, view.render())

                try:
                    result = self.acp.prompt(
                        session_id,
                        text,
                        on_update=on_update,
                        on_tick=lambda: self._stream_flush(stream),
                        timeout=self.config.turn_timeout,
                    )
                    stop_reason = result.stop_reason
                except AcpTimeout as exc:
                    error_text = f"⏱️ {exc}"
                except AgentCrashed:
                    error_text = (
                        "💥 the agent crashed mid-turn; it is restarting. Send the prompt again."
                    )
                except AcpError as exc:
                    if binding.session_id:
                        self._forget_live(binding.session_id)
                    error_text = f"⚠️ agent error: {exc}"

            final = view.render()
            if error_text:
                final = f"{final}\n\n{error_text}" if final else error_text
            else:
                mark = _STOP_MARKS.get(
                    str(stop_reason), f"⏹️ stopped ({stop_reason or 'unknown'})"
                )
                if view.tool_calls:
                    mark = f"{mark} · {view.tool_calls} tool call(s)"
                final = f"{final}\n\n{mark}"
            self._stream_close(stream, final)
        finally:
            # A turn that is over must not leave live approval buttons behind.
            self.approvals.cancel_chat(chat_id, "the turn ended")

    # ------------------------------------------------------------------ streaming

    def _stream_push(self, stream: MessageStream, text: str) -> bool:
        """Coalesced push that never lets a Telegram failure kill a turn."""
        try:
            stream.push(text)
        except TelegramError as exc:
            self.log.error("Telegram push failed for chat %s: %s", stream.chat_id, exc)
            return False
        return True

    def _stream_flush(self, stream: MessageStream) -> None:
        try:
            stream.flush()
        except TelegramError as exc:
            self.log.error("Telegram edit failed for chat %s: %s", stream.chat_id, exc)

    def _stream_close(self, stream: MessageStream, text: str) -> None:
        try:
            stream.close(text)
        except TelegramError as exc:
            self.log.error("Telegram final edit failed for chat %s: %s", stream.chat_id, exc)
            if not stream.messages:
                self._reply(stream.chat_id, f"⚠️ could not deliver the answer: {exc}")

    # ------------------------------------------------------------------ sessions

    def _ensure_agent(self) -> AgentCapabilities:
        """Start the agent (if needed) and complete the handshake once per process."""
        if not self.acp.running:
            self.acp.start()
            self.capabilities = None
        if self.capabilities is None:
            self.capabilities = self.acp.initialize()
        return self.capabilities

    def _ensure_session(self, chat_id: int, binding: Binding, *, force_new: bool = False) -> str:
        """Return a session id that is live in this process.

        Order of preference, per the spec: reuse the session we already attached;
        re-attach the bound session (``session/load``); resume the most recent
        session for that ``cwd`` (``session/list``); otherwise create a new one.
        """
        caps = self._ensure_agent()
        session_id = binding.session_id
        with self._sessions_lock:
            if not force_new and session_id and session_id in self._live_sessions:
                return session_id
        cwd = binding.project_root

        if not force_new and session_id and caps.load_session:
            try:
                loaded = self.acp.load_session(session_id, cwd)
                session_id = loaded.session_id
                self._record_session(chat_id, binding, loaded)
                self._apply_approval_posture(chat_id, session_id, loaded)
            except AcpError as exc:
                self.log.info("could not re-attach session %s: %s", session_id, exc)
                session_id = None

        if not force_new and session_id is None and caps.supports_list and caps.load_session:
            newest = self._newest_session(cwd)
            if newest is not None:
                try:
                    loaded = self.acp.load_session(newest.session_id, cwd)
                    session_id = loaded.session_id
                    self._record_session(chat_id, binding, loaded)
                    self._apply_approval_posture(chat_id, session_id, loaded)
                    self.log.info("resumed most recent session %s for %s", session_id, cwd)
                except AcpError as exc:
                    self.log.info("could not resume session %s: %s", newest.session_id, exc)
                    session_id = None

        if session_id is None:
            created = self.acp.new_session(cwd)
            session_id = created.session_id
            self._record_session(chat_id, binding, created)
            self._apply_approval_posture(chat_id, session_id, created)

        with self._sessions_lock:
            self._live_sessions.add(session_id)
        return session_id

    def _record_session(self, chat_id: int, binding: Binding, session: Any) -> None:
        model = session.model if hasattr(session, "model") else None
        mode = None
        approval = session.approval_posture if hasattr(session, "approval_posture") else None
        raw = getattr(session, "raw", None)
        if isinstance(raw, Mapping):
            modes = raw.get("modes")
            if isinstance(modes, Mapping):
                mode = modes.get("currentModeId")
        self.router.set_session(
            chat_id,
            session.session_id,
            model=str(model) if model is not None else None,
            mode=str(mode) if mode is not None else None,
            approval=str(approval) if approval is not None else None,
        )

    def _apply_approval_posture(self, chat_id: int, session_id: str, session: Any) -> None:
        """Ask for the configured ``tool_approval`` posture when it differs.

        The spec's default is ``ask``. The reference agent already defaults to it,
        so this is a no-op there; if the option is missing (or the agent refuses the
        call) the agent's own posture stays in charge and we say so in the log.
        """
        desired = self.config.approval_posture
        if not desired:
            return
        option = session.option("tool_approval") if hasattr(session, "option") else None
        if option is None:
            return
        current = session.approval_posture if hasattr(session, "approval_posture") else None
        if current is not None and str(current) == desired:
            return
        try:
            self.acp.set_config_option(session_id, "tool_approval", desired)
        except AcpError as exc:
            self.log.info(
                "could not set tool_approval=%s on session %s (%s); keeping %r",
                desired,
                session_id[:8],
                exc,
                current,
            )
            return
        self.log.info("tool_approval set to %s for session %s", desired, session_id[:8])
        self.router.set_session(chat_id, session_id, approval=desired)

    def _newest_session(self, cwd: str) -> SessionInfo | None:
        try:
            sessions = self.acp.list_sessions(cwd)
        except AcpError as exc:
            self.log.info("session/list failed for %s: %s", cwd, exc)
            return None
        if not sessions:
            return None
        with_updated = [session for session in sessions if session.updated_at]
        if with_updated:
            return max(with_updated, key=lambda session: str(session.updated_at))
        return sessions[0]

    def _forget_live(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._sessions_lock:
            self._live_sessions.discard(session_id)

    def _on_agent_restart(self) -> None:
        """Called from the ACP reader thread: do not block on ACP requests here."""
        self.log.warning("agent restarted; live session handles dropped, they will be resumed")
        self.capabilities = None
        with self._sessions_lock:
            self._live_sessions.clear()

    def _chat_for_session(self, session_id: str) -> int | None:
        for chat_id, binding in self.router.bindings.items():
            if binding.session_id == session_id:
                return chat_id
        return None

    # ------------------------------------------------------------------ access

    def _offer_pairing(
        self, user_id: Any, chat_id: int, user: Mapping[str, Any], chat_type: str
    ) -> None:
        if user_id is None:
            return
        username = user.get("username") or user.get("first_name")
        request = self.access.request_pairing(int(user_id), chat_id, username)
        if request is None:
            return
        self.router.save()
        self.log.warning(
            "PAIRING: user %s (%s) in %s chat %s is not allowed. Code %s (valid %.0fs). "
            "Approve with: python -m acp_im_gateway pairing approve %s",
            request.user_id,
            username or "no username",
            chat_type,
            chat_id,
            request.code,
            self.config.pairing_ttl,
            request.code,
        )
        self._reply(
            chat_id,
            "🔒 Not authorised.\n"
            "A one-time pairing code was written to the gateway log. Ask the operator to "
            "approve it, then send /projects.",
        )

    def _merge_external_state(self) -> None:
        """Adopt allowlist/pairing changes made by the CLI in another process."""
        try:
            changed = self.router.merge_external()
        except Exception:  # pragma: no cover - state file races
            self.log.exception("failed to merge external state")
            return
        if not changed:
            return
        self.log.info("allowlist updated from the state file")
        for request in self.access.pending_requests():
            if request.approved_at is None or request.notified:
                continue
            self._reply(
                request.chat_id,
                "✅ You are approved. Send /projects to see candidates, then /bind <name>.",
            )
            self.access.mark_notified(request.code)
        self.router.save()

    # ------------------------------------------------------------------ outbound

    def _reply(self, chat_id: int, text: str) -> None:
        try:
            self.telegram.send_message(chat_id, text, chunk=True)
        except TelegramError as exc:
            self.log.error("sendMessage to chat %s failed: %s", chat_id, exc)
