"""Approval bridge: ``session/request_permission`` -> Telegram buttons -> ACP response.

The agent asks the *client* for permission before a gated tool call. The gateway
answers it in one of two ways:

* the approval tiers (:mod:`acp_im_gateway.tiers`) approve it silently when the
  call is in ``auto_allow`` — no tap, logged at INFO with the reason — or force a
  tap when it matches ``always_ask``, which is a code gate and wins over
  ``auto_allow``;
* otherwise one Telegram message with inline buttons, keeping the ACP request
  open (``DEFER``) and answering it with the exact ``optionId`` the agent
  advertised once the user taps.

A callback from an unauthorised user is refused. Unanswered approvals expire and
are answered as ``cancelled`` — the bridge never lets the agent hang.
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from .acp import (
    DECLINE,
    DEFER,
    InboundRequest,
    permission_cancelled,
    permission_selected,
)
from .telegram import TelegramClient, TelegramError, inline_keyboard
from .tiers import ApprovalDecision, ApprovalTiers, decide_request

_logger = logging.getLogger("acp_im_gateway.approvals")

CALLBACK_PREFIX = "ap"
MAX_CALLBACK_DATA = 64
MAX_OPEN_PER_CHAT = 4  # safety valve: never flood a chat with approval prompts

_KIND_ICONS = {
    "read": "📖",
    "edit": "✏️",
    "execute": "▶️",
    "think": "💭",
    "fetch": "🌐",
    "search": "🔎",
    "other": "🔧",
}
_ALLOW_KINDS = ("allow_once", "allow_always", "allow")
_REJECT_KINDS = ("reject_once", "reject_always", "reject", "deny")


# --------------------------------------------------------------------------- parsing


@dataclass(frozen=True)
class PermissionOption:
    """One option the agent advertised for this request."""

    option_id: str
    name: str
    kind: str = "other"

    @property
    def is_allow(self) -> bool:
        return any(token in self.kind.lower() for token in _ALLOW_KINDS)

    @property
    def is_reject(self) -> bool:
        return any(token in self.kind.lower() for token in _REJECT_KINDS)

    @classmethod
    def from_raw(cls, raw: Mapping[str, Any]) -> "PermissionOption":
        return cls(
            option_id=str(raw.get("optionId") or raw.get("id") or ""),
            name=str(raw.get("name") or raw.get("optionId") or "Option"),
            kind=str(raw.get("kind") or "other"),
        )


@dataclass
class PermissionRequest:
    """A parsed ``session/request_permission`` payload."""

    request_id: Any
    session_id: str
    tool_call: dict[str, Any] = field(default_factory=dict)
    options: list[PermissionOption] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_call_id(self) -> str:
        return str(self.tool_call.get("toolCallId") or "")

    @property
    def kind(self) -> str:
        return str(self.tool_call.get("kind") or self.tool_call.get("toolKind") or "other")

    @property
    def title(self) -> str:
        return str(
            self.tool_call.get("title")
            or self.tool_call.get("name")
            or self.tool_call_id
            or "tool call"
        )

    def ordered_options(self) -> list[PermissionOption]:
        """Allow options first, then the rest (stable within each group)."""
        allows = [option for option in self.options if option.is_allow]
        rejects = [option for option in self.options if option.is_reject]
        middle = [
            option
            for option in self.options
            if not option.is_allow and not option.is_reject
        ]
        return allows + middle + rejects

    def has_reject(self) -> bool:
        return any(option.is_reject for option in self.options)

    def detail(self, limit: int = 400) -> str:
        """A short, human-readable hint about what is being requested."""
        bits: list[str] = []
        raw_input = self.tool_call.get("rawInput")
        if isinstance(raw_input, Mapping):
            preferred = ("command", "file_path", "path", "pattern", "url", "query", "description")
            for key in preferred:
                if raw_input.get(key) not in (None, ""):
                    bits.append(f"{key}: {_short(raw_input[key])}")
            extra = [key for key in raw_input if key not in preferred][:2]
            for key in extra:
                bits.append(f"{key}: {_short(raw_input[key])}")
        elif isinstance(raw_input, str) and raw_input.strip():
            bits.append(_short(raw_input))

        locations = self.tool_call.get("locations")
        if not bits and isinstance(locations, Iterable) and not isinstance(locations, (str, bytes)):
            paths: list[str] = []
            for location in locations or ():
                if isinstance(location, Mapping) and location.get("path"):
                    paths.append(str(location["path"]))
            if paths:
                bits.append(", ".join(paths[:3]))

        content = self.tool_call.get("content")
        if not bits and isinstance(content, Iterable) and not isinstance(content, (str, bytes)):
            for block in content or ():
                if isinstance(block, Mapping) and block.get("type") == "text":
                    text = str(block.get("text") or "").strip()
                    if text:
                        bits.append(_short(text))
                        break

        text = " · ".join(bits).strip()
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return text

    @classmethod
    def from_params(cls, params: Mapping[str, Any], request_id: Any = None) -> "PermissionRequest":
        tool_call = params.get("toolCall")
        options_raw = params.get("options") or []
        return cls(
            request_id=request_id,
            session_id=str(params.get("sessionId") or ""),
            tool_call=dict(tool_call) if isinstance(tool_call, Mapping) else {},
            options=[
                PermissionOption.from_raw(option)
                for option in options_raw
                if isinstance(option, Mapping)
            ],
            raw=dict(params),
        )


def _short(value: Any, limit: int = 160) -> str:
    if isinstance(value, (Mapping, list, tuple)):
        text = str(dict(value) if isinstance(value, Mapping) else list(value))
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_request(request: PermissionRequest) -> str:
    """The plain-text approval message (no parse_mode, so nothing can break)."""
    icon = _KIND_ICONS.get(request.kind.lower(), "🔐")
    lines = [f"{icon} Approval needed — {request.title}"]
    if request.kind:
        lines.append(f"kind: {request.kind}")
    detail = request.detail()
    if detail:
        lines.append(detail)
    if request.tool_call_id:
        lines.append(f"tool call: {request.tool_call_id}")
    lines.append("Tap a button to answer the agent.")
    return "\n".join(lines)


def build_keyboard(request: PermissionRequest, token: str) -> dict[str, Any]:
    """Inline keyboard: the agent's options, with a Deny fallback when it offers none."""
    rows: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    for index, option in enumerate(request.ordered_options()):
        label = option.name.strip() or option.option_id or f"Option {index + 1}"
        if len(label) > 28:
            label = label[:27] + "…"
        current.append((label, f"{CALLBACK_PREFIX}:{token}:{index}"))
        if len(current) == 2:
            rows.append(current)
            current = []
    if current:
        rows.append(current)
    if not request.has_reject():
        rows.append([("Deny", f"{CALLBACK_PREFIX}:{token}:x")])
    return inline_keyboard(rows)


# --------------------------------------------------------------------------- bridge


@dataclass
class PendingApproval:
    """An approval message waiting for a tap."""

    token: str
    request: PermissionRequest
    inbound: InboundRequest
    chat_id: int
    message_id: int | None
    options: list[PermissionOption]
    created_at: float
    expires_at: float
    resolved: bool = False
    decision: str | None = None


class ApprovalBridge:
    """Wire ACP permission requests to Telegram inline buttons and back."""

    def __init__(
        self,
        telegram: TelegramClient,
        *,
        chat_for_session: Callable[[str], int | None],
        timeout: float = 300.0,
        clock: Callable[[], float] = time.time,
        log: logging.Logger | None = None,
        token_factory: Callable[[], str] | None = None,
        max_open_per_chat: int = MAX_OPEN_PER_CHAT,
        tiers: ApprovalTiers | None = None,
        chat_thread: Callable[[int], int | None] | None = None,
    ) -> None:
        self.telegram = telegram
        self.chat_for_session = chat_for_session
        self.timeout = max(float(timeout), 0.0)
        self.clock = clock
        self.log = log or _logger
        self.token_factory = token_factory or (lambda: secrets.token_hex(4))
        self.max_open_per_chat = max(1, int(max_open_per_chat))
        #: Approval tiers: ``auto_allow`` approves silently, ``always_ask`` is a
        #: code gate that forces a tap. ``None`` means "always ask".
        self.tiers = tiers
        #: Forum topics: reply into the same thread the chat is talking in.
        self.chat_thread = chat_thread
        self.pending: dict[str, PendingApproval] = {}
        #: Guards ``pending``: it is touched by the ACP reader thread (new requests),
        #: background delivery threads, the polling loop (expiry) and turn threads.
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ inbound

    def handle_inbound(self, request: InboundRequest) -> Any:
        """``AcpClient.on_request`` handler. Returns DEFER/DECLINE, never blocks.

        The prompt is delivered from a short-lived background thread: this handler
        runs on the ACP *reader* thread, and a slow Bot API round trip must not
        stall the stdout stream that carries every update for every session.
        """
        if request.method != "session/request_permission":
            return DECLINE

        parsed = PermissionRequest.from_params(request.params, request.id)
        chat_id = self.chat_for_session(parsed.session_id)
        if chat_id is None:
            self.log.warning(
                "permission request for unbound session %s declined", parsed.session_id
            )
            request.fail(-32601, "no chat is bound to this session")
            return DEFER

        decision = decide_request(self.tiers, chat_id, parsed)
        if decision is not None and decision.auto_approve:
            if self._approve_without_a_tap(request, parsed, chat_id, decision):
                return DEFER
        elif decision is not None and decision.forced:
            self.log.info(
                "forced tap in chat %s for %s: %s", chat_id, parsed.title, decision.reason
            )

        if self.open_count(chat_id) >= self.max_open_per_chat:
            self.log.warning("too many open approvals for chat %s; declining", chat_id)
            request.fail(-32603, "too many approval prompts are already open in this chat")
            return DEFER

        moment = self.clock()
        approval = PendingApproval(
            token=self.token_factory(),
            request=parsed,
            inbound=request,
            chat_id=chat_id,
            message_id=None,
            options=parsed.ordered_options(),
            created_at=moment,
            expires_at=moment + self.timeout if self.timeout > 0 else float("inf"),
        )
        # Register before delivering so open_count()/expiry already account for it.
        with self._lock:
            self.pending[approval.token] = approval
        threading.Thread(
            target=self._deliver,
            args=(approval,),
            name=f"approval-{approval.token}",
            daemon=True,
        ).start()
        return DEFER

    def _approve_without_a_tap(
        self,
        request: InboundRequest,
        parsed: PermissionRequest,
        chat_id: int,
        decision: ApprovalDecision,
    ) -> bool:
        """Answer the agent straight away (``auto_allow``). False -> ask instead."""
        option = choose_allow_option(parsed)
        if option is None:
            self.log.warning(
                "auto-approve wanted for %s in chat %s but the agent offered no allow "
                "option; asking instead",
                parsed.title,
                chat_id,
            )
            return False
        if not request.respond(permission_selected(option.option_id)):
            self.log.warning(
                "could not deliver the automatic approval for %s; asking instead", parsed.title
            )
            return False
        self.log.info(
            "auto-approved %s in chat %s as %r (%s)",
            parsed.title,
            chat_id,
            option.option_id,
            decision.reason,
        )
        return True

    def _thread_for(self, chat_id: int) -> int | None:
        if self.chat_thread is None:
            return None
        try:
            return self.chat_thread(int(chat_id))
        except Exception:  # pragma: no cover - defensive
            self.log.debug("could not resolve a forum thread for chat %s", chat_id, exc_info=True)
            return None

    def _deliver(self, approval: PendingApproval) -> None:
        """Send the buttons for a registered approval (background thread)."""
        try:
            sent = self.telegram.send_message(
                approval.chat_id,
                render_request(approval.request),
                reply_markup=build_keyboard(approval.request, approval.token),
                message_thread_id=self._thread_for(approval.chat_id),
            )
        except TelegramError as exc:
            self.log.error("cannot deliver approval prompt: %s", exc)
            with self._lock:
                self.pending.pop(approval.token, None)
            approval.inbound.fail(-32603, "could not deliver the approval prompt to the chat")
            return

        if sent:
            try:
                approval.message_id = int(sent[0].get("message_id"))
            except (TypeError, ValueError):
                approval.message_id = None
        self.log.info(
            "approval requested in chat %s for %s (%d option(s))",
            approval.chat_id,
            approval.request.title,
            len(approval.request.options),
        )
        if approval.inbound.answered or approval.resolved:
            # Answered, expired or cancelled while we were sending: no stale buttons.
            self._finish_message(approval, approval.decision or "already answered")

    # ------------------------------------------------------------------ callback

    def parse_callback_data(self, data: str) -> tuple[str, str] | None:
        """``ap:<token>:<index>`` -> ``(token, index)``; None for other callbacks."""
        if not data or not data.startswith(f"{CALLBACK_PREFIX}:"):
            return None
        parts = data.split(":")
        if len(parts) != 3:
            return None
        return parts[1], parts[2]

    def handle_callback(self, callback_query: Mapping[str, Any], *, authorized: bool) -> str | None:
        """Answer an approval tap. Returns the text to show as a callback toast."""
        parsed = self.parse_callback_data(str(callback_query.get("data") or ""))
        if parsed is None:
            return None
        token, index = parsed
        with self._lock:
            approval = self.pending.get(token)
        if approval is None:
            return "This approval is no longer valid."
        if not authorized:
            self.log.warning("unauthorised callback for approval %s refused", token)
            return "Not authorised."

        message = callback_query.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        if chat_id is not None and int(chat_id) != approval.chat_id:
            self.log.warning(
                "callback for approval %s came from chat %s, expected %s",
                token,
                chat_id,
                approval.chat_id,
            )
            return "Not authorised."

        return self.resolve(token, index)

    def resolve(self, token: str, index: str) -> str:
        """Answer the ACP request for ``token`` with option ``index`` (``x`` = deny)."""
        with self._lock:
            approval = self.pending.get(token)
            if approval is None:
                return "This approval is no longer valid."
            if approval.inbound.answered:
                self.pending.pop(token, None)
                return "Already answered."

            if index == "x":
                result: dict[str, Any] = permission_cancelled()
                label = "Denied"
            else:
                try:
                    option = approval.options[int(index)]
                except (ValueError, IndexError):
                    # Keep the prompt alive so the user can still tap a valid button:
                    # dropping it here would leave the agent waiting forever.
                    self.log.warning("approval %s: bad option index %r", token, index)
                    return "Unknown option."
                # The agent's own optionId, wrapped in the nested outcome it decodes.
                result = permission_selected(option.option_id)
                label = option.name

            self.pending.pop(token, None)
            approval.resolved = True
            approval.decision = label

        answered = approval.inbound.respond(result)
        if not answered:
            self.log.warning("approval %s could not be delivered (agent gone?)", token)
        self._finish_message(approval, label)
        self.log.info("approval %s answered: %s", token, label)
        return f"Answered: {label}"

    def expire_due(self, *, now: float | None = None) -> int:
        """Resolve approvals nobody answered as ``cancelled``. Returns how many."""
        moment = self.clock() if now is None else now
        due: list[PendingApproval] = []
        with self._lock:
            for token, approval in list(self.pending.items()):
                if not approval.resolved and moment >= approval.expires_at:
                    del self.pending[token]
                    approval.resolved = True
                    approval.decision = "expired"
                    due.append(approval)
        for approval in due:
            approval.inbound.respond(permission_cancelled())
            self._finish_message(approval, "expired — answered as cancelled")
        if due:
            self.log.info("expired %d unanswered approval(s)", len(due))
        return len(due)

    def cancel_chat(self, chat_id: int, reason: str = "turn ended") -> int:
        """Answer every open approval in a chat as ``cancelled`` (turn over / stopped).

        Keeps stale buttons from outliving the turn they belong to.
        """
        chat_id = int(chat_id)
        cancelled: list[PendingApproval] = []
        with self._lock:
            for token, approval in list(self.pending.items()):
                if approval.chat_id == chat_id and not approval.resolved:
                    del self.pending[token]
                    approval.resolved = True
                    approval.decision = reason
                    cancelled.append(approval)
        for approval in cancelled:
            approval.inbound.respond(permission_cancelled())
            self._finish_message(approval, f"no longer needed — {reason}")
        if cancelled:
            self.log.info("cancelled %d open approval(s) in chat %s", len(cancelled), chat_id)
        return len(cancelled)

    def open_count(self, chat_id: int) -> int:
        chat_id = int(chat_id)
        with self._lock:
            return sum(1 for approval in self.pending.values() if approval.chat_id == chat_id)

    def _finish_message(self, approval: PendingApproval, suffix: str) -> None:
        """Replace the buttons with the outcome, so nobody taps a stale prompt."""
        if approval.message_id is None:
            return
        original = render_request(approval.request)
        text = f"{original}\n\n➡️ {suffix}"
        try:
            # reply_markup={} removes the keyboard.
            self.telegram.edit_message_text(
                approval.chat_id, approval.message_id, text, reply_markup={}
            )
        except TelegramError as exc:  # pragma: no cover - message could be deleted
            self.log.debug("could not mark approval %s as answered: %s", approval.token, exc)


def option_labels(options: Sequence[PermissionOption]) -> list[str]:
    """Convenience for tests/logs."""
    return [option.name for option in options]


def choose_allow_option(request: PermissionRequest) -> PermissionOption | None:
    """The option to answer with when the tiers approve without a tap.

    ``allow_once`` wins over ``allow_always``: a silent approval should not
    persist a grant the user never saw.
    """
    allows = [option for option in request.ordered_options() if option.is_allow]
    if not allows:
        return None
    for option in allows:
        haystack = f"{option.kind} {option.option_id}".lower()
        if "once" in haystack:
            return option
    return allows[0]
