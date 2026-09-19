"""Approval bridge: ACP permission request -> inline buttons -> ACP response."""

from __future__ import annotations

from typing import Any, Mapping

import threading

import pytest

from acp_im_gateway.acp import DECLINE, DEFER, InboundRequest
from acp_im_gateway.approvals import (
    ApprovalBridge,
    PermissionOption,
    PermissionRequest,
    build_keyboard,
    render_request,
)
from acp_im_gateway.telegram import TelegramError

from .helpers import FakeClock, FakeTelegram, approval_buttons, wait_until

CHAT_ID = 111

PERMISSION_PARAMS: dict[str, Any] = {
    "sessionId": "sess-1",
    "toolCall": {
        "toolCallId": "call-42",
        "title": "Run the test suite",
        "kind": "execute",
        "status": "pending",
        "rawInput": {"command": "pytest -q", "cwd": "/home/dev/app"},
    },
    "options": [
        {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
        {"optionId": "allow_always", "name": "Always allow", "kind": "allow_always"},
        {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
    ],
}


class RecordingAgent:
    """Stands in for AcpClient: records the JSON-RPC replies the bridge would send."""

    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    def _write(self, message: Mapping[str, Any]) -> bool:
        self.writes.append(dict(message))
        return True

    def _forget_inbound(self, request_id: Any) -> None:
        return None


def make_inbound(
    agent: RecordingAgent,
    *,
    params: Mapping[str, Any] | None = None,
    method: str = "session/request_permission",
    request_id: int = 7,
) -> InboundRequest:
    return InboundRequest(agent, request_id, method, params if params is not None else PERMISSION_PARAMS)


def wait_delivery(timeout: float = 5.0) -> None:
    """Wait for the bridge's background delivery threads to settle."""
    settled = wait_until(
        lambda: not any(thread.name.startswith("approval-") for thread in threading.enumerate()),
        timeout=timeout,
    )
    assert settled, "approval delivery did not settle"


def open_request(
    bridge: ApprovalBridge,
    agent: RecordingAgent,
    *,
    params: Mapping[str, Any] | None = None,
    request_id: int = 7,
) -> InboundRequest:
    """Register a permission request and wait for the prompt to be delivered."""
    inbound = make_inbound(agent, params=params, request_id=request_id)
    assert bridge.handle_inbound(inbound) is DEFER
    wait_delivery()
    return inbound

class BlockingTelegram(FakeTelegram):
    """Telegram double whose sends block until the test releases them."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def send_message(self, chat_id: int, text: str, **kwargs: Any) -> list[dict[str, Any]]:
        assert self.release.wait(timeout=5.0), "the test never released the send"
        return super().send_message(chat_id, text, **kwargs)


def make_bridge(
    telegram: FakeTelegram,
    clock: FakeClock,
    *,
    chat_id: int | None = CHAT_ID,
    tokens: list[str] | None = None,
    **kwargs: Any,
) -> ApprovalBridge:
    codes = list(tokens or ["tok12345"])
    return ApprovalBridge(
        telegram,  # type: ignore[arg-type]
        chat_for_session=lambda session_id: chat_id,
        clock=clock,
        token_factory=lambda: codes.pop(0),
        **kwargs,
    )


# --------------------------------------------------------------------------- parsing


def test_request_parsing_and_option_ordering() -> None:
    request = PermissionRequest.from_params(PERMISSION_PARAMS, 7)
    assert request.session_id == "sess-1"
    assert request.title == "Run the test suite"
    assert request.kind == "execute"
    assert request.tool_call_id == "call-42"
    assert "command: pytest -q" in request.detail()
    assert [option.option_id for option in request.ordered_options()] == [
        "allow_once",
        "allow_always",
        "reject_once",
    ]
    assert request.has_reject() is True


def test_render_is_plain_text_with_the_essentials() -> None:
    text = render_request(PermissionRequest.from_params(PERMISSION_PARAMS, 7))
    assert "Approval needed — Run the test suite" in text
    assert "kind: execute" in text
    assert "pytest -q" in text
    assert "Tap a button" in text


def test_keyboard_uses_advertised_options_plus_deny_fallback() -> None:
    request = PermissionRequest.from_params(PERMISSION_PARAMS, 7)
    markup = build_keyboard(request, "tok12345")
    buttons = markup["inline_keyboard"][0]
    assert [button["callback_data"] for button in buttons] == ["ap:tok12345:0", "ap:tok12345:1"]
    assert markup["inline_keyboard"][1][0]["callback_data"] == "ap:tok12345:2"
    assert markup["inline_keyboard"][1][0]["text"] == "Reject"

    without_reject = dict(PERMISSION_PARAMS)
    without_reject["options"] = [
        {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"}
    ]
    plain = build_keyboard(PermissionRequest.from_params(without_reject, 7), "tok12345")
    assert plain["inline_keyboard"][-1][0] == {"text": "Deny", "callback_data": "ap:tok12345:x"}


def test_parse_callback_data() -> None:
    bridge = make_bridge(FakeTelegram(), FakeClock())
    assert bridge.parse_callback_data("ap:tok12345:2") == ("tok12345", "2")
    assert bridge.parse_callback_data("ap:tok12345:x") == ("tok12345", "x")
    assert bridge.parse_callback_data("something-else") is None
    assert bridge.parse_callback_data("ap:broken") is None


# --------------------------------------------------------------------------- inbound


def test_permission_request_becomes_one_message_with_buttons_and_defers() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    bridge = make_bridge(telegram, clock)
    agent = RecordingAgent()

    open_request(bridge, agent)

    assert agent.writes == []  # not answered yet: the user has to tap
    assert len(telegram.sent(CHAT_ID)) == 1
    buttons = approval_buttons(telegram, CHAT_ID)
    assert [button["text"] for button in buttons] == ["Allow once", "Always allow", "Reject"]
    assert bridge.open_count(CHAT_ID) == 1


def test_approval_delivery_does_not_block_the_caller() -> None:
    """The handler runs on the ACP reader thread: a slow send must not stall it."""
    telegram = BlockingTelegram()
    bridge = make_bridge(telegram, FakeClock())
    agent = RecordingAgent()

    assert bridge.handle_inbound(make_inbound(agent)) is DEFER
    assert bridge.open_count(CHAT_ID) == 1  # registered before delivery finishes
    assert telegram.sent() == []  # still in flight

    telegram.release.set()
    wait_delivery()
    assert len(telegram.sent(CHAT_ID)) == 1


def test_cancelled_before_delivery_leaves_no_stale_buttons() -> None:
    telegram = BlockingTelegram()
    bridge = make_bridge(telegram, FakeClock())
    agent = RecordingAgent()
    bridge.handle_inbound(make_inbound(agent))
    assert bridge.cancel_chat(CHAT_ID, "the turn ended") == 1
    telegram.release.set()
    wait_delivery()
    prompt_id = telegram.sent(CHAT_ID)[0].message_id
    decorated = [
        message
        for message in telegram.messages
        if message.message_id == prompt_id and message.reply_markup == {}
    ]
    assert decorated, "the delivered prompt should have had its buttons removed"


def test_non_permission_requests_are_declined() -> None:
    bridge = make_bridge(FakeTelegram(), FakeClock())
    agent = RecordingAgent()
    request = make_inbound(agent, method="fs/read_text_file", params={"sessionId": "sess-1"})
    assert bridge.handle_inbound(request) is DECLINE


def test_request_for_an_unbound_session_is_declined_cleanly() -> None:
    bridge = make_bridge(FakeTelegram(), FakeClock(), chat_id=None)
    agent = RecordingAgent()
    open_request(bridge, agent)
    assert agent.writes[0]["error"]["code"] == -32601
    assert bridge.pending == {}


def test_telegram_failure_declines_instead_of_hanging() -> None:
    telegram = FakeTelegram()
    telegram.fail_send = TelegramError("chat not found", status=400)
    bridge = make_bridge(telegram, FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent)
    assert agent.writes[0]["error"]["message"]
    assert bridge.pending == {}


def test_too_many_open_approvals_in_one_chat_are_declined() -> None:
    telegram = FakeTelegram()
    bridge = make_bridge(telegram, FakeClock(), tokens=["tok00001", "tok00002"], max_open_per_chat=1)
    first, second = RecordingAgent(), RecordingAgent()
    assert bridge.handle_inbound(make_inbound(first)) is DEFER
    assert bridge.handle_inbound(make_inbound(second)) is DEFER
    wait_delivery()
    assert second.writes[0]["error"]["code"] == -32603
    assert bridge.open_count(CHAT_ID) == 1


# --------------------------------------------------------------------------- callback


def test_callback_answers_the_agent_and_clears_the_buttons() -> None:
    telegram = FakeTelegram()
    bridge = make_bridge(telegram, FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent)

    message = telegram.sent(CHAT_ID)[0]
    answer = bridge.handle_callback(
        {"data": "ap:tok12345:1", "message": {"chat": {"id": CHAT_ID}}}, authorized=True
    )

    assert answer == "Answered: Always allow"
    assert agent.writes == [
        {
            "jsonrpc": "2.0",
            "id": 7,
            "result": {"outcome": "selected", "optionId": "allow_always"},
        }
    ]
    assert bridge.pending == {}
    assert bridge.open_count(CHAT_ID) == 0
    final_edit = telegram.edits(CHAT_ID)[-1]
    assert final_edit.message_id == message.message_id
    assert final_edit.reply_markup == {}
    assert "➡️ Always allow" in final_edit.text


def test_deny_button_with_an_advertised_reject_option() -> None:
    bridge = make_bridge(FakeTelegram(), FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent)
    bridge.handle_callback(
        {"data": "ap:tok12345:2", "message": {"chat": {"id": CHAT_ID}}}, authorized=True
    )
    assert agent.writes[0]["result"] == {"outcome": "selected", "optionId": "reject_once"}


def test_deny_button_without_a_reject_option_cancels() -> None:
    params = dict(PERMISSION_PARAMS)
    params["options"] = [{"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"}]
    bridge = make_bridge(FakeTelegram(), FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent, params=params)
    bridge.handle_callback(
        {"data": "ap:tok12345:x", "message": {"chat": {"id": CHAT_ID}}}, authorized=True
    )
    assert agent.writes[0]["result"] == {"outcome": "cancelled"}


def test_callback_from_an_unauthorised_user_is_refused() -> None:
    telegram = FakeTelegram()
    bridge = make_bridge(telegram, FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent)

    answer = bridge.handle_callback(
        {"data": "ap:tok12345:0", "message": {"chat": {"id": CHAT_ID}}}, authorized=False
    )
    assert answer == "Not authorised."
    assert agent.writes == []  # nothing answered
    assert bridge.open_count(CHAT_ID) == 1


def test_callback_from_another_chat_is_refused() -> None:
    telegram = FakeTelegram()
    bridge = make_bridge(telegram, FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent)
    answer = bridge.handle_callback(
        {"data": "ap:tok12345:0", "message": {"chat": {"id": 999}}}, authorized=True
    )
    assert answer == "Not authorised."
    assert agent.writes == []


def test_unknown_or_reused_tokens_are_reported() -> None:
    bridge = make_bridge(FakeTelegram(), FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent)
    assert (
        bridge.handle_callback({"data": "ap:nope1234:0", "message": {"chat": {"id": CHAT_ID}}}, authorized=True)
        == "This approval is no longer valid."
    )
    bridge.handle_callback(
        {"data": "ap:tok12345:0", "message": {"chat": {"id": CHAT_ID}}}, authorized=True
    )
    assert (
        bridge.handle_callback({"data": "ap:tok12345:0", "message": {"chat": {"id": CHAT_ID}}}, authorized=True)
        == "This approval is no longer valid."
    )


def test_bad_option_index_keeps_the_prompt_alive() -> None:
    bridge = make_bridge(FakeTelegram(), FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent)
    assert (
        bridge.handle_callback(
            {"data": "ap:tok12345:99", "message": {"chat": {"id": CHAT_ID}}}, authorized=True
        )
        == "Unknown option."
    )
    assert agent.writes == []
    assert bridge.open_count(CHAT_ID) == 1
    # A valid tap afterwards still works.
    bridge.handle_callback(
        {"data": "ap:tok12345:0", "message": {"chat": {"id": CHAT_ID}}}, authorized=True
    )
    assert agent.writes[0]["result"]["optionId"] == "allow_once"


def test_expiry_answers_cancelled_and_updates_the_message() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    bridge = make_bridge(telegram, clock, timeout=60.0)
    agent = RecordingAgent()
    open_request(bridge, agent)

    assert bridge.expire_due() == 0
    clock.advance(61.0)
    assert bridge.expire_due() == 1
    assert agent.writes[0]["result"] == {"outcome": "cancelled"}
    assert "expired" in telegram.edits(CHAT_ID)[-1].text
    assert bridge.open_count(CHAT_ID) == 0


def test_cancel_chat_clears_stale_buttons() -> None:
    telegram = FakeTelegram()
    bridge = make_bridge(telegram, FakeClock())
    agent = RecordingAgent()
    open_request(bridge, agent)
    assert bridge.cancel_chat(CHAT_ID, "the turn ended") == 1
    assert agent.writes[0]["result"] == {"outcome": "cancelled"}
    assert "the turn ended" in telegram.edits(CHAT_ID)[-1].text
    assert bridge.open_count(CHAT_ID) == 0
    assert bridge.cancel_chat(CHAT_ID) == 0


def test_option_labels_helper() -> None:
    from acp_im_gateway.approvals import option_labels

    request = PermissionRequest.from_params(PERMISSION_PARAMS, 7)
    assert option_labels(request.ordered_options()) == ["Allow once", "Always allow", "Reject"]
    assert PermissionOption("x", "X", "allow_once").is_allow is True


def test_pending_state_is_thread_safe_under_concurrent_use() -> None:
    """Delivery threads, expiry, cancellation and resolution all touch ``pending``."""
    telegram = FakeTelegram()
    telegram.fail_send = TelegramError("no network", status=500)
    tokens = [f"tok{index:05d}" for index in range(40)]
    bridge = make_bridge(telegram, FakeClock(), tokens=tokens, max_open_per_chat=50, timeout=1.0)
    agents = [RecordingAgent() for _ in tokens]
    errors: list[BaseException] = []
    stop = threading.Event()

    def stress() -> None:
        try:
            while not stop.is_set():
                bridge.expire_due(now=bridge.clock() + 60)  # force expiry of everything
                bridge.open_count(CHAT_ID)
                bridge.cancel_chat(CHAT_ID)
        except BaseException as exc:  # pragma: no cover - only on a real race
            errors.append(exc)

    workers = [threading.Thread(target=stress, name=f"stress-{index}") for index in range(3)]
    for worker in workers:
        worker.start()
    try:
        for agent in agents:
            assert bridge.handle_inbound(make_inbound(agent)) is DEFER
    finally:
        stop.set()
        for worker in workers:
            worker.join(timeout=10)
    wait_delivery()

    assert errors == []
    assert bridge.pending == {}
    assert all(agent.writes for agent in agents), "every request must be answered or declined"
