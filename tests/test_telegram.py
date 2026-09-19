"""Telegram adapter: chunking, rate limits, 429 handling, in-place streaming."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from acp_im_gateway.telegram import (
    MAX_MESSAGE_LENGTH,
    MessageStream,
    TelegramClient,
    TelegramError,
    TelegramTransportError,
    Throttle,
    chunk_message,
    inline_keyboard,
    reassemble,
    scan_fences,
)

from .helpers import FakeClock, FakeTelegram


# --------------------------------------------------------------------------- chunking


def test_short_text_is_one_part() -> None:
    assert chunk_message("hello") == ["hello"]
    assert chunk_message("") == []
    assert chunk_message("x" * MAX_MESSAGE_LENGTH) == ["x" * MAX_MESSAGE_LENGTH]


def test_text_is_split_into_ordered_parts_of_at_most_4096() -> None:
    text = "x" * 5000
    parts = chunk_message(text)
    assert len(parts) == 2
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    assert reassemble(parts) == text


def test_long_paragraphs_split_on_boundaries() -> None:
    text = "\n".join(f"line {index} " + "y" * 60 for index in range(200))
    parts = chunk_message(text)
    assert len(parts) > 1
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    assert reassemble(parts) == text


def test_code_fences_stay_balanced_across_parts() -> None:
    text = "```python\n" + "\n".join(f"print({index})" for index in range(800)) + "\n```\n"
    parts = chunk_message(text)
    assert len(parts) > 1
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    # Every part is self-contained Markdown: no part leaves a fence open.
    for part in parts:
        assert scan_fences(part) is None, part[:80]
        assert part.count("```") % 2 == 0
    # Closing/reopening the fence preserves the code content.
    assert reassemble(parts) == text


def test_fence_is_reopened_with_its_language() -> None:
    text = "```bash\n" + "echo hi\n" * 700 + "```\n"
    parts = chunk_message(text)
    assert len(parts) > 1
    assert parts[1].startswith("```bash\n")


def test_prose_then_fence_spanning_the_boundary() -> None:
    text = "preamble\n" * 200 + "```\n" + "code line\n" * 600 + "```\n"
    parts = chunk_message(text)
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    assert all(scan_fences(part) is None for part in parts)
    assert reassemble(parts) == text


def test_small_limit_still_preserves_content() -> None:
    text = "\n".join(f"row {index}" for index in range(200))
    parts = chunk_message(text, limit=256)
    assert len(parts) > 1
    assert all(len(part) <= 256 for part in parts)
    assert reassemble(parts) == text


def test_limit_below_the_safety_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="limit must be"):
        chunk_message("x" * 1000, limit=10)


def test_unicode_is_split_without_corruption() -> None:
    text = "🎈 hola ñandú — 漢字\n" * 500
    parts = chunk_message(text)
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    assert reassemble(parts) == text


def test_scan_fences_detects_state() -> None:
    assert scan_fences("plain text") is None
    assert scan_fences("```python\ncode") == "python"
    assert scan_fences("```python\ncode\n```") is None
    assert scan_fences("```\ncode", "python") is None  # closing a fence from before


# --------------------------------------------------------------------------- throttle


def test_throttle_enforces_minimum_interval_per_key() -> None:
    clock = FakeClock()
    throttle = Throttle(1.0, clock=clock, sleeper=clock.sleep)
    assert throttle.wait_for("a") == 0.0
    assert throttle.wait_for("a") == pytest.approx(1.0)
    assert throttle.wait_for("b") == 0.0  # other chats are not blocked
    assert throttle.wait_for("a") == pytest.approx(1.0)


def test_throttle_is_a_noop_when_disabled() -> None:
    clock = FakeClock()
    throttle = Throttle(0.0, clock=clock, sleeper=clock.sleep)
    assert throttle.wait_for("a") == 0.0
    assert throttle.wait_for("a") == 0.0
    assert clock.slept == []


def test_inline_keyboard_shape() -> None:
    markup = inline_keyboard([[("Allow", "ap:t:0"), ("Deny", "ap:t:x")]])
    assert markup == {
        "inline_keyboard": [[{"text": "Allow", "callback_data": "ap:t:0"}, {"text": "Deny", "callback_data": "ap:t:x"}]]
    }


# --------------------------------------------------------------------------- client


class ScriptedClient(TelegramClient):
    """A TelegramClient whose transport is a scripted list of responses."""

    def __init__(self, responses: list[Any], **kwargs: Any) -> None:
        self.responses = list(responses)
        self.payloads: list[dict[str, Any]] = []
        self.clock = FakeClock()
        super().__init__(
            "test-token",
            dry_run=False,
            sleeper=self.clock.sleep,
            clock=self.clock,
            **kwargs,
        )

    def _http_post(self, url: str, payload: Mapping[str, Any], timeout: float | None = None) -> Any:
        self.payloads.append(dict(payload))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_call_returns_result_on_success() -> None:
    client = ScriptedClient([{"ok": True, "result": {"message_id": 7}}])
    assert client.call("sendMessage", {"chat_id": 1, "text": "hi"}) == {"message_id": 7}
    assert client.payloads == [{"chat_id": 1, "text": "hi"}]


def test_429_is_honoured_with_retry_after_and_does_not_crash() -> None:
    client = ScriptedClient(
        [
            {"ok": False, "error_code": 429, "description": "Too Many Requests", "parameters": {"retry_after": 3}},
            {"ok": True, "result": {"message_id": 8}},
        ]
    )
    result = client.call("sendMessage", {"chat_id": 1, "text": "hi"})
    assert result == {"message_id": 8}
    assert client.clock.slept == [3.0]


def test_persistent_429_raises_after_retries() -> None:
    client = ScriptedClient(
        [
            {"ok": False, "error_code": 429, "description": "slow down", "parameters": {"retry_after": 1}},
            {"ok": False, "error_code": 429, "description": "slow down", "parameters": {"retry_after": 1}},
            {"ok": False, "error_code": 429, "description": "slow down", "parameters": {"retry_after": 2}},
        ],
        max_retries=2,
    )
    with pytest.raises(TelegramError) as excinfo:
        client.call("sendMessage", {"chat_id": 1, "text": "hi"})
    assert excinfo.value.status == 429
    assert excinfo.value.retry_after == 2.0


def test_client_error_is_raised_without_retry() -> None:
    client = ScriptedClient([{"ok": False, "error_code": 400, "description": "chat not found"}])
    with pytest.raises(TelegramError, match="chat not found"):
        client.call("sendMessage", {"chat_id": 1, "text": "hi"})
    assert len(client.payloads) == 1


def test_transport_errors_are_retried_then_surface() -> None:
    client = ScriptedClient(
        [
            TelegramTransportError("boom"),
            {"ok": True, "result": []},
        ]
    )
    assert client.call("getUpdates", {"timeout": 0}) == []
    assert client.clock.slept == [1.0]

    failing = ScriptedClient([TelegramTransportError("boom")] * 3, max_retries=2)
    with pytest.raises(TelegramTransportError):
        failing.call("getUpdates", {})


def test_send_message_respects_the_per_chat_rate_limit() -> None:
    client = ScriptedClient([{"ok": True, "result": {"message_id": 1}}] * 2, send_interval=1.0)
    client.send_message(1, "one")
    client.send_message(1, "two")
    assert client.clock.slept == [1.0]


def test_dry_run_never_touches_the_network() -> None:
    client = TelegramClient("", dry_run=True, send_interval=0)
    sent = client.send_message(5, "hello")
    assert sent[0]["message_id"] > 0
    assert client.calls[0][0] == "sendMessage"
    assert client.calls[0][1]["text"] == "hello"


# --------------------------------------------------------------------------- streaming


def make_stream(telegram: FakeTelegram, clock: FakeClock, **kwargs: Any) -> MessageStream:
    return MessageStream(telegram, 111, clock=clock, log=None, **kwargs)  # type: ignore[arg-type]


def test_stream_sends_once_and_edits_in_place() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock)
    assert stream.push("first") is True
    assert len(telegram.sent()) == 1
    assert telegram.sent()[0].text == "first"
    assert stream.message_id is not None

    clock.advance(2.0)
    assert stream.push("first second") is True
    assert len(telegram.sent()) == 1  # still ONE message
    assert telegram.last_sent_text(111) == "first"
    assert telegram.edits()[0].text == "first second"
    assert telegram.edits()[0].message_id == stream.message_id


def test_stream_coalesces_updates_inside_the_edit_interval() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock, edit_interval=1.2)
    stream.push("token 1")
    for index in range(2, 40):
        clock.advance(0.05)
        stream.push(f"token {index}")
    # 38 pushes over ~1.9s, with a 1.2s edit interval: exactly one coalesced edit.
    assert len(telegram.edits()) == 1
    # Deterministic: the interval is first satisfied on the 25th tick (IEEE-754
    # accumulation of 0.05 lands just under 1.2 on tick 24).
    assert telegram.edits()[0].text == "token 26"

    # The newest text is what the next allowed flush sends, not the oldest.
    clock.advance(1.2)
    stream.push("token 40")
    assert telegram.edits()[-1].text == "token 40"


def test_stream_flush_is_cheap_when_nothing_changed() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock)
    stream.push("same")
    before = len(telegram.messages)
    clock.advance(5.0)
    assert stream.flush() is False
    assert len(telegram.messages) == before


def test_stream_close_forces_the_final_text() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock)
    stream.push("in progress")
    clock.advance(0.1)
    assert stream.close("in progress\n\ndone") is True
    assert telegram.current_text(111) == "in progress\n\ndone"
    # After closing, further pushes are ignored.
    assert stream.push("late") is False


def test_stream_splits_overflow_into_ordered_parts() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock)
    text = "z" * 5000
    stream.push(text)
    sent = telegram.sent()
    assert len(sent) == 2
    assert all(len(message.text) <= MAX_MESSAGE_LENGTH for message in sent)
    assert "".join(message.text for message in sent) == text
    clock.advance(2.0)
    stream.push(text + " tail")
    assert all(len(message.text) <= MAX_MESSAGE_LENGTH for message in telegram.sent())
