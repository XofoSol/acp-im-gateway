"""Telegram adapter: chunking, rate limits, 429 handling, in-place streaming."""

from __future__ import annotations

import re
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
    html_balanced,
    inline_keyboard,
    is_parse_error,
    reassemble,
    scan_fences,
    strip_markup,
)

from .helpers import PARSE_400_DESCRIPTION, FakeClock, FakeTelegram


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


def test_stream_seal_freezes_the_message_and_starts_a_new_one() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock, edit_interval=0.0)
    stream.push("first block")
    assert len(telegram.sent()) == 1
    assert stream.seal() == 1

    stream.push("first block\n\nsecond block")
    sent = telegram.sent()
    assert len(sent) == 2, "a sealed message is never reused"
    assert sent[0].text == "first block"
    assert sent[1].text == "\n\nsecond block"
    assert stream.sealed_text == "first block"
    assert [message.message_id for message in stream.sealed_messages] == [sent[0].message_id]

    clock.advance(5.0)
    stream.push("first block\n\nsecond block\n\nthird")
    # The sealed message keeps its frozen text; the live one is edited in place.
    current = {message.message_id: message.text for message in telegram.messages}
    assert current[sent[0].message_id] == "first block"
    assert current[sent[1].message_id] == "\n\nsecond block\n\nthird"


def test_stream_seal_is_a_noop_when_nothing_is_on_screen_yet() -> None:
    telegram = FakeTelegram()
    stream = make_stream(telegram, FakeClock())
    assert stream.seal() == 0
    stream.push("hello")
    assert stream.up_to_date is True
    assert stream.live_message_id == stream.message_id


def test_stream_seal_refuses_while_an_edit_is_pending() -> None:
    """Freezing stale content would drop whatever the pending edit carried."""
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock, edit_interval=10.0)
    stream.push("shown")
    stream.push("shown plus something newer")  # coalesced: not on screen yet
    assert stream.up_to_date is False
    assert stream.seal() == 0
    assert len(stream.sealed_messages) == 0

    clock.advance(20.0)
    stream.flush()
    assert stream.up_to_date is True
    assert stream.seal() == 1


def test_stream_close_flushes_the_tail_into_the_new_message() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock, edit_interval=0.0)
    stream.push("block one")
    stream.seal()
    stream.push("block one\n\nblock two")
    clock.advance(5.0)
    stream.close("block one\n\nblock two\n\n✅ done")
    sent = telegram.sent()
    assert len(sent) == 2
    current = {message.message_id: message.text for message in telegram.messages}
    assert current[sent[0].message_id] == "block one"
    assert current[sent[1].message_id] == "\n\nblock two\n\n✅ done"


def test_html_chunking_keeps_tags_balanced_and_content_intact() -> None:
    text = (
        "<tg-spoiler>" + "reasoning " * 400 + "</tg-spoiler>\n\n"
        "<pre>" + "\n".join(f"line {index}" for index in range(500)) + "</pre>"
    )
    parts = chunk_message(text, html=True)
    assert len(parts) > 1
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    from acp_im_gateway.telegram import scan_html

    assert all(scan_html(part) == () for part in parts)
    strip = lambda chunk: re.sub(  # noqa: E731
        r"</?(?:pre|code|b|i|u|s|tg-spoiler|blockquote)(?:\s[^>]*)?>", "", chunk
    )
    assert strip("".join(parts)) == strip(text)


def test_html_chunking_keeps_nested_tags_balanced() -> None:
    text = (
        "<tg-spoiler><pre>"
        + "".join(f"line {index} " * 12 + "\n" for index in range(300))
        + "</pre></tg-spoiler>"
    )
    parts = chunk_message(text, html=True)
    assert len(parts) > 1
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    from acp_im_gateway.telegram import scan_html

    assert all(scan_html(part) == () for part in parts)
    # No part ends in the middle of a tag, which Telegram would reject.
    assert not any(re.search(r"</?[a-z-]*$", part) for part in parts)


def test_html_chunking_rejects_a_limit_it_cannot_honour() -> None:
    with pytest.raises(ValueError, match="limit must be"):
        chunk_message("x" * 1000, limit=10, html=True)


def test_stream_seal_can_freeze_a_prefix_of_what_was_pushed() -> None:
    """Freezing *less* than what was pushed must not repeat the difference."""
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock, edit_interval=0.0)
    full = "header\n\nbody starts here"
    stream.push(full)
    assert telegram.current_text(111) == full

    frozen = "header"
    stream.push(frozen)
    stream.flush(force=True)
    assert stream.seal(frozen) == 1
    assert stream.sealed_text == frozen

    stream.push(full + "\n\nand more")
    current = {message.message_id: message.text for message in telegram.messages}
    sent = telegram.sent()
    assert current[sent[0].message_id] == frozen
    assert current[sent[1].message_id] == "\n\nbody starts here\n\nand more", "no duplication"
    assert (current[sent[0].message_id] + current[sent[1].message_id]) == full + "\n\nand more"


# --------------------------------------------------------------------------- markup fallback


def test_is_parse_error_recognises_a_markup_rejection() -> None:
    assert is_parse_error(
        TelegramError(f"sendMessage failed: {PARSE_400_DESCRIPTION}", status=400)
    )
    assert is_parse_error(
        TelegramError("editMessageText failed: can't find end tag", status=400)
    )
    # A generic 400 (or any other failure) is not a markup problem.
    assert not is_parse_error(TelegramError("chat not found", status=400))
    assert not is_parse_error(TelegramError("Too Many Requests", status=429))


def test_strip_markup_removes_gateway_tags_and_decodes_entities() -> None:
    body = "<pre>$ ls &amp;&amp; echo '&lt;b&gt;'</pre>"
    assert strip_markup(body) == "$ ls && echo '<b>'"
    # A literal ``&lt;/b&gt;`` is *text*, not a tag: stripping leaves it alone.
    assert strip_markup("<tg-spoiler>a &lt;/b&gt; b</tg-spoiler>") == "a </b> b"


def test_html_balanced_flags_stray_and_unclosed_tags() -> None:
    assert html_balanced("<b>x</b>")
    assert html_balanced("<tg-spoiler><pre>code</pre></tg-spoiler>")
    assert html_balanced("no tags at all")
    assert not html_balanced("<pre>unclosed")
    assert not html_balanced("stray </b>")
    assert not html_balanced("<b><i>x</b></i>")


def test_chunking_alone_cannot_repair_a_stray_closing_tag() -> None:
    """Why :func:`html_balanced` exists: chunking balances *its own* cuts but
    hands a short, already-broken body straight through."""
    body = "hello </b> world"
    assert chunk_message(body, html=True) == [body]
    assert not html_balanced(body)


def test_message_stream_replays_a_parse_rejection_as_plain_text() -> None:
    """A: a body Telegram refuses for its markup is re-sent unformatted."""
    telegram = FakeTelegram()
    telegram.reject_html = True  # every HTML call is refused, plain text is not
    stream = make_stream(telegram, FakeClock(), parse_mode="HTML")
    assert stream.push("<pre>hola &amp; adiós</pre>") is True

    assert telegram.current_text(111) == "hola & adiós"  # content survives, unformatted
    # The HTML attempt was refused (only successful calls are recorded), then the
    # plain re-send landed with the markup dropped.
    assert telegram.send_attempts == 2
    sent = telegram.calls_named("sendMessage")
    assert sent[-1]["parse_mode"] is None
    assert stream.live_message_id is not None


def test_message_stream_replays_a_parse_rejection_on_edit_as_plain_text() -> None:
    telegram = FakeTelegram()
    clock = FakeClock()
    stream = make_stream(telegram, clock, parse_mode="HTML", edit_interval=0.0)
    stream.push("<pre>first</pre>")
    assert telegram.current_text(111) == "<pre>first</pre>"

    telegram.reject_html = True  # markup starts being refused
    clock.advance(5.0)
    assert stream.push("<pre>first and second</pre>") is True
    assert telegram.current_text(111) == "first and second"
    assert telegram.edit_attempts == 2  # HTML edit refused, then the plain re-edit
    assert telegram.calls_named("editMessageText")[-1]["parse_mode"] is None


def test_message_stream_delivers_an_unbalanced_body_as_plain_text() -> None:
    """C/4: a body the balancer cannot make valid is never sent as HTML."""
    telegram = FakeTelegram()
    stream = make_stream(telegram, FakeClock(), parse_mode="HTML")
    assert stream.push("before <pre>oops") is True
    assert telegram.calls_named("sendMessage")[-1]["parse_mode"] is None
    assert telegram.current_text(111) == "before oops"


def test_client_retries_a_markup_rejection_as_plain_text() -> None:
    """A: the same fallback protects the direct (non-streaming) send path."""
    client = ScriptedClient(
        [
            {"ok": False, "error_code": 400, "description": PARSE_400_DESCRIPTION},
            {"ok": True, "result": {"message_id": 9}},
        ]
    )
    sent = client.send_message(1, "<b>hi</b>", parse_mode="HTML")
    assert sent == [{"message_id": 9}]
    assert client.payloads[0]["parse_mode"] == "HTML"
    assert "parse_mode" not in client.payloads[1]  # plain text: markup dropped
    assert client.payloads[1]["text"] == "hi"
