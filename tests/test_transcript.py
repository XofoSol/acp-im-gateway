"""Part A end to end: a turn must read like the CLI conversation.

Everything here is hermetic — the ACP side is ``tests/fake_agent.py`` and the
Telegram side is the in-memory double — and no ``session/prompt`` ever reaches a
real agent.
"""

from __future__ import annotations

from typing import Any

from acp_im_gateway.telegram import MAX_MESSAGE_LENGTH, chunk_message, scan_html

from .helpers import make_harness  # noqa: F401  (pytest fixture)

OUTPUT = "\n".join(f"tests/test_api.py::test_{index} PASSED" for index in range(6))
OUTPUT += "\n1 failed, 6 passed in 0.42s"


def sent_messages(harness: Any) -> list[Any]:
    """The turn's messages only: /bind's reply is cleared away first."""
    return harness.telegram.sent(harness.chat_id)


def test_a_turn_reads_like_the_cli_conversation(make_harness: Any) -> None:
    harness = make_harness(
        "--tool-command",
        "pytest -q",
        "--tool-output",
        OUTPUT,
    )
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.send("run the tests and tell me what broke")
    assert harness.wait_idle()

    chat = harness.visible_text()
    # 1. start notice with the project and the model
    assert "⏳ alpha · model fake-model" in chat
    # 2. what the agent said
    assert "echo: run the tests and tell me what broke" in chat
    # 3. the command verbatim, with "completed", and its real output fenced
    assert "$ pytest -q [completed]" in chat
    assert "<pre>tests/test_api.py::test_0 PASSED" in chat
    assert "1 failed, 6 passed in 0.42s" in chat
    # 4. the finish line, with the test summary
    assert "✅ done" in chat and "tests: 6 passed, 1 failed" in chat
    # the whole thing is a transcript, in order
    assert chat.index("⏳ alpha") < chat.index("$ pytest -q")
    assert chat.index("$ pytest -q") < chat.index("✅ done")


def test_a_turn_updates_one_message_in_place(make_harness: Any) -> None:
    harness = make_harness("--tool-command", "git status --short")
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.send("is the tree clean?")
    assert harness.wait_idle()

    assert len(sent_messages(harness)) == 1, "a turn edits one message in place"
    assert len(harness.telegram.edits(harness.chat_id)) >= 1
    assert "$ git status --short" in harness.current_text()


def test_edits_are_never_more_often_than_the_configured_interval(make_harness: Any) -> None:
    """A fast turn must not turn into one Bot API call per token."""
    harness = make_harness("--tool-command", "pytest -q", "--tool-output", OUTPUT, edit_interval=60.0)
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.send("run the tests")
    assert harness.wait_idle()

    # Everything inside the interval is coalesced; only the forced final edit lands.
    assert len(sent_messages(harness)) == 1
    assert len(harness.telegram.edits(harness.chat_id)) == 1


def test_a_long_turn_is_frozen_into_ordered_messages(make_harness: Any) -> None:
    long_output = "\n".join(f"output line {index:03d} " + "x" * 40 for index in range(200))
    harness = make_harness(
        "--tool-command", "pytest -q", "--tool-output", long_output, overflow_chars=600
    )
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.send("run the tests")
    assert harness.wait_idle()

    sent = sent_messages(harness)
    assert len(sent) >= 2, "past the overflow limit a new message starts"
    assert all(len(message.text) <= MAX_MESSAGE_LENGTH for message in sent)

    # The transcript is still complete and in order across the messages…
    chat = harness.visible_text()
    assert "⏳ alpha · model fake-model" in chat
    assert "$ pytest -q [completed]" in chat
    assert chat.index("$ pytest -q") < chat.rindex("✅ done")

    # Nothing is duplicated and nothing is lost across the freeze: each marker of
    # the transcript is in the chat exactly once.
    current = {message.message_id: message.text for message in harness.telegram.messages}
    whole = "\n".join(current[message.message_id] for message in sent)
    for marker in (
        "⏳ alpha · model fake-model",
        "echo: run the tests",
        "$ pytest -q [completed]",
        "output line 199",
        "✅ done",
    ):
        assert whole.count(marker) == 1, marker

    # …and a frozen message is never edited again once the next one started.
    frozen_id = sent[0].message_id
    sends = [index for index, (method, _) in enumerate(harness.telegram.calls) if method == "sendMessage"]
    after = [
        payload
        for method, payload in harness.telegram.calls[sends[1] :]
        if method == "editMessageText" and payload.get("message_id") == frozen_id
    ]
    assert after == [], "a frozen message must not be rewritten"


def test_reasoning_is_quiet_but_available(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()
    chat = harness.visible_text()
    assert "<tg-spoiler>considering</tg-spoiler>" in chat


def test_reasoning_is_omitted_when_thinking_is_off(make_harness: Any) -> None:
    harness = make_harness(show_thinking=False)
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()
    chat = harness.visible_text()
    assert "considering" not in chat
    assert "<tg-spoiler>" not in chat


def test_the_transcript_is_sent_as_html(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.send("hello")
    assert harness.wait_idle()
    payloads = harness.telegram.calls_named("sendMessage")
    assert payloads and all(payload.get("parse_mode") == "HTML" for payload in payloads)
    edits = harness.telegram.calls_named("editMessageText")
    assert edits and all(payload.get("parse_mode") == "HTML" for payload in edits)
    # Both the sends and the edits are valid HTML: no tag is left open.
    for message in harness.telegram.messages:
        if message.chat_id == harness.chat_id:
            assert scan_html(message.text) == ()


def test_a_file_edit_shows_its_path_and_diff(make_harness: Any) -> None:
    harness = make_harness("--edit-path", "tests/test_api.py")
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.send("fix the failing test")
    assert harness.wait_idle()
    chat = harness.visible_text()
    assert "✏️ tests/test_api.py" in chat
    assert "-" in chat and "+" in chat


def test_the_heartbeat_shows_the_turn_is_alive(make_harness: Any) -> None:
    harness = make_harness("--slow", "1.0", heartbeat_seconds=0.2)
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.send("a slow turn")
    assert harness.wait_idle()

    assert any(
        "⏳ still working" in message.text for message in harness.telegram.edits(harness.chat_id)
    ), "a silent stretch must be visible, not look hung"
    # The pulse is transient: the final message carries the verdict, not the beat.
    assert "still working" not in harness.current_text()
    assert "✅ done" in harness.current_text()


def test_no_heartbeat_when_the_turn_keeps_talking(make_harness: Any) -> None:
    harness = make_harness(heartbeat_seconds=0.0)
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()
    assert "still working" not in harness.all_text()


def test_threads_are_answered_in_their_own_topic(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/bind alpha", message_thread_id=42)
    bind_reply = harness.last_reply()
    assert "✅ Bound to alpha" in bind_reply
    harness.telegram.reset()

    harness.send("hello", message_thread_id=42)
    assert harness.wait_idle()
    sends = harness.telegram.calls_named("sendMessage")
    assert sends and all(payload.get("message_thread_id") == 42 for payload in sends)
    assert "✅ done" in harness.current_text()


def test_replies_outside_a_topic_carry_no_thread_id(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/bind alpha")
    sends = harness.telegram.calls_named("sendMessage")
    assert sends and all(payload.get("message_thread_id") is None for payload in sends)


def test_chunking_keeps_html_tags_balanced() -> None:
    text = "<tg-spoiler>" + "reasoning " * 500 + "</tg-spoiler>\n\n<pre>" + "code\n" * 900 + "</pre>"
    parts = chunk_message(text, html=True)
    assert len(parts) > 1
    assert all(len(part) <= MAX_MESSAGE_LENGTH for part in parts)
    assert all(scan_html(part) == () for part in parts)
