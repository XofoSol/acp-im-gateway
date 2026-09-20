"""Renderer: the chat must read like the CLI conversation.

These tests pin the v1.1 Part A rendering rules: ordered blocks, tool calls with
the exact command and its real output, file edits with a diff, reasoning inside a
spoiler, lifecycle notices and a test summary. Nothing here talks to Telegram or
to an agent.
"""

from __future__ import annotations

from typing import Any

from acp_im_gateway.render import (
    TestSummary,
    TurnView,
    error_notice,
    escape,
    format_duration,
    heartbeat_notice,
    is_test_command,
    parse_test_summary,
    start_notice,
    truncate_lines,
)

CHAT = 111


def update(session_update: str, **fields: Any) -> dict[str, Any]:
    return {"params": {"update": {"sessionUpdate": session_update, **fields}}}


def tool(view: TurnView, **fields: Any) -> None:
    view.apply(update("tool_call", **fields))


def tool_update(view: TurnView, tool_id: str, **fields: Any) -> None:
    view.apply(update("tool_call_update", toolCallId=tool_id, **fields))


def say(view: TurnView, text: str) -> None:
    view.apply(update("agent_message_chunk", content={"type": "text", "text": text}))


def think(view: TurnView, text: str) -> None:
    view.apply(update("agent_thought_chunk", content={"type": "text", "text": text}))


# --------------------------------------------------------------------------- notices


def test_start_notice_names_the_project_and_the_model() -> None:
    assert start_notice("alpha", "fake-model") == "⏳ alpha · model fake-model"
    assert start_notice("alpha", None) == "⏳ alpha"
    assert start_notice(None, "fake-model") == "⏳ model fake-model"
    assert start_notice(None, None) == "⏳ working…"


def test_start_notice_is_the_first_block_of_the_transcript() -> None:
    view = TurnView()
    view.start("alpha", "fake-model")
    say(view, "on it")
    rendered = view.render()
    assert rendered.startswith("⏳ alpha · model fake-model")
    assert rendered.index("⏳") < rendered.index("on it")


def test_finish_notice_carries_tool_calls_duration_and_the_test_summary() -> None:
    view = TurnView()
    view.start("alpha", "fake-model")
    tool(view, toolCallId="c1", title="Bash", kind="execute", status="in_progress",
         rawInput={"command": "pytest -q"})
    tool_update(view, "c1", status="completed",
                content=[{"type": "content", "content": {"type": "text", "text": "5 passed in 0.1s"}}])
    final = view.finish("end_turn", elapsed=12.4)
    assert final == "✅ done · 1 tool call(s) · 12s · tests: 5 passed"
    assert view.render().endswith(final)


def test_finish_notice_without_a_test_command() -> None:
    view = TurnView()
    tool(view, toolCallId="c1", title="Read file", kind="read", status="completed")
    assert view.finish("end_turn", elapsed=0.4) == "✅ done · 1 tool call(s) · 0.4s"
    assert TurnView().finish("end_turn") == "✅ done"


def test_cancelled_and_error_notices() -> None:
    assert TurnView().finish("cancelled") == "⏹️ cancelled"
    assert TurnView().finish("max_tokens") == "⚠️ stopped: token limit"
    assert TurnView().finish("weird") == "⏹️ stopped (weird)"
    assert error_notice("agent error: boom") == "❌ agent error: boom"
    view = TurnView()
    view.fail("the agent crashed")
    assert view.render() == "❌ the agent crashed"


def test_heartbeat_shows_the_elapsed_time() -> None:
    assert heartbeat_notice(3.2) == "⏳ still working · 3.2s elapsed"
    assert heartbeat_notice(72) == "⏳ still working · 1m 12s elapsed"
    assert format_duration(0.4) == "0.4s"
    assert format_duration(3725) == "1h 02m"
    assert format_duration(None) == ""


# --------------------------------------------------------------------------- tool calls


def test_shell_call_shows_the_exact_command_and_its_output() -> None:
    view = TurnView()
    tool(view, toolCallId="c1", title="Bash", kind="execute", status="in_progress",
         rawInput={"command": "git status --short"})
    tool_update(view, "c1", status="completed",
                content=[{"type": "content", "content": {"type": "text", "text": " M app.py\n?? new.py"}}])
    rendered = view.render()
    assert "$ git status --short" in rendered  # piped straight from the payload
    assert "[completed]" in rendered
    assert "<pre> M app.py\n?? new.py</pre>" in rendered
    assert view.tool_calls == 1


def test_tool_output_is_truncated_with_a_more_lines_marker() -> None:
    view = TurnView(tool_output_lines=3)
    output = "\n".join(f"line {index}" for index in range(10))
    tool(view, toolCallId="c1", title="Bash", kind="execute", status="in_progress",
         rawInput={"command": "pytest -q"})
    tool_update(view, "c1", status="completed",
                content=[{"type": "content", "content": {"type": "text", "text": output}}])
    rendered = view.render()
    assert "… (+7 more lines)" in rendered
    assert "line 9" in rendered  # the tail is what survives: the verdict lives there
    assert "line 0" not in rendered
    assert rendered.count("<pre>") == 1


def test_raw_output_object_is_understood() -> None:
    view = TurnView()
    tool(view, toolCallId="c1", title="Bash", kind="execute", status="in_progress",
         rawInput={"command": "make build"})
    tool_update(view, "c1", status="failed", rawOutput={"stdout": "ok", "stderr": "boom"})
    rendered = view.render()
    assert "[failed]" in rendered
    assert "ok" in rendered and "boom" in rendered


def test_file_edit_shows_the_path_and_a_short_diff() -> None:
    view = TurnView()
    tool(view, toolCallId="c1", title="Edit file", kind="edit", status="in_progress",
         locations=[{"path": "src/app.py"}])
    tool_update(view, "c1", status="completed",
                content=[{"type": "diff", "path": "src/app.py",
                          "oldText": "value = 1\n", "newText": "value = 42\n"}])
    rendered = view.render()
    assert "✏️ src/app.py [completed]" in rendered
    assert "-value = 1" in rendered and "+value = 42" in rendered
    assert view.edits == 1


def test_edit_falls_back_to_raw_input_paths() -> None:
    view = TurnView()
    tool(view, toolCallId="c1", title="Write", kind="edit", status="in_progress",
         rawInput={"file_path": "docs/readme.md", "content": "hello"})
    assert "✏️ docs/readme.md" in view.render()


def test_tool_call_without_a_command_keeps_the_icon_title_status_line() -> None:
    view = TurnView()
    tool(view, toolCallId="c1", title="Run tests", kind="execute", status="pending")
    tool_update(view, "c1", status="completed")
    assert "▶️ Run tests [completed]" in view.render()


def test_plans_render_as_a_checklist() -> None:
    view = TurnView()
    view.apply(update("plan", entries=[{"content": "step one", "status": "completed"},
                                       {"content": "step two"}]))
    assert "📋 plan:" in view.render()
    assert "✅ step one" in view.render()
    assert "▫️ step two" in view.render()


# --------------------------------------------------------------------------- reasoning


def test_reasoning_is_rendered_in_a_telegram_spoiler() -> None:
    view = TurnView()
    think(view, "the test fails because of a missing import")
    say(view, "fixed it")
    rendered = view.render()
    assert "<tg-spoiler>the test fails because of a missing import</tg-spoiler>" in rendered
    # Ordered: the reasoning came first, the answer after it.
    assert rendered.index("<tg-spoiler>") < rendered.index("fixed it")


def test_reasoning_is_omitted_when_thinking_is_off() -> None:
    view = TurnView(show_thinking=False)
    think(view, "secret reasoning")
    say(view, "here is the answer")
    rendered = view.render()
    assert "secret reasoning" not in rendered
    assert "<tg-spoiler>" not in rendered
    assert rendered == "here is the answer"


def test_reasoning_longer_than_the_cap_is_trimmed() -> None:
    view = TurnView()
    think(view, "x" * 2000)
    rendered = view.render()
    assert "…</tg-spoiler>" in rendered
    assert len(rendered) < 1200


# --------------------------------------------------------------------------- safety


def test_agent_output_is_escaped_so_it_cannot_break_the_message() -> None:
    view = TurnView()
    say(view, "<b>bold</b> & <script>alert(1)</script>")
    rendered = view.render()
    assert "<b>bold</b>" not in rendered
    assert "&lt;b&gt;bold&lt;/b&gt; &amp; &lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert escape("<>&") == "&lt;&gt;&amp;"


def test_escaping_survives_the_spoiler_and_verbatim_prompts() -> None:
    view = TurnView()
    view.start("alpha", "m")
    think(view, "<i>hmm</i>")
    say(view, "echo: <b>not markup</b>")
    rendered = view.render()
    assert "<tg-spoiler>&lt;i&gt;hmm&lt;/i&gt;</tg-spoiler>" in rendered
    assert "&lt;b&gt;not markup&lt;/b&gt;" in rendered


# --------------------------------------------------------------------------- summaries


def test_test_commands_are_recognised() -> None:
    for command in ("pytest -q", "python -m pytest tests/", "npm test", "npm run test",
                    "go test ./...", "cargo test", "make test", "make check",
                    "npx vitest run", "dotnet test"):
        assert is_test_command(command), command
    for command in ("git status", "rm -rf build", "npm publish", "make build", ""):
        assert not is_test_command(command), command


def test_test_summary_is_parsed_from_the_output() -> None:
    assert parse_test_summary("no counts here") is None
    summary = parse_test_summary("===== 12 passed, 1 failed, 2 skipped in 3.4s =====")
    assert summary == TestSummary(passed=12, failed=1, skipped=2)
    assert summary is not None and summary.label() == "12 passed, 1 failed, 2 skipped"
    assert TestSummary().label() == ""
    assert TestSummary(errors=1).label() == "1 error"


def test_only_a_test_command_feeds_the_finish_summary() -> None:
    view = TurnView()
    tool(view, toolCallId="c1", title="Bash", kind="execute", status="in_progress",
         rawInput={"command": "git status"})
    tool_update(view, "c1", status="completed",
                content=[{"type": "content", "content": {"type": "text", "text": "5 passed"}}])
    assert view.test_run is None
    assert "tests:" not in view.finish("end_turn")

    tool(view, toolCallId="c2", title="Bash", kind="execute", status="in_progress",
         rawInput={"command": "python -m pytest -q"})
    tool_update(view, "c2", status="completed",
                content=[{"type": "content", "content": {"type": "text", "text": "7 passed in 0.2s"}}])
    assert view.test_run == TestSummary(passed=7)
    assert view.finish("end_turn").endswith("tests: 7 passed")


def test_truncate_lines_keeps_the_tail() -> None:
    body, omitted = truncate_lines("a\nb\nc\nd", 2)
    assert body == "c\nd"
    assert omitted == 2
    assert truncate_lines("a\nb", 5) == ("a\nb", 0)


# --------------------------------------------------------------------------- blocks


def test_blocks_are_ordered_and_only_the_last_one_can_change() -> None:
    view = TurnView()
    view.start("alpha", "fake-model")
    think(view, "reasoning")
    say(view, "answer")
    tool(view, toolCallId="c1", title="Bash", kind="execute", status="in_progress",
         rawInput={"command": "git status"})
    blocks = view.blocks()
    assert blocks[0] == "⏳ alpha · model fake-model"
    assert "reasoning" in blocks[1]
    assert blocks[2] == "answer"
    assert blocks[3].startswith("$ git status")
    assert view.render() == "\n\n".join(blocks)


def test_streaming_text_stays_a_single_growing_block() -> None:
    view = TurnView()
    say(view, "hello ")
    say(view, "world")
    assert view.blocks() == ["hello world"]
