"""Gateway behaviour end to end: real ACP client + fake agent + fake Telegram.

These tests never touch the network and never send a ``session/prompt`` to a real
agent: the ACP side is the scriptable fake in ``tests/fake_agent.py``.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any

import pytest

from acp_im_gateway.acp import STEER_METHOD
from acp_im_gateway.gateway import MAX_DELIVERY_FAILURES, TurnView
from acp_im_gateway.telegram import TelegramError

from .helpers import (
    CHAT,
    PARSE_400_DESCRIPTION,
    events_named,
    make_harness,  # noqa: F401  (pytest fixture: used by name in the signatures)
    wait_until,
)

#: Telegram's permanent 400 for unparsable markup, as the streaming path sees it.
PARSE_400 = TelegramError(f"editMessageText failed: {PARSE_400_DESCRIPTION}", status=400)

# --------------------------------------------------------------------------- access


def test_unbound_text_explains_how_to_bind(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("hello agent")
    reply = harness.last_reply()
    assert "/projects" in reply and "/bind" in reply
    assert harness.agent_requests("session/new") == []


def test_unknown_sender_is_denied_and_gets_a_code_in_the_log_only(
    make_harness: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    harness = make_harness()
    harness.send("let me in", user_id=666)

    reply = harness.last_reply()
    assert "Not authorised" in reply
    match = re.search(r"Code ([A-Z0-9]{8})", caplog.text)
    assert match, "the pairing code must be written to the gateway log"
    code = match.group(1)
    assert code not in reply  # never sent to the unknown sender
    assert code in harness.state()["access"]["pending"]
    assert harness.gateway.access.is_user_allowed(666) is False
    assert harness.agent_requests("session/new") == []


def test_allowlisted_user_auto_enables_a_new_group_and_keeps_processing(
    make_harness: Any,
) -> None:
    """A group starts disabled; an allowlisted sender enables it as a side effect."""
    harness = make_harness(allowed_chats=(), chat_id=-500)
    assert harness.gateway.access.is_chat_allowed(-500, "supergroup") is False

    harness.send("hello", chat_type="supergroup")

    # Enabled in memory and persisted, with a short confirmation in the chat…
    assert harness.gateway.access.is_chat_allowed(-500, "supergroup") is True
    assert -500 in harness.state()["access"]["allowed_chat_ids"]
    assert any("enabled automatically" in text for text in harness.replies())
    # …and the message was processed, not dropped (unbound -> how to bind).
    assert "/bind" in harness.last_reply()

    # The very same chat now drives a full turn like any other.
    harness.send("/bind alpha", chat_type="supergroup")
    assert "✅ Bound to alpha" in harness.last_reply()
    harness.send("hi agent", chat_type="supergroup")
    assert harness.wait_idle()
    assert [prompt["text"] for prompt in harness.prompts()] == ["hi agent"]


def test_a_pre_listed_group_is_used_without_a_confirmation(make_harness: Any) -> None:
    harness = make_harness(allowed_chats=(-500,), chat_id=-500)
    harness.send("hello", chat_type="supergroup")
    assert not any("enabled automatically" in text for text in harness.replies())
    assert "/bind" in harness.last_reply()


def test_unknown_user_in_a_new_group_gets_pairing_and_never_enables_it(
    make_harness: Any, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING)
    harness = make_harness(allowed_chats=(), chat_id=-500)

    harness.send("let me in", chat_type="supergroup", user_id=666)

    assert "Not authorised" in harness.last_reply()
    match = re.search(r"Code ([A-Z0-9]{8})", caplog.text)
    assert match, "the pairing code must be written to the gateway log"
    assert harness.gateway.access.is_user_allowed(666) is False
    assert harness.gateway.access.is_chat_allowed(-500, "supergroup") is False
    assert -500 not in harness.state()["access"]["allowed_chat_ids"]
    assert harness.agent_requests("session/new") == []


def test_direct_messages_are_unaffected_by_group_auto_enabling(make_harness: Any) -> None:
    harness = make_harness(allowed_chats=(), chat_id=CHAT)  # CHAT is a private chat
    harness.send("hello")  # chat_type defaults to "private"

    assert harness.gateway.access.allowed_chat_ids == set()
    assert not any("enabled automatically" in text for text in harness.replies())
    assert "/bind" in harness.last_reply()

    # A DM still runs a normal turn, exactly as before.
    harness.send("/bind alpha")
    harness.send("hi agent")
    assert harness.wait_idle()
    assert [prompt["text"] for prompt in harness.prompts()] == ["hi agent"]


def test_auto_enabled_chat_survives_a_restart(make_harness: Any) -> None:
    first = make_harness(allowed_chats=(), chat_id=-500)
    first.send("hello", chat_type="supergroup")
    assert -500 in first.state()["access"]["allowed_chat_ids"]

    # A fresh gateway on the same state file starts with the chat already enabled.
    second = make_harness(allowed_chats=(), chat_id=-500)
    assert second.gateway.access.is_chat_allowed(-500, "supergroup") is True
    second.send("hello again", chat_type="supergroup")
    assert not any("enabled automatically" in text for text in second.replies())
    assert "/bind" in second.last_reply()


def test_bind_outside_the_allowed_root_is_refused(make_harness: Any, tmp_path: Path) -> None:
    harness = make_harness()
    outside = tmp_path / "not-inside-the-root"
    outside.mkdir()
    harness.send(f"/bind {outside}")
    assert "🚫" in harness.last_reply()
    assert harness.gateway.router.get(CHAT) is None


def test_projects_bind_and_status_flow(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/projects")
    listing = harness.last_reply()
    assert "alpha" in listing and "beta" in listing
    assert "✅" not in listing

    harness.send("/bind alpha")
    assert "✅ Bound to alpha" in harness.last_reply()

    harness.send("/projects")
    assert "✅ alpha" in harness.last_reply()

    harness.send("/status")
    status = harness.last_reply()
    assert "alpha" in status
    assert "state: idle" in status

    harness.send("/unbind")
    assert "Unbound" in harness.last_reply()
    assert harness.gateway.router.get(CHAT) is None


def test_unknown_command_is_reported(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/nonsense")
    assert "Unknown command /nonsense" in harness.last_reply()


def test_help_lists_the_commands(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/help")
    help_text = harness.last_reply()
    for command in ("/projects", "/bind", "/new", "/stop", "/status"):
        assert command in help_text


# --------------------------------------------------------------------------- turns


def test_turn_streams_into_one_message_with_verbatim_prompt(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/bind alpha")
    harness.telegram.reset()

    prompt = "please run the tests — verbatim ñ\nsecond line"
    harness.send(prompt)
    assert harness.wait_idle()

    sent = harness.telegram.sent(CHAT)
    assert len(sent) == 1, "a turn must edit one message in place"
    assert len(harness.telegram.edits(CHAT)) >= 1

    final = harness.current_text()
    assert f"echo: {prompt}" in final
    assert "▶️ Run tests [completed]" in final
    assert "✅ done" in final and "1 tool call(s)" in final
    assert harness.prompts()[0]["text"] == prompt


def test_turn_view_renders_thoughts_tool_calls_and_plans() -> None:
    view = TurnView()
    assert view.render() == "💭 working…"

    view.apply({"params": {"update": {"sessionUpdate": "agent_thought_chunk"}}})
    assert "thinking" in view.render()

    view.apply(
        {
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "hola"},
                }
            }
        }
    )
    assert view.render().startswith("hola")

    view.apply(
        {
            "params": {
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "c1",
                    "title": "Read file",
                    "kind": "read",
                    "status": "pending",
                }
            }
        }
    )
    view.apply(
        {
            "params": {
                "update": {"sessionUpdate": "tool_call_update", "toolCallId": "c1", "status": "completed"}
            }
        }
    )
    rendered = view.render()
    assert "📖 Read file [completed]" in rendered

    view.apply(
        {
            "params": {
                "update": {
                    "sessionUpdate": "plan",
                    "entries": [{"content": "step one", "status": "completed"}, {"content": "step two"}],
                }
            }
        }
    )
    assert "📋 plan" in view.render() and "✅ step one" in view.render()
    assert view.tool_calls == 1


def test_message_during_a_turn_is_steered_when_advertised(make_harness: Any) -> None:
    harness = make_harness("--permission", "--steer")
    harness.send("/bind alpha")
    harness.send("first prompt")
    assert harness.wait_for_buttons(), "the fake agent should ask for permission"

    harness.send("also check the docs")
    assert "Steered" in harness.last_reply()
    steers = harness.agent_requests(STEER_METHOD)
    assert steers and steers[0]["params"]["prompt"][0]["text"] == "also check the docs"

    harness.callback(harness.buttons()[0]["callback_data"])
    assert harness.wait_idle()
    assert harness.any_text("✅ done")
    assert [event["text"] for event in harness.prompts()] == ["first prompt"]


def test_message_during_a_turn_is_queued_when_steering_is_not_advertised(make_harness: Any) -> None:
    harness = make_harness("--permission")  # no --steer: queue instead
    harness.send("/bind alpha")
    harness.send("first prompt")
    assert harness.wait_for_buttons()

    harness.send("second prompt")
    assert "queued" in harness.last_reply().lower()
    assert harness.runtime.queue_size == 1

    harness.callback(harness.buttons()[0]["callback_data"])
    assert harness.wait_for_buttons(), "the queued turn runs when the first one ends"
    harness.callback(harness.buttons()[0]["callback_data"])
    assert harness.wait_idle()
    assert [event["text"] for event in harness.prompts()] == [
        "first prompt",
        "second prompt",
    ]


def test_stop_cancels_the_turn_and_drops_the_queue(make_harness: Any) -> None:
    harness = make_harness("--permission")
    harness.send("/bind alpha")
    harness.send("long turn")
    assert harness.wait_for_buttons()

    harness.send("queued while busy")
    harness.send("/stop")
    assert "Dropped 1 queued message(s)" in harness.last_reply()
    assert harness.wait_idle()

    assert events_named(harness.agent_events(), "cancel"), "the agent should be told to cancel"
    assert harness.any_text("cancelled")
    # Stale approval buttons are cleared once the turn is over.
    assert harness.buttons() == []
    assert "no longer needed" in harness.telegram.edits(CHAT)[-1].text


def test_status_reports_session_model_and_queue(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()
    harness.send("/status")
    status = harness.last_reply()
    assert "fake-model" in status
    assert "approval posture: ask" in status
    assert "session: sess-1" in status
    assert "queued: 0" in status


def test_new_starts_a_fresh_session(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()
    assert len(harness.agent_requests("session/new")) == 1

    harness.send("/new")
    assert "🆕 Fresh session" in harness.last_reply()
    assert len(harness.agent_requests("session/new")) == 2


# --------------------------------------------------------------------------- approvals


def test_approval_buttons_reach_the_chat_and_the_answer_reaches_the_agent(make_harness: Any) -> None:
    harness = make_harness("--permission")
    harness.send("/bind alpha")
    harness.send("run the tests")
    assert harness.wait_for_buttons()

    buttons = harness.buttons()
    assert [button["text"] for button in buttons] == ["Allow once", "Always", "Reject"]
    assert "Run the test suite" in harness.telegram.sent(CHAT)[-1].text

    harness.callback(buttons[1]["callback_data"])
    assert harness.wait_idle()
    responses = harness.permission_responses()
    assert responses[0]["message"]["result"] == {
        "outcome": {"outcome": "selected", "optionId": "allow_always"}
    }
    # End-to-end proof of the fix: the agent decodes the tap as an ALLOW, not a
    # decline. A flat outcome would be rejected by its unmarshaller and the tool
    # call would read as denied, which is the bug this guards against.
    decisions = events_named(harness.agent_events(), "permission_decision")
    assert decisions and decisions[0]["allowed"] is True


def test_callback_from_an_unauthorised_user_is_refused_by_the_gateway(make_harness: Any) -> None:
    harness = make_harness("--permission")
    harness.send("/bind alpha")
    harness.send("run the tests")
    assert harness.wait_for_buttons()

    harness.callback(harness.buttons()[0]["callback_data"], user_id=666)
    assert harness.telegram.callback_answers[-1][1] == "Not authorised."
    assert harness.permission_responses() == []

    harness.send("/stop")
    assert harness.wait_idle()


def test_turn_is_skipped_when_telegram_is_unreachable(make_harness: Any) -> None:
    """No message means no turn: never spend a turn nobody could see."""
    from acp_im_gateway.telegram import TelegramError

    harness = make_harness()
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.telegram.fail_send = TelegramError("telegram is down", status=500)

    harness.send("this prompt must not reach the agent")
    assert wait_until(lambda: not harness.runtime.busy)

    assert harness.prompts() == []
    assert harness.agent_requests("session/new") == []


def test_stream_failure_mid_turn_does_not_kill_the_gateway(make_harness: Any) -> None:
    from acp_im_gateway.telegram import TelegramError

    harness = make_harness()
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()
    harness.telegram.reset()
    harness.telegram.fail_edit = TelegramError("flaky edits", status=500)

    harness.send("another prompt")
    assert wait_until(lambda: not harness.runtime.busy)
    # The turn still ran and produced its message; only the edits failed, and the
    # failure was logged instead of killing the gateway.
    assert len(harness.prompts()) == 2
    assert len(harness.telegram.sent(CHAT)) == 1
    assert harness.telegram.edits(CHAT) == []


def test_turn_loop_reports_dropped_messages_after_an_internal_error(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/bind alpha")
    runtime = harness.runtime
    runtime.busy = True
    runtime.enqueue("queued one")
    runtime.enqueue("queued two")

    def boom(chat_id: int, binding: Any, text: str) -> None:
        raise RuntimeError("kaboom")

    harness.gateway._execute_turn = boom  # type: ignore[method-assign]
    harness.gateway._turn_loop(CHAT, "first")

    assert runtime.busy is False
    assert runtime.queue_size == 0
    assert "Dropped 2 queued message(s)" in harness.last_reply()


def test_the_agent_posture_is_pinned_to_ask_the_gateway_is_the_only_gatekeeper(
    make_harness: Any,
) -> None:
    """v1.2 §2: never loosen the agent's ``tool_approval`` posture.

    If the agent stopped asking, ``always_ask`` would never see the call and money
    could be spent silently. So neither a widening gateway posture nor ``/aprobar
    auto`` may ever turn the *agent's* posture down: the gateway answers every
    request itself.
    """
    for posture in ("auto", "yolo"):
        harness = make_harness(
            "--permission",
            "--permission-command",
            "make build",
            approval_posture=posture,
        )
        harness.send("/bind alpha")
        harness.send("/aprobar auto")
        harness.send("do something")
        assert harness.wait_idle()

        requests = harness.agent_requests("session/set_config_option")
        assert requests == [], (
            f"the gateway must never request a tool_approval posture from the "
            f"agent (posture={posture}): {requests}"
        )
        # The command was still answered by the gateway, without a tap…
        assert harness.buttons() == []
        assert harness.permission_responses(), "the agent must still be answered"
        # …and the fake agent's own posture is untouched: it keeps asking.
        assert harness.gateway.tiers.posture_for(harness.chat_id) == "auto"


def test_aprobar_auto_silences_the_unknown_but_still_taps_always_ask(
    make_harness: Any, tmp_path: Path
) -> None:
    """``/aprobar auto`` widens the *gateway*: unknown silent, always_ask still taps."""
    # An unknown command stops asking in an auto chat…
    widened = make_harness(
        "--permission", "--permission-command", "make build",
        state_file=tmp_path / "state-auto.json",
    )
    widened.send("/bind alpha")
    widened.send("/aprobar auto")
    assert "auto" in widened.last_reply()
    widened.send("build it")
    assert widened.wait_idle()
    assert widened.buttons() == [], "an unknown command must be silent at /aprobar auto"
    assert widened.permission_responses(), "the agent must still be answered"

    # …while always_ask is still a code gate: the tap is required.
    gated = make_harness(
        "--permission", "--permission-command", "git push origin main",
        state_file=tmp_path / "state-gated.json",
    )
    gated.send("/bind alpha")
    gated.send("/aprobar auto")
    gated.send("ship it")
    assert gated.wait_for_buttons(), "always_ask must still tap at /aprobar auto"
    assert gated.permission_responses() == []
    gated.callback(gated.buttons()[0]["callback_data"])
    assert gated.wait_idle()
    assert gated.permission_responses()

    # The default (preguntar) keeps asking for the unknown. Its own state file
    # keeps it clear of the "auto" binding the earlier harnesses persisted.
    asking = make_harness(
        "--permission", "--permission-command", "make build",
        chat_id=222, state_file=tmp_path / "state-asking.json",
    )
    asking.send("/bind alpha")
    asking.send("build it")
    assert asking.wait_for_buttons(), "the default posture still asks"
    asking.callback(asking.buttons()[0]["callback_data"])
    assert asking.wait_idle()


def test_aprobar_persists_in_the_binding_and_shows_in_status(make_harness: Any) -> None:
    harness = make_harness()
    harness.send("/aprobar")  # not bound yet
    assert "/bind" in harness.last_reply()

    harness.send("/bind alpha")
    harness.send("/aprobar auto")
    assert "auto" in harness.last_reply()
    assert harness.gateway.router.get(CHAT).posture == "auto"
    assert harness.state()["bindings"][str(CHAT)]["approval"] == "auto"

    harness.send("/status")
    status = harness.last_reply()
    assert "approval posture: auto" in status
    assert "agent stays ask" in status

    harness.send("/aprobar preguntar")
    assert "preguntar" in harness.last_reply()
    assert harness.gateway.router.get(CHAT).posture == "ask"

    harness.send("/aprobar nonsense")
    assert "preguntar" in harness.last_reply() and "auto" in harness.last_reply()


def test_status_shows_the_effective_gateway_posture(make_harness: Any) -> None:
    """A widening GATEWAY_APPROVAL_POSTURE shows in /status (the agent still stays ask)."""
    harness = make_harness(approval_posture="auto")
    harness.send("/bind alpha")
    harness.send("/status")
    assert "approval posture: auto" in harness.last_reply()
    assert "agent stays ask" in harness.last_reply()

    harness.send("/aprobar preguntar")
    harness.send("/status")
    assert "approval posture: ask" in harness.last_reply()


def test_aprobar_survives_a_gateway_restart(make_harness: Any) -> None:
    """The posture is persisted in the binding, so a restart keeps it."""
    first = make_harness("--permission", "--permission-command", "make build")
    first.send("/bind alpha")
    first.send("/aprobar auto")

    second = make_harness("--permission", "--permission-command", "make build")
    second.send("still silent after a restart")
    assert second.wait_idle()
    assert second.buttons() == [], "the persisted posture must be applied after a restart"
    assert second.permission_responses()


def test_approval_posture_is_never_a_way_to_stop_the_agent_asking(make_harness: Any) -> None:
    """A widening config posture must not reach the agent's config options."""
    harness = make_harness("--permission", approval_posture="auto")
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()
    assert harness.agent_requests("session/set_config_option") == []
    assert harness.any_text("✅ done")  # the turn ran regardless


def test_matching_posture_is_not_requested(make_harness: Any) -> None:
    harness = make_harness()  # default posture: ask, which the fake already reports
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()
    assert harness.agent_requests("session/set_config_option") == []


def test_turns_after_the_first_one_start_immediately(make_harness: Any) -> None:
    """Each message runs its own turn: nothing is stranded behind a finished turn."""
    harness = make_harness()
    harness.send("/bind alpha")

    calls: list[str] = []
    original = harness.gateway._execute_turn

    def stub(chat_id: int, binding: Any, text: str) -> None:
        calls.append(text)
        original(chat_id, binding, text)

    harness.gateway._execute_turn = stub  # type: ignore[method-assign]
    harness.send("first")
    assert harness.wait_idle()
    harness.send("second")
    assert harness.wait_idle()

    assert calls == ["first", "second"]
    assert harness.runtime.queue_size == 0
    assert "Dropped" not in harness.last_reply()
    assert [prompt["text"] for prompt in harness.prompts()] == ["first", "second"]


# --------------------------------------------------------------------------- delivery failures


def test_a_permanently_rejected_body_stops_retrying_and_unblocks_the_chat(
    make_harness: Any,
) -> None:
    """Regression for the hang: a body Telegram rejects *for good* must not be
    retried on every poll forever. The turn has to complete so the chat unlocks
    and the queue drains."""
    harness = make_harness("--slow", "2.0", heartbeat_seconds=0.05, edit_interval=0.0)
    harness.send("/bind alpha")
    harness.send("first")
    assert harness.wait_idle()

    harness.telegram.reset()
    harness.telegram.fail_edit = PARSE_400  # every edit is refused, permanently

    harness.send("second")
    harness.send("third")  # queued behind the running turn
    assert harness.runtime.queue_size == 1

    # The gateway gives up and says so, in plain text, instead of looping.
    assert wait_until(
        lambda: any("stopped retrying" in text for text in harness.replies()), timeout=10
    )

    settled = harness.telegram.edit_attempts
    time.sleep(0.6)
    assert harness.telegram.edit_attempts == settled, "retrying must stay bounded"
    # Each of the MAX_DELIVERY_FAILURES pushes makes at most an HTML + a plain try.
    assert settled <= MAX_DELIVERY_FAILURES * 2

    # …and the chat is not left blocked: both turns run and the queue drains.
    assert harness.wait_idle(timeout=15)
    assert harness.runtime.busy is False
    assert harness.runtime.queue_size == 0
    assert [prompt["text"] for prompt in harness.prompts()] == ["first", "second", "third"]


def test_stop_takes_effect_while_delivery_is_failing(make_harness: Any) -> None:
    harness = make_harness("--permission", heartbeat_seconds=0.05, edit_interval=0.0)
    harness.send("/bind alpha")
    harness.send("long turn")
    assert harness.wait_for_buttons()

    harness.telegram.fail_edit = PARSE_400  # streaming edits now fail

    harness.send("/stop")
    assert harness.wait_idle(timeout=10)
    assert harness.runtime.busy is False
    assert events_named(harness.agent_events(), "cancel"), "the agent must be told to cancel"
    assert any("Cancelling" in text for text in harness.replies())


def test_a_turn_refused_for_its_markup_is_still_delivered_as_plain_text(
    make_harness: Any,
) -> None:
    """A/4 end to end: when Telegram refuses the HTML, the turn's content still
    reaches the chat (unformatted) and the turn completes normally."""
    harness = make_harness(
        "--tool-command", "pytest -q", "--tool-output", "1 passed in 0.10s"
    )
    harness.send("/bind alpha")
    harness.telegram.reset()
    harness.telegram.reject_html = True  # every HTML call is refused

    harness.send("run the tests")
    assert harness.wait_idle()

    assert harness.runtime.busy is False
    chat = harness.current_text()
    assert "$ pytest -q [completed]" in chat  # tags stripped, content kept
    assert "<pre>" not in chat
    assert "1 passed in 0.10s" in chat
