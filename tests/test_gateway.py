"""Gateway behaviour end to end: real ACP client + fake agent + fake Telegram.

These tests never touch the network and never send a ``session/prompt`` to a real
agent: the ACP side is the scriptable fake in ``tests/fake_agent.py``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import pytest

from acp_im_gateway.acp import STEER_METHOD
from acp_im_gateway.config import Config
from acp_im_gateway.gateway import Gateway, TurnView

from .helpers import (
    FAKE_AGENT,
    PYTHON,
    FakeTelegram,
    approval_buttons,
    callback_update,
    events_named,
    message_update,
    read_agent_log,
    wait_until,
)

LOG = logging.getLogger("tests.gateway")
CHAT = 111
USER = 900


@dataclass
class Harness:
    gateway: Gateway
    telegram: FakeTelegram
    log_path: Path
    chat_id: int = CHAT

    # -- driving ---------------------------------------------------------------

    def send(self, text: str, **kwargs: Any) -> None:
        kwargs.setdefault("chat_id", self.chat_id)
        kwargs.setdefault("user_id", USER)
        self.gateway.handle_update(message_update(text, **kwargs))

    def callback(self, data: str, **kwargs: Any) -> None:
        kwargs.setdefault("chat_id", self.chat_id)
        kwargs.setdefault("user_id", USER)
        self.gateway.handle_update(callback_update(data, **kwargs))

    # -- inspecting ------------------------------------------------------------

    @property
    def runtime(self) -> Any:
        return self.gateway.router.runtime(self.chat_id)

    def replies(self) -> list[str]:
        return self.telegram.texts(self.chat_id)

    def last_reply(self) -> str:
        sent = self.telegram.sent(self.chat_id)
        return sent[-1].text if sent else ""

    def current_text(self) -> str:
        return self.telegram.current_text(self.chat_id) or ""

    def agent_events(self) -> list[dict[str, Any]]:
        return read_agent_log(self.log_path)

    def agent_requests(self, method: str) -> list[dict[str, Any]]:
        """JSON-RPC requests the agent received, by method."""
        return [
            event
            for event in self.agent_events()
            if event.get("event") == "request" and event.get("method") == method
        ]

    def prompts(self) -> list[dict[str, Any]]:
        """Prompts the agent received, with the text exactly as it arrived."""
        return events_named(self.agent_events(), "prompt")

    def any_text(self, needle: str) -> bool:
        return any(
            needle in message.text
            for message in self.telegram.messages
            if message.chat_id == self.chat_id
        )

    def permission_responses(self) -> list[dict[str, Any]]:
        return events_named(self.agent_events(), "permission_response")

    def buttons(self) -> list[dict[str, Any]]:
        return approval_buttons(self.telegram, self.chat_id)

    def wait_idle(self, timeout: float = 10.0) -> bool:
        return wait_until(lambda: not self.runtime.busy, timeout=timeout)

    def wait_for_buttons(self, timeout: float = 10.0) -> bool:
        return wait_until(lambda: bool(self.buttons()), timeout=timeout)

    def state(self) -> dict[str, Any]:
        return json.loads(self.gateway.store.path.read_text(encoding="utf-8"))


@pytest.fixture()
def make_harness(tmp_path: Path, config: Config) -> Any:
    created: list[Gateway] = []

    def factory(
        *agent_flags: str,
        allowed_users: Sequence[int] = (USER,),
        allowed_chats: Sequence[int] = (),
        chat_id: int = CHAT,
        **overrides: Any,
    ) -> Harness:
        log_path = tmp_path / f"agent-{len(created)}.jsonl"
        cfg = replace(
            config,
            agent_cmd=(PYTHON, str(FAKE_AGENT), "--log", str(log_path), *agent_flags),
            allowed_user_ids=frozenset(allowed_users),
            allowed_chat_ids=frozenset(allowed_chats),
            **overrides,
        )
        telegram = FakeTelegram()
        gateway = Gateway(cfg, telegram=telegram, log=LOG)
        gateway.start()
        created.append(gateway)
        return Harness(gateway=gateway, telegram=telegram, log_path=log_path, chat_id=chat_id)

    yield factory
    for gateway in created:
        gateway.stop()


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


def test_group_chats_need_to_be_listed(make_harness: Any) -> None:
    refused = make_harness(allowed_chats=(), chat_id=-500)
    refused.send("hello", chat_type="supergroup")
    assert "ALLOWED_CHAT_IDS" in refused.last_reply()

    allowed = make_harness(allowed_chats=(-500,), chat_id=-500)
    allowed.send("hello", chat_type="supergroup")
    assert "/bind" in allowed.last_reply()


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
        "outcome": "selected",
        "optionId": "allow_always",
    }


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


def test_approval_posture_is_requested_when_it_differs(make_harness: Any) -> None:
    harness = make_harness(approval_posture="auto")
    harness.send("/bind alpha")
    harness.send("hello")
    assert harness.wait_idle()

    requests = harness.agent_requests("session/set_config_option")
    assert requests, "the gateway should ask for the configured posture"
    assert requests[0]["params"]["value"] == "auto"
    assert requests[0]["params"]["configId"] == "tool_approval"
    assert harness.any_text("✅ done")  # a refusal is logged, not fatal


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
