"""ACP client behaviour, driven against a scriptable fake agent on stdio."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Mapping

import pytest

from acp_im_gateway.acp import (
    DECLINE,
    DEFER,
    AcpClient,
    AcpError,
    AgentCrashed,
    AgentNotRunning,
    AcpTimeout,
    InboundRequest,
    STEER_METHOD,
)

from .helpers import FAKE_AGENT, PYTHON, events_named, read_agent_log, wait_until

LOG = logging.getLogger("tests.acp")


def make_client(log_path: Path, *flags: str, **kwargs: Any) -> AcpClient:
    command = [PYTHON, str(FAKE_AGENT), "--log", str(log_path), *flags]
    kwargs.setdefault("request_timeout", 10.0)
    kwargs.setdefault("restart_backoff", (0.02,))
    kwargs.setdefault("restart_backoff_max", 0.05)
    return AcpClient(command, log=LOG, **kwargs)


def test_initialize_exposes_capabilities(tmp_path: Path) -> None:
    client = make_client(tmp_path / "log.jsonl", "--steer", "--load-session")
    with client:
        capabilities = client.initialize()
    assert capabilities.protocol_version == 1
    assert capabilities.agent_name == "fake-agent"
    assert capabilities.agent_version == "0.0.1"
    assert capabilities.load_session is True
    assert capabilities.supports_list is True
    assert capabilities.supports_close is True
    assert capabilities.supports_steer is True
    assert capabilities.steer_method == STEER_METHOD
    assert capabilities.supports_images is False and capabilities.supports_audio is False


def test_steering_is_absent_when_the_agent_does_not_advertise_it(tmp_path: Path) -> None:
    client = make_client(tmp_path / "log.jsonl")
    with client:
        capabilities = client.initialize()
    assert capabilities.supports_steer is False
    assert capabilities.steer_method is None


def test_prompt_forwards_text_verbatim_and_streams_updates(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    client = make_client(log_path)
    updates: list[Mapping[str, Any]] = []
    tricky = "línea 1\nlínea 2\t tab\n```python\nprint('$HOME')\n```\n(no rewriting)"
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        assert session.session_id == "sess-1"
        assert session.model == "fake-model"
        assert session.approval_posture == "ask"
        ticks: list[int] = []
        result = client.prompt(
            session.session_id, tricky, on_update=updates.append, on_tick=lambda: ticks.append(1)
        )

    assert result.stop_reason == "end_turn"
    assert result.updates == len(updates) >= 4

    kinds = [update["params"]["update"]["sessionUpdate"] for update in updates]
    assert "agent_message_chunk" in kinds
    assert "tool_call" in kinds

    events = read_agent_log(log_path)
    prompts = events_named(events, "prompt")
    assert len(prompts) == 1
    assert prompts[0]["text"] == tricky  # byte-for-byte, no LLM in the middle
    assert prompts[0]["sessionId"] == "sess-1"
    assert ticks, "the prompt loop should tick so callers can flush streaming output"


def test_non_json_stdout_lines_are_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="tests.acp")
    client = make_client(tmp_path / "log.jsonl", "--junk")
    with client:
        capabilities = client.initialize()
        session = client.new_session(tmp_path)
        result = client.prompt(session.session_id, "hello", timeout=10.0)
    assert capabilities.agent_name == "fake-agent"
    assert result.stop_reason == "end_turn"
    assert "not JSON" in caplog.text


def test_unknown_notification_is_ignored(tmp_path: Path) -> None:
    seen: list[str] = []
    client = make_client(tmp_path / "log.jsonl", on_notification=lambda m: seen.append(str(m.get("method"))))
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        result = client.prompt(session.session_id, "hello", timeout=10.0)
    assert result.stop_reason == "end_turn"
    assert "fake/unknown_notification" in seen


def test_unknown_inbound_request_is_declined(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    client = make_client(log_path, "--unknown-request")
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        result = client.prompt(session.session_id, "hello", timeout=10.0)
    assert result.stop_reason == "end_turn"
    responses = events_named(read_agent_log(log_path), "unknown_response")
    assert len(responses) == 1
    message = responses[0]["message"]
    assert message["error"]["code"] == -32601  # declined cleanly, never a hang


def test_permission_request_is_deferred_then_answered(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    pending: list[InboundRequest] = []
    received = threading.Event()

    def handler(request: InboundRequest) -> Any:
        if request.method == "session/request_permission":
            pending.append(request)
            received.set()
            return DEFER
        return DECLINE

    client = make_client(log_path, "--permission", on_request=handler)
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        outcome: list[Any] = []

        def run_turn() -> None:
            outcome.append(client.prompt(session.session_id, "run the tests", timeout=10.0))

        worker = threading.Thread(target=run_turn, daemon=True)
        worker.start()
        assert received.wait(10.0), "the agent never asked for permission"

        request = pending[0]
        assert request.params["toolCall"]["title"] == "Run the test suite"
        assert [option["optionId"] for option in request.params["options"]] == [
            "allow_once",
            "allow_always",
            "reject_once",
        ]
        assert request.answered is False
        request.respond({"outcome": "selected", "optionId": "allow_always"})
        worker.join(timeout=10.0)

    assert outcome and outcome[0].stop_reason == "end_turn"
    responses = events_named(read_agent_log(log_path), "permission_response")
    assert responses[0]["message"]["result"] == {"outcome": "selected", "optionId": "allow_always"}


def test_default_handler_cancels_permission_requests_instead_of_hanging(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    client = make_client(log_path, "--permission")
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        result = client.prompt(session.session_id, "run the tests", timeout=10.0)
    assert result.stop_reason == "end_turn"
    responses = events_named(read_agent_log(log_path), "permission_response")
    assert responses[0]["message"]["result"] == {"outcome": "cancelled"}


def test_crash_mid_turn_raises_and_the_agent_comes_back(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    restarts: list[int] = []
    client = make_client(log_path, "--crash-on-prompt", on_restart=lambda: restarts.append(1))
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        with pytest.raises(AgentCrashed):
            client.prompt(session.session_id, "this will die", timeout=10.0)
        assert wait_until(lambda: client.running and restarts, timeout=10.0)
        # The respawned agent is usable again.
        assert client.initialize().agent_name == "fake-agent"


def test_respawn_after_exit_and_request_failure_without_respawn(tmp_path: Path) -> None:
    restarts: list[int] = []
    client = make_client(tmp_path / "log.jsonl", "--exit-after", "1", on_restart=lambda: restarts.append(1))
    with client:
        client.initialize()  # 1 request: the fake exits right after answering
        assert wait_until(lambda: bool(restarts), timeout=10.0)
        assert client.initialize().agent_name == "fake-agent"

    plain = make_client(tmp_path / "log2.jsonl", "--exit-after", "1", auto_respawn=False)
    with plain:
        plain.initialize()
        assert wait_until(lambda: not plain.running, timeout=10.0)
        with pytest.raises(AgentNotRunning):
            plain.list_sessions(tmp_path)


def test_prompt_timeout_is_enforced(tmp_path: Path) -> None:
    # A handler that defers forever: the turn never finishes on its own.
    client = make_client(tmp_path / "log.jsonl", "--permission", on_request=lambda request: DEFER)
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        with pytest.raises(AcpTimeout):
            client.prompt(session.session_id, "run the tests", timeout=0.3)


def test_steer_and_cancel_and_session_lifecycle(tmp_path: Path) -> None:
    log_path = tmp_path / "log.jsonl"
    client = make_client(log_path, "--steer")
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        assert client.steer(session.session_id, "also check the docs") == {}
        assert client.cancel(session.session_id) is True
        listed = client.list_sessions(tmp_path)
        assert [entry.session_id for entry in listed] == ["sess-1"]
        client.close_session(session.session_id)
        assert client.list_sessions(tmp_path) == []

    events = read_agent_log(log_path)
    steer = events_named(events, "request")
    assert any(event["method"] == STEER_METHOD for event in steer)
    assert len(events_named(events, "cancel")) == 1


def test_steer_without_advertisement_raises(tmp_path: Path) -> None:
    client = make_client(tmp_path / "log.jsonl")
    with client:
        client.initialize()
        session = client.new_session(tmp_path)
        with pytest.raises(AcpError, match="does not advertise session steering"):
            client.steer(session.session_id, "hello")


def test_load_session_without_capability_raises(tmp_path: Path) -> None:
    client = make_client(tmp_path / "log.jsonl")
    with client:
        client.initialize()
        with pytest.raises(AcpError, match="loadSession"):
            client.load_session("sess-1", tmp_path)


def test_load_session_when_advertised(tmp_path: Path) -> None:
    client = make_client(tmp_path / "log.jsonl", "--load-session")
    with client:
        client.initialize()
        loaded = client.load_session("sess-existing", tmp_path)
    assert loaded.session_id == "sess-existing"


def test_stop_is_clean_and_blocks_further_requests(tmp_path: Path) -> None:
    client = make_client(tmp_path / "log.jsonl")
    client.start()
    client.initialize()
    client.stop()
    client.stop()  # idempotent
    assert client.running is False
    with pytest.raises(AgentNotRunning):
        client.initialize()


def test_missing_agent_command_reports_clearly() -> None:
    client = AcpClient(("definitely-not-an-agent-binary", "acp"), log=LOG)
    with pytest.raises(AgentNotRunning, match="not found"):
        client.start()


class _RecordingClient:
    """Minimal stand-in for AcpClient that records JSON-RPC replies."""

    def __init__(self) -> None:
        self.writes: list[Mapping[str, Any]] = []

    def _write(self, message: Mapping[str, Any]) -> bool:
        self.writes.append(dict(message))
        return True

    def _forget_inbound(self, request_id: Any) -> None:
        return None


def test_an_inbound_request_is_answered_at_most_once() -> None:
    client = _RecordingClient()
    request = InboundRequest(client, 5, "session/request_permission", {"sessionId": "s1"})
    assert request.respond({"outcome": "cancelled"}) is True
    assert request.respond({"outcome": "selected", "optionId": "allow_once"}) is False
    assert request.fail(-32603, "too late") is False
    assert request.abandon("agent went away") is None
    assert len(client.writes) == 1


def test_concurrent_answers_to_one_request_write_exactly_one_response() -> None:
    client = _RecordingClient()
    request = InboundRequest(client, 6, "session/request_permission", {"sessionId": "s1"})
    outcomes: list[bool] = []
    outcomes_lock = threading.Lock()

    def answer(kind: str) -> None:
        result = (
            request.respond({"outcome": "cancelled"})
            if kind == "respond"
            else request.fail(-32603, "declined")
        )
        with outcomes_lock:
            outcomes.append(result)

    threads = [
        threading.Thread(target=answer, args=("respond" if index % 2 else "fail",))
        for index in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)

    assert outcomes.count(True) == 1
    assert len(client.writes) == 1
    assert request.answered is True
