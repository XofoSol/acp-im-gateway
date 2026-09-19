"""Integration test against the real ACP agent.

It performs the handshake required by the spec — ``initialize`` -> ``session/new``
-> ``session/close`` — and **never** sends ``session/prompt``: that would spend the
user's money. The test skips cleanly when the agent binary is not installed.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pytest

from acp_im_gateway.acp import AcpClient

LOG = logging.getLogger("tests.integration")

pytestmark = pytest.mark.integration


def test_real_agent_handshake_initialize_new_session_close(tmp_path: Path) -> None:
    binary = shutil.which("reasonix")
    if binary is None:
        pytest.skip("the real agent binary (reasonix) is not on PATH")

    client = AcpClient([binary, "acp"], request_timeout=120.0, auto_respawn=False, log=LOG)
    client.start()
    try:
        capabilities = client.initialize()
        assert capabilities.protocol_version is not None
        assert capabilities.agent_name, "the agent must identify itself"
        caps = capabilities.raw.get("agentCapabilities") or {}
        assert isinstance(caps, dict)

        session = client.new_session(tmp_path)
        assert session.session_id
        assert session.config_options, "session/new must advertise configOptions"
        # The spec's default posture: the agent asks before gated tool calls.
        assert session.approval_posture == "ask"

        sessions = client.list_sessions(tmp_path)
        assert any(entry.session_id == session.session_id for entry in sessions), (
            "the new session should be listed for its cwd"
        )

        client.close_session(session.session_id)
    finally:
        client.stop()

    # The whole point of this test: no turn was ever run.
    assert "session/prompt" not in client.sent_methods
