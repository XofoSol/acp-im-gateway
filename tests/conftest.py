"""Pytest fixtures shared by the suite.

The runtime is standard-library only, but ``pytest`` is a development dependency
(``pip install -e .[dev]`` or just ``pip install pytest``). A bare checkout is
also supported: helpers.py puts the repository root on ``sys.path``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from acp_im_gateway.config import Config

from .helpers import FAKE_AGENT, PYTHON, FakeClock, FakeTelegram, make_project


@pytest.fixture()
def projects_root(tmp_path: Path) -> Path:
    """An allowed root with two git repositories and one non-repository."""
    root = tmp_path / "Projects"
    root.mkdir()
    make_project(root, "alpha")
    make_project(root, "beta")
    (root / "not-a-repo").mkdir()
    return root


@pytest.fixture()
def config(tmp_path: Path, projects_root: Path) -> Config:
    """A dry-run-safe config: private root, local state file, no secrets."""
    return Config(
        telegram_bot_token="",
        projects_root=projects_root,
        allowed_roots=(projects_root,),
        agent_index_dir=tmp_path / "agent-index",
        state_file=tmp_path / "state" / "state.json",
        agent_cmd=(PYTHON, str(FAKE_AGENT)),
        edit_interval=1.2,
        send_interval=1.0,
        poll_timeout=1,
        busy_mode="steer",
        pairing_ttl=900.0,
        approval_timeout=300.0,
        discovery_depth=1,
        log_level="DEBUG",
        dry_run=True,
    )


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def telegram() -> FakeTelegram:
    return FakeTelegram()


@pytest.fixture()
def caplog_info(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.DEBUG)
    return caplog
