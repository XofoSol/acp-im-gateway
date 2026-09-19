"""CLI behaviour: ``--help``, config errors, pairing, bindings, dry-run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from acp_im_gateway.__main__ import build_parser, main
from acp_im_gateway.router import StateStore

from .helpers import FAKE_AGENT, PYTHON

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_cli(
    args: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 90.0,
) -> subprocess.CompletedProcess[str]:
    """Run the CLI in a subprocess with a minimal, isolated environment."""
    base_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "PYTHONPATH": str(REPO_ROOT),
        "PYTHONUNBUFFERED": "1",
    }
    base_env.update(env or {})
    return subprocess.run(
        [PYTHON, "-m", "acp_im_gateway", *args],
        capture_output=True,
        text=True,
        env=base_env,
        cwd=str(cwd or REPO_ROOT),
        timeout=timeout,
    )


def isolated_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    """Env with no ambient token/TOML: every test owns its own paths."""
    env = {
        "GATEWAY_CONFIG": str(tmp_path / "missing.toml"),
        "GATEWAY_ENV_FILE": str(tmp_path / "missing.env"),
        "GATEWAY_STATE_FILE": str(tmp_path / "state.json"),
        "REASONIX_PROJECTS_DIR": str(tmp_path / "agent-project-index"),
        "TELEGRAM_BOT_TOKEN": "",
    }
    env.update(extra)
    return env


def seed_pending_pairing(state_file: Path, code: str, *, user_id: int = 901) -> None:
    """Add one pending pairing code to the state file, keeping everything else."""
    import time

    store = StateStore(state_file)
    data: dict[str, Any] = store.load() or {"version": 1, "telegram_offset": 0, "bindings": {}}
    data.setdefault("version", 1)
    access = data.setdefault("access", {"allowed_user_ids": [], "allowed_chat_ids": []})
    access.setdefault("allowed_user_ids", [])
    access.setdefault("allowed_chat_ids", [])
    pending = access.setdefault("pending", {})
    moment = time.time()
    pending[code] = {
        "code": code,
        "user_id": user_id,
        "chat_id": 111,
        "username": "newcomer",
        "created_at": moment,
        "expires_at": moment + 900.0,
        "approved_at": None,
        "notified": False,
    }
    store.save(data)

# --------------------------------------------------------------------------- parser


def test_help_lists_the_subcommands() -> None:
    result = run_cli(["--help"])
    assert result.returncode == 0, result.stderr
    for command in ("run", "pairing", "bindings", "projects", "config"):
        assert command in result.stdout
    assert "commands:" in result.stdout


def test_nested_help_lists_actions() -> None:
    pairing = run_cli(["pairing", "--help"])
    assert pairing.returncode == 0
    for action in ("list", "approve", "reject"):
        assert action in pairing.stdout

    bindings = run_cli(["bindings", "--help"])
    assert bindings.returncode == 0
    for action in ("list", "add", "remove"):
        assert action in bindings.stdout


def test_flags_work_before_and_after_the_subcommand() -> None:
    parser = build_parser()
    before = parser.parse_args(["--dry-run", "projects"])
    after = parser.parse_args(["projects", "--dry-run"])
    assert before.dry_run is True and after.dry_run is True
    assert parser.parse_args(["run"]).command == "run"
    with pytest.raises(SystemExit):
        parser.parse_args([])  # a command is required


def test_main_is_callable_in_process(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config"]) == 0
    assert "acp-im-gateway" in capsys.readouterr().out


# --------------------------------------------------------------------------- run


def test_run_without_a_token_fails_with_an_actionable_message(tmp_path: Path) -> None:
    env = isolated_env(tmp_path)
    env.pop("TELEGRAM_BOT_TOKEN")
    result = run_cli(["run"], env=env, cwd=tmp_path)
    assert result.returncode == 2
    assert "TELEGRAM_BOT_TOKEN" in result.stderr


def test_run_dry_run_starts_polls_and_exits_cleanly(tmp_path: Path, projects_root: Path) -> None:
    env = isolated_env(
        tmp_path,
        TELEGRAM_BOT_TOKEN="123456:dry-run-token",
        GATEWAY_DRY_RUN="1",
        GATEWAY_POLL_TIMEOUT="1",
        PROJECTS_ROOT=str(projects_root),
        REASONIX_ACP_CMD=f"{PYTHON} {FAKE_AGENT}",
    )
    result = run_cli(["run", "--max-polls", "2"], env=env, cwd=tmp_path, timeout=90.0)
    assert result.returncode == 0, result.stderr
    assert "starting:" in result.stderr
    assert "agent ready" in result.stderr  # the ACP handshake completed
    # State is written even on a dry run.
    assert (tmp_path / "state.json").is_file()


def test_run_reports_a_missing_agent_command(tmp_path: Path, projects_root: Path) -> None:
    env = isolated_env(
        tmp_path,
        TELEGRAM_BOT_TOKEN="123456:dry-run-token",
        GATEWAY_DRY_RUN="1",
        GATEWAY_POLL_TIMEOUT="1",
        PROJECTS_ROOT=str(projects_root),
        REASONIX_ACP_CMD="definitely-not-an-agent",
    )
    result = run_cli(["run", "--max-polls", "1"], env=env, cwd=tmp_path, timeout=90.0)
    assert result.returncode == 1
    assert "agent error" in result.stderr


# --------------------------------------------------------------------------- pairing


def test_pairing_list_approve_and_reject(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    env = isolated_env(tmp_path)
    seed_pending_pairing(state_file, "ABCD2345", user_id=901)

    listed = run_cli(["pairing", "list"], env=env, cwd=tmp_path)
    assert listed.returncode == 0
    assert "ABCD2345" in listed.stdout and "901" in listed.stdout

    approved = run_cli(["pairing", "approve", "ABCD2345"], env=env, cwd=tmp_path)
    assert approved.returncode == 0, approved.stderr
    assert "approved user 901" in approved.stdout
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["access"]["allowed_user_ids"] == [901]

    seed_pending_pairing(state_file, "REJECT99", user_id=902)
    rejected = run_cli(["pairing", "reject", "REJECT99"], env=env, cwd=tmp_path)
    assert rejected.returncode == 0
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert "REJECT99" not in state["access"]["pending"]
    assert state["access"]["allowed_user_ids"] == [901]


def test_pairing_approve_with_an_unknown_code_fails(tmp_path: Path) -> None:
    env = isolated_env(tmp_path)
    result = run_cli(["pairing", "approve", "NOPE1234"], env=env, cwd=tmp_path)
    assert result.returncode == 1
    assert "unknown pairing code" in result.stderr


def test_pairing_list_when_empty(tmp_path: Path) -> None:
    result = run_cli(["pairing", "list"], env=isolated_env(tmp_path), cwd=tmp_path)
    assert result.returncode == 0
    assert "No pending pairing codes" in result.stdout


# --------------------------------------------------------------------------- bindings


def test_bindings_add_list_and_remove(tmp_path: Path, projects_root: Path) -> None:
    env = isolated_env(tmp_path, PROJECTS_ROOT=str(projects_root))
    project = projects_root / "alpha"

    added = run_cli(["bindings", "add", "222", str(project)], env=env, cwd=tmp_path)
    assert added.returncode == 0, added.stderr
    assert "bound chat 222" in added.stdout

    listed = run_cli(["bindings", "list"], env=env, cwd=tmp_path)
    assert "222" in listed.stdout and "alpha" in listed.stdout
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["bindings"]["222"]["project_root"] == str(project.resolve())

    removed = run_cli(["bindings", "remove", "222"], env=env, cwd=tmp_path)
    assert removed.returncode == 0
    assert "No bindings yet" in run_cli(["bindings", "list"], env=env, cwd=tmp_path).stdout
    assert run_cli(["bindings", "remove", "222"], env=env, cwd=tmp_path).returncode == 1


def test_bindings_add_outside_the_allowed_root_is_refused(tmp_path: Path, projects_root: Path) -> None:
    outside = tmp_path / "outside-the-root"
    outside.mkdir()
    env = isolated_env(tmp_path, PROJECTS_ROOT=str(projects_root))
    result = run_cli(["bindings", "add", "222", str(outside)], env=env, cwd=tmp_path)
    assert result.returncode == 1
    assert "outside the allowed roots" in result.stderr
    assert not (tmp_path / "state.json").exists() or json.loads(
        (tmp_path / "state.json").read_text(encoding="utf-8")
    )["bindings"] == {}


def test_bindings_list_when_empty(tmp_path: Path) -> None:
    result = run_cli(["bindings", "list"], env=isolated_env(tmp_path), cwd=tmp_path)
    assert result.returncode == 0
    assert "No bindings yet" in result.stdout


# --------------------------------------------------------------------------- projects and config


def test_projects_command_lists_candidates(tmp_path: Path, projects_root: Path) -> None:
    env = isolated_env(tmp_path, PROJECTS_ROOT=str(projects_root))
    result = run_cli(["projects"], env=env, cwd=tmp_path)
    assert result.returncode == 0
    assert "alpha" in result.stdout and "beta" in result.stdout
    assert str(projects_root) in result.stdout


def test_projects_command_when_nothing_is_found(tmp_path: Path) -> None:
    empty = tmp_path / "empty-root"
    empty.mkdir()
    env = isolated_env(tmp_path, PROJECTS_ROOT=str(empty))
    result = run_cli(["projects"], env=env, cwd=tmp_path)
    assert result.returncode == 0
    assert "No projects discovered" in result.stdout


def test_config_command_redacts_the_token(tmp_path: Path) -> None:
    env = isolated_env(tmp_path, TELEGRAM_BOT_TOKEN="123456:super-secret")
    result = run_cli(["config"], env=env, cwd=tmp_path)
    assert result.returncode == 0
    assert "super-secret" not in result.stdout
    assert "telegram_bot_token" in result.stdout


def test_version_flag() -> None:
    result = run_cli(["--version"])
    assert result.returncode == 0
    assert "acp-im-gateway" in result.stdout


def test_python_m_entry_point_is_importable() -> None:
    assert sys.version_info >= (3, 11)
