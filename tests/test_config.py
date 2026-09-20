"""Configuration precedence and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from acp_im_gateway.config import (
    Config,
    ConfigError,
    load_dotenv,
    parse_bool,
    parse_id_list,
    parse_path_list,
    split_command,
)


def _no_toml(tmp_path: Path) -> Path:
    return tmp_path / "does-not-exist.toml"


def test_neutral_defaults_are_not_personal() -> None:
    config = Config()
    assert config.projects_root == Path("~/Projects").expanduser()
    assert config.agent_index_dir == Path("~/.reasonix/projects").expanduser()
    assert config.agent_cmd == ("reasonix", "acp")
    assert config.telegram_bot_token == ""
    assert config.allowed_user_ids == frozenset()
    assert config.telegram_api_base == "https://api.telegram.org"


def test_defaults_from_empty_environment(tmp_path: Path) -> None:
    config = Config.from_env(env={}, toml_path=_no_toml(tmp_path))
    assert config.agent_cmd == ("reasonix", "acp")
    assert config.edit_interval == 1.2
    assert config.busy_mode == "steer"
    assert config.dry_run is False


def test_environment_overrides_defaults(tmp_path: Path) -> None:
    env = {
        "TELEGRAM_BOT_TOKEN": "123:abc",
        "ALLOWED_USER_IDS": "7, 8",
        "ALLOWED_CHAT_IDS": "-1001",
        "PROJECTS_ROOT": str(tmp_path / "work"),
        "ALLOWED_ROOTS": f"{tmp_path / 'one'}, {tmp_path / 'two'}",
        "REASONIX_ACP_CMD": "python3 -m my_agent --acp",
        "GATEWAY_EDIT_INTERVAL": "2.5",
        "GATEWAY_SEND_INTERVAL": "0",
        "GATEWAY_BUSY_MODE": "QUEUE",
        "GATEWAY_POLL_TIMEOUT": "5",
        "GATEWAY_DRY_RUN": "1",
    }
    config = Config.from_env(env=env, toml_path=_no_toml(tmp_path))
    assert config.telegram_bot_token == "123:abc"
    assert config.allowed_user_ids == frozenset({7, 8})
    assert config.allowed_chat_ids == frozenset({-1001})
    assert config.projects_root == tmp_path / "work"
    assert config.allowed_roots == (tmp_path / "one", tmp_path / "two")
    assert config.agent_cmd == ("python3", "-m", "my_agent", "--acp")
    assert config.edit_interval == 2.5
    assert config.send_interval == 0.0
    assert config.busy_mode == "queue"
    assert config.poll_timeout == 5
    assert config.dry_run is True


def test_toml_is_read_and_environment_wins(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text(
        """
[telegram]
bot_token = "from-toml"
allowed_user_ids = [1, 2]

[paths]
projects_root = "~/from-toml"

[agent]
command = "agent --stdio"

[runtime]
edit_interval = 3.0
busy_mode = "queue"
""",
        encoding="utf-8",
    )
    config = Config.from_env(env={}, toml_path=toml)
    assert config.telegram_bot_token == "from-toml"
    assert config.allowed_user_ids == frozenset({1, 2})
    assert config.agent_cmd == ("agent", "--stdio")
    assert config.edit_interval == 3.0
    assert config.busy_mode == "queue"

    overridden = Config.from_env(env={"GATEWAY_EDIT_INTERVAL": "0.5"}, toml_path=toml)
    assert overridden.edit_interval == 0.5
    assert overridden.busy_mode == "queue"  # untouched TOML value survives


def test_dotenv_file_is_loaded_and_real_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nexport TELEGRAM_BOT_TOKEN=from-file\nGATEWAY_BUSY_MODE=queue\nEMPTY=\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("GATEWAY_BUSY_MODE", "steer")
    config = Config.from_env(env_file=env_file, toml_path=_no_toml(tmp_path))
    assert config.telegram_bot_token == "from-file"
    assert config.busy_mode == "steer"


def test_load_dotenv_handles_quotes_and_comments(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "A=1\n# B=2\nC = 'three'\nD=\"four\"\nnot-a-pair\n\n", encoding="utf-8"
    )
    values = load_dotenv(path)
    assert values == {"A": "1", "C": "three", "D": "four"}
    assert load_dotenv(tmp_path / "missing") == {}


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    toml = tmp_path / "broken.toml"
    toml.write_text("this is = not [ valid", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid TOML"):
        Config.from_env(env={}, toml_path=toml)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, frozenset()),
        ("", frozenset()),
        ("1,2,3", frozenset({1, 2, 3})),
        (" 4 , 5 ", frozenset({4, 5})),
        ("-1001;42", frozenset({-1001, 42})),
        ([1, "2"], frozenset({1, 2})),
    ],
)
def test_parse_id_list(raw: object, expected: frozenset[int]) -> None:
    assert parse_id_list(raw) == expected


def test_parse_id_list_rejects_junk() -> None:
    with pytest.raises(ConfigError, match="not a numeric id"):
        parse_id_list("1,abc")


def test_parse_path_list(tmp_path: Path) -> None:
    assert parse_path_list(f"{tmp_path / 'a'},{tmp_path / 'b'}") == (tmp_path / "a", tmp_path / "b")
    assert parse_path_list(None) == ()


def test_split_command() -> None:
    assert split_command("reasonix acp") == ("reasonix", "acp")
    assert split_command('agent --flag "two words"') == ("agent", "--flag", "two words")
    with pytest.raises(ConfigError):
        split_command("   ")


@pytest.mark.parametrize(
    "raw,expected",
    [("1", True), ("true", True), ("YES", True), ("0", False), ("off", False), (None, False), (True, True)],
)
def test_parse_bool(raw: object, expected: bool) -> None:
    assert parse_bool(raw, False) is expected


def test_validate_requires_a_token_and_sane_numbers(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="TELEGRAM_BOT_TOKEN"):
        Config().validate()
    Config(telegram_bot_token="x").validate()
    with pytest.raises(ConfigError, match="GATEWAY_BUSY_MODE"):
        Config(telegram_bot_token="x", busy_mode="nope").validate(require_token=False)
    with pytest.raises(ConfigError, match="GATEWAY_EDIT_INTERVAL"):
        Config(telegram_bot_token="x", edit_interval=-1).validate(require_token=False)
    with pytest.raises(ConfigError, match="GATEWAY_POLL_TIMEOUT"):
        Config(telegram_bot_token="x", poll_timeout=0).validate(require_token=False)


def test_safe_table_redacts_the_token(projects_root: Path) -> None:
    config = Config(telegram_bot_token="123456:secret", projects_root=projects_root)
    table = dict(config.safe_table())
    assert table["telegram_bot_token"] == "set"
    assert "secret" not in " ".join(table.values())
    assert table["allowed_roots"] == str(projects_root)
    assert "DMs only" in table["allowed_chat_ids"]


# --------------------------------------------------------------------------- v1.1 knobs


def test_render_and_tier_defaults(tmp_path: Path) -> None:
    from acp_im_gateway.tiers import DEFAULT_ALWAYS_ASK, DEFAULT_AUTO_ALLOW

    config = Config.from_env(env={}, toml_path=_no_toml(tmp_path))
    assert config.tool_output_lines == 15
    assert config.overflow_chars == 3500
    assert config.show_thinking is True
    assert config.heartbeat_seconds == 60.0
    assert config.auto_allow == DEFAULT_AUTO_ALLOW
    assert config.always_ask == DEFAULT_ALWAYS_ASK
    assert config.tier_overrides == {}
    assert "git status" in config.auto_allow and "rm -rf" in config.always_ask


def test_render_knobs_come_from_the_environment(tmp_path: Path) -> None:
    config = Config.from_env(
        env={
            "GATEWAY_TOOL_OUTPUT_LINES": "3",
            "GATEWAY_OVERFLOW_CHARS": "2000",
            "GATEWAY_SHOW_THINKING": "off",
            "GATEWAY_HEARTBEAT_SECONDS": "15",
            "GATEWAY_AUTO_ALLOW": "git status, pytest",
            "GATEWAY_ALWAYS_ASK": "",
        },
        toml_path=_no_toml(tmp_path),
    )
    assert config.tool_output_lines == 3
    assert config.overflow_chars == 2000
    assert config.show_thinking is False
    assert config.heartbeat_seconds == 15.0
    assert config.auto_allow == ("git status", "pytest")
    assert config.always_ask == ()


def test_tier_lists_and_per_chat_overrides_from_toml(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text(
        """
[approvals]
auto_allow = ["git status", "ls -la"]
always_ask = ["rm -rf", "stripe"]

[approvals.chats."42"]
auto_allow = ["make build"]
always_ask = []
""",
        encoding="utf-8",
    )
    config = Config.from_env(env={}, toml_path=toml)
    assert config.auto_allow == ("git status", "ls -la")
    assert config.always_ask == ("rm -rf", "stripe")
    assert config.tier_overrides[42].auto_allow == ("make build",)
    assert config.tier_overrides[42].always_ask == ()


def test_per_chat_override_needs_a_numeric_chat_id(tmp_path: Path) -> None:
    toml = tmp_path / "config.toml"
    toml.write_text('[approvals.chats."not-a-chat"]\nauto_allow = ["ls"]\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="numeric chat id"):
        Config.from_env(env={}, toml_path=toml)


def test_render_validation_rejects_impossible_numbers() -> None:
    with pytest.raises(ConfigError, match="GATEWAY_TOOL_OUTPUT_LINES"):
        Config(telegram_bot_token="x", tool_output_lines=0).validate()
    with pytest.raises(ConfigError, match="GATEWAY_OVERFLOW_CHARS"):
        Config(telegram_bot_token="x", overflow_chars=100).validate()
    with pytest.raises(ConfigError, match="GATEWAY_HEARTBEAT_SECONDS"):
        Config(telegram_bot_token="x", heartbeat_seconds=-1).validate()


def test_safe_table_lists_the_new_knobs() -> None:
    rows = dict(Config(telegram_bot_token="secret").safe_table())
    assert rows["tool_output_lines"] == "15"
    assert rows["overflow_chars"] == "3500"
    assert rows["show_thinking"] == "True"
    assert "git status" in rows["auto_allow"]
    assert "rm -rf" in rows["always_ask"]
