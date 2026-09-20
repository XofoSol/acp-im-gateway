"""Approval tiers: `auto_allow` (silent) and `always_ask` (a code gate).

The unit tests pin the matching rules; the gateway tests prove the whole path —
a bash payload, a generic tool-call payload, and an unknown command falling back
to `ask` — against the fake agent and the in-memory Telegram double.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from acp_im_gateway.approvals import choose_allow_option
from acp_im_gateway.tiers import (
    DEFAULT_ALWAYS_ASK,
    DEFAULT_AUTO_ALLOW,
    ASK,
    AUTO_ALLOW,
    POSTURE_ASK,
    POSTURE_AUTO,
    TIER_ALWAYS_ASK,
    TIER_AUTO_ALLOW,
    TIER_POSTURE,
    ApprovalTiers,
    TierRules,
    normalize_posture,
    parse_tier_list,
    shell_heads,
    subject_from_tool_call,
)

from .helpers import make_harness  # noqa: F401  (pytest fixture)

CHAT = 111


def bash(command: str, *, title: str = "Bash", kind: str = "execute"):
    return subject_from_tool_call({"kind": kind, "title": title, "rawInput": {"command": command}})


def generic(title: str, raw: Any, *, kind: str = "other"):
    return subject_from_tool_call({"kind": kind, "title": title, "rawInput": raw})


def action(tiers: ApprovalTiers, subject: Any, chat_id: int = CHAT) -> str:
    return tiers.evaluate(chat_id, subject).action


# --------------------------------------------------------------------------- tiers


def test_auto_allow_covers_the_default_read_only_commands() -> None:
    tiers = ApprovalTiers()
    for command in (
        "git status",
        "git status --short",
        "git diff HEAD~1",
        "git log --oneline",
        "git show HEAD",
        "git branch -a",
        "git remote -v",
        "ls -la",
        "cat README.md",
        "head -n 20 file",
        "tail -f log",
        "grep -rn todo src",
        "rg --files",
        "find . -name '*.py'",
        "pwd",
        "wc -l src/app.py",
        "pytest -q",
        "python -m pytest tests/",
    ):
        assert action(tiers, bash(command)) == AUTO_ALLOW, command


def test_auto_allow_is_case_insensitive() -> None:
    assert action(ApprovalTiers(), bash("GIT STATUS")) == AUTO_ALLOW
    assert action(ApprovalTiers(), bash("Pytest -q")) == AUTO_ALLOW


def test_every_default_always_ask_pattern_forces_a_tap() -> None:
    tiers = ApprovalTiers()
    samples = {
        "fal.ai": "python -c 'import fal.ai'",
        "fal_client": "python run.py --fal_client",
        "FAL_KEY": "printenv FAL_KEY",
        "elevenlabs": "python tts.py --provider elevenlabs",
        "openai": "python -c 'import openai'",
        "anthropic": "python -c 'import anthropic'",
        "stripe": "python charges.py stripe",
        "deploy": "./deploy.sh production",
        "rsync": "rsync -av build/ host:/srv",
        "scp": "scp dist.tar.gz host:/tmp",
        "wp": "wp post create --post_title=hello",
        "rm -rf": "rm -rf node_modules",
        "git push": "git push origin main",
        "git reset --hard": "git reset --hard HEAD~3",
        "docker push": "docker push registry/app:latest",
        "npm publish": "npm publish --access public",
        "aws": "aws s3 rm s3://bucket --recursive",
        "gcloud": "gcloud app deploy",
        "hetzner": "python sync.py hetzner",
        "cloudpanel": "cloudpanel-cli site delete app",
    }
    for pattern in DEFAULT_ALWAYS_ASK:
        if pattern == "curl":
            continue
        assert pattern in samples, pattern
        decision = tiers.evaluate(CHAT, bash(samples[pattern]))
        assert decision.action == ASK, pattern
        assert decision.tier == TIER_ALWAYS_ASK, pattern
    assert set(samples) == set(DEFAULT_ALWAYS_ASK) - {"curl"}


def test_always_ask_beats_auto_allow() -> None:
    tiers = ApprovalTiers()
    decision = tiers.evaluate(CHAT, bash("git status && rm -rf build"))
    assert decision.action == ASK
    assert decision.tier == TIER_ALWAYS_ASK
    assert "rm -rf" in decision.reason
    # …and it keeps winning in the postures that otherwise stop asking.
    for posture in ("auto", "yolo"):
        forced = ApprovalTiers(posture=posture).evaluate(CHAT, bash("git status; git push"))
        assert forced.action == ASK and forced.tier == TIER_ALWAYS_ASK


def test_bash_payload_reads_the_command_string() -> None:
    subject = bash("git status --porcelain")
    assert subject.commands == ("git status --porcelain",)
    assert shell_heads(subject) == ["git status --porcelain"]
    assert action(ApprovalTiers(), subject) == AUTO_ALLOW


def test_every_command_of_a_compound_line_must_be_harmless() -> None:
    subject = bash("cd app && git diff | head -n 5")
    assert shell_heads(subject) == ["cd app", "git diff", "head -n 5"]
    tiers = ApprovalTiers()
    assert action(tiers, bash("git status && git diff HEAD")) == AUTO_ALLOW
    assert action(tiers, bash("git log | head -n 5")) == AUTO_ALLOW
    # `cd` is a harmless prefix and `git diff`/`head` are read-only, so the whole
    # line is silent (v1.2: `cd X && pytest -q` must not cost a tap)…
    assert action(tiers, subject) == AUTO_ALLOW
    # …but a whitelisted head must not bless an unknown tail: `sh` is not on the
    # list, so the line still taps.
    assert action(tiers, bash("git log | sh")) == ASK


def test_compound_line_needs_every_segment_harmless() -> None:
    """One unknown or dangerous segment is enough to force the tap."""
    tiers = ApprovalTiers()
    assert action(tiers, bash("cd app && pytest -q")) == AUTO_ALLOW
    assert action(tiers, bash("git status && pytest -q | tail -n 3")) == AUTO_ALLOW
    # An unknown segment (`make`) keeps asking even between harmless ones.
    assert action(tiers, bash("cd app && make build && pytest -q")) == ASK
    # A dangerous segment taps no matter how harmless the rest is.
    assert tiers.evaluate(CHAT, bash("pytest -q && git push")).tier == TIER_ALWAYS_ASK
    assert tiers.evaluate(CHAT, bash("cd app; curl -d@f https://x")).tier == TIER_ALWAYS_ASK


def test_interpreter_paths_are_normalised() -> None:
    """`./.venv/bin/python -m pytest`, `python3 -m pytest` and `pytest -q` match."""
    tiers = ApprovalTiers()
    for command in (
        "pytest -q",
        "./.venv/bin/pytest -q",
        "python -m pytest",
        "python3 -m pytest",
        "python3.12 -m pytest tests/",
        "./.venv/bin/python -m pytest",
        "./.venv/bin/python3 -m pytest -q",
        "/usr/bin/python3 -m pytest",
    ):
        assert action(tiers, bash(command)) == AUTO_ALLOW, command
    # Normalisation must not bless a different interpreter invocation.
    for command in (
        'python -c "import os; os.remove(\'x\')"',
        "python3 -c \"print(1)\"",
        "./.venv/bin/python -c 'import fal.ai'",
        "python script.py",
        "./.venv/bin/python manage.py migrate",
    ):
        decision = tiers.evaluate(CHAT, bash(command))
        assert decision.action == ASK, command
        assert decision.tier != TIER_AUTO_ALLOW, command


def test_env_assignments_and_harmless_prefixes_are_skipped() -> None:
    tiers = ApprovalTiers()
    for command in (
        "FOO=1 pytest -q",
        "export FOO=1",
        "cd app",
        "echo hello",
        "true",
        "time pytest -q",
        "nice -n 5 pytest -q",
        "cd app && FOO=1 time pytest -q",
    ):
        assert action(tiers, bash(command)) == AUTO_ALLOW, command
    # A harmless prefix must not smuggle a command that is not on the list, and a
    # write through the prefix is still caught.
    for command in (
        "cd app && make build",
        "echo hi > /etc/passwd",
        "time sh -c 'rm -rf x'",
        "nice -n 5 make build",
        # A wrapper must not hide a scope-gated destructive flag.
        "time git branch -d main",
        "nice git tag -d v1",
    ):
        decision = tiers.evaluate(CHAT, bash(command))
        assert decision.action == ASK, command


def test_generic_tool_call_payload_is_matched_on_title_and_arguments() -> None:
    tiers = ApprovalTiers()
    # Title carries the vendor: money, so a tap.
    decision = tiers.evaluate(CHAT, generic("Call OpenAI", {"model": "gpt-4"}))
    assert decision.action == ASK and "openai" in decision.reason
    # Arguments carry it too, even without a command string.
    decision = tiers.evaluate(CHAT, generic("Charge card", {"provider": "stripe", "amount": 500}))
    assert decision.action == ASK and "stripe" in decision.reason
    # A generic call with no matching tier is not silently approved…
    assert action(tiers, generic("Search the docs", {"query": "how to bind a chat"})) == ASK
    # An argv list is not a command string, so it is not silently approved…
    assert action(tiers, generic("run", ["git", "status"])) == ASK
    # …but a shell-shaped tool call is, whatever key carries the command.
    assert action(tiers, generic("run", {"command": "git status"}, kind="execute")) == AUTO_ALLOW
    assert action(tiers, generic("run", {"script": "git diff HEAD"}, kind="bash")) == AUTO_ALLOW
    # A shell-shaped call also needs every command of the line on the list.
    assert action(tiers, generic("run", {"command": "git diff && npm publish"}, kind="execute")) == ASK


def test_always_ask_matches_file_locations_too() -> None:
    subject = subject_from_tool_call(
        {
            "kind": "edit",
            "title": "Write file",
            "locations": [{"path": "/srv/cloudpanel/config.json"}],
        }
    )
    decision = ApprovalTiers().evaluate(CHAT, subject)
    assert decision.action == ASK and decision.tier == TIER_ALWAYS_ASK


def test_an_unknown_command_falls_back_to_the_posture() -> None:
    tiers = ApprovalTiers()
    decision = tiers.evaluate(CHAT, bash("make build"))
    assert decision.action == ASK
    assert decision.tier == TIER_POSTURE
    assert "posture" in decision.reason

    for posture in ("auto", "yolo"):
        widened = ApprovalTiers(posture=posture).evaluate(CHAT, bash("make build"))
        assert widened.action == AUTO_ALLOW and widened.tier == TIER_POSTURE
    # …but anything gated still taps in those postures.
    assert action(ApprovalTiers(posture="yolo"), bash("git push")) == ASK


def test_auto_allow_never_matches_inside_a_word() -> None:
    """`ls` must not bless `false ls`, nor `cat` a `concatenate` tool."""
    tiers = ApprovalTiers()
    for command in ("false ls", "results", "analysis of the log", "git status-report"):
        assert action(tiers, bash(command)) == ASK, command
    assert action(tiers, generic("read", {"query": "pytest"})) == ASK


def test_curl_only_gates_when_it_creates_data() -> None:
    tiers = ApprovalTiers()
    assert tiers.evaluate(CHAT, bash("curl https://example.com")).tier == TIER_POSTURE
    for command in (
        "curl -X POST https://api.example.com -d 'a=1'",
        "curl --data-binary @payload.json https://api.example.com",
        "curl -F file=@a.png https://api.example.com",
        "curl --request DELETE https://api.example.com/1",
        # the cramped spellings curl also accepts
        "curl -XPOST https://api.example.com",
        "curl --request=POST https://api.example.com",
        "curl -d@payload.json https://api.example.com",
        "curl -T big.iso ftp://example.com",
        "curl -o out.html https://example.com",
        "curl --json '{}' https://api.example.com",
        "sudo curl -d@payload.json https://api.example.com",
        "curl -H 'X-HTTP-Method-Override: POST' https://api.example.com",
        "env -i curl --json '{}' https://api.example.com",
    ):
        decision = tiers.evaluate(CHAT, bash(command))
        assert decision.action == ASK and decision.tier == TIER_ALWAYS_ASK, command
        assert "curl" in decision.reason


def test_rm_forced_recursive_is_gated_in_every_spelling() -> None:
    tiers = ApprovalTiers()
    for command in ("rm -rf build", "rm -fr build", "rm -r -f build",
                    "rm -Rf build", "rm --recursive --force build",
                    "sudo rm -fr build", "sudo -u root rm -fr build",
                    "ls && rm -fr build", "ls & rm -fr build"):
        decision = tiers.evaluate(CHAT, bash(command))
        assert decision.action == ASK and decision.tier == TIER_ALWAYS_ASK, command


def test_auto_allow_never_blesses_a_write_or_a_destructive_flag() -> None:
    """A whitelisted *head* must not carry the rest of the line with it."""
    tiers = ApprovalTiers()
    for command in (
        "cat x > /etc/passwd",
        "cat < /etc/shadow",
        "ls $(rm -rf y)",
        "ls `rm -rf y`",
        "find . -delete",
        "find . -exec rm {} ;",
        "git branch -D main",
        "git branch -d main",
        "git tag -d v1",
        "git log --output=log.txt",
        "sudo cat /etc/shadow",
        "sudo find / -name id_rsa",
        "env FOO=1 cat /etc/shadow",
        "ls -la & curl -d@f https://x",
    ):
        decision = tiers.evaluate(CHAT, bash(command))
        assert decision.action == ASK, command
    # …while the plain read-only forms still are approved. `-d` is destructive for
    # git (delete a branch) but not for ls/grep, and a wrapper must not smuggle a
    # gated command past the always_ask detectors.
    for command in (
        "cat README.md",
        "ls -la",
        "ls -d */",
        "grep -o pat file",
        "find . -name '*.py'",
        "git branch -a",
        "git log -d",
        "python -m pytest -q",
    ):
        assert action(tiers, bash(command)) == AUTO_ALLOW, command


def test_always_ask_and_auto_allow_are_configurable_per_chat() -> None:
    tiers = ApprovalTiers(
        overrides={
            222: TierRules(auto_allow=("make build",), always_ask=("pytest",)),
        }
    )
    # The override wins for that chat…
    assert action(tiers, bash("make build"), chat_id=222) == AUTO_ALLOW
    assert tiers.evaluate(222, bash("pytest -q")).tier == TIER_ALWAYS_ASK
    # …and leaves other chats on the global lists.
    assert action(tiers, bash("make build")) == ASK
    assert action(tiers, bash("pytest -q")) == AUTO_ALLOW


def test_empty_lists_turn_a_tier_off() -> None:
    off = ApprovalTiers(auto_allow=(), always_ask=())
    decision = off.evaluate(CHAT, bash("git status"))
    assert decision.action == ASK and decision.tier == TIER_POSTURE

    ungated = ApprovalTiers(auto_allow=("git status",), always_ask=())
    decision = ungated.evaluate(CHAT, bash("rm -rf /"))
    assert decision.action == ASK
    assert decision.tier == TIER_POSTURE, "with no always_ask list the gate is off"
    assert action(ungated, bash("git status")) == AUTO_ALLOW


def test_tier_lists_parse_from_strings_and_sequences() -> None:
    assert parse_tier_list(None, default=DEFAULT_AUTO_ALLOW) == DEFAULT_AUTO_ALLOW
    assert parse_tier_list("", default=DEFAULT_AUTO_ALLOW) == ()
    assert parse_tier_list("git status, pytest", default=()) == ("git status", "pytest")
    assert parse_tier_list(["a", "b", "a"], default=()) == ("a", "b")


def test_aprobar_posture_is_per_chat_and_normalised() -> None:
    tiers = ApprovalTiers()  # default posture: ask
    assert tiers.posture_for(CHAT) == POSTURE_ASK
    assert normalize_posture("preguntar") == POSTURE_ASK
    assert normalize_posture("auto") == POSTURE_AUTO
    assert normalize_posture("YOLO") == POSTURE_AUTO
    assert normalize_posture("nonsense") == POSTURE_ASK  # the cautious direction

    tiers.set_posture(CHAT, "auto")
    assert tiers.posture_for(CHAT) == POSTURE_AUTO
    # The widened chat approves the unknown silently…
    assert action(tiers, bash("make build")) == AUTO_ALLOW
    # …but always_ask is still a code gate, and other chats are untouched.
    assert tiers.evaluate(CHAT, bash("git push")).tier == TIER_ALWAYS_ASK
    assert action(tiers, bash("make build"), chat_id=999) == ASK

    tiers.set_posture(CHAT, "preguntar")
    assert action(tiers, bash("make build")) == ASK


def test_postures_can_be_seeded_per_chat() -> None:
    tiers = ApprovalTiers(posture="ask", postures={CHAT: "auto"})
    assert tiers.posture_for(CHAT) == POSTURE_AUTO
    assert tiers.posture_for(999) == POSTURE_ASK
    tiers.clear_posture(CHAT)
    assert tiers.posture_for(CHAT) == POSTURE_ASK


def test_decisions_are_logged_at_info_with_the_reason(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    tiers = ApprovalTiers()
    tiers.decide(CHAT, bash("git status --short"))
    tiers.decide(CHAT, bash("git push origin main"))
    text = caplog.text
    assert "auto_allow: 'git status' is read-only" in text
    assert "always_ask gate: 'git push' is money or irreversible" in text
    assert "-> auto_allow" in text and "-> ask" in text


def test_choose_allow_option_prefers_a_one_shot_grant() -> None:
    from acp_im_gateway.approvals import PermissionRequest

    request = PermissionRequest.from_params(
        {
            "sessionId": "s1",
            "toolCall": {"toolCallId": "c1", "title": "t", "kind": "execute"},
            "options": [
                {"optionId": "allow_always", "name": "Always", "kind": "allow_always"},
                {"optionId": "allow_once", "name": "Once", "kind": "allow_once"},
                {"optionId": "reject_once", "name": "No", "kind": "reject_once"},
            ],
        },
        1,
    )
    chosen = choose_allow_option(request)
    assert chosen is not None and chosen.option_id == "allow_once"

    no_allow = PermissionRequest.from_params(
        {
            "sessionId": "s1",
            "toolCall": {"toolCallId": "c1", "title": "t"},
            "options": [{"optionId": "reject_once", "name": "No", "kind": "reject_once"}],
        },
        1,
    )
    assert choose_allow_option(no_allow) is None


# --------------------------------------------------------------------------- gateway


def test_auto_allow_is_approved_without_a_tap(make_harness: Any, caplog: pytest.LogCaptureFixture) -> None:
    """A read-only command never reaches the phone, and the agent is told so."""
    caplog.set_level(logging.INFO)
    harness = make_harness("--permission", "--permission-command", "git status --short")
    harness.send("/bind alpha")
    harness.send("is the tree clean?")
    assert harness.wait_idle()

    assert harness.buttons() == [], "an auto_allow command must not ask for a tap"
    responses = harness.permission_responses()
    assert responses, "the agent must still be answered"
    assert responses[0]["message"]["result"] == {
        "outcome": {"outcome": "selected", "optionId": "allow_once"}
    }
    decisions = harness.permission_decisions()
    assert decisions and decisions[0]["allowed"] is True
    assert "auto-approved" in caplog.text and "auto_allow" in caplog.text
    assert "✅ done" in harness.all_text()


def test_always_ask_forces_a_tap_even_in_a_widening_posture(make_harness: Any) -> None:
    harness = make_harness(
        "--permission",
        "--permission-command",
        "git push origin main",
        approval_posture="yolo",
    )
    harness.send("/bind alpha")
    harness.send("ship it")
    assert harness.wait_for_buttons(), "always_ask is a code gate: it must still tap"
    assert harness.permission_responses() == []

    actions = harness.gateway.tiers.evaluate(harness.chat_id, bash("git push")).action
    assert actions == ASK
    harness.callback(harness.buttons()[0]["callback_data"])
    assert harness.wait_idle()
    assert harness.permission_responses()[0]["message"]["result"]["outcome"]["optionId"] == "allow_once"


def test_a_generic_tool_call_payload_asks(make_harness: Any, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    harness = make_harness("--permission", "--permission-generic")
    harness.send("/bind alpha")
    harness.send("charge the customer")
    assert harness.wait_for_buttons()
    assert "stripe" in caplog.text
    assert harness.permission_responses() == []

    harness.callback(harness.buttons()[0]["callback_data"])
    assert harness.wait_idle()
    assert harness.permission_responses()


def test_an_unknown_command_still_asks(make_harness: Any) -> None:
    harness = make_harness("--permission", "--permission-command", "make build")
    harness.send("/bind alpha")
    harness.send("build it")
    assert harness.wait_for_buttons()
    assert harness.permission_responses() == []
