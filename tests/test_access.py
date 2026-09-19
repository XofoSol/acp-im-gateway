"""Access control: allowlist, group gate and pairing code lifetime."""

from __future__ import annotations

import pytest

from acp_im_gateway.access import AccessPolicy, PairingError

from .helpers import FakeClock


def make_policy(**kwargs: object) -> tuple[AccessPolicy, FakeClock]:
    clock = FakeClock()
    policy = AccessPolicy(
        allowed_user_ids=set(kwargs.pop("allowed_user_ids", {900})),
        allowed_chat_ids=set(kwargs.pop("allowed_chat_ids", set())),
        pairing_ttl=float(kwargs.pop("pairing_ttl", 900.0)),
        clock=clock,
        code_factory=kwargs.pop("code_factory", None),
    )
    return policy, clock


def test_deny_by_default() -> None:
    policy, _ = make_policy(allowed_user_ids={900})
    assert policy.is_user_allowed(900) is True
    assert policy.is_user_allowed(901) is False
    assert policy.is_user_allowed(None) is False
    assert policy.authorize(901, 111, "private") is False


def test_direct_messages_need_no_chat_allowlist() -> None:
    policy, _ = make_policy()
    assert policy.is_chat_allowed(111, "private") is True
    assert policy.authorize(900, 111, "private") is True


def test_groups_need_an_allowlisted_chat_and_user() -> None:
    policy, _ = make_policy(allowed_chat_ids={-500})
    assert policy.authorize(900, -500, "supergroup") is True
    assert policy.authorize(900, -501, "supergroup") is False
    assert policy.authorize(901, -500, "group") is False


def test_pairing_code_flow() -> None:
    policy, _ = make_policy(allowed_user_ids={900}, code_factory=lambda: "CODE1234")
    assert policy.request_pairing(900, 111) is None  # already allowed
    request = policy.request_pairing(901, 111, "newcomer")
    assert request is not None
    assert request.code == "CODE1234"
    assert policy.is_user_allowed(901) is False  # still denied until approved
    approved = policy.approve("code1234")  # case-insensitive
    assert approved.user_id == 901
    assert policy.is_user_allowed(901) is True
    assert approved.approved_at is not None


def test_pairing_code_expires() -> None:
    policy, clock = make_policy(code_factory=lambda: "EXPIRES1", pairing_ttl=60.0)
    request = policy.request_pairing(901, 111)
    assert request is not None
    clock.advance(59.0)
    policy.approve("EXPIRES1")
    assert policy.is_user_allowed(901) is True

    policy2, clock2 = make_policy(code_factory=lambda: "EXPIRES2", pairing_ttl=60.0)
    policy2.request_pairing(902, 112)
    clock2.advance(61.0)
    with pytest.raises(PairingError, match="expired"):
        policy2.approve("EXPIRES2")
    assert policy2.is_user_allowed(902) is False
    assert "EXPIRES2" not in policy2.pending  # expired codes are pruned


def test_unknown_code_is_rejected() -> None:
    policy, _ = make_policy()
    with pytest.raises(PairingError, match="unknown pairing code"):
        policy.approve("NOPE")
    with pytest.raises(PairingError):
        policy.reject("NOPE")


def test_reject_drops_the_code_without_allowing_the_user() -> None:
    policy, _ = make_policy(code_factory=lambda: "REJECT01")
    policy.request_pairing(901, 111)
    rejected = policy.reject("REJECT01")
    assert rejected.user_id == 901
    assert policy.is_user_allowed(901) is False
    with pytest.raises(PairingError):
        policy.approve("REJECT01")


def test_requesting_again_invalidates_the_previous_code() -> None:
    codes = iter(["FIRST234", "SECOND23"])
    policy, _ = make_policy(code_factory=lambda: next(codes))
    first = policy.request_pairing(901, 111)
    second = policy.request_pairing(901, 111)
    assert first is not None and second is not None
    assert "FIRST234" not in policy.pending
    assert "SECOND23" in policy.pending


def test_prune_and_pending_list() -> None:
    policy, clock = make_policy(code_factory=lambda: "PENDING1", pairing_ttl=10.0)
    policy.request_pairing(901, 111)
    assert len(policy.pending_requests()) == 1
    clock.advance(11.0)
    assert policy.pending_requests() == []
    assert policy.prune() == 1


def test_persistence_round_trip_and_merge() -> None:
    policy, clock = make_policy(code_factory=lambda: "MERGE123")
    policy.request_pairing(901, 111, "newcomer")
    payload = policy.to_dict()
    assert payload["allowed_user_ids"] == [900]
    assert "MERGE123" in payload["pending"]

    fresh, _ = make_policy(allowed_user_ids={900})
    assert fresh.merge(payload) is True
    assert "MERGE123" in fresh.pending
    assert fresh.merge(payload) is False  # idempotent

    # A CLI approval written to disk must reach the running process.
    policy.approve("MERGE123")
    approved_payload = policy.to_dict()
    other, _ = make_policy()
    assert other.merge(approved_payload) is True
    assert other.is_user_allowed(901) is True
    assert other.pending["MERGE123"].approved_at is not None
    assert other.merge(approved_payload) is False


def test_merge_ignores_expired_and_malformed_entries() -> None:
    policy, clock = make_policy()
    stale = clock.now - 10_000
    changed = policy.merge(
        {
            "allowed_user_ids": ["not-an-int-would-fail-later", 42],
            "pending": {
                "OLDCODE1": {
                    "code": "OLDCODE1",
                    "user_id": 5,
                    "chat_id": 1,
                    "created_at": stale,
                    "expires_at": stale,
                },
                "BROKEN": {"nope": True},
            },
        }
    )
    # The garbage entry is ignored, the numeric one is adopted.
    assert changed is True
    assert policy.is_user_allowed(42) is True
    assert policy.pending == {}


def test_from_config_defaults_to_deny() -> None:
    policy = AccessPolicy.from_config()
    assert policy.is_user_allowed(1) is False
