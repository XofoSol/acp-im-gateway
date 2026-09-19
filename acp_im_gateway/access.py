"""Access control: allowlist, group gate and pairing codes.

Deny by default. An unknown sender never reaches the agent; instead they get a
message saying they are not authorised, and the gateway logs a short-lived
one-time pairing code for the operator to approve from the CLI:

    python -m acp_im_gateway pairing approve <code>

Codes expire (:attr:`AccessPolicy.pairing_ttl`), are single-use, and are only
ever written to the gateway log — never sent to the requester.

Group chats need *both* an allowlisted user and a chat id listed in
``ALLOWED_CHAT_IDS`` (an empty list means direct messages only).
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

# Ambiguous glyphs (0/O, 1/I/L) are excluded so codes survive being read off a log.
CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
CODE_LENGTH = 8


class PairingError(Exception):
    """Unknown, expired or already-used pairing code."""


@dataclass
class PairingRequest:
    """A pending (or approved) request from an unknown sender."""

    code: str
    user_id: int
    chat_id: int
    username: str | None = None
    created_at: float = 0.0
    expires_at: float = 0.0
    approved_at: float | None = None
    notified: bool = False

    def is_expired(self, now: float) -> bool:
        return self.approved_at is None and self.expires_at <= now

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "user_id": self.user_id,
            "chat_id": self.chat_id,
            "username": self.username,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "approved_at": self.approved_at,
            "notified": self.notified,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PairingRequest":
        return cls(
            code=str(data["code"]),
            user_id=int(data["user_id"]),
            chat_id=int(data.get("chat_id", 0)),
            username=data.get("username"),
            created_at=float(data.get("created_at", 0.0)),
            expires_at=float(data.get("expires_at", 0.0)),
            approved_at=(None if data.get("approved_at") is None else float(data["approved_at"])),
            notified=bool(data.get("notified", False)),
        )


@dataclass
class AccessPolicy:
    """Allowlist plus pairing-code bookkeeping. Deny by default, always."""

    allowed_user_ids: set[int] = field(default_factory=set)
    allowed_chat_ids: set[int] = field(default_factory=set)
    pairing_ttl: float = 900.0
    pending: dict[str, PairingRequest] = field(default_factory=dict)
    #: Codes rejected in this process. A stale copy of the state file must not be
    #: able to resurrect a code the operator already said no to.
    rejected: set[str] = field(default_factory=set)
    clock: Callable[[], float] = time.time
    code_factory: Callable[[], str] | None = None

    def __post_init__(self) -> None:
        self.allowed_user_ids = {int(uid) for uid in self.allowed_user_ids}
        self.allowed_chat_ids = {int(cid) for cid in self.allowed_chat_ids}

    # ------------------------------------------------------------------ gating

    def is_user_allowed(self, user_id: int | None) -> bool:
        """Unknown sender is denied by default."""
        if user_id is None:
            return False
        return int(user_id) in self.allowed_user_ids

    def is_chat_allowed(self, chat_id: int | None, chat_type: str | None = None) -> bool:
        """Group traffic needs an explicit ``ALLOWED_CHAT_IDS`` entry; DMs are fine."""
        if chat_type in ("group", "supergroup", "channel"):
            if chat_id is None:
                return False
            return int(chat_id) in self.allowed_chat_ids
        return True

    def authorize(self, user_id: int | None, chat_id: int | None, chat_type: str | None) -> bool:
        """Full gate: allowlisted user *and*, for groups, allowlisted chat."""
        return self.is_user_allowed(user_id) and self.is_chat_allowed(chat_id, chat_type)

    # ------------------------------------------------------------------ pairing

    def new_code(self) -> str:
        if self.code_factory is not None:
            return self.code_factory()
        return "".join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))

    def request_pairing(
        self,
        user_id: int,
        chat_id: int,
        username: str | None = None,
        *,
        now: float | None = None,
    ) -> PairingRequest | None:
        """Create a fresh one-time code for an unknown sender.

        Returns ``None`` when the user is already allowed. Any previous code for
        the same user is invalidated.
        """
        if self.is_user_allowed(user_id):
            return None
        moment = self.clock() if now is None else now
        self.prune(now=moment)
        for code, request in list(self.pending.items()):
            if request.user_id == int(user_id):
                del self.pending[code]
        code = self.new_code()
        request = PairingRequest(
            code=code,
            user_id=int(user_id),
            chat_id=int(chat_id),
            username=username,
            created_at=moment,
            expires_at=moment + self.pairing_ttl,
        )
        self.pending[code] = request
        return request

    def get(self, code: str) -> PairingRequest:
        request = self.pending.get(code.strip().upper())
        if request is None:
            raise PairingError(f"unknown pairing code: {code!r}")
        return request

    def approve(self, code: str, *, now: float | None = None) -> PairingRequest:
        """Add the requester to the allowlist. Raises for unknown/expired codes."""
        moment = self.clock() if now is None else now
        request = self.get(code)
        if request.is_expired(moment):
            del self.pending[request.code]
            raise PairingError(f"pairing code {request.code} expired at {request.expires_at:.0f}")
        request.approved_at = moment
        self.allowed_user_ids.add(request.user_id)
        return request

    def reject(self, code: str, *, now: float | None = None) -> PairingRequest:
        request = self.get(code)
        self.prune(now=self.clock() if now is None else now)
        self.pending.pop(request.code, None)
        self.rejected.add(request.code)
        return request

    def mark_notified(self, code: str) -> None:
        request = self.pending.get(code)
        if request is not None:
            request.notified = True

    def prune(self, *, now: float | None = None) -> int:
        """Drop expired codes. Returns how many were removed."""
        moment = self.clock() if now is None else now
        stale = [code for code, req in self.pending.items() if req.is_expired(moment)]
        for code in stale:
            del self.pending[code]
        return len(stale)

    def pending_requests(self, *, now: float | None = None) -> list[PairingRequest]:
        moment = self.clock() if now is None else now
        return sorted(
            (req for req in self.pending.values() if not req.is_expired(moment)),
            key=lambda req: req.created_at,
        )

    # -------------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        self.prune()
        return {
            "allowed_user_ids": sorted(self.allowed_user_ids),
            "allowed_chat_ids": sorted(self.allowed_chat_ids),
            "pending": {code: req.to_dict() for code, req in self.pending.items()},
        }

    def merge(self, data: Mapping[str, Any] | None) -> bool:
        """Union in allowlist/pairing data written by another process (the CLI).

        Returns True when something changed. Approval state is preserved: an
        approved code is never resurrected as pending, and allowlist entries are
        only ever added.
        """
        if not data:
            return False
        changed = False
        for raw_user_id in data.get("allowed_user_ids") or ():
            try:
                uid = int(raw_user_id)
            except (TypeError, ValueError):
                continue
            if uid not in self.allowed_user_ids:
                self.allowed_user_ids.add(uid)
                changed = True
        for raw_chat_id in data.get("allowed_chat_ids") or ():
            try:
                cid = int(raw_chat_id)
            except (TypeError, ValueError):
                continue
            if cid not in self.allowed_chat_ids:
                self.allowed_chat_ids.add(cid)
                changed = True
        incoming_codes = set(data.get("pending") or {})
        for code, raw in (data.get("pending") or {}).items():
            if code in self.rejected:
                continue
            try:
                incoming = PairingRequest.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                continue
            local = self.pending.get(code)
            if local is None:
                if incoming.approved_at is None and not incoming.is_expired(self.clock()):
                    self.pending[code] = incoming
                    changed = True
                elif incoming.approved_at is not None and incoming.user_id in self.allowed_user_ids:
                    self.pending[code] = incoming
                    changed = True
                continue
            if local.approved_at is None and incoming.approved_at is not None:
                local.approved_at = incoming.approved_at
                changed = True
            if not local.notified and incoming.notified:
                local.notified = True
                changed = True
        # A rejection only needs to be remembered while the stale entry exists.
        self.rejected.intersection_update(incoming_codes)
        return changed

    @classmethod
    def from_config(
        cls,
        *,
        allowed_user_ids: Iterable[int] = (),
        allowed_chat_ids: Iterable[int] = (),
        pairing_ttl: float = 900.0,
        clock: Callable[[], float] = time.time,
        code_factory: Callable[[], str] | None = None,
    ) -> "AccessPolicy":
        return cls(
            allowed_user_ids={int(uid) for uid in allowed_user_ids},
            allowed_chat_ids={int(cid) for cid in allowed_chat_ids},
            pairing_ttl=float(pairing_ttl),
            clock=clock,
            code_factory=code_factory,
        )
