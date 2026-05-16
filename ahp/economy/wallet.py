"""Atomic credit wallets, backed by Redis WATCH/MULTI/EXEC.

Two layers:

* :class:`WalletState` is the pure state of one wallet: a balance, a
  set of outstanding holds. Helpers compute the new state given an
  operation. No I/O. Trivially testable.
* :class:`Wallet` wraps WalletState in Redis persistence. ``hold``,
  ``settle``, ``refund`` are atomic via optimistic concurrency
  (WATCH on the wallet key + the hold key). A settlement that writes
  to four wallets (server, compute, broker, commons) is done in a
  single ``MULTI/EXEC`` block.

Invariants enforced:

* Balance never goes negative. Operations that would do so raise
  :class:`InsufficientFundsError`.
* A hold can only be settled or refunded once. Re-settling the same
  hold_id raises :class:`UnknownHoldError`.
* Holds carry a Redis TTL so they self-clean if the broker dies
  mid-call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Final


INITIAL_FUND: Final[float] = 100.0
"""Credits a wallet is seeded with on first creation."""


HOLD_TTL_SECONDS: Final[int] = 300
"""How long a hold remains valid before Redis garbage-collects it.

Set well above any reasonable per-message timeout so we don't lose
in-flight money on slow models, but well below 'forever' so a crashed
broker doesn't permanently lock funds.
"""


class InsufficientFundsError(Exception):
    """Raised when an operation would push a wallet negative."""


class UnknownHoldError(Exception):
    """Raised when settling or refunding a hold that doesn't exist."""


class HoldExpiredError(Exception):
    """Raised when a hold's TTL has elapsed (Redis evicted the key)."""


# ── pure state ────────────────────────────────────────────────────────


@dataclass
class WalletState:
    """In-memory snapshot of one wallet.

    The Redis-backed :class:`Wallet` reconstructs this from the
    persisted JSON, computes the next state, and writes it back
    in a single transaction.
    """

    owner: str
    balance: float = INITIAL_FUND
    held_total: float = 0.0           # sum of outstanding holds
    holds: dict[str, float] = field(default_factory=dict)
    # Audit trail (most recent first). Bounded; the broker can also
    # mirror these to an external audit sink.
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available(self) -> float:
        return self.balance - self.held_total

    def to_json(self) -> str:
        return json.dumps({
            "owner": self.owner,
            "balance": self.balance,
            "held_total": self.held_total,
            "holds": dict(self.holds),
            "history": list(self.history)[-50:],  # cap on persisted history
        })

    @classmethod
    def from_json(cls, raw: str | bytes, *, owner: str) -> "WalletState":
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        data = json.loads(raw) if raw else {}
        return cls(
            owner=owner,
            balance=float(data.get("balance", INITIAL_FUND)),
            held_total=float(data.get("held_total", 0.0)),
            holds={k: float(v) for k, v in (data.get("holds") or {}).items()},
            history=list(data.get("history") or []),
        )


def _record(state: WalletState, entry: dict[str, Any]) -> None:
    entry = dict(entry)
    entry["ts"] = time.time()
    state.history.append(entry)


def apply_hold(
    state: WalletState, *, hold_id: str, amount: float, reason: str = "",
) -> WalletState:
    """Reserve ``amount`` against the wallet, marked with ``hold_id``.

    Reduces ``available`` immediately; the underlying ``balance`` is
    unchanged until settlement.
    """
    if amount < 0:
        raise ValueError(f"hold amount must be non-negative, got {amount}")
    if hold_id in state.holds:
        # Idempotent: same hold_id, same amount → no-op. Different
        # amount with same id → error so we don't silently overwrite.
        if abs(state.holds[hold_id] - amount) < 1e-9:
            return state
        raise ValueError(
            f"hold_id {hold_id!r} already exists with different amount "
            f"({state.holds[hold_id]} vs {amount})"
        )
    if state.available < amount:
        raise InsufficientFundsError(
            f"wallet {state.owner!r}: available {state.available:.4f} "
            f"< hold amount {amount:.4f}"
        )
    state.holds[hold_id] = float(amount)
    state.held_total += float(amount)
    _record(state, {
        "op": "hold", "id": hold_id, "amount": amount, "reason": reason,
    })
    return state


def apply_release(
    state: WalletState, *, hold_id: str, debit: float, reason: str = "",
) -> WalletState:
    """Release a hold, debiting ``debit`` from balance.

    ``debit`` is the actual cost; the held amount minus debit is
    returned to ``available``. If ``debit > held`` we raise — the
    broker should call :func:`apply_topup` separately if the actual
    cost exceeds the hold.
    """
    held = state.holds.get(hold_id)
    if held is None:
        raise UnknownHoldError(f"no hold {hold_id!r} on wallet {state.owner!r}")
    if debit < 0:
        raise ValueError(f"debit must be non-negative, got {debit}")
    if debit > held + 1e-9:
        raise InsufficientFundsError(
            f"debit {debit:.4f} exceeds hold {held:.4f} on {hold_id!r}; "
            f"increase the hold first"
        )
    state.holds.pop(hold_id)
    state.held_total -= held
    state.balance -= debit
    _record(state, {
        "op": "release", "id": hold_id, "held": held,
        "debit": debit, "reason": reason,
    })
    return state


def apply_credit(state: WalletState, *, amount: float, reason: str = "") -> WalletState:
    """Add ``amount`` to the balance."""
    if amount < 0:
        raise ValueError(f"credit amount must be non-negative, got {amount}")
    state.balance += float(amount)
    _record(state, {"op": "credit", "amount": amount, "reason": reason})
    return state


def apply_refund(state: WalletState, *, hold_id: str, reason: str = "") -> WalletState:
    """Cancel a hold entirely — released back to ``available``."""
    held = state.holds.get(hold_id)
    if held is None:
        raise UnknownHoldError(f"no hold {hold_id!r} on wallet {state.owner!r}")
    state.holds.pop(hold_id)
    state.held_total -= held
    _record(state, {"op": "refund", "id": hold_id, "amount": held, "reason": reason})
    return state


# ── Redis-backed Wallet (thin wrapper) ────────────────────────────────


_WALLET_KEY = "ahp:wallet:{owner}"


class Wallet:
    """Redis-backed wallet handle.

    Each ``hold/settle/refund`` performs one optimistic-concurrency
    transaction. The settlement helper :meth:`settle_four_way` writes
    to multiple wallets in a single transaction so the four-recipient
    split is atomic.
    """

    def __init__(self, redis_client: Any, owner: str) -> None:
        self._redis = redis_client
        self.owner = owner

    @staticmethod
    def key(owner: str) -> str:
        return _WALLET_KEY.format(owner=owner)

    async def get_state(self) -> WalletState:
        raw = await self._redis.get(self.key(self.owner))
        if raw is None:
            return WalletState(owner=self.owner)
        return WalletState.from_json(raw, owner=self.owner)

    async def _save(self, state: WalletState) -> None:
        await self._redis.set(self.key(self.owner), state.to_json())

    async def topup(self, amount: float, *, reason: str = "topup") -> WalletState:
        """One-shot credit. Used for initial funding or admin top-ups."""
        # Optimistic CAS via WATCH/MULTI/EXEC. We retry on conflict —
        # the typical case is one writer per wallet so contention is
        # rare; but the broker may write tax credits concurrently.
        for _ in range(8):
            async with self._redis.pipeline(transaction=True) as pipe:
                await pipe.watch(self.key(self.owner))
                state = await self.get_state()
                state = apply_credit(state, amount=amount, reason=reason)
                pipe.multi()
                pipe.set(self.key(self.owner), state.to_json())
                results = await pipe.execute()
                if results is not None:
                    return state
        raise RuntimeError(f"wallet {self.owner}: topup retries exhausted")

    async def hold(
        self, *, hold_id: str, amount: float, reason: str = "",
    ) -> WalletState:
        for _ in range(8):
            async with self._redis.pipeline(transaction=True) as pipe:
                await pipe.watch(self.key(self.owner))
                state = await self.get_state()
                state = apply_hold(
                    state, hold_id=hold_id, amount=amount, reason=reason,
                )
                pipe.multi()
                pipe.set(self.key(self.owner), state.to_json())
                # Mirror to a small expirable key so a crashed broker
                # doesn't leak funds — broker can scan and refund holds
                # whose mirror key has expired.
                pipe.setex(
                    f"ahp:hold:{self.owner}:{hold_id}",
                    HOLD_TTL_SECONDS,
                    str(amount),
                )
                results = await pipe.execute()
                if results is not None:
                    return state
        raise RuntimeError(f"wallet {self.owner}: hold retries exhausted")

    async def refund(self, hold_id: str, *, reason: str = "") -> WalletState:
        for _ in range(8):
            async with self._redis.pipeline(transaction=True) as pipe:
                await pipe.watch(self.key(self.owner))
                state = await self.get_state()
                state = apply_refund(state, hold_id=hold_id, reason=reason)
                pipe.multi()
                pipe.set(self.key(self.owner), state.to_json())
                pipe.delete(f"ahp:hold:{self.owner}:{hold_id}")
                results = await pipe.execute()
                if results is not None:
                    return state
        raise RuntimeError(f"wallet {self.owner}: refund retries exhausted")

    async def settle_against_hold(
        self,
        *,
        hold_id: str,
        debit: float,
        reason: str = "",
    ) -> WalletState:
        """Release ``hold_id``, debiting ``debit`` from balance."""
        for _ in range(8):
            async with self._redis.pipeline(transaction=True) as pipe:
                await pipe.watch(self.key(self.owner))
                state = await self.get_state()
                state = apply_release(
                    state, hold_id=hold_id, debit=debit, reason=reason,
                )
                pipe.multi()
                pipe.set(self.key(self.owner), state.to_json())
                pipe.delete(f"ahp:hold:{self.owner}:{hold_id}")
                results = await pipe.execute()
                if results is not None:
                    return state
        raise RuntimeError(f"wallet {self.owner}: settle retries exhausted")
