"""Tests for the wallet primitives.

Pure-state tests run synchronously; the Redis-backed Wallet is
exercised by the fakeredis fixture from conftest.
"""

from __future__ import annotations

import json
import math

import pytest

from ahp.economy.wallet import (
    INITIAL_FUND,
    InsufficientFundsError,
    UnknownHoldError,
    Wallet,
    WalletState,
    apply_credit,
    apply_hold,
    apply_refund,
    apply_release,
)


# ── pure state ────────────────────────────────────────────────────────


def test_hold_reduces_available_not_balance():
    s = WalletState(owner="alice")
    s = apply_hold(s, hold_id="h1", amount=10.0)
    assert s.balance == INITIAL_FUND
    assert s.available == INITIAL_FUND - 10.0
    assert s.held_total == 10.0


def test_hold_rejects_insufficient():
    s = WalletState(owner="alice", balance=5.0)
    with pytest.raises(InsufficientFundsError):
        apply_hold(s, hold_id="h1", amount=10.0)


def test_release_debits_balance_and_returns_remainder():
    s = WalletState(owner="alice")
    s = apply_hold(s, hold_id="h1", amount=10.0)
    s = apply_release(s, hold_id="h1", debit=4.0)
    assert s.balance == INITIAL_FUND - 4.0
    assert s.held_total == 0.0
    assert "h1" not in s.holds


def test_release_unknown_hold_raises():
    s = WalletState(owner="alice")
    with pytest.raises(UnknownHoldError):
        apply_release(s, hold_id="never", debit=1.0)


def test_refund_returns_held_amount_to_available():
    s = WalletState(owner="alice")
    s = apply_hold(s, hold_id="h1", amount=15.0)
    s = apply_refund(s, hold_id="h1")
    assert s.balance == INITIAL_FUND
    assert s.available == INITIAL_FUND
    assert s.held_total == 0.0


def test_credit_increases_balance():
    s = WalletState(owner="alice")
    s = apply_credit(s, amount=50.0)
    assert s.balance == INITIAL_FUND + 50.0


def test_to_json_roundtrip():
    s = WalletState(owner="alice")
    s = apply_hold(s, hold_id="h1", amount=10.0)
    raw = s.to_json()
    s2 = WalletState.from_json(raw, owner="alice")
    assert s2.balance == s.balance
    assert s2.held_total == s.held_total
    assert s2.holds == s.holds


# ── Redis-backed ──────────────────────────────────────────────────────


async def test_wallet_creates_with_initial_fund(redis_client):
    w = Wallet(redis_client, owner="alice")
    state = await w.get_state()
    assert state.balance == INITIAL_FUND
    assert state.available == INITIAL_FUND


async def test_wallet_topup_persists(redis_client):
    w = Wallet(redis_client, owner="alice")
    await w.topup(50.0, reason="initial seed")
    state = await w.get_state()
    assert state.balance == INITIAL_FUND + 50.0


async def test_wallet_hold_then_settle(redis_client):
    w = Wallet(redis_client, owner="alice")
    await w.hold(hold_id="h1", amount=10.0, reason="dispatch")
    mid = await w.get_state()
    assert mid.available == INITIAL_FUND - 10.0

    await w.settle_against_hold(hold_id="h1", debit=4.0, reason="actual cost")
    final = await w.get_state()
    assert final.balance == INITIAL_FUND - 4.0
    assert final.available == INITIAL_FUND - 4.0
    assert "h1" not in final.holds


async def test_wallet_hold_then_refund(redis_client):
    w = Wallet(redis_client, owner="alice")
    await w.hold(hold_id="h1", amount=20.0)
    await w.refund("h1", reason="dispatch failed")
    final = await w.get_state()
    assert final.balance == INITIAL_FUND
    assert final.available == INITIAL_FUND


async def test_wallet_settle_unknown_hold_raises(redis_client):
    w = Wallet(redis_client, owner="alice")
    with pytest.raises(UnknownHoldError):
        await w.settle_against_hold(hold_id="ghost", debit=1.0)


async def test_wallet_hold_mirror_key_written(redis_client):
    """The hold should leave a ahp:hold:<owner>:<id> mirror so a
    crashed broker can clean up via TTL.
    """
    w = Wallet(redis_client, owner="alice")
    await w.hold(hold_id="h1", amount=10.0)
    mirror = await redis_client.get("ahp:hold:alice:h1")
    assert mirror is not None
    assert float(mirror) == 10.0
