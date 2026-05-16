"""Tests for the AgentBanker — funding + draining per lifecycle.

These tests exercise the banker in isolation. The integration with
AgentFactory.invite_and_start is covered elsewhere; here we want to
verify the funding/draining math and the per-lifecycle routing
without any factory machinery in the way.
"""

from __future__ import annotations

import math

import pytest

from ahp.core.address import AgentAddress
from ahp.economy.agent_banker import (
    AgentBanker,
    BROKER_WALLET,
    COMMONS_WALLET,
    STARTING_FUND_BY_LIFECYCLE,
    funding_source_for,
    refund_destination_for,
)
from ahp.economy.wallet import INITIAL_FUND, InsufficientFundsError, Wallet


# ── pure routing rules ───────────────────────────────────────────────


def test_session_funded_by_host_server():
    assert funding_source_for("session", "myhost") == "myhost"


def test_session_falls_back_to_commons_when_no_host():
    assert funding_source_for("session", None) == COMMONS_WALLET


def test_longterm_funded_by_commons():
    assert funding_source_for("longterm", "myhost") == COMMONS_WALLET


def test_ephemeral_funded_by_commons():
    assert funding_source_for("ephemeral", "myhost") == COMMONS_WALLET


def test_session_refunds_to_host():
    assert refund_destination_for("session", "myhost") == "myhost"


def test_longterm_persists_balance():
    assert refund_destination_for("longterm", "myhost") is None


def test_ephemeral_returns_to_commons():
    assert refund_destination_for("ephemeral", "myhost") == COMMONS_WALLET


# ── funding flows ────────────────────────────────────────────────────


def _addr(lifecycle: str = "session", inst: str = "a0") -> AgentAddress:
    return AgentAddress.parse(f"acme.researcher.x.y.s.{lifecycle}.{inst}")


async def test_session_agent_funded_from_host_server(redis_client):
    banker = AgentBanker(redis_client)
    host = Wallet(redis_client, owner="myhost")
    await host.topup(50.0, reason="seed")  # ensure host has funds

    addr = _addr("session")
    amount = await banker.fund_agent(addr, host_server_id="myhost")
    assert math.isclose(amount, STARTING_FUND_BY_LIFECYCLE["session"])

    # Host's wallet went down by the funded amount.
    host_state = await host.get_state()
    expected_host = INITIAL_FUND + 50.0 - STARTING_FUND_BY_LIFECYCLE["session"]
    assert math.isclose(host_state.balance, expected_host, abs_tol=1e-9)

    # Agent's wallet went up by the same amount.
    agent_w = Wallet(redis_client, owner=str(addr))
    agent_state = await agent_w.get_state()
    expected_agent = INITIAL_FUND + STARTING_FUND_BY_LIFECYCLE["session"]
    assert math.isclose(agent_state.balance, expected_agent, abs_tol=1e-9)


async def test_longterm_agent_funded_from_commons(redis_client):
    banker = AgentBanker(redis_client)
    commons = Wallet(redis_client, owner=COMMONS_WALLET)
    await commons.topup(100.0, reason="seed commons")

    addr = _addr("longterm")
    amount = await banker.fund_agent(addr, host_server_id="myhost")
    assert math.isclose(amount, STARTING_FUND_BY_LIFECYCLE["longterm"])

    commons_state = await commons.get_state()
    expected = INITIAL_FUND + 100.0 - STARTING_FUND_BY_LIFECYCLE["longterm"]
    assert math.isclose(commons_state.balance, expected, abs_tol=1e-9)


async def test_unfunded_host_fails_provisioning(redis_client):
    """If the host can't afford to fund the agent, raise loudly."""
    banker = AgentBanker(redis_client)
    # Don't top up the host; default INITIAL_FUND=100 is enough though,
    # so we explicitly drain it.
    host = Wallet(redis_client, owner="poorhost")
    # Use a session agent (host-funded) where amount > INITIAL_FUND.
    addr = _addr("session")
    with pytest.raises(InsufficientFundsError):
        await banker.fund_agent(addr, host_server_id="poorhost", amount=1000.0)


async def test_drain_session_returns_to_host(redis_client):
    banker = AgentBanker(redis_client)
    addr = _addr("session")
    await banker.fund_agent(addr, host_server_id="myhost")

    # Agent has been doing work — give it more credits.
    agent_w = Wallet(redis_client, owner=str(addr))
    await agent_w.topup(20.0, reason="earned credits")

    state_before = await agent_w.get_state()
    drained = await banker.drain_agent(addr, host_server_id="myhost")
    assert math.isclose(drained, state_before.balance, abs_tol=1e-9)

    # Agent's balance is now zero.
    after = await agent_w.get_state()
    assert math.isclose(after.balance, 0.0, abs_tol=1e-9)


async def test_drain_longterm_keeps_balance(redis_client):
    banker = AgentBanker(redis_client)
    commons = Wallet(redis_client, owner=COMMONS_WALLET)
    await commons.topup(100.0, reason="seed")

    addr = _addr("longterm")
    await banker.fund_agent(addr, host_server_id="myhost")
    drained = await banker.drain_agent(addr, host_server_id="myhost")
    assert drained == 0.0

    # Agent's wallet still has its starting balance + INITIAL_FUND.
    agent_w = Wallet(redis_client, owner=str(addr))
    state = await agent_w.get_state()
    assert state.balance > 0.0


async def test_drain_ephemeral_returns_to_commons(redis_client):
    banker = AgentBanker(redis_client)
    commons = Wallet(redis_client, owner=COMMONS_WALLET)
    await commons.topup(50.0, reason="seed")

    addr = _addr("ephemeral")
    await banker.fund_agent(addr, host_server_id="myhost")

    # Commons paid out, then drains should return.
    commons_before = (await commons.get_state()).balance
    drained = await banker.drain_agent(addr, host_server_id="myhost")
    commons_after = (await commons.get_state()).balance

    assert drained > 0
    assert math.isclose(commons_after, commons_before + drained, abs_tol=1e-9)


async def test_drain_zero_balance_is_noop(redis_client):
    """A drained-then-redrain agent doesn't double-pay."""
    banker = AgentBanker(redis_client)
    addr = _addr("session")
    await banker.fund_agent(addr, host_server_id="myhost")

    # Drain the agent's wallet down to zero by repeated drain calls.
    # First drain returns the actual balance; subsequent drains should
    # return 0 (balance already zero from prior drain + Wallet's
    # default INITIAL_FUND was kept after creation).
    first = await banker.drain_agent(addr, host_server_id="myhost")
    assert first > 0
    second = await banker.drain_agent(addr, host_server_id="myhost")
    assert second == 0.0
