"""Engine integration: SEND-GET with the broker wired.

Verifies the economic hot path. Each test sets up a broker + a single
echo agent; calls SEND-GET; checks wallet movements + reputation
updates.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from ahp.adapters.base import AHPAgent
from ahp.broker import Broker, ServerMeta
from ahp.core import AgentAddress, Code, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import ReputationEntry, VISIBILITY_FULL_AT
from ahp.economy.wallet import INITIAL_FUND
from ahp.engine.router import ProtocolEngine
from ahp.engine.thread_manager import ThreadManager
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


class _EchoAgent(AHPAgent):
    """Replies with body=text from request body."""

    async def handle_message(self, message: Message) -> Message | None:
        return Message(
            source=self.address,
            target=message.source,
            verb="SEND",
            code=message.code,
            body=f"echo: {message.body}",
            thread=message.thread,
        )


@pytest.fixture
async def broker_stack(redis_client):
    """Engine + broker + a single echo agent registered under an
    owning server with a self-hosted compute provider.
    """
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    broker = Broker(redis_client)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0, broker=broker,
    )

    # Register a server that owns the echo agent's org.
    server = ServerMeta(
        server_id="acme-server",
        org="acme",
        base_rate=0.0002,
        compute_binding="acme-server.small.echo",
        supported_tiers=["small"],
    )
    await broker.register_server(server)

    # Self-hosted compute provider with one matching leaf.
    await broker.register_compute_provider(
        ComputeProvider(provider_id="acme-server"),
    )
    await broker.register_leaf(MenuLeaf(
        provider_id="acme-server", tier="small", model="echo",
        rate_per_1k_chars=0.0,  # self-hosted: compute slice flows back
    ))

    # Give the server full visibility so the router doesn't gate it out
    # on coin-flips during the test.
    await broker.set_reputation(ReputationEntry(
        owner="acme-server", reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))

    # The echo agent.
    addr = AgentAddress.parse(
        "acme.researcher.x.y.s.session.echo-0",
    )
    agent = _EchoAgent(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    # Subscribe-then-publish race: give the consumer loop a beat.
    await asyncio.sleep(0.1)

    # Caller wallet — seed with extra credits.
    caller = AgentAddress.parse("you.human.x.y.s.session.caller")
    await broker.wallet(str(caller)).topup(50.0, reason="test seed")
    await registry.register(caller)

    class Stack:
        pass
    s = Stack()
    s.engine = engine
    s.broker = broker
    s.bus = bus
    s.target = addr
    s.caller = caller
    s.server = server
    try:
        yield s
    finally:
        await agent.stop()
        await bus.close()


# ── happy path ───────────────────────────────────────────────────────


async def test_send_get_settles_on_response(broker_stack):
    s = broker_stack
    caller_before = (await s.broker.wallet(str(s.caller)).get_state()).balance
    server_before = (await s.broker.wallet("acme-server").get_state()).balance
    broker_pool_before = (await s.broker.wallet("__broker__").get_state()).balance
    commons_before = (await s.broker.wallet("__commons__").get_state()).balance

    msg = Message(
        source=s.caller, target=s.target,
        code=Code.INTERVIEW_TEXT, verb="SEND-GET",
        body="hello", thread="t::1",
    )
    response = await s.engine.handle(msg, timeout=2.0)
    assert response is not None
    assert "echo: hello" in str(response.body)

    caller_after = (await s.broker.wallet(str(s.caller)).get_state()).balance
    server_after = (await s.broker.wallet("acme-server").get_state()).balance
    broker_pool_after = (await s.broker.wallet("__broker__").get_state()).balance
    commons_after = (await s.broker.wallet("__commons__").get_state()).balance

    # Caller paid something.
    assert caller_after < caller_before
    # Server earned something.
    assert server_after > server_before
    # Tax flowed (small numbers, but non-zero).
    assert broker_pool_after >= broker_pool_before
    assert commons_after >= commons_before
    # Conservation: caller's debit ≈ sum of credits to server + broker + commons
    # (and to the compute provider, which in this self-hosted setup is
    # the same as acme-server's wallet so the credits are already
    # bundled).
    debit = caller_before - caller_after
    credit = (
        (server_after - server_before)
        + (broker_pool_after - broker_pool_before)
        + (commons_after - commons_before)
    )
    assert math.isclose(debit, credit, abs_tol=1e-6)


# ── refund on timeout ───────────────────────────────────────────────


async def test_send_get_timeout_refunds_caller(broker_stack):
    s = broker_stack
    # Address a target that doesn't exist — engine returns None.
    dead = AgentAddress.parse(
        "acme.researcher.x.y.s.session.does-not-exist",
    )
    caller_before = (await s.broker.wallet(str(s.caller)).get_state()).balance

    msg = Message(
        source=s.caller, target=dead,
        code=Code.INTERVIEW_TEXT, verb="SEND-GET",
        body="hello", thread="t::dead",
    )
    response = await s.engine.handle(msg, timeout=1.0)
    assert response is None

    caller_after = (await s.broker.wallet(str(s.caller)).get_state()).balance
    # Caller wasn't charged because the target was not alive (engine
    # short-circuits before the broker hold).
    assert math.isclose(caller_before, caller_after, abs_tol=1e-9)


# ── insufficient funds ──────────────────────────────────────────────


async def test_send_get_returns_none_on_insufficient_funds(broker_stack):
    """A caller with no balance can't dispatch and we return None
    rather than raising — caller-side code already handles None
    responses cleanly.
    """
    s = broker_stack
    # Bankrupt the caller by holding all their balance.
    bal = (await s.broker.wallet(str(s.caller)).get_state()).balance
    await s.broker.wallet(str(s.caller)).hold(
        hold_id="permahold", amount=bal,
        reason="bankrupt the test caller",
    )

    msg = Message(
        source=s.caller, target=s.target,
        code=Code.INTERVIEW_TEXT, verb="SEND-GET",
        body="anything", thread="t::broke",
    )
    response = await s.engine.handle(msg, timeout=1.0)
    assert response is None


# ── reputation updates ─────────────────────────────────────────────


async def test_reputation_updates_after_successful_settlement(broker_stack):
    s = broker_stack
    rep_before = await s.broker.get_reputation("acme-server")
    completed_before = (
        rep_before.completed_accepted if rep_before else 0
    )

    msg = Message(
        source=s.caller, target=s.target,
        code=Code.INTERVIEW_TEXT, verb="SEND-GET",
        body="hello", thread="t::rep",
    )
    await s.engine.handle(msg, timeout=2.0)

    rep_after = await s.broker.get_reputation("acme-server")
    assert rep_after is not None
    assert rep_after.completed_accepted == completed_before + 1
    # Reputation should have nudged up by REP_REWARD_SUCCESS.
    assert rep_after.reputation > rep_before.reputation


# ── backward compatibility ─────────────────────────────────────────


async def test_engine_without_broker_still_dispatches(redis_client):
    """A bare engine (no broker wired) behaves exactly as before."""
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
        # no broker= argument
    )
    addr = AgentAddress.parse(
        "acme.researcher.x.y.s.session.echo-1",
    )
    agent = _EchoAgent(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.1)

    caller = AgentAddress.parse("you.human.x.y.s.session.caller")
    await registry.register(caller)

    msg = Message(
        source=caller, target=addr,
        code=Code.INTERVIEW_TEXT, verb="SEND-GET",
        body="hello", thread="t::nobroker",
    )
    response = await engine.handle(msg, timeout=2.0)
    assert response is not None
    assert "echo: hello" in str(response.body)

    await agent.stop()
    await bus.close()
