"""AgentRegistry.update_reputation + broker integration tests.

Two clusters:

1. Registry primitive — CAS nudge clamps to [0, 1], returns None for
   unregistered addresses, emits audit events.
2. Broker integration — when a registry reference is wired, settlement
   feeds back into the responder's AgentMeta.reputation. SEND-GET
   end-to-end test through the engine confirms the path is reachable
   in practice (not just in unit form).
"""

from __future__ import annotations

import asyncio

import pytest

from ahp.adapters.base import AHPAgent
from ahp.audit import InMemoryAuditSink
from ahp.broker import Broker, ServerMeta
from ahp.core import AgentAddress, Code, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import (
    REP_PENALTY_FAILURE,
    REP_REWARD_SUCCESS,
    ReputationEntry,
    VISIBILITY_FULL_AT,
)
from ahp.engine.router import ProtocolEngine
from ahp.engine.thread_manager import ThreadManager
from ahp.registry.registry import AgentMeta, AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


# ── registry primitive ───────────────────────────────────────────────


async def test_update_reputation_nudges_up(redis_client):
    registry = AgentRegistry(redis_client)
    addr = AgentAddress.parse("acme.r.x.y.s.session.a-0")
    await registry.register(addr, AgentMeta(reputation=0.5))

    meta = await registry.update_reputation(addr, +0.1)
    assert meta is not None
    assert abs(meta.reputation - 0.6) < 1e-9
    # Persisted.
    fetched = await registry.get(addr)
    assert abs(fetched.reputation - 0.6) < 1e-9


async def test_update_reputation_nudges_down(redis_client):
    registry = AgentRegistry(redis_client)
    addr = AgentAddress.parse("acme.r.x.y.s.session.a-1")
    await registry.register(addr, AgentMeta(reputation=0.5))

    meta = await registry.update_reputation(addr, -0.2)
    assert meta is not None
    assert abs(meta.reputation - 0.3) < 1e-9


async def test_update_reputation_clamps_to_unit_interval(redis_client):
    registry = AgentRegistry(redis_client)
    addr = AgentAddress.parse("acme.r.x.y.s.session.a-2")
    await registry.register(addr, AgentMeta(reputation=0.95))

    # Up-clamp.
    meta = await registry.update_reputation(addr, +0.5)
    assert meta is not None
    assert meta.reputation == 1.0

    # Down-clamp.
    meta = await registry.update_reputation(addr, -5.0)
    assert meta is not None
    assert meta.reputation == 0.0


async def test_update_reputation_unknown_returns_none(redis_client):
    registry = AgentRegistry(redis_client)
    addr = AgentAddress.parse("acme.r.x.y.s.session.never-registered")
    result = await registry.update_reputation(addr, +0.1)
    assert result is None


async def test_update_reputation_emits_audit(redis_client):
    sink = InMemoryAuditSink()
    registry = AgentRegistry(redis_client, audit=sink)
    addr = AgentAddress.parse("acme.r.x.y.s.session.a-3")
    await registry.register(addr, AgentMeta(reputation=0.5))

    await registry.update_reputation(addr, +0.05)
    ops = [e.op for e in sink.events]
    assert "registry.update_reputation" in ops


# ── broker integration ──────────────────────────────────────────────


async def _seed_broker_with_registry(
    redis_client,
) -> tuple[Broker, AgentRegistry, AgentAddress]:
    registry = AgentRegistry(redis_client, heartbeat_ttl=60)
    broker = Broker(redis_client, registry=registry)
    await broker.register_server(ServerMeta(
        server_id="acme", org="acme", base_rate=0.0002,
        compute_binding="acme.small.echo",
        supported_tiers=["small"],
    ))
    await broker.register_compute_provider(ComputeProvider(provider_id="acme"))
    await broker.register_leaf(MenuLeaf(
        provider_id="acme", tier="small", model="echo",
        rate_per_1k_chars=0.0,
    ))
    await broker.set_reputation(ReputationEntry(
        owner="acme", reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))
    addr = AgentAddress.parse("acme.r.x.y.s.session.responder")
    await registry.register(addr, AgentMeta(reputation=0.5))
    return broker, registry, addr


async def test_settlement_nudges_responder_up_on_accepted(redis_client):
    broker, registry, addr = await _seed_broker_with_registry(redis_client)
    caller = "you.human.x.y.s.session.caller"
    await broker.wallet(caller).topup(10.0, reason="seed")
    await broker.hold(caller=caller, amount=1.0, hold_id="msg:1")

    rep_before = (await registry.get(addr)).reputation
    await broker.calculate_and_settle(
        caller=caller, hold_id="msg:1",
        server=await broker.servers.get("acme"),
        leaf=MenuLeaf(
            provider_id="acme", tier="small", model="echo",
            rate_per_1k_chars=0.0,
        ),
        response_chars=200, max_response_chars=1000,
        actual_latency_ms=400.0, completed_with_caller=0,
        responder=addr,
    )
    rep_after = (await registry.get(addr)).reputation
    assert abs((rep_after - rep_before) - REP_REWARD_SUCCESS) < 1e-9


async def test_settlement_nudges_responder_down_on_sub_tier(redis_client):
    broker, registry, addr = await _seed_broker_with_registry(redis_client)
    caller = "you.human.x.y.s.session.caller"
    await broker.wallet(caller).topup(10.0, reason="seed")
    await broker.hold(caller=caller, amount=1.0, hold_id="msg:2")

    rep_before = (await registry.get(addr)).reputation
    await broker.calculate_and_settle(
        caller=caller, hold_id="msg:2",
        server=await broker.servers.get("acme"),
        leaf=MenuLeaf(
            provider_id="acme", tier="small", model="echo",
            rate_per_1k_chars=0.0,
        ),
        response_chars=200, max_response_chars=1000,
        actual_latency_ms=400.0, completed_with_caller=0,
        tier_verdict="sub_tier",
        responder=addr,
    )
    rep_after = (await registry.get(addr)).reputation
    assert abs((rep_before - rep_after) - REP_PENALTY_FAILURE) < 1e-9


async def test_no_registry_means_no_agent_rep_update(redis_client):
    """Without a registry reference, the broker still settles but the
    AgentMeta.reputation field is untouched (no error, just no nudge)."""
    broker = Broker(redis_client)  # no registry=
    await broker.register_server(ServerMeta(
        server_id="acme", org="acme", base_rate=0.0002,
        compute_binding="acme.small.echo",
        supported_tiers=["small"],
    ))
    await broker.register_compute_provider(ComputeProvider(provider_id="acme"))
    await broker.register_leaf(MenuLeaf(
        provider_id="acme", tier="small", model="echo",
        rate_per_1k_chars=0.0,
    ))
    await broker.set_reputation(ReputationEntry(
        owner="acme", reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))

    registry = AgentRegistry(redis_client)
    addr = AgentAddress.parse("acme.r.x.y.s.session.r0")
    await registry.register(addr, AgentMeta(reputation=0.5))

    caller = "you.human.x.y.s.session.caller"
    await broker.wallet(caller).topup(10.0, reason="seed")
    await broker.hold(caller=caller, amount=1.0, hold_id="msg:no-reg")
    await broker.calculate_and_settle(
        caller=caller, hold_id="msg:no-reg",
        server=await broker.servers.get("acme"),
        leaf=None,
        response_chars=200, max_response_chars=1000,
        actual_latency_ms=400.0, completed_with_caller=0,
        responder=addr,
    )
    # Untouched.
    assert (await registry.get(addr)).reputation == 0.5


# ── engine end-to-end ──────────────────────────────────────────────


class _EchoAgent(AHPAgent):
    async def handle_message(self, message: Message) -> Message | None:
        return Message(
            source=self.address, target=message.source,
            verb="SEND", code=message.code,
            body=f"echo: {message.body}", thread=message.thread,
        )


async def test_engine_send_get_nudges_responder_reputation(redis_client):
    """A real SEND-GET through ProtocolEngine should bump the
    responder's AgentMeta.reputation up by REP_REWARD_SUCCESS."""
    registry = AgentRegistry(redis_client, heartbeat_ttl=60)
    broker = Broker(redis_client, registry=registry)
    await broker.register_server(ServerMeta(
        server_id="acme", org="acme", base_rate=0.0002,
        compute_binding="acme.small.echo",
        supported_tiers=["small"],
    ))
    await broker.register_compute_provider(ComputeProvider(provider_id="acme"))
    await broker.register_leaf(MenuLeaf(
        provider_id="acme", tier="small", model="echo",
        rate_per_1k_chars=0.0,
    ))
    await broker.set_reputation(ReputationEntry(
        owner="acme", reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))

    bus = RedisBus(redis_client)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0, broker=broker,
    )

    addr = AgentAddress.parse("acme.r.x.y.s.session.echoer")
    agent = _EchoAgent(address=addr, engine=engine)
    await agent.register()  # marks alive AND seeds AgentMeta
    await agent.start()
    await asyncio.sleep(0.1)

    # Override the reputation we want to nudge from.
    meta = await registry.get(addr) or AgentMeta()
    meta.reputation = 0.5
    await registry.register(addr, meta)

    caller = AgentAddress.parse("you.human.x.y.s.session.caller")
    await registry.register(caller)
    await broker.wallet(str(caller)).topup(20.0, reason="seed")

    try:
        msg = Message(
            source=caller, target=addr,
            code=Code.INTERVIEW_TEXT, verb="SEND-GET",
            body="hello", thread="t::rep",
        )
        response = await engine.handle(msg, timeout=2.0)
        assert response is not None

        # The responder's AgentMeta.reputation took the nudge.
        post = await registry.get(addr)
        assert post is not None
        assert post.reputation > 0.5
        assert abs(post.reputation - (0.5 + REP_REWARD_SUCCESS)) < 1e-9
    finally:
        await agent.stop()
        await bus.close()
