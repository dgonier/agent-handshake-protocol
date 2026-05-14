"""Federation test: two AHP engines on one shared Redis form one network.

The key claim from the protocol design: agent addresses are universal
strings. Once two processes share a Redis substrate, they share the
registry, the bus, the cache, and the tap — so an agent on "node B"
can address an agent on "node A" by URI alone, with no inter-process
plumbing.

This test shares ONE FakeRedis client between two complete AHP stacks
(bus + registry + cache + engine), which is sufficient to prove the
protocol's federation semantics:

* Registry HSET on engine A is visible to engine B's resolver.
* Liveness markers (TTL'd keys) work across engines.
* engine_b.handle(SEND-GET to A's agent) reaches A's bus consumer and
  the reply returns to B.
* engine_b.handle(CAST-GET to a pattern) discovers and reaches BOTH
  A's and B's matching agents.

The two-process / two-FastAPI variant lives in ``examples/federation/``;
running it requires a real Redis (pubsub is per-connection in
fakeredis, which makes a strict two-client test environment harder
than it's worth).
"""

from __future__ import annotations

import asyncio

import fakeredis.aioredis
import pytest

from ahp.adapters.base import AHPAgent
from ahp.adapters.factory import AgentFactory
from ahp.adapters.groups import GroupRegistry
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern
from ahp.engine import ProtocolEngine
from ahp.registry import AgentRegistry
from ahp.transport import ProtocolCache, RedisBus


# ── helpers ────────────────────────────────────────────────────────────


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


class _LabeledEcho(AHPAgent):
    """Echoes the body with a node label so we can prove which node replied."""

    def __init__(self, address, engine, *, label):
        super().__init__(address, engine, heartbeat_interval=0)
        self.label = label

    async def handle_message(self, message: Message):
        if not message.expects_response:
            return None
        return Message(
            source=self.address, target=message.source, verb="SEND",
            code=message.code, thread=message.thread,
            body=f"{self.label}:{message.body}",
        )


def _build_node(shared_redis):
    """Construct an independent AHP stack (bus + registry + cache + engine).

    Both nodes call this with the SAME shared_redis client, modeling
    two participants on one Redis network.
    """
    bus = RedisBus(shared_redis)
    registry = AgentRegistry(shared_redis, heartbeat_ttl=60)
    cache = ProtocolCache(shared_redis)
    engine = ProtocolEngine(bus, registry, cache, default_timeout=2.0)
    return bus, registry, cache, engine


@pytest.fixture
async def federation():
    """Two complete AHP stacks sharing one fakeredis client."""
    shared = fakeredis.aioredis.FakeRedis(decode_responses=True)

    bus_a, reg_a, cache_a, engine_a = _build_node(shared)
    bus_b, reg_b, cache_b, engine_b = _build_node(shared)

    class _Federation:
        pass

    fed = _Federation()
    fed.shared = shared
    fed.a = type("Node", (), {
        "bus": bus_a, "registry": reg_a, "cache": cache_a, "engine": engine_a,
    })()
    fed.b = type("Node", (), {
        "bus": bus_b, "registry": reg_b, "cache": cache_b, "engine": engine_b,
    })()
    try:
        yield fed
    finally:
        await bus_a.close()
        await bus_b.close()
        await shared.aclose()


# ── core federation guarantees ─────────────────────────────────────────


async def test_registry_is_shared_across_nodes(federation):
    """A registers an agent; B's registry sees it without any handshake."""
    bull = _addr("tifin.adversarial.finance.equities.s.session.bull")
    await federation.a.registry.register(bull)

    assert await federation.b.registry.is_alive(bull)
    pattern = AddressPattern.parse("*.adversarial.finance.*.s.*.*")
    resolved = await federation.b.registry.resolve(pattern)
    assert bull in resolved


async def test_node_b_can_call_node_a_agent_by_address(federation):
    """B's engine.handle reaches an agent hosted on A's bus consumer."""
    bull_addr = _addr("tifin.adversarial.finance.equities.s.session.bull")
    bull = _LabeledEcho(bull_addr, federation.a.engine, label="bull@nodeA")
    await bull.register()
    await bull.start()
    await asyncio.sleep(0.05)

    try:
        # Construct + send the request entirely from B's perspective.
        req = Message(
            source=_addr("tifin.collaborative.finance.equities.s.session.alice"),
            target=bull_addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::fed::a-from-b",
            body="hi from node B",
        )
        reply = await federation.b.engine.handle(req, timeout=2.0)
        assert reply is not None
        assert reply.body == "bull@nodeA:hi from node B"
        # The reply's source must be A's bull (universal address — same string
        # both nodes see).
        assert reply.source == bull_addr
    finally:
        await bull.stop()


async def test_broadcast_reaches_agents_on_both_nodes(federation):
    """A CAST-GET issued by B discovers agents on A AND on B."""
    bull_addr = _addr("tifin.adversarial.finance.equities.s.session.bull")
    bear_addr = _addr("tifin.adversarial.finance.equities.s.session.bear")

    bull = _LabeledEcho(bull_addr, federation.a.engine, label="bull@A")
    bear = _LabeledEcho(bear_addr, federation.b.engine, label="bear@B")
    for a in (bull, bear):
        await a.register()
        await a.start()
    await asyncio.sleep(0.05)

    try:
        # B initiates the broadcast.
        req = Message(
            source=_addr("tifin.collaborative.finance.equities.s.session.alice"),
            target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
            verb="CAST-GET", code=Code.ADVERSARIAL_DEBATE,
            thread="thread::fed::debate", body="Tesla",
        )
        replies = await federation.b.engine.handle(req, timeout=2.0)
        bodies = sorted(r.body for r in replies)
        assert bodies == ["bear@B:Tesla", "bull@A:Tesla"]
    finally:
        await bull.stop()
        await bear.stop()


# ── group registry semantics work cross-node too ──────────────────────


async def test_group_lookup_resolves_to_cross_node_pattern(federation):
    """A group registered on B's factory routes to A's agents over the wire."""
    bull_addr = _addr("tifin.adversarial.finance.equities.s.session.bull")
    bull = _LabeledEcho(bull_addr, federation.a.engine, label="bull@A")
    await bull.register(); await bull.start()
    await asyncio.sleep(0.05)

    # B's factory has its own group registry — group names are *local*
    # configuration, but the pattern they resolve to is universal.
    b_groups = GroupRegistry()
    b_groups.register("debaters", "*.adversarial.finance.*.s.*.*")
    AgentFactory(federation.b.engine, groups=b_groups)

    alice = _LabeledEcho(
        _addr("tifin.collaborative.finance.equities.s.session.alice"),
        federation.b.engine, label="alice@B",
    )

    try:
        replies = await alice.broadcast_to(
            "debaters",                     # local name on node B
            code=Code.ADVERSARIAL_DEBATE,
            body="argue Tesla",
            timeout=2.0,
        )
        assert any(r.source == bull_addr for r in replies)
        assert any("bull@A" in r.body for r in replies)
    finally:
        await bull.stop()


# ── cache hits cross node boundaries ──────────────────────────────────


async def test_cache_is_shared_across_nodes(federation):
    """A SEND-GET cached by A's engine is served from cache when B asks
    the same question — same Redis key derivation, same backing store."""
    bull_addr = _addr("tifin.adversarial.finance.equities.s.longterm.bull")
    bull = _LabeledEcho(bull_addr, federation.a.engine, label="bull@A")
    await bull.register(); await bull.start()
    await asyncio.sleep(0.05)

    try:
        req_a = Message(
            source=_addr("tifin.collaborative.finance.equities.s.session.alice"),
            target=bull_addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::fed::cache",
            body="Tesla",
        )
        reply_a = await federation.a.engine.handle(req_a, timeout=2.0)
        assert reply_a is not None

        # Now stop the agent. Without the cache, B's identical request
        # would time out. With the cache, B sees A's stored response.
        await bull.stop()
        await bull.deregister()

        req_b = Message(
            source=_addr("tifin.collaborative.finance.equities.s.session.alice"),
            target=bull_addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::fed::cache-b",
            body="Tesla",
        )
        reply_b = await federation.b.engine.handle(req_b, timeout=0.3)
        assert reply_b is not None
        assert reply_b.body == reply_a.body
    finally:
        await bull.stop()
