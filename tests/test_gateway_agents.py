"""Gateway agent tests.

Three clusters:

1. Unit: GatewayAgent constructor wiring (accept-tier sanity, metadata
   stamping), translate() override required, default handler shape.
2. Concrete stand-ins: EmbeddingToTextGateway, JsonToTextGateway
   produce the right output tier from representative inputs.
3. End-to-end: SEND-GET through ProtocolEngine to an EmbeddingToText
   gateway routes correctly under the existing CompatibilityMatrix
   and the response carries a string-tier body.
4. Relaying: RelayingGatewayAgent forwards to a downstream target and
   translates the round-trip when a relay_to is supplied.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from ahp.adapters import (
    EmbeddingToTextGateway,
    GatewayAgent,
    JsonToTextGateway,
    RelayingGatewayAgent,
)
from ahp.adapters.base import AHPAgent
from ahp.core import AcceptTier, AgentAddress, Code, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.engine.router import ProtocolEngine
from ahp.engine.thread_manager import ThreadManager
from ahp.registry.registry import AgentMeta, AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


# ── unit: constructor + metadata stamping ────────────────────────────


def _engine(redis_client) -> ProtocolEngine:
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    return ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )


async def test_constructor_rejects_address_without_source_tier(redis_client):
    """An e→s gateway whose address.accept doesn't include 'e' is
    a bug: the protocol would route source-tier messages away from
    this agent. Constructor catches it."""
    engine = _engine(redis_client)
    addr = AgentAddress.parse("acme.gw.x.y.s.session.bad-gateway")  # s, not e
    with pytest.raises(ValueError, match="SOURCE_TIER"):
        EmbeddingToTextGateway(address=addr, engine=engine)


async def test_constructor_stamps_metadata(redis_client):
    """The base class stamps the gateway direction into AgentMeta.extra
    and adds a 'gateway' capability tag for discovery."""
    engine = _engine(redis_client)
    addr = AgentAddress.parse("acme.gw.x.y.se.session.es-gateway")
    agent = EmbeddingToTextGateway(address=addr, engine=engine)
    assert "gateway" in agent.metadata.capabilities
    assert agent.metadata.extra["gateway"] == {
        "source_tier": "e",
        "output_tier": "s",
    }
    await engine.bus.close()


async def test_constructor_preserves_caller_supplied_metadata(redis_client):
    """If the caller passes their own AgentMeta, the gateway stamp is
    merged in rather than overwriting."""
    engine = _engine(redis_client)
    addr = AgentAddress.parse("acme.gw.x.y.se.session.merged")
    meta = AgentMeta(
        capabilities=["search"],
        extra={"custom": "value"},
        description="my gateway",
    )
    agent = EmbeddingToTextGateway(address=addr, engine=engine, metadata=meta)
    assert "search" in agent.metadata.capabilities
    assert "gateway" in agent.metadata.capabilities
    assert agent.metadata.extra["custom"] == "value"
    assert agent.metadata.extra["gateway"]["source_tier"] == "e"
    assert agent.metadata.description == "my gateway"
    await engine.bus.close()


async def test_base_translate_raises(redis_client):
    """A subclass that forgets to override translate() should fail
    loudly, not silently echo."""
    engine = _engine(redis_client)

    class BareGateway(GatewayAgent):
        SOURCE_TIER = AcceptTier.STRING
        OUTPUT_TIER = AcceptTier.STRING
        # no translate override

    addr = AgentAddress.parse("acme.gw.x.y.s.session.bare")
    agent = BareGateway(address=addr, engine=engine)
    msg = Message(
        source=AgentAddress.parse("you.h.x.y.s.session.caller"),
        target=addr, code=Code.INTERVIEW_TEXT,
        verb="SEND-GET", body="hi", thread="t::bare",
    )
    with pytest.raises(NotImplementedError):
        await agent.handle_message(msg)
    await engine.bus.close()


# ── concrete: EmbeddingToTextGateway ─────────────────────────────────


async def test_embedding_gateway_describes_bytes(redis_client):
    engine = _engine(redis_client)
    addr = AgentAddress.parse("acme.gw.x.y.se.session.e2s")
    agent = EmbeddingToTextGateway(address=addr, engine=engine)
    payload = b"\x00" * 1536
    msg = Message(
        source=AgentAddress.parse("you.h.x.y.s.session.observer"),
        target=addr, code=Code.INTERVIEW_EMBEDDINGS,
        verb="SEND-GET", body=payload, thread="t::e2s-bytes",
    )
    reply = await agent.handle_message(msg)
    assert reply is not None
    assert isinstance(reply.body, str)
    assert "1536 bytes" in reply.body
    assert hashlib.sha256(payload).hexdigest()[:12] in reply.body
    await engine.bus.close()


async def test_embedding_gateway_describes_float_vector(redis_client):
    engine = _engine(redis_client)
    addr = AgentAddress.parse("acme.gw.x.y.se.session.e2s-vec")
    agent = EmbeddingToTextGateway(address=addr, engine=engine)
    payload = [0.1, 0.2, 0.3, 0.4, 0.5]  # vector
    msg = Message(
        source=AgentAddress.parse("you.h.x.y.s.session.observer"),
        target=addr, code=Code.INTERVIEW_EMBEDDINGS,
        verb="SEND-GET", body=payload, thread="t::e2s-vec",
    )
    reply = await agent.handle_message(msg)
    assert reply is not None
    assert isinstance(reply.body, str)
    assert "dim=5" in reply.body
    assert "0.100" in reply.body  # first element formatted
    await engine.bus.close()


# ── concrete: JsonToTextGateway ──────────────────────────────────────


async def test_json_gateway_pretty_prints(redis_client):
    engine = _engine(redis_client)
    addr = AgentAddress.parse("acme.gw.x.y.sj.session.j2s")
    agent = JsonToTextGateway(address=addr, engine=engine)
    payload = {"score": 0.9, "verdict": "ok", "items": [1, 2, 3]}
    msg = Message(
        source=AgentAddress.parse("you.h.x.y.s.session.observer"),
        target=addr, code=Code.ADVERSARIAL_AUDIT,
        verb="SEND-GET", body=payload, thread="t::j2s",
    )
    reply = await agent.handle_message(msg)
    assert reply is not None
    assert isinstance(reply.body, str)
    # Pretty-printed: 2-space indent, sorted keys.
    parsed = json.loads(reply.body)
    assert parsed == payload
    assert '"items"' in reply.body  # key with quotes survives
    # Sorted by key — items comes before score comes before verdict.
    items_idx = reply.body.index('"items"')
    score_idx = reply.body.index('"score"')
    verdict_idx = reply.body.index('"verdict"')
    assert items_idx < score_idx < verdict_idx
    await engine.bus.close()


# ── end-to-end: routing through the engine ───────────────────────────


async def test_end_to_end_engine_routes_to_embedding_gateway(redis_client):
    """A SEND-GET to an EmbeddingToText gateway with INTERVIEW_EMBEDDINGS
    routes correctly under CompatibilityMatrix (e-tier accept satisfies
    the {b,e} requirement) and returns the translated string body."""
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    addr = AgentAddress.parse("acme.gw.x.y.se.session.e2s-live")
    agent = EmbeddingToTextGateway(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.1)

    caller = AgentAddress.parse("you.h.x.y.s.session.observer")
    await registry.register(caller)
    try:
        # Use a float-vector payload because the bus's JSON envelope
        # rejects raw bytes (callers base64-encode for transport).
        # The unit tests cover the raw-bytes describe path directly.
        msg = Message(
            source=caller, target=addr,
            code=Code.INTERVIEW_EMBEDDINGS, verb="SEND-GET",
            body=[0.1, 0.2, 0.3], thread="t::e2e",
        )
        response = await engine.handle(msg, timeout=2.0)
        assert response is not None
        assert isinstance(response.body, str)
        assert "dim=3" in response.body
    finally:
        await agent.stop()
        await bus.close()


async def test_engine_blocks_routing_when_gateway_misses_tier(redis_client):
    """A gateway address that does NOT include the source tier in its
    accept set can't be constructed — but if a non-gateway agent with
    accept='s' is targeted with INTERVIEW_EMBEDDINGS, the engine's
    CompatibilityMatrix should still refuse. This is the property
    gateways are designed to bridge."""
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    class _StringOnly(AHPAgent):
        async def handle_message(self, message: Message) -> Message | None:
            return Message(
                source=self.address, target=message.source,
                verb="SEND", code=message.code, thread=message.thread,
                body="echo",
            )

    addr = AgentAddress.parse("acme.r.x.y.s.session.string-only")
    agent = _StringOnly(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.1)

    caller = AgentAddress.parse("you.h.x.y.s.session.observer")
    await registry.register(caller)
    try:
        msg = Message(
            source=caller, target=addr,
            code=Code.INTERVIEW_EMBEDDINGS, verb="SEND-GET",
            body=b"\x00" * 32, thread="t::block",
        )
        # CompatibilityMatrix rejects this — engine raises (the
        # protocol behavior we're confirming is in place).
        from ahp.engine.errors import ProtocolError
        with pytest.raises(ProtocolError):
            await engine.handle(msg, timeout=1.0)
    finally:
        await agent.stop()
        await bus.close()


# ── relaying ─────────────────────────────────────────────────────────


class _JsonEchoAgent(AHPAgent):
    """Downstream JSON-tier agent. Returns the input wrapped in a dict
    so we can tell relay-routed traffic from one-shot translation."""

    async def handle_message(self, message: Message) -> Message | None:
        return Message(
            source=self.address, target=message.source,
            verb="SEND", code=message.code, thread=message.thread,
            body={"echo": message.body, "via": str(self.address)},
        )


async def test_relaying_gateway_forwards_and_translates_back(redis_client):
    """Caller (s-tier) -> j→s gateway with relay_to -> JSON service ->
    gateway translates the reply back -> caller."""
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    # Downstream JSON service.
    downstream = AgentAddress.parse("acme.r.x.y.sj.session.downstream")
    service = _JsonEchoAgent(address=downstream, engine=engine)
    await service.register()
    await service.start()

    # Relaying gateway: source tier 'j' so it can receive JSON
    # messages; subclass that pretty-prints the response back to
    # text via translate_response.
    class _RelayingJsonGateway(RelayingGatewayAgent):
        SOURCE_TIER = AcceptTier.JSON
        OUTPUT_TIER = AcceptTier.STRING

        async def translate(self, message: Message):
            return message.body  # forward JSON to downstream unchanged

        async def translate_response(self, response, original):
            return json.dumps(response.body, sort_keys=True, default=str)

    gw_addr = AgentAddress.parse("acme.gw.x.y.sj.session.relay-gw")
    gateway = _RelayingJsonGateway(address=gw_addr, engine=engine)
    await gateway.register()
    await gateway.start()
    await asyncio.sleep(0.1)

    caller = AgentAddress.parse("you.h.x.y.s.session.observer")
    await registry.register(caller)
    try:
        msg = Message(
            source=caller, target=gw_addr,
            code=Code.COLLAB_CONSENSUS, verb="SEND-GET",
            body={
                "relay_to": str(downstream),
                "payload": {"q": "score this"},
            },
            thread="t::relay",
        )
        response = await engine.handle(msg, timeout=3.0)
        assert response is not None
        assert isinstance(response.body, str)
        parsed = json.loads(response.body)
        # Downstream wrapped the body in {"echo": ..., "via": ...}.
        assert "echo" in parsed
        assert parsed["via"] == str(downstream)
    finally:
        await gateway.stop()
        await service.stop()
        await bus.close()


async def test_relaying_gateway_without_relay_falls_back_to_translate(
    redis_client,
):
    """No relay_to → behaves like a one-shot translator."""
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    class _RelayingJsonGateway(RelayingGatewayAgent):
        SOURCE_TIER = AcceptTier.JSON
        OUTPUT_TIER = AcceptTier.STRING

        async def translate(self, message: Message):
            return json.dumps(message.body, sort_keys=True, default=str)

    addr = AgentAddress.parse("acme.gw.x.y.sj.session.relay-fb")
    agent = _RelayingJsonGateway(address=addr, engine=engine)
    msg = Message(
        source=AgentAddress.parse("you.h.x.y.s.session.observer"),
        target=addr,
        code=Code.COLLAB_CONSENSUS, verb="SEND-GET",
        body={"no": "relay_to here"},
        thread="t::fb",
    )
    reply = await agent.handle_message(msg)
    assert reply is not None
    assert reply.body == json.dumps(
        {"no": "relay_to here"}, sort_keys=True, default=str,
    )
    await bus.close()


async def test_relaying_gateway_invalid_relay_address_returns_error_string(
    redis_client,
):
    """A malformed relay_to value yields an error string rather than
    raising; the caller gets a useful diagnostic in the reply body."""
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    class _RelayingJsonGateway(RelayingGatewayAgent):
        SOURCE_TIER = AcceptTier.JSON
        OUTPUT_TIER = AcceptTier.STRING

        async def translate(self, message: Message):
            return ""

    addr = AgentAddress.parse("acme.gw.x.y.sj.session.relay-bad")
    agent = _RelayingJsonGateway(address=addr, engine=engine)
    msg = Message(
        source=AgentAddress.parse("you.h.x.y.s.session.observer"),
        target=addr, code=Code.COLLAB_CONSENSUS, verb="SEND-GET",
        body={"relay_to": "not.a.valid.address"},
        thread="t::bad",
    )
    reply = await agent.handle_message(msg)
    assert reply is not None
    assert "gateway error" in reply.body
    assert "not.a.valid.address" in reply.body
    await bus.close()
