"""CLI tests for `ahp tap`, `ahp send`, `ahp describe-agent`.

Same Redis-injection pattern as the other Redis-touching CLI tests:
monkeypatch ``ahp.cli._connect_redis`` to hand back the test's
``redis_client`` fakeredis and call the async worker directly.

Caveat noted in CLAUDE.md: fakeredis pubsub is per-client. ``tap`` and
``send`` both build their own ``RedisBus`` instances, so we share the
SAME ``redis_client`` across the producer and the subscriber by
patching ``_connect_redis`` to always return the same instance.
"""

from __future__ import annotations

import asyncio
import io

import pytest

import ahp.cli
from ahp.adapters.base import AHPAgent
from ahp.broker import Broker, ServerMeta
from ahp.core import AgentAddress, Code, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import (
    ReputationEntry,
    VISIBILITY_FULL_AT,
)
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta, AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


async def _arun(cmd: str, *argv: str) -> tuple[int, str]:
    parser = ahp.cli.build_parser()
    args = parser.parse_args([cmd, *argv])
    buf = io.StringIO()
    if cmd == "tap":
        rc = await ahp.cli._tap_async(args, buf)
    elif cmd == "send":
        rc = await ahp.cli._send_async(args, buf)
    elif cmd == "describe-agent":
        rc = await ahp.cli._describe_agent_async(args, buf)
    else:
        raise AssertionError(f"unexpected cmd {cmd}")
    return rc, buf.getvalue()


# ── tap ───────────────────────────────────────────────────────────────


async def test_tap_streams_published_messages(redis_client, monkeypatch):
    """Tap should pick up messages published to the bus via the tap mirror."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    # Use a small limit so the tap worker exits cleanly without Ctrl-C.
    bus = RedisBus(redis_client)

    async def publish_some() -> None:
        # Wait a beat for the tap subscriber to be ready.
        await asyncio.sleep(0.2)
        for i in range(3):
            src = AgentAddress.parse(
                f"tifin.researcher.x.y.s.session.alice-{i}"
            )
            tgt = AgentAddress.parse(
                f"tifin.researcher.x.y.s.session.bob-{i}"
            )
            msg = Message(
                source=src, target=tgt,
                code=Code.INTERVIEW_TEXT, verb="SEND",
                body=f"hello {i}", thread=f"t::{i}",
            )
            await bus.send(msg)

    # Run publisher concurrently with the tap reader.
    publisher = asyncio.create_task(publish_some())
    try:
        rc, out = await _arun(
            "tap",
            "--redis-url", "redis://test/0",
            "--limit", "3",
        )
    finally:
        await publisher
        await bus.close()

    assert rc == 0
    # Each of the three messages renders one line.
    assert out.count("alice-") >= 3
    assert "hello 0" in out
    assert "hello 2" in out


async def test_tap_filters_by_code(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    bus = RedisBus(redis_client)

    async def publish() -> None:
        await asyncio.sleep(0.2)
        src = AgentAddress.parse("tifin.researcher.x.y.s.session.alice")
        tgt = AgentAddress.parse("tifin.researcher.x.y.s.session.bob")
        # One interview message + one that won't match.
        await bus.send(Message(
            source=src, target=tgt,
            code=Code.INTERVIEW_TEXT, verb="SEND",
            body="match me", thread="t::1",
        ))
        await bus.send(Message(
            source=src, target=tgt,
            code="audit.event", verb="SEND",
            body="ignore me", thread="t::2",
        ))

    publisher = asyncio.create_task(publish())
    try:
        rc, out = await _arun(
            "tap",
            "--redis-url", "redis://test/0",
            "--code", "interview.*",
            "--limit", "1",
        )
    finally:
        await publisher
        await bus.close()

    assert rc == 0
    assert "match me" in out
    assert "ignore me" not in out


# ── send ──────────────────────────────────────────────────────────────


class _EchoAgent(AHPAgent):
    async def handle_message(self, message: Message) -> Message | None:
        body = message.body
        return Message(
            source=self.address, target=message.source,
            verb="SEND", code=message.code,
            body=f"echo: {body}", thread=message.thread,
        )


async def test_send_fire_and_forget(redis_client, monkeypatch):
    """`ahp send` (no --get) delivers a SEND and prints a delivery line."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(),
        default_timeout=2.0,
    )
    addr = AgentAddress.parse("tifin.researcher.x.y.s.session.echo-fire")
    agent = _EchoAgent(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.1)

    try:
        rc, out = await _arun(
            "send",
            "--redis-url", "redis://test/0",
            "--target", str(addr),
            "--code", Code.INTERVIEW_TEXT,
            "--body", "hello",
        )
    finally:
        await agent.stop()
        await bus.close()

    assert rc == 0
    assert "sent to" in out
    assert str(addr) in out


async def test_send_get_prints_response(redis_client, monkeypatch):
    """`ahp send --get` waits for a response and prints it."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(),
        default_timeout=2.0,
    )
    addr = AgentAddress.parse("tifin.researcher.x.y.s.session.echo-get")
    agent = _EchoAgent(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.1)

    try:
        rc, out = await _arun(
            "send",
            "--redis-url", "redis://test/0",
            "--target", str(addr),
            "--code", Code.INTERVIEW_TEXT,
            "--body", "ping",
            "--get",
            "--timeout", "2.0",
        )
    finally:
        await agent.stop()
        await bus.close()

    assert rc == 0
    assert "echo: ping" in out


async def test_send_body_parses_as_json_when_valid(
    redis_client, monkeypatch,
):
    """A JSON-shaped --body becomes a dict in the message body."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(),
        default_timeout=2.0,
    )
    addr = AgentAddress.parse("tifin.researcher.x.y.s.session.echo-json")
    agent = _EchoAgent(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.1)

    try:
        rc, out = await _arun(
            "send",
            "--redis-url", "redis://test/0",
            "--target", str(addr),
            "--code", Code.INTERVIEW_TEXT,
            "--body", '{"q": "what is dark matter?"}',
            "--get",
        )
    finally:
        await agent.stop()
        await bus.close()

    assert rc == 0
    # The echo includes "echo: " + repr of the dict.
    assert "dark matter" in out


# ── describe-agent ────────────────────────────────────────────────────


async def test_describe_agent_unregistered(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, out = await _arun(
        "describe-agent",
        "tifin.researcher.x.y.s.session.nobody",
        "--redis-url", "redis://test/0",
    )
    assert rc == 0
    assert "no registry metadata" in out


async def test_describe_agent_shows_full_state(redis_client, monkeypatch):
    """Registered agent + broker reputation + owning server should
    all surface in the output."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    addr = AgentAddress.parse(
        "tifin.researcher.x.y.s.session.alice"
    )
    registry = AgentRegistry(redis_client, heartbeat_ttl=60)
    await registry.register(
        addr,
        AgentMeta(
            capabilities=["search", "summarize"],
            reputation=0.7,
            description="alice the researcher",
        ),
    )

    broker = Broker(redis_client)
    await broker.register_server(ServerMeta(
        server_id="tifin", org="tifin",
        base_rate=0.0002,
        compute_binding="tifin.small.echo",
        supported_tiers=["small"],
    ))
    await broker.register_compute_provider(ComputeProvider(provider_id="tifin"))
    await broker.register_leaf(MenuLeaf(
        provider_id="tifin", tier="small", model="echo",
        rate_per_1k_chars=0.0,
    ))
    await broker.set_reputation(ReputationEntry(
        owner=str(addr), reputation=0.85,
        completed_accepted=42, completed_total=44, failed=2,
        sum_latency_ms=4400.0, csat=0.9, csat_samples=3,
    ))

    rc, out = await _arun(
        "describe-agent", str(addr),
        "--redis-url", "redis://test/0",
    )
    assert rc == 0
    # Registry plane.
    assert str(addr) in out
    assert "alive" in out
    assert "search, summarize" in out
    assert "alice the researcher" in out
    # Broker reputation plane.
    assert "0.850" in out  # reputation
    assert "42/44" in out  # completed
    assert "failed: 2" in out
    assert "csat" in out
    # Owning server plane.
    assert "tifin.small.echo" in out  # compute_binding
    assert "best leaf" in out


async def test_describe_agent_invalid_address(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, _ = await _arun(
        "describe-agent",
        "not-a-valid-address",
        "--redis-url", "redis://test/0",
    )
    assert rc == 2
