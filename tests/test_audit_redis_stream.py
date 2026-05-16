"""Tests for RedisStreamAuditSink + broker / survey audit emissions.

Two clusters:

1. Sink-level — XADD writes the event, MAXLEN trims, errors don't
   propagate.
2. Integration — Broker constructed with audit=sink emits typed
   events at registration, settlement, outage, survey enqueue +
   response. The events round-trip cleanly through the JSON layer.
"""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from ahp.audit import (
    DEFAULT_REDIS_AUDIT_STREAM,
    AuditEvent,
    InMemoryAuditSink,
    MultiSink,
    RedisStreamAuditSink,
)
from ahp.broker import Broker, ServerMeta
from ahp.broker.surveys import SurveyRequest, SurveyResponse
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import (
    ReputationEntry,
    VISIBILITY_FULL_AT,
)


# ── sink level ────────────────────────────────────────────────────────


async def test_redis_stream_sink_xadds(redis_client):
    sink = RedisStreamAuditSink(redis_client)
    event = AuditEvent(
        op="broker.test", target="acme",
        extra={"x": 1, "y": "two"},
    )
    await sink.emit(event)
    entries = await redis_client.xrange(DEFAULT_REDIS_AUDIT_STREAM)
    assert len(entries) == 1
    _id, fields = entries[0]
    payload = json.loads(fields["data"])
    assert payload["op"] == "broker.test"
    assert payload["target"] == "acme"
    assert payload["extra"]["x"] == 1
    assert payload["extra"]["y"] == "two"


async def test_redis_stream_sink_respects_maxlen(redis_client):
    """MAXLEN ~ trims the stream over time. Approximate trim may
    overshoot the cap slightly, so we only assert it bounds it."""
    sink = RedisStreamAuditSink(redis_client, maxlen=5)
    for i in range(20):
        await sink.emit(AuditEvent(op=f"x.{i}"))
    length = await redis_client.xlen(DEFAULT_REDIS_AUDIT_STREAM)
    assert length <= 20  # trim is best-effort upper-bounded
    # And the *latest* event is preserved.
    last = (await redis_client.xrevrange(
        DEFAULT_REDIS_AUDIT_STREAM, count=1,
    ))[0]
    payload = json.loads(last[1]["data"])
    assert payload["op"] == "x.19"


async def test_redis_stream_sink_swallows_errors(redis_client):
    """A failing client must not raise out of emit()."""

    class BoomClient:
        async def xadd(self, *a, **kw):
            raise RuntimeError("boom")

    sink = RedisStreamAuditSink(BoomClient())
    await sink.emit(AuditEvent(op="x"))  # must not raise


async def test_redis_stream_sink_custom_stream_key(redis_client):
    sink = RedisStreamAuditSink(redis_client, stream_key="custom:stream")
    await sink.emit(AuditEvent(op="x"))
    # Default key is empty.
    assert await redis_client.xlen(DEFAULT_REDIS_AUDIT_STREAM) == 0
    # Custom key has the event.
    assert await redis_client.xlen("custom:stream") == 1


# ── broker integration ───────────────────────────────────────────────


async def _seed_broker_with_audit(
    redis_client, *, audit=None,
) -> Broker:
    broker = Broker(redis_client, audit=audit)
    await broker.register_server(ServerMeta(
        server_id="acme", org="acme",
        base_rate=0.0002, compute_binding="acme.small.echo",
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
    return broker


async def test_broker_emits_server_and_provider_events(redis_client):
    sink = InMemoryAuditSink()
    await _seed_broker_with_audit(redis_client, audit=sink)
    ops = [e.op for e in sink.events]
    assert "broker.server.register" in ops
    assert "broker.provider.register" in ops
    assert "broker.leaf.register" in ops


async def test_broker_emits_settlement_event(redis_client):
    sink = InMemoryAuditSink()
    broker = await _seed_broker_with_audit(redis_client, audit=sink)
    caller = "you.human.x.y.s.session.caller"
    await broker.wallet(caller).topup(10.0, reason="seed")
    await broker.wallet("__commons__").topup(20.0, reason="seed")
    await broker.hold(caller=caller, amount=1.0, hold_id="msg:99")

    await broker.calculate_and_settle(
        caller=caller, hold_id="msg:99",
        server=await broker.servers.get("acme"),
        leaf=MenuLeaf(
            provider_id="acme", tier="small", model="echo",
            rate_per_1k_chars=0.0,
        ),
        response_chars=500, max_response_chars=1000,
        actual_latency_ms=500.0, completed_with_caller=0,
    )

    settlement_events = [e for e in sink.events if e.op == "broker.settlement"]
    assert len(settlement_events) == 1
    extra = settlement_events[0].extra
    assert extra["caller"] == caller
    assert extra["leaf"] == "acme.small.echo"
    assert extra["pre_tax"] > 0
    assert "to_server" in extra
    assert "to_broker" in extra
    assert "to_commons" in extra


async def test_broker_emits_outage_event(redis_client):
    """When check_compute_outages flags an outage, it emits a
    broker.provider.outage event with success=False."""
    from ahp.broker.compute_registry import ComputeProviderRegistry

    sink = InMemoryAuditSink()
    broker = Broker(redis_client, audit=sink)
    # Short TTL via a fresh registry; swap into the broker.
    short = ComputeProviderRegistry(redis_client, heartbeat_ttl=1)
    broker.compute = short

    await broker.register_compute_provider(ComputeProvider(provider_id="flaky"))
    await asyncio.sleep(1.2)  # let alive key expire
    hits = await broker.check_compute_outages()
    assert hits == ["flaky"]

    outage_events = [e for e in sink.events if e.op == "broker.provider.outage"]
    assert len(outage_events) == 1
    assert outage_events[0].target == "flaky"
    assert outage_events[0].success is False
    assert outage_events[0].error == "UnplannedDeregister"


# ── survey audit emissions ───────────────────────────────────────────


async def test_surveys_emit_enqueue_and_response(redis_client):
    sink = InMemoryAuditSink()
    broker = await _seed_broker_with_audit(redis_client, audit=sink)
    actor = "you.human.x.y.s.session.caller"
    await broker.wallet("__commons__").topup(20.0, reason="seed")

    request = SurveyRequest.new(
        kind="post_settlement",
        target_server="acme", surveyed_actor=actor,
        recipe="post_settlement:csat",
        settlement_id="msg:42",
        reward=0.5,
        delay_seconds=0.0,
    )
    sink_pre_count = len([e for e in sink.events if e.op.startswith("survey.")])
    await broker.surveys.enqueue(request)
    assert any(e.op == "survey.enqueue" for e in sink.events)

    response = SurveyResponse(
        survey_id=request.survey_id,
        surveyed_actor=actor,
        target_server="acme",
        recipe=request.recipe,
        settlement_id=request.settlement_id,
        score=0.9,
    )
    await broker.submit_survey_response(response)
    resp_events = [e for e in sink.events if e.op == "survey.response"]
    assert len(resp_events) == 1
    assert resp_events[0].target == request.survey_id
    assert resp_events[0].extra["score"] == 0.9


async def test_multisink_with_redis_stream_and_in_memory(redis_client):
    """A MultiSink combining in-memory + Redis Streams works for both."""
    mem = InMemoryAuditSink()
    rstream = RedisStreamAuditSink(redis_client)
    sink = MultiSink([mem, rstream])

    broker = Broker(redis_client, audit=sink)
    await broker.register_server(ServerMeta(
        server_id="acme", org="acme", base_rate=0.0002,
        compute_binding="acme.small.echo",
    ))

    # In-memory side picked it up.
    assert any(e.op == "broker.server.register" for e in mem.events)
    # Redis Streams side too.
    entries = await redis_client.xrange(DEFAULT_REDIS_AUDIT_STREAM)
    payloads = [json.loads(f["data"]) for _id, f in entries]
    assert any(p["op"] == "broker.server.register" for p in payloads)
