"""Tests for the audit layer: events, sinks, and emit wire-up."""

from __future__ import annotations

import json
import logging

import pytest

from ahp.audit import (
    AuditEvent,
    AuditSink,
    InMemoryAuditSink,
    LoggingAuditSink,
    MultiSink,
    NullAuditSink,
)
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.engine.router import ProtocolEngine
from ahp.registry.auth import (
    AddressClaimPolicy,
    Principal,
    UnauthorizedRegistrationError,
)
from ahp.registry.registry import AgentRegistry


# ── AuditEvent shape ────────────────────────────────────────────────────


def test_audit_event_serializes_compactly():
    e = AuditEvent(op="registry.register", target="acme.r.d.s.s.session.0")
    payload = json.loads(e.to_json())
    assert payload["op"] == "registry.register"
    assert payload["target"] == "acme.r.d.s.s.session.0"
    assert payload["success"] is True
    assert payload["error"] is None
    assert isinstance(payload["timestamp"], float)


def test_audit_event_to_dict_roundtrip():
    e = AuditEvent(op="engine.send", success=False, error="X")
    d = e.to_dict()
    assert d["op"] == "engine.send"
    assert d["success"] is False
    assert d["error"] == "X"


# ── InMemorySink ────────────────────────────────────────────────────────


async def test_in_memory_sink_buffers_and_clears():
    sink = InMemoryAuditSink(capacity=4)
    for i in range(6):
        await sink.emit(AuditEvent(op=f"op.{i}"))
    # Bounded — first two evicted.
    assert len(sink) == 4
    assert sink.events[0].op == "op.2"
    sink.clear()
    assert len(sink) == 0


def test_in_memory_sink_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        InMemoryAuditSink(capacity=0)


# ── LoggingSink ─────────────────────────────────────────────────────────


async def test_logging_sink_writes_json(caplog):
    sink = LoggingAuditSink(logger_name="ahp.audit.test", level=logging.INFO)
    caplog.set_level(logging.INFO, logger="ahp.audit.test")
    await sink.emit(AuditEvent(op="engine.send", code="x.y", verb="SEND"))
    records = [r for r in caplog.records if r.name == "ahp.audit.test"]
    assert len(records) == 1
    payload = json.loads(records[0].getMessage())
    assert payload["op"] == "engine.send"
    assert payload["verb"] == "SEND"


# ── MultiSink ───────────────────────────────────────────────────────────


async def test_multi_sink_fans_out():
    a = InMemoryAuditSink()
    b = InMemoryAuditSink()
    multi = MultiSink([a, b])
    await multi.emit(AuditEvent(op="x"))
    assert len(a) == 1 and len(b) == 1


async def test_multi_sink_swallows_exceptions(caplog):
    class _Broken:
        async def emit(self, event: AuditEvent) -> None:
            raise RuntimeError("boom")

    good = InMemoryAuditSink()
    multi = MultiSink([_Broken(), good])
    caplog.set_level(logging.ERROR, logger="ahp.audit.sinks")
    await multi.emit(AuditEvent(op="x"))
    # The good sink still received the event.
    assert len(good) == 1
    assert any("audit sink" in r.getMessage() for r in caplog.records)


async def test_null_sink_drops():
    sink = NullAuditSink()
    await sink.emit(AuditEvent(op="x"))


def test_audit_sink_is_runtime_checkable():
    assert isinstance(NullAuditSink(), AuditSink)
    assert isinstance(InMemoryAuditSink(), AuditSink)


# ── Registry emits ──────────────────────────────────────────────────────


async def test_registry_emits_on_register_and_deregister(redis_client):
    sink = InMemoryAuditSink()
    principal = Principal(id="alice")
    registry = AgentRegistry(
        redis_client, audit=sink, principal=principal,
    )
    addr = AgentAddress.parse("acme.researcher.fin.eq.s.session.0")
    await registry.register(addr)
    await registry.heartbeat(addr)
    await registry.deregister(addr)
    ops = [e.op for e in sink.events]
    assert ops == [
        "registry.register",
        "registry.heartbeat",
        "registry.deregister",
    ]
    assert all(e.principal == "alice" for e in sink.events)
    assert all(e.target == str(addr) for e in sink.events)
    assert all(e.success for e in sink.events)


async def test_registry_emits_on_unauthorized_register(redis_client):
    sink = InMemoryAuditSink()
    # Alice can only claim her own org.
    principal = Principal.with_claims("alice", "alice.*.*.*.*.*.*")
    registry = AgentRegistry(
        redis_client,
        audit=sink,
        principal=principal,
        policy=AddressClaimPolicy(),
    )
    forbidden = AgentAddress.parse("bob.researcher.fin.eq.s.session.0")
    with pytest.raises(UnauthorizedRegistrationError):
        await registry.register(forbidden)
    assert len(sink.events) == 1
    e = sink.events[0]
    assert e.op == "registry.register"
    assert e.success is False
    assert e.error == "UnauthorizedRegistrationError"


async def test_registry_heartbeat_unregistered_emits_failure(redis_client):
    sink = InMemoryAuditSink()
    registry = AgentRegistry(redis_client, audit=sink)
    addr = AgentAddress.parse("acme.researcher.fin.eq.s.session.0")
    ok = await registry.heartbeat(addr)
    assert ok is False
    assert len(sink.events) == 1
    assert sink.events[0].error == "NotRegistered"


# ── Engine emits ────────────────────────────────────────────────────────


async def test_engine_emits_on_send_to_dead_target(stack):
    sink = InMemoryAuditSink()
    stack.engine.audit = sink
    src = AgentAddress.parse("acme.r.fin.eq.s.session.src")
    dst = AgentAddress.parse("acme.r.fin.eq.s.session.dst")
    msg = Message(
        source=src, target=dst, code=Code.INTERVIEW_TEXT,
        verb="SEND", body={"q": "x"}, thread="t::1",
    )
    delivered = await stack.engine.handle(msg)
    assert delivered == 0
    assert len(sink.events) == 1
    e = sink.events[0]
    assert e.op == "engine.send"
    assert e.verb == "SEND"
    assert e.code == Code.INTERVIEW_TEXT
    assert e.success is True
    assert e.extra == {"count": 0}


async def test_engine_emits_on_dispatch_error(stack):
    """An invalid verb path raises — the failure should be observed."""
    sink = InMemoryAuditSink()
    stack.engine.audit = sink
    src = AgentAddress.parse("acme.r.fin.eq.s.session.src")
    # INVALIDATE needs an AddressPattern target; pass a concrete address
    # to provoke InvalidTargetTypeError after _handle_invalidate runs.
    dst = AgentAddress.parse("acme.r.fin.eq.s.session.dst")
    msg = Message(
        source=src, target=dst, code=Code.INTERVIEW_TEXT,
        verb="INVALIDATE", body={}, thread="t::1",
    )
    with pytest.raises(Exception):
        await stack.engine.handle(msg)
    assert len(sink.events) == 1
    assert sink.events[0].success is False
    assert sink.events[0].error is not None


# ── CloudWatch sink ─────────────────────────────────────────────────────


class _FakeLogsClient:
    """Stand-in for boto3.client('logs')."""

    def __init__(self, *, fail_first_put: bool = False) -> None:
        self.created_groups: list[str] = []
        self.created_streams: list[tuple[str, str]] = []
        self.put_batches: list[dict] = []
        self._fail_first_put = fail_first_put
        self._puts = 0

    class exceptions:  # noqa: N801 — mimics boto3 shape
        class ResourceAlreadyExistsException(Exception):
            pass

    def create_log_group(self, *, logGroupName):
        self.created_groups.append(logGroupName)

    def create_log_stream(self, *, logGroupName, logStreamName):
        self.created_streams.append((logGroupName, logStreamName))

    def put_log_events(self, *, logGroupName, logStreamName, logEvents):
        self._puts += 1
        if self._fail_first_put and self._puts == 1:
            raise RuntimeError("transient")
        self.put_batches.append({
            "group": logGroupName,
            "stream": logStreamName,
            "events": list(logEvents),
        })


async def test_cloudwatch_sink_buffers_and_flushes():
    from ahp.audit.cloudwatch import CloudWatchLogsSink

    client = _FakeLogsClient()
    sink = CloudWatchLogsSink(
        log_group="/ahp/test", log_stream="s1",
        client=client, batch_size=3, flush_interval=60,
    )
    await sink.emit(AuditEvent(op="x.1"))
    await sink.emit(AuditEvent(op="x.2"))
    assert client.put_batches == []  # not yet at batch_size
    await sink.emit(AuditEvent(op="x.3"))
    # Hit batch_size → flushed.
    assert len(client.put_batches) == 1
    batch = client.put_batches[0]
    assert batch["group"] == "/ahp/test"
    assert batch["stream"] == "s1"
    assert len(batch["events"]) == 3
    # Each event is a JSON string with an ms timestamp.
    for evt in batch["events"]:
        assert isinstance(evt["timestamp"], int)
        json.loads(evt["message"])  # parses


async def test_cloudwatch_sink_creates_group_and_stream_once():
    from ahp.audit.cloudwatch import CloudWatchLogsSink

    client = _FakeLogsClient()
    sink = CloudWatchLogsSink(
        log_group="/ahp/test", log_stream="s1",
        client=client, batch_size=1, flush_interval=60,
    )
    await sink.emit(AuditEvent(op="x.1"))
    await sink.emit(AuditEvent(op="x.2"))
    assert client.created_groups == ["/ahp/test"]
    assert client.created_streams == [("/ahp/test", "s1")]


async def test_cloudwatch_sink_tolerates_already_exists():
    from ahp.audit.cloudwatch import CloudWatchLogsSink

    client = _FakeLogsClient()
    AlreadyExists = client.exceptions.ResourceAlreadyExistsException

    def raising_create_group(*, logGroupName):
        raise AlreadyExists("already there")
    def raising_create_stream(*, logGroupName, logStreamName):
        raise AlreadyExists("already there")

    client.create_log_group = raising_create_group  # type: ignore[method-assign]
    client.create_log_stream = raising_create_stream  # type: ignore[method-assign]

    sink = CloudWatchLogsSink(
        log_group="/ahp/test", log_stream="s1",
        client=client, batch_size=1, flush_interval=60,
    )
    # Should not raise.
    await sink.emit(AuditEvent(op="x.1"))
    assert len(client.put_batches) == 1


async def test_cloudwatch_sink_rebuffers_on_failure(caplog):
    from ahp.audit.cloudwatch import CloudWatchLogsSink

    client = _FakeLogsClient(fail_first_put=True)
    sink = CloudWatchLogsSink(
        log_group="/ahp/test", log_stream="s1",
        client=client, batch_size=1, flush_interval=60,
    )
    caplog.set_level(logging.ERROR, logger="ahp.audit.cloudwatch")
    await sink.emit(AuditEvent(op="x.1"))
    # First put failed; event is back in the buffer; nothing delivered.
    assert client.put_batches == []
    # Second emit (next event) triggers another flush which now succeeds
    # and clears both buffered entries.
    await sink.emit(AuditEvent(op="x.2"))
    assert len(client.put_batches) == 1
    assert len(client.put_batches[0]["events"]) == 2
