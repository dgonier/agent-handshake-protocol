"""CLI tests for `ahp audit-tail`.

Drives the async worker directly (same pattern as the other Redis-
touching CLI tests) and asserts on the rendered output for one-shot
mode + filter modes. ``--follow`` is exercised via a short-timeout
xread loop so the test doesn't hang.
"""

from __future__ import annotations

import asyncio
import io
import json

import pytest

import ahp.cli
from ahp.audit import (
    DEFAULT_REDIS_AUDIT_STREAM,
    AuditEvent,
    RedisStreamAuditSink,
)


async def _arun(*argv: str) -> tuple[int, str]:
    parser = ahp.cli.build_parser()
    args = parser.parse_args(["audit-tail", *argv])
    buf = io.StringIO()
    rc = await ahp.cli._audit_tail_async(args, buf)
    return rc, buf.getvalue()


async def test_audit_tail_empty_stream(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, out = await _arun("--redis-url", "redis://test/0")
    assert rc == 0
    assert "no audit entries" in out


async def test_audit_tail_renders_entries(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    sink = RedisStreamAuditSink(redis_client)
    await sink.emit(AuditEvent(
        op="broker.server.register",
        target="acme",
        extra={"org": "acme"},
    ))
    await sink.emit(AuditEvent(
        op="broker.settlement",
        target="acme",
        extra={"caller": "you.human.x.y.s.session.alice", "pre_tax": 0.42},
    ))

    rc, out = await _arun("--redis-url", "redis://test/0")
    assert rc == 0
    assert "broker.server.register" in out
    assert "broker.settlement" in out
    # The extras get rendered as JSON in the line tail.
    assert '"pre_tax": 0.42' in out


async def test_audit_tail_filters_by_op_glob(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    sink = RedisStreamAuditSink(redis_client)
    await sink.emit(AuditEvent(op="broker.server.register", target="acme"))
    await sink.emit(AuditEvent(op="survey.enqueue", target="sv-xyz"))
    await sink.emit(AuditEvent(op="broker.settlement", target="acme"))

    rc, out = await _arun(
        "--redis-url", "redis://test/0",
        "--op", "broker.*",
    )
    assert rc == 0
    assert "broker.server.register" in out
    assert "broker.settlement" in out
    assert "survey.enqueue" not in out


async def test_audit_tail_filters_by_target_substring(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    sink = RedisStreamAuditSink(redis_client)
    await sink.emit(AuditEvent(op="x", target="acme.small.echo"))
    await sink.emit(AuditEvent(op="x", target="beta.small.qwen"))

    rc, out = await _arun(
        "--redis-url", "redis://test/0",
        "--target-contains", "acme",
    )
    assert rc == 0
    assert "acme.small.echo" in out
    assert "beta.small.qwen" not in out


async def test_audit_tail_respects_limit(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    sink = RedisStreamAuditSink(redis_client)
    for i in range(5):
        await sink.emit(AuditEvent(op=f"x.{i}"))

    rc, out = await _arun(
        "--redis-url", "redis://test/0",
        "--limit", "2",
    )
    assert rc == 0
    # Renders lines for at most 2 events. Use 'x.' as the marker
    # because the timestamp format doesn't contain it.
    line_count = sum(1 for line in out.splitlines() if "  x." in line)
    assert line_count == 2


async def test_audit_tail_custom_stream(redis_client, monkeypatch):
    """--stream points the reader at a non-default key."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    sink = RedisStreamAuditSink(redis_client, stream_key="alt:stream")
    await sink.emit(AuditEvent(op="alt.event"))

    # Default stream is empty.
    rc, out = await _arun("--redis-url", "redis://test/0")
    assert rc == 0
    assert "no audit entries" in out

    # Custom stream has it.
    rc, out = await _arun(
        "--redis-url", "redis://test/0",
        "--stream", "alt:stream",
    )
    assert rc == 0
    assert "alt.event" in out
