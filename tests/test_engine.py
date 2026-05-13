"""Tests for ProtocolEngine — verb dispatch over the full stack."""

from __future__ import annotations

import asyncio

import pytest

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern
from ahp.engine.errors import IncompatibleTargetError, InvalidTargetTypeError
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


ALICE_URI = "demo.collaborative.finance.equities.s.session.alice"
BOB_URI = "demo.adversarial.finance.equities.s.longterm.bob"
CAROL_URI = "demo.adversarial.finance.equities.s.session.carol"
JBOT_URI = "demo.interview.finance.equities.j.session.jbot"


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


def _msg(
    *,
    source: str = ALICE_URI,
    target: str | AddressPattern = BOB_URI,
    verb: str = "SEND",
    code: str = Code.INTERVIEW_TEXT,
    thread: str = "thread::test",
    body=None,
) -> Message:
    tgt = target if isinstance(target, AddressPattern) else _addr(target)
    return Message(
        source=_addr(source), target=tgt, verb=verb,
        code=code, thread=thread, body=body,
    )


async def _register(stack, *uris, reputation=0.0):
    for u in uris:
        await stack.registry.register(_addr(u), AgentMeta(reputation=reputation))


# ── SEND ────────────────────────────────────────────────────────────────


async def test_send_to_live_target_delivers(stack):
    await _register(stack, BOB_URI)
    sub = await stack.bus.listen(_addr(BOB_URI))
    await asyncio.sleep(0.01)
    try:
        delivered = await stack.engine.handle(_msg(body="hi"))
        assert delivered >= 1
        received = await sub.get_one(timeout=1.0)
        assert received is not None and received.body == "hi"
    finally:
        await sub.close()


async def test_send_to_unregistered_target_returns_zero(stack):
    delivered = await stack.engine.handle(_msg(body="hi"))
    assert delivered == 0


async def test_send_rejects_pattern_target(stack):
    msg = _msg(target=AddressPattern.parse("*.*.*.*.*.*.*"), verb="CAST")
    msg.verb = "SEND"  # type: ignore[assignment]
    with pytest.raises(InvalidTargetTypeError):
        await stack.engine.handle(msg)


async def test_send_blocked_by_compatibility(stack):
    # bob accepts only 's', but interview.schema requires 'j'
    await _register(stack, BOB_URI)
    with pytest.raises(IncompatibleTargetError):
        await stack.engine.handle(
            _msg(code=Code.INTERVIEW_SCHEMA, body={"k": "v"})
        )


# ── SEND-GET ────────────────────────────────────────────────────────────


async def _spawn_autoresponder(stack, addr_uri: str, body_fn):
    """Spin up a background task that replies to one message addressed to ``addr_uri``."""
    addr = _addr(addr_uri)
    ready = asyncio.Event()

    async def run():
        sub = await stack.bus.listen(addr)
        ready.set()
        try:
            req = await sub.get_one(timeout=2.0)
            if req is None:
                return
            resp = Message(
                source=addr, target=req.source, verb="SEND",
                code=req.code, thread=req.thread, body=body_fn(req),
            )
            await stack.bus.send_reply(req, resp)
        finally:
            await sub.close()

    task = asyncio.create_task(run())
    await ready.wait()
    return task


async def test_send_get_returns_response_and_caches(stack):
    await _register(stack, BOB_URI)
    task = await _spawn_autoresponder(stack, BOB_URI, lambda r: f"reply:{r.body}")
    try:
        reply = await stack.engine.handle(
            _msg(verb="SEND-GET", body="q1"), timeout=2.0,
        )
        assert reply is not None
        assert reply.body == "reply:q1"
    finally:
        await task

    # Second identical request hits cache — no responder needed.
    cached = await stack.engine.handle(
        _msg(verb="SEND-GET", body="q1"), timeout=0.2,
    )
    assert cached is not None
    assert cached.body == "reply:q1"


async def test_send_get_dead_target_returns_none(stack):
    # Not registered ⇒ not alive ⇒ engine skips the bus call.
    result = await stack.engine.handle(
        _msg(verb="SEND-GET", body="q"), timeout=0.3,
    )
    assert result is None


async def test_send_get_does_not_cache_ephemeral_target(stack):
    ephemeral_uri = "demo.adversarial.finance.equities.s.ephemeral.flash"
    await _register(stack, ephemeral_uri)
    task = await _spawn_autoresponder(stack, ephemeral_uri, lambda r: "once")
    try:
        first = await stack.engine.handle(
            _msg(target=ephemeral_uri, verb="SEND-GET", body="q"), timeout=2.0,
        )
        assert first is not None
        assert first.body == "once"
    finally:
        await task

    # No responder this time — cache should be empty for ephemeral lifecycle.
    second = await stack.engine.handle(
        _msg(target=ephemeral_uri, verb="SEND-GET", body="q"), timeout=0.2,
    )
    assert second is None


async def test_send_get_compatibility_violation_raises(stack):
    await _register(stack, BOB_URI)
    with pytest.raises(IncompatibleTargetError):
        await stack.engine.handle(
            _msg(verb="SEND-GET", code=Code.INTERVIEW_SCHEMA, body={"k": "v"}),
        )


# ── CAST ────────────────────────────────────────────────────────────────


async def test_cast_fans_out_through_registry(stack):
    await _register(stack, BOB_URI, CAROL_URI)
    sub_b = await stack.bus.listen(_addr(BOB_URI))
    sub_c = await stack.bus.listen(_addr(CAROL_URI))
    await asyncio.sleep(0.01)
    try:
        delivered = await stack.engine.handle(_msg(
            target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
            verb="CAST",
            body="hey",
        ))
        assert delivered >= 2
        rb = await sub_b.get_one(timeout=1.0)
        rc = await sub_c.get_one(timeout=1.0)
        assert rb is not None and rb.body == "hey"
        assert rc is not None and rc.body == "hey"
    finally:
        await sub_b.close()
        await sub_c.close()


async def test_cast_filters_incompatible_targets(stack):
    # jbot accepts j only; carol accepts s. Code requires s → only carol.
    await _register(stack, CAROL_URI, JBOT_URI)
    sub_c = await stack.bus.listen(_addr(CAROL_URI))
    sub_j = await stack.bus.listen(_addr(JBOT_URI))
    await asyncio.sleep(0.01)
    try:
        delivered = await stack.engine.handle(_msg(
            target=AddressPattern.parse("*.*.finance.equities.*.*.*"),
            verb="CAST",
            code=Code.INTERVIEW_TEXT,  # requires s
            body="hi",
        ))
        # Only carol satisfies; pattern matches both but matrix filters jbot.
        assert delivered == 1
        got_c = await sub_c.get_one(timeout=1.0)
        got_j = await sub_j.get_one(timeout=0.2)
        assert got_c is not None
        assert got_j is None
    finally:
        await sub_c.close()
        await sub_j.close()


async def test_cast_excludes_dead_agents(stack):
    await _register(stack, BOB_URI, CAROL_URI)
    # Kill carol's liveness marker.
    from ahp.transport.keys import Keys
    await stack.redis.delete(Keys.alive_key(_addr(CAROL_URI)))

    sub_b = await stack.bus.listen(_addr(BOB_URI))
    sub_c = await stack.bus.listen(_addr(CAROL_URI))
    await asyncio.sleep(0.01)
    try:
        delivered = await stack.engine.handle(_msg(
            target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
            verb="CAST",
            body="hi",
        ))
        assert delivered == 1
        got_b = await sub_b.get_one(timeout=1.0)
        got_c = await sub_c.get_one(timeout=0.2)
        assert got_b is not None
        assert got_c is None
    finally:
        await sub_b.close()
        await sub_c.close()


async def test_cast_no_matches_returns_zero(stack):
    delivered = await stack.engine.handle(_msg(
        target=AddressPattern.parse("nobody.*.*.*.*.*.*"),
        verb="CAST",
    ))
    assert delivered == 0


# ── CAST-GET ────────────────────────────────────────────────────────────


async def test_cast_get_collects_responses(stack):
    await _register(stack, BOB_URI, CAROL_URI)

    ready = asyncio.Barrier(3)

    async def auto_reply(uri: str, body: str):
        addr = _addr(uri)
        sub = await stack.bus.listen(addr)
        await ready.wait()
        try:
            req = await sub.get_one(timeout=2.0)
            assert req is not None
            resp = Message(
                source=addr, target=req.source, verb="SEND",
                code=Code.ADVERSARIAL_DEBATE, thread=req.thread, body=body,
            )
            await stack.bus.send_reply(req, resp)
        finally:
            await sub.close()

    t1 = asyncio.create_task(auto_reply(BOB_URI, "bob"))
    t2 = asyncio.create_task(auto_reply(CAROL_URI, "carol"))
    try:
        await ready.wait()
        replies = await stack.engine.handle(
            _msg(
                target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
                verb="CAST-GET",
                code=Code.ADVERSARIAL_DEBATE,
                body="argue",
            ),
            timeout=2.0,
        )
        assert sorted(r.body for r in replies) == ["bob", "carol"]
    finally:
        await asyncio.gather(t1, t2)


async def test_cast_get_max_responses_caps_collection(stack):
    await _register(stack, BOB_URI, CAROL_URI)

    ready = asyncio.Barrier(3)

    async def auto_reply(uri: str):
        addr = _addr(uri)
        sub = await stack.bus.listen(addr)
        await ready.wait()
        try:
            req = await sub.get_one(timeout=2.0)
            if req is None:
                return
            resp = Message(
                source=addr, target=req.source, verb="SEND",
                code=Code.ADVERSARIAL_DEBATE, thread=req.thread, body=uri,
            )
            await stack.bus.send_reply(req, resp)
        finally:
            await sub.close()

    t1 = asyncio.create_task(auto_reply(BOB_URI))
    t2 = asyncio.create_task(auto_reply(CAROL_URI))
    try:
        await ready.wait()
        replies = await stack.engine.handle(
            _msg(
                target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
                verb="CAST-GET",
                code=Code.ADVERSARIAL_DEBATE, body="argue",
            ),
            timeout=2.0,
            max_responses=1,
        )
        assert len(replies) == 1
    finally:
        await asyncio.gather(t1, t2, return_exceptions=True)


async def test_cast_get_empty_targets_returns_empty(stack):
    replies = await stack.engine.handle(
        _msg(
            target=AddressPattern.parse("nobody.*.*.*.*.*.*"),
            verb="CAST-GET",
            code=Code.ADVERSARIAL_DEBATE, body="x",
        ),
        timeout=0.3,
    )
    assert replies == []


# ── INVALIDATE ──────────────────────────────────────────────────────────


async def test_invalidate_clears_matching_cache(stack):
    # Pre-populate cache with two responses.
    await _register(stack, BOB_URI)
    task = await _spawn_autoresponder(stack, BOB_URI, lambda r: "answer")
    try:
        await stack.engine.handle(_msg(verb="SEND-GET", body="q"), timeout=2.0)
    finally:
        await task

    # Sanity: a second SEND-GET hits cache (no responder running).
    cached = await stack.engine.handle(_msg(verb="SEND-GET", body="q"), timeout=0.2)
    assert cached is not None

    # INVALIDATE all finance-adversarial cache entries.
    inv_msg = Message(
        source=_addr(ALICE_URI),
        target=AddressPattern.parse("*.adversarial.finance.*.*.*.*"),
        verb="INVALIDATE",
        code=Code.ERROR_INTERNAL,  # filler; INVALIDATE doesn't use code routing
        thread="thread::invalidate",
        body={},
    )
    invalidated = await stack.engine.handle(inv_msg)
    assert invalidated >= 1

    # Third call: cache miss + no responder ⇒ None.
    third = await stack.engine.handle(_msg(verb="SEND-GET", body="q"), timeout=0.2)
    assert third is None


async def test_invalidate_requires_pattern_target(stack):
    bad = _msg(target=BOB_URI, verb="INVALIDATE", body={})
    with pytest.raises(InvalidTargetTypeError):
        await stack.engine.handle(bad)


async def test_invalidate_with_params(stack):
    # Cache two different parameterized targets.
    bob_tesla = "demo.adversarial.finance.equities.s.longterm.bob?stock=Tesla"
    bob_ford = "demo.adversarial.finance.equities.s.longterm.bob?stock=Ford"
    await stack.registry.register(_addr(bob_tesla))
    await stack.registry.register(_addr(bob_ford))

    t1 = await _spawn_autoresponder(stack, bob_tesla, lambda r: "tesla-ans")
    try:
        await stack.engine.handle(
            _msg(target=bob_tesla, verb="SEND-GET", body="q"), timeout=2.0,
        )
    finally:
        await t1

    t2 = await _spawn_autoresponder(stack, bob_ford, lambda r: "ford-ans")
    try:
        await stack.engine.handle(
            _msg(target=bob_ford, verb="SEND-GET", body="q"), timeout=2.0,
        )
    finally:
        await t2

    inv = Message(
        source=_addr(ALICE_URI),
        target=AddressPattern.parse("*.adversarial.finance.*.*.*.*"),
        verb="INVALIDATE", code=Code.ERROR_INTERNAL,
        thread="thread::inv", body={"params": {"stock": "Tesla"}},
    )
    n = await stack.engine.handle(inv)
    assert n == 1
    # Tesla cache gone, Ford still served from cache.
    tesla_after = await stack.engine.handle(
        _msg(target=bob_tesla, verb="SEND-GET", body="q"), timeout=0.2,
    )
    ford_after = await stack.engine.handle(
        _msg(target=bob_ford, verb="SEND-GET", body="q"), timeout=0.2,
    )
    assert tesla_after is None
    assert ford_after is not None and ford_after.body == "ford-ans"


# ── CAST-SUB / spawn_thread ─────────────────────────────────────────────


async def test_cast_sub_not_implemented(stack):
    msg = _msg(
        target=AddressPattern.parse("*.*.*.*.*.*.*"),
        verb="CAST-SUB",
        body=None,
    )
    with pytest.raises(NotImplementedError):
        await stack.engine.handle(msg)


async def test_spawn_thread_creates_durable_thread(stack):
    initiator = _addr(ALICE_URI)
    tid = await stack.engine.spawn_thread("Tesla outlook", initiator)
    fetched = await stack.threads.get(tid)
    assert fetched is not None
    assert fetched.initiator == initiator
    assert "tesla-outlook" in tid


async def test_join_thread_adds_participant(stack):
    tid = await stack.engine.spawn_thread("topic", _addr(ALICE_URI))
    await stack.engine.join_thread(tid, _addr(BOB_URI))
    parts = sorted(map(str, await stack.threads.participants(tid)))
    assert str(_addr(BOB_URI)) in parts


# ── construction & defaults ─────────────────────────────────────────────


async def test_engine_falls_back_to_default_matrix_and_threads(redis_client):
    from ahp.registry.registry import AgentRegistry
    from ahp.transport.cache import ProtocolCache
    from ahp.transport.redis_bus import RedisBus

    bus = RedisBus(redis_client)
    try:
        engine = ProtocolEngine(
            bus,
            AgentRegistry(redis_client),
            ProtocolCache(redis_client),
        )
        assert engine.matrix is not None
        assert engine.threads is not None
        assert engine.default_timeout > 0
    finally:
        await bus.close()
