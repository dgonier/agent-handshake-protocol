"""Async tests for RedisBus over fakeredis."""

from __future__ import annotations

import asyncio

import pytest

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern
from ahp.transport.redis_bus import RedisBus, Subscription


ALICE = "demo.collaborative.finance.equities.s.session.alice"
BOB = "demo.adversarial.finance.equities.s.session.bob"
CAROL = "demo.adversarial.finance.equities.s.session.carol"


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


def _msg(
    *,
    source: str = ALICE,
    target: str | AddressPattern = BOB,
    verb: str = "SEND",
    code: str = Code.INTERVIEW_TEXT,
    thread: str = "thread::test",
    body=None,
) -> Message:
    tgt = target if isinstance(target, AddressPattern) else _addr(target)
    return Message(
        source=_addr(source),
        target=tgt,
        verb=verb,
        code=code,
        thread=thread,
        body=body,
    )


@pytest.fixture
async def bus(redis_client):
    b = RedisBus(redis_client)
    try:
        yield b
    finally:
        await b.close()


# ── thread history ──────────────────────────────────────────────────────


async def test_append_and_read_thread(bus: RedisBus):
    msgs = [_msg(body=f"hello-{i}") for i in range(3)]
    for m in msgs:
        await bus.append_thread(m)

    history = await bus.get_thread("thread::test")
    assert len(history) == 3
    assert [m.body for m in history] == ["hello-0", "hello-1", "hello-2"]


async def test_thread_length(bus: RedisBus):
    assert await bus.thread_length("thread::empty") == 0
    await bus.append_thread(_msg(body="x"))
    assert await bus.thread_length("thread::test") == 1


# ── point-to-point ──────────────────────────────────────────────────────


async def test_send_delivers_to_listener(bus: RedisBus):
    bob = _addr(BOB)
    sub: Subscription = await bus.listen(bob)
    await asyncio.sleep(0.01)
    try:
        msg = _msg(target=BOB, body="ping")
        delivered = await bus.send(msg)
        assert delivered >= 1

        received = await sub.get_one(timeout=1.0)
        assert received is not None
        assert received.body == "ping"
        assert received.source == _addr(ALICE)
        assert received.target == bob
    finally:
        await sub.close()


async def test_send_rejects_pattern_target(bus: RedisBus):
    pat = AddressPattern.parse("*.*.*.*.*.*.*")
    # Build a CAST-typed message (pattern target legal at Message level)
    msg = _msg(target=pat, verb="CAST")
    with pytest.raises(ValueError, match="AgentAddress"):
        await bus.send(msg)


async def test_send_get_returns_single_response(bus: RedisBus):
    bob = _addr(BOB)
    ready = asyncio.Event()

    async def responder():
        sub = await bus.listen(bob)
        ready.set()
        try:
            req = await sub.get_one(timeout=2.0)
            assert req is not None
            assert req.reply_to is not None
            resp = Message(
                source=bob, target=_addr(ALICE), verb="SEND",
                code=Code.INTERVIEW_TEXT, thread=req.thread, body=f"reply:{req.body}",
            )
            await bus.send_reply(req, resp)
        finally:
            await sub.close()

    task = asyncio.create_task(responder())
    try:
        await ready.wait()
        reply = await bus.send_get(_msg(target=BOB, verb="SEND-GET", body="q"), timeout=2.0)
        assert reply is not None
        assert reply.body == "reply:q"
    finally:
        await task


async def test_send_get_times_out(bus: RedisBus):
    # No listener — should hit timeout and return None.
    reply = await bus.send_get(
        _msg(target=BOB, verb="SEND-GET", body="hello"),
        timeout=0.3,
    )
    assert reply is None


# ── broadcast ───────────────────────────────────────────────────────────


async def test_cast_fans_out_to_targets(bus: RedisBus):
    bob, carol = _addr(BOB), _addr(CAROL)
    sub_b = await bus.listen(bob)
    sub_c = await bus.listen(carol)
    # Give the underlying pub/sub a beat to register both subscriptions.
    await asyncio.sleep(0.01)
    try:
        msg = _msg(
            target=AddressPattern.parse("*.adversarial.*.*.s.*.*"),
            verb="CAST",
            body="hey all",
        )
        delivered = await bus.cast(msg, [bob, carol])
        assert delivered >= 2

        rb = await sub_b.get_one(timeout=1.0)
        rc = await sub_c.get_one(timeout=1.0)
        assert rb is not None and rb.body == "hey all"
        assert rc is not None and rc.body == "hey all"
    finally:
        await sub_b.close()
        await sub_c.close()


async def test_cast_get_collects_multiple_responses(bus: RedisBus):
    bob, carol = _addr(BOB), _addr(CAROL)
    ready = asyncio.Barrier(3)  # bob + carol + main

    async def auto_reply(addr: AgentAddress, marker: str):
        sub = await bus.listen(addr)
        await ready.wait()
        try:
            req = await sub.get_one(timeout=2.0)
            assert req is not None
            resp = Message(
                source=addr, target=_addr(ALICE), verb="SEND",
                code=Code.ADVERSARIAL_DEBATE, thread=req.thread, body=marker,
            )
            await bus.send_reply(req, resp)
        finally:
            await sub.close()

    t1 = asyncio.create_task(auto_reply(bob, "bob-says"))
    t2 = asyncio.create_task(auto_reply(carol, "carol-says"))
    try:
        await ready.wait()
        request = _msg(
            target=AddressPattern.parse("*.adversarial.*.*.s.*.*"),
            verb="CAST-GET",
            code=Code.ADVERSARIAL_DEBATE,
            body="argue",
        )
        replies = await bus.cast_get(request, [bob, carol], timeout=2.0)
        bodies = sorted(r.body for r in replies)
        assert bodies == ["bob-says", "carol-says"]
    finally:
        await asyncio.gather(t1, t2)


async def test_cast_get_stops_at_max_responses(bus: RedisBus):
    bob, carol = _addr(BOB), _addr(CAROL)
    ready = asyncio.Barrier(3)

    async def auto_reply(addr: AgentAddress):
        sub = await bus.listen(addr)
        await ready.wait()
        try:
            req = await sub.get_one(timeout=2.0)
            if req is None:
                return
            resp = Message(
                source=addr, target=_addr(ALICE), verb="SEND",
                code=Code.ADVERSARIAL_DEBATE, thread=req.thread, body=str(addr),
            )
            await bus.send_reply(req, resp)
        finally:
            await sub.close()

    t1 = asyncio.create_task(auto_reply(bob))
    t2 = asyncio.create_task(auto_reply(carol))
    try:
        await ready.wait()
        request = _msg(
            target=AddressPattern.parse("*.adversarial.*.*.s.*.*"),
            verb="CAST-GET",
            code=Code.ADVERSARIAL_DEBATE, body="argue",
        )
        replies = await bus.cast_get(
            request, [bob, carol], timeout=2.0, max_responses=1,
        )
        assert len(replies) == 1
    finally:
        await asyncio.gather(t1, t2, return_exceptions=True)


# ── consume() — background handler tasks ────────────────────────────────


async def test_consume_dispatches_to_handler(bus: RedisBus):
    bob = _addr(BOB)
    received: list[Message] = []
    started = asyncio.Event()

    async def handler(msg: Message) -> None:
        received.append(msg)
        started.set()

    task = bus.consume(bob, handler)
    try:
        # Give consume() a beat to subscribe before we publish.
        await asyncio.sleep(0.05)
        await bus.send(_msg(target=BOB, body="hello-consume"))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert len(received) == 1
        assert received[0].body == "hello-consume"
    finally:
        task.cancel()


# ── serialization edge cases ────────────────────────────────────────────


async def test_dict_body_round_trips(bus: RedisBus):
    bob = _addr(BOB)
    sub = await bus.listen(bob)
    await asyncio.sleep(0.01)
    try:
        body = {"q": "Tesla", "horizon": 12, "tags": ["finance", "auto"]}
        await bus.send(_msg(target=BOB, body=body))
        received = await sub.get_one(timeout=1.0)
        assert received is not None
        assert received.body == body
    finally:
        await sub.close()


async def test_bytes_body_rejected_at_send(bus: RedisBus):
    with pytest.raises(TypeError, match="bytes"):
        await bus.send(_msg(target=BOB, body=b"raw-binary"))
