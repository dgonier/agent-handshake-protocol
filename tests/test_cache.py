"""Async tests for ProtocolCache."""

from __future__ import annotations

import asyncio

import pytest

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import LIFECYCLE_TTL, Message
from ahp.core.pattern import AddressPattern
from ahp.transport.cache import ProtocolCache


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


def _request(target: str = "demo.adversarial.finance.equities.s.longterm.frank",
             code: str = Code.INTERVIEW_TEXT,
             source: str = "demo.collaborative.finance.equities.s.session.alice",
             body=None) -> Message:
    return Message(
        source=_addr(source), target=_addr(target), verb="SEND-GET",
        code=code, thread="thread::cache", body=body,
    )


def _response_to(req: Message, body: str) -> Message:
    if isinstance(req.target, AddressPattern):
        raise AssertionError
    return Message(
        source=req.target,
        target=req.source,
        verb="SEND",
        code=req.code,
        thread=req.thread,
        body=body,
    )


@pytest.fixture
async def cache(redis_client):
    return ProtocolCache(redis_client)


# ── basic put / get ─────────────────────────────────────────────────────


async def test_put_then_get_returns_response(cache: ProtocolCache):
    req = _request()
    resp = _response_to(req, "the answer")
    stored = await cache.put(req, resp)
    assert stored is True

    cached = await cache.get(req)
    assert cached is not None
    assert cached.body == "the answer"


async def test_miss_returns_none(cache: ProtocolCache):
    assert await cache.get(_request()) is None


# ── lifecycle-derived TTL ───────────────────────────────────────────────


async def test_ephemeral_lifecycle_not_cached(cache: ProtocolCache):
    req = _request(target="demo.adversarial.finance.equities.s.ephemeral.frank")
    stored = await cache.put(req, _response_to(req, "x"))
    assert stored is False
    assert await cache.get(req) is None


async def test_ttl_is_set_from_lifecycle(cache: ProtocolCache, redis_client):
    req = _request(target="demo.adversarial.finance.equities.s.session.frank")
    await cache.put(req, _response_to(req, "x"))
    from ahp.transport.keys import Keys
    key = Keys.cache_key(ProtocolCache.derive_key(req))
    ttl = await redis_client.ttl(key)
    assert 0 < ttl <= LIFECYCLE_TTL["session"]


async def test_pattern_target_not_cached(cache: ProtocolCache):
    req = Message(
        source=_addr("o.r.d.sd.s.session.i"),
        target=AddressPattern.parse("*.*.*.*.*.*.*"),
        verb="CAST-GET", code=Code.INTERVIEW_TEXT, thread="t", body=None,
    )
    resp = Message(
        source=_addr("o.r.d.sd.s.session.x"),
        target=_addr("o.r.d.sd.s.session.i"),
        verb="SEND", code=Code.INTERVIEW_TEXT, thread="t", body="x",
    )
    stored = await cache.put(req, resp)
    assert stored is False
    with pytest.raises(ValueError):
        await cache.get(req)


# ── key derivation ──────────────────────────────────────────────────────


async def test_same_request_same_key(cache: ProtocolCache):
    a = _request()
    b = _request()
    assert ProtocolCache.derive_key(a) == ProtocolCache.derive_key(b)


async def test_different_code_different_key(cache: ProtocolCache):
    a = _request(code=Code.INTERVIEW_TEXT)
    b = _request(code=Code.INTERVIEW_SCHEMA)
    assert ProtocolCache.derive_key(a) != ProtocolCache.derive_key(b)


async def test_param_order_irrelevant_for_key(cache: ProtocolCache):
    t1 = "demo.adversarial.finance.equities.s.longterm.frank?a=1&b=2"
    t2 = "demo.adversarial.finance.equities.s.longterm.frank?b=2&a=1"
    a = _request(target=t1)
    b = _request(target=t2)
    assert ProtocolCache.derive_key(a) == ProtocolCache.derive_key(b)


async def test_distinct_bodies_yield_distinct_keys(cache: ProtocolCache):
    """Different request bodies must NOT collide on the same cache slot."""
    a = _request(body="Tesla")
    b = _request(body="Apple")
    assert ProtocolCache.derive_key(a) != ProtocolCache.derive_key(b)


async def test_dict_body_order_irrelevant_for_key(cache: ProtocolCache):
    a = _request(body={"horizon": 12, "ticker": "Tesla"})
    b = _request(body={"ticker": "Tesla", "horizon": 12})
    assert ProtocolCache.derive_key(a) == ProtocolCache.derive_key(b)


async def test_distinct_bodies_dont_share_cached_response(cache: ProtocolCache):
    """End-to-end: two different queries against the same (target, code)
    must return their respective responses, not the first-seen answer."""
    req_tesla = _request(body="Tesla")
    req_apple = _request(body="Apple")
    await cache.put(req_tesla, _response_to(req_tesla, "tesla-answer"))
    await cache.put(req_apple, _response_to(req_apple, "apple-answer"))

    tesla_hit = await cache.get(req_tesla)
    apple_hit = await cache.get(req_apple)
    assert tesla_hit is not None and tesla_hit.body == "tesla-answer"
    assert apple_hit is not None and apple_hit.body == "apple-answer"


# ── invalidation ────────────────────────────────────────────────────────


async def test_invalidate_by_pattern(cache: ProtocolCache):
    r_finance = _request(target="demo.adversarial.finance.equities.s.longterm.frank")
    r_science = _request(target="demo.adversarial.science.biology.s.longterm.frank")
    await cache.put(r_finance, _response_to(r_finance, "fin"))
    await cache.put(r_science, _response_to(r_science, "sci"))

    n = await cache.invalidate(AddressPattern.parse("*.*.finance.*.*.*.*"))
    assert n == 1
    assert await cache.get(r_finance) is None
    assert await cache.get(r_science) is not None


async def test_invalidate_by_params(cache: ProtocolCache):
    r_tesla = _request(target="demo.adversarial.finance.equities.s.longterm.frank?stock=Tesla")
    r_ford = _request(target="demo.adversarial.finance.equities.s.longterm.frank?stock=Ford")
    await cache.put(r_tesla, _response_to(r_tesla, "tesla"))
    await cache.put(r_ford, _response_to(r_ford, "ford"))

    n = await cache.invalidate(
        AddressPattern.parse("*.adversarial.finance.*.*.*.*"),
        params={"stock": "Tesla"},
    )
    assert n == 1
    assert await cache.get(r_tesla) is None
    assert await cache.get(r_ford) is not None


async def test_invalidate_no_match_returns_zero(cache: ProtocolCache):
    req = _request()
    await cache.put(req, _response_to(req, "x"))
    n = await cache.invalidate(AddressPattern.parse("other.*.*.*.*.*.*"))
    assert n == 0
    assert await cache.get(req) is not None


async def test_clear_wipes_everything(cache: ProtocolCache):
    for i in range(3):
        req = _request(target=f"demo.adversarial.finance.equities.s.longterm.agent-{i}")
        await cache.put(req, _response_to(req, str(i)))
    n = await cache.clear()
    assert n == 3
