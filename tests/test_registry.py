"""Async tests for AgentRegistry."""

from __future__ import annotations

import asyncio

import pytest

from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern
from ahp.registry.registry import AgentMeta, AgentRegistry
from ahp.transport.keys import Keys


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


@pytest.fixture
async def registry(redis_client):
    return AgentRegistry(redis_client, heartbeat_ttl=30)


# ── lifecycle ───────────────────────────────────────────────────────────


async def test_register_then_get(registry: AgentRegistry):
    addr = _addr("demo.adversarial.finance.equities.s.session.frank")
    meta = AgentMeta(
        capabilities=["debate", "rebut"], reputation=0.8,
        description="bear case generator",
    )
    await registry.register(addr, meta)

    stored = await registry.get(addr)
    assert stored is not None
    assert stored.capabilities == ["debate", "rebut"]
    assert stored.reputation == 0.8
    assert stored.description == "bear case generator"
    assert await registry.is_alive(addr)


async def test_register_without_meta_uses_default(registry: AgentRegistry):
    addr = _addr("demo.adversarial.finance.equities.s.session.frank")
    await registry.register(addr)
    meta = await registry.get(addr)
    assert meta is not None
    assert meta.reputation == 0.0
    assert meta.capabilities == []


async def test_deregister_clears_entry_and_liveness(registry: AgentRegistry):
    addr = _addr("demo.adversarial.finance.equities.s.session.frank")
    await registry.register(addr)
    await registry.deregister(addr)

    assert await registry.get(addr) is None
    assert not await registry.is_alive(addr)


async def test_heartbeat_refreshes_liveness(registry: AgentRegistry, redis_client):
    addr = _addr("demo.adversarial.finance.equities.s.session.frank")
    await registry.register(addr)
    # Set a short TTL so we can verify refresh actually extends it.
    await redis_client.expire(Keys.alive_key(addr), 1)
    ttl_before = await redis_client.ttl(Keys.alive_key(addr))
    assert ttl_before <= 1

    refreshed = await registry.heartbeat(addr)
    assert refreshed is True

    ttl_after = await redis_client.ttl(Keys.alive_key(addr))
    assert ttl_after > ttl_before


async def test_heartbeat_unknown_agent_returns_false(registry: AgentRegistry):
    addr = _addr("demo.adversarial.finance.equities.s.session.ghost")
    refreshed = await registry.heartbeat(addr)
    assert refreshed is False


async def test_invalid_heartbeat_ttl_rejected(redis_client):
    with pytest.raises(ValueError):
        AgentRegistry(redis_client, heartbeat_ttl=0)


# ── liveness expiry ─────────────────────────────────────────────────────


async def test_resolve_alive_only_excludes_expired(registry: AgentRegistry, redis_client):
    addr = _addr("demo.adversarial.finance.equities.s.session.frank")
    await registry.register(addr)
    # Forcibly delete the liveness marker to simulate expiry.
    await redis_client.delete(Keys.alive_key(addr))

    alive = await registry.resolve(
        AddressPattern.parse("*.*.*.*.*.*.*"), alive_only=True,
    )
    assert alive == []

    everyone = await registry.resolve(
        AddressPattern.parse("*.*.*.*.*.*.*"), alive_only=False,
    )
    assert addr in everyone


# ── pattern resolution ─────────────────────────────────────────────────


async def test_resolve_filters_by_pattern(registry: AgentRegistry):
    addrs = [
        _addr("demo.adversarial.finance.equities.s.session.frank"),
        _addr("demo.adversarial.science.biology.s.session.zoe"),
        _addr("demo.collaborative.finance.equities.s.session.alice"),
        _addr("public.human.general.x.s.session.devin"),
    ]
    for a in addrs:
        await registry.register(a)

    fin_adv = await registry.resolve(
        AddressPattern.parse("*.adversarial.finance.*.*.*.*"),
    )
    assert fin_adv == [addrs[0]]

    any_adv = await registry.resolve(
        AddressPattern.parse("*.adversarial.*.*.*.*.*"),
    )
    assert sorted(map(str, any_adv)) == sorted(map(str, [addrs[0], addrs[1]]))

    everyone = await registry.resolve(AddressPattern.all())
    assert len(everyone) == len(addrs)


async def test_resolve_accept_subset_semantics(registry: AgentRegistry):
    a = _addr("demo.adversarial.finance.equities.s.session.frank")
    b = _addr("demo.adversarial.finance.equities.sj.session.gertrude")
    c = _addr("demo.adversarial.finance.equities.j.session.henri")
    for x in (a, b, c):
        await registry.register(x)

    # Pattern requires 's' — only those that include 's' in accept match.
    matches = await registry.resolve(
        AddressPattern.parse("*.adversarial.*.*.s.*.*"),
    )
    found = sorted(str(m) for m in matches)
    assert found == sorted([str(a), str(b)])


# ── discovery ───────────────────────────────────────────────────────────


async def test_discover_by_capability_and_reputation(registry: AgentRegistry):
    expert = _addr("demo.adversarial.finance.equities.s.session.frank")
    novice = _addr("demo.adversarial.finance.equities.s.session.greg")
    await registry.register(expert, AgentMeta(
        capabilities=["debate", "valuation"], reputation=0.9,
    ))
    await registry.register(novice, AgentMeta(
        capabilities=["debate"], reputation=0.1,
    ))

    high_rep = await registry.discover(
        role="adversarial", domain="finance", min_reputation=0.5,
    )
    assert len(high_rep) == 1
    assert high_rep[0][0] == expert

    valuation = await registry.discover(
        role="adversarial", capability="valuation",
    )
    assert [a for a, _ in valuation] == [expert]


async def test_discover_returns_metadata(registry: AgentRegistry):
    addr = _addr("demo.adversarial.finance.equities.s.session.frank")
    await registry.register(addr, AgentMeta(capabilities=["x"], reputation=0.5))
    results = await registry.discover(role="adversarial")
    assert len(results) == 1
    _, meta = results[0]
    assert meta.capabilities == ["x"]
    assert meta.reputation == 0.5


# ── enumeration ─────────────────────────────────────────────────────────


async def test_count_total_vs_alive(registry: AgentRegistry, redis_client):
    addrs = [
        _addr(f"demo.adversarial.finance.equities.s.session.a{i}")
        for i in range(3)
    ]
    for a in addrs:
        await registry.register(a)

    assert await registry.count() == 3
    assert await registry.count(alive_only=True) == 3

    # Kill one liveness marker.
    await redis_client.delete(Keys.alive_key(addrs[0]))
    assert await registry.count() == 3
    assert await registry.count(alive_only=True) == 2


async def test_list_all(registry: AgentRegistry):
    addrs = [
        _addr(f"demo.adversarial.finance.equities.s.session.a{i}")
        for i in range(2)
    ]
    for a in addrs:
        await registry.register(a)
    listed = await registry.list_all()
    assert sorted(map(str, listed)) == sorted(map(str, addrs))
