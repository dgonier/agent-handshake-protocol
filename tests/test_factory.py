"""Tests for AgentFactory — pattern-keyed construction + bulk spawning."""

from __future__ import annotations

import pytest

from ahp.adapters.base import AHPAgent
from ahp.adapters.factory import AgentFactory
from ahp.core.address import AgentAddress
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern


class _NoopAgent(AHPAgent):
    """Minimal agent that does nothing — only used to verify factory wiring."""

    label: str = "noop"

    async def handle_message(self, message: Message):
        return None


class _AdversarialAgent(_NoopAgent):
    label = "adversarial"


class _CollaborativeAgent(_NoopAgent):
    label = "collab"


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── registration + create ───────────────────────────────────────────────


def test_create_uses_first_matching_builder(stack):
    f = AgentFactory(stack.engine)
    f.register(
        AddressPattern.parse("*.adversarial.*.*.*.*.*"),
        lambda a, e, p: _AdversarialAgent(a, e),
    )
    f.register(
        AddressPattern.parse("*.collaborative.*.*.*.*.*"),
        lambda a, e, p: _CollaborativeAgent(a, e),
    )

    a1 = f.create("demo.adversarial.finance.equities.s.session.frank")
    a2 = f.create("demo.collaborative.finance.equities.s.session.alice")
    assert isinstance(a1, _AdversarialAgent)
    assert isinstance(a2, _CollaborativeAgent)
    assert a1.address.instance == "frank"
    assert a2.address.instance == "alice"


def test_priority_overrides_order(stack):
    f = AgentFactory(stack.engine)
    f.register(
        AddressPattern.parse("*.*.*.*.*.*.*"),
        lambda a, e, p: _NoopAgent(a, e),
        priority=0,
    )
    f.register(
        AddressPattern.parse("*.adversarial.*.*.*.*.*"),
        lambda a, e, p: _AdversarialAgent(a, e),
        priority=10,
    )

    addr = _addr("o.adversarial.d.sd.s.session.x")
    built = f.create(addr)
    assert isinstance(built, _AdversarialAgent)  # higher priority wins


def test_create_raises_on_no_match(stack):
    f = AgentFactory(stack.engine)
    f.register(
        AddressPattern.parse("*.adversarial.*.*.*.*.*"),
        lambda a, e, p: _AdversarialAgent(a, e),
    )
    with pytest.raises(LookupError):
        f.create("o.collaborative.d.sd.s.session.x")


def test_can_create(stack):
    f = AgentFactory(stack.engine)
    f.register(
        AddressPattern.parse("*.adversarial.*.*.*.*.*"),
        lambda a, e, p: _AdversarialAgent(a, e),
    )
    assert f.can_create("o.adversarial.d.sd.s.session.x")
    assert not f.can_create("o.collaborative.d.sd.s.session.x")


def test_register_accepts_string_pattern(stack):
    f = AgentFactory(stack.engine)
    f.register("*.adversarial.*.*.*.*.*", lambda a, e, p: _AdversarialAgent(a, e))
    assert f.can_create("o.adversarial.d.sd.s.session.x")


def test_unregister_all_clears(stack):
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))
    assert f.registrations()
    f.unregister_all()
    assert not f.registrations()


# ── spawn (provisioning) ────────────────────────────────────────────────


async def test_spawn_with_prefix_N(stack):
    f = AgentFactory(stack.engine)
    f.register("*.adversarial.*.*.*.*.*", lambda a, e, p: _AdversarialAgent(a, e))

    result = await f.spawn("4*.adversarial.finance.2*.s.session.*")
    assert len(result.new) == 4
    assert result.reused == []
    orgs = [a.address.org for a in result.new]
    subs = [a.address.subdomain for a in result.new]
    assert orgs == ["org0", "org1", "org2", "org3"]
    assert subs == ["subdomain0", "subdomain1", "subdomain0", "subdomain1"]
    assert all(isinstance(a, _AdversarialAgent) for a in result.new)


async def test_spawn_with_cross_join(stack):
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))
    result = await f.spawn("*4.adversarial.finance.*2.s.session.*")
    assert len(result.new) == 8


async def test_spawn_with_named_pool(stack):
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))

    pool = {
        "org": ["nike", "adidas", "coke", "pepsi"],
        "subdomain": ["sales", "marketing"],
        "instance": ["alpha"],
    }
    result = await f.spawn(
        "4*.adversarial.finance.2*.s.session.*",
        namer=lambda field, i: pool[field][i],
    )
    orgs = [a.address.org for a in result.new]
    subs = [a.address.subdomain for a in result.new]
    assert orgs == ["nike", "adidas", "coke", "pepsi"]
    assert subs == ["sales", "marketing", "sales", "marketing"]


def test_spawn_fresh_is_sync(stack):
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))
    agents = f.spawn_fresh("3*.adversarial.finance.x.s.session.y")
    assert len(agents) == 3


async def test_spawn_and_start_registers_and_consumes(stack):
    """spawn_and_start should register + start each agent atomically."""
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e, heartbeat_interval=0))
    try:
        result = await f.spawn_and_start("3*.adversarial.finance.x.s.session.y")
        assert len(result.new) == 3
        for a in result.new:
            assert await stack.registry.is_alive(a.address)
    finally:
        for a in result.new:
            await a.stop()
            await a.deregister()


# ── reuse vs fresh semantics ────────────────────────────────────────────


async def test_spawn_reuses_existing_when_no_dash(stack):
    """Without a dash, existing alive agents are pulled from the registry."""
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))

    # Pre-register two agents that the spec could match.
    existing_a = _addr("acme.adversarial.finance.equities.s.session.frank")
    existing_b = _addr("bcorp.adversarial.finance.equities.s.session.gertrude")
    await stack.registry.register(existing_a)
    await stack.registry.register(existing_b)

    result = await f.spawn("4*.adversarial.finance.equities.s.session.*")
    # 2 existing reused + 2 new built.
    assert len(result.reused) == 2
    assert len(result.new) == 2
    reused_uris = sorted(map(str, result.reused))
    assert reused_uris == sorted([str(existing_a), str(existing_b)])


async def test_dash_syntax_ignores_registry(stack):
    """`4-*` always builds 4 fresh agents regardless of what's registered."""
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))

    existing = _addr("acme.adversarial.finance.equities.s.session.frank")
    await stack.registry.register(existing)

    result = await f.spawn("4-*.adversarial.finance.equities.s.session.*")
    assert len(result.new) == 4
    assert result.reused == []
    # None of the new agents should have the existing org name.
    assert "acme" not in [a.address.org for a in result.new]


async def test_reuse_tops_up_when_short(stack):
    """If existing count < N, fresh names fill the gap."""
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))

    await stack.registry.register(
        _addr("acme.adversarial.finance.equities.s.session.frank"),
    )
    result = await f.spawn("3*.adversarial.finance.equities.s.session.*")
    assert len(result.reused) == 1
    assert len(result.new) == 2
    # Fresh names start their index past the existing count to avoid collision.
    new_orgs = [a.address.org for a in result.new]
    assert all(o.startswith("org") for o in new_orgs)
    assert "acme" not in new_orgs


async def test_reuse_exact_count_match_uses_only_existing(stack):
    """If N agents already exist that match the skeleton, no new are built."""
    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))

    for org in ("acme", "bcorp"):
        await stack.registry.register(
            _addr(f"{org}.adversarial.finance.equities.s.session.frank"),
        )

    result = await f.spawn("2*.adversarial.finance.equities.s.session.frank")
    assert len(result.new) == 0
    assert len(result.reused) == 2


async def test_dead_agents_not_reused(stack):
    """Registered but expired agents don't count as reusable."""
    from ahp.transport.keys import Keys

    f = AgentFactory(stack.engine)
    f.register("*.*.*.*.*.*.*", lambda a, e, p: _NoopAgent(a, e))

    addr = _addr("acme.adversarial.finance.equities.s.session.frank")
    await stack.registry.register(addr)
    await stack.redis.delete(Keys.alive_key(addr))  # simulate expiry

    result = await f.spawn("1*.adversarial.finance.equities.s.session.*")
    assert len(result.reused) == 0
    assert len(result.new) == 1
