"""Tests for the upgraded Skill dataclass + SkillRegistry + factory wiring + CLI.

Five clusters:

1. Skill dataclass shape — new fields default correctly; old call
   sites (positional name+description) still work.
2. SkillRegistry — register / unregister / decorator / allowed_for
   convention / collisions.
3. Factory wiring — profile_for pulls matching skills; collision
   errors surface from the factory.
4. Real compiled LangGraph DAG in the graph field — invoke it
   end-to-end through the skill, not just store-and-retrieve.
5. `ahp list-skills` CLI — happy path, --for filter, --tag filter,
   empty output.
"""

from __future__ import annotations

import io
from typing import TypedDict

import pytest

import ahp.cli
from ahp.adapters import (
    AgentFactory,
    CapabilityProvider,
    CapabilityRegistry,
    DEFAULT_SKILL_REGISTRY,
    Skill,
    SkillBinding,
    SkillRegistry,
    SKILL_KIND,
    ToolNameCollisionError,
    skill,
)
from ahp.adapters.tool_address import ResourceAddress, ToolAddress
from ahp.core import AgentAddress
from ahp.core.pattern import AddressPattern


# ── reusable LangGraph stand-in ──────────────────────────────────────


def _make_compiled_graph():
    """A trivially compiled LangGraph DAG with one increment node.

    Used to confirm the skill's `graph` field accepts and round-trips
    a real CompiledStateGraph instance.
    """
    from langgraph.graph import StateGraph, START, END

    class S(TypedDict):
        n: int

    def inc(s: S) -> S:
        return {"n": s["n"] + 1}

    builder = StateGraph(S)
    builder.add_node("inc", inc)
    builder.add_edge(START, "inc")
    builder.add_edge("inc", END)
    return builder.compile()


# ── Skill dataclass ──────────────────────────────────────────────────


def test_skill_legacy_call_still_works():
    """Existing call sites using only name/description/tools/prompt_fragment
    should continue to work — backwards compat for any code from before
    the upgrade."""
    s = Skill(name="x", description="y")
    assert s.name == "x"
    assert s.description == "y"
    assert s.tools == ()
    assert s.prompt_fragment == ""


def test_skill_new_fields_default_empty():
    s = Skill(name="x", description="y")
    assert s.when_to_use == ""
    assert s.graph is None
    assert s.suggested_tools == ()
    assert s.suggested_specialists == ()
    assert s.suggested_loras == ()
    assert s.suggested_information_sources == ()


def test_skill_carries_compiled_graph():
    """The graph field accepts a real CompiledStateGraph and is
    invokable directly off the skill."""
    g = _make_compiled_graph()
    s = Skill(name="incrementer", description="adds one", graph=g)
    assert s.graph is g
    out = s.graph.invoke({"n": 5})
    assert out["n"] == 6


def test_skill_carries_all_four_address_bundles():
    s = Skill(
        name="refund-investigation",
        description="Investigate a refund request",
        when_to_use="customer asks for a refund",
        suggested_tools=(
            ToolAddress.parse("acme.api.*.orders.lookup_order"),
            ToolAddress.parse("acme.api.*.refunds.issue_refund"),
        ),
        suggested_specialists=(
            AgentAddress.parse(
                "legalcorp.tos-reviewer.consumer.refunds.s.longterm.primary"
            ),
        ),
        suggested_loras=(
            ResourceAddress.parse("acme.lora.support.refunds.de-escalation"),
        ),
        suggested_information_sources=(
            ResourceAddress.parse("acme.data.policy.refunds.policy-text"),
            ResourceAddress.parse(
                "legalcorp.data.regulation.consumer-protection.us-text"
            ),
        ),
    )
    assert len(s.suggested_tools) == 2
    assert len(s.suggested_specialists) == 1
    assert len(s.suggested_loras) == 1
    assert len(s.suggested_information_sources) == 2


# ── SkillRegistry ────────────────────────────────────────────────────


def _fresh_registry() -> SkillRegistry:
    return SkillRegistry()


def test_register_via_decorator_uses_function_name():
    reg = _fresh_registry()

    @reg.skill("acme", "support", "refunds")
    def make_refund_investigation():
        return Skill(name="refund-investigation", description="…")

    assert len(reg) == 1
    addrs = reg.addresses()
    assert str(addrs[0]) == "acme.skill.support.refunds.refund-investigation"


def test_register_with_explicit_name():
    reg = _fresh_registry()

    @reg.skill("acme", "support", "refunds", name="custom-skill-id")
    def whatever():
        return Skill(name="custom-skill-id", description="…")

    addrs = reg.addresses()
    assert str(addrs[0]) == "acme.skill.support.refunds.custom-skill-id"


def test_register_rejects_non_skill_factory_return():
    reg = _fresh_registry()

    def bad():
        return "not a skill"

    with pytest.raises(TypeError, match="expected Skill"):
        reg.register(bad, "acme", "x", "y")


def test_duplicate_address_raises():
    reg = _fresh_registry()

    @reg.skill("acme", "x", "y", name="dupe")
    def first():
        return Skill(name="dupe", description="first")

    with pytest.raises(ValueError, match="already registered"):
        @reg.skill("acme", "x", "y", name="dupe")
        def second():
            return Skill(name="dupe", description="second")


def test_allowed_for_default_convention():
    """Default allowed_for is {scope}.*.{domain}.{subdomain}.*.*.*"""
    reg = _fresh_registry()

    @reg.skill("acme", "support", "refunds", name="s1")
    def factory():
        return Skill(name="s1", description="d")

    binding = next(iter(reg.bindings()))
    # Default pattern: acme.*.support.refunds.*.*.*
    addr_in_scope = AgentAddress.parse(
        "acme.triage-bot.support.refunds.s.session.t1"
    )
    addr_out_of_scope = AgentAddress.parse(
        "acme.triage-bot.support.tickets.s.session.t1"
    )
    assert binding.allowed_for.matches(addr_in_scope)
    assert not binding.allowed_for.matches(addr_out_of_scope)


def test_allowed_for_explicit_override():
    reg = _fresh_registry()

    @reg.skill(
        "acme", "support", "refunds", name="s1",
        allowed_for="*.engineer-on-call.*.*.*.*.*",
    )
    def factory():
        return Skill(name="s1", description="d")

    binding = next(iter(reg.bindings()))
    eng = AgentAddress.parse("acme.engineer-on-call.support.x.s.longterm.alex")
    triage = AgentAddress.parse("acme.triage-bot.support.x.s.session.t1")
    assert binding.allowed_for.matches(eng)
    assert not binding.allowed_for.matches(triage)


def test_for_address_returns_only_visible():
    reg = _fresh_registry()

    @reg.skill("acme", "support", "refunds", name="visible")
    def f1():
        return Skill(name="visible", description="…")

    @reg.skill("acme", "support", "billing", name="invisible")
    def f2():
        return Skill(name="invisible", description="…")

    addr = AgentAddress.parse("acme.triage-bot.support.refunds.s.session.t1")
    skills = reg.for_address(addr)
    names = [s.name for s in skills]
    assert "visible" in names
    assert "invisible" not in names


def test_tag_filter():
    reg = _fresh_registry()

    @reg.skill("acme", "x", "y", name="tagged", tags=["primary"])
    def f1():
        return Skill(name="tagged", description="…")

    @reg.skill("acme", "x", "y", name="untagged")
    def f2():
        return Skill(name="untagged", description="…")

    addr = AgentAddress.parse("acme.a.x.y.s.session.t1")
    primary_only = reg.for_address(addr, tags=["primary"])
    assert [s.name for s in primary_only] == ["tagged"]


def test_unregister():
    reg = _fresh_registry()

    @reg.skill("acme", "x", "y", name="s1")
    def f1():
        return Skill(name="s1", description="…")

    addr = next(iter(reg.addresses()))
    assert reg.unregister(addr) is True
    assert len(reg) == 0
    assert reg.unregister(addr) is False  # second unregister is no-op


# ── module-level @skill decorator writes to DEFAULT ──────────────────


def test_module_level_skill_decorator_writes_to_default():
    """The module-level `skill` decorator writes to
    DEFAULT_SKILL_REGISTRY just like @tool / @resource."""
    DEFAULT_SKILL_REGISTRY.clear()

    @skill("module-test", "x", "y", name="default-target")
    def factory():
        return Skill(name="default-target", description="…")

    assert any(
        "default-target" in str(a)
        for a in DEFAULT_SKILL_REGISTRY.addresses()
    )
    DEFAULT_SKILL_REGISTRY.clear()


# ── AgentFactory wiring ──────────────────────────────────────────────


def test_factory_pulls_matching_skills_into_profile(redis_client):
    """profile_for() returns a profile whose skills include any in
    the registry whose allowed_for matches the agent."""
    from ahp.core.compatibility import CompatibilityMatrix
    from ahp.engine.router import ProtocolEngine
    from ahp.engine.thread_manager import ThreadManager
    from ahp.registry.registry import AgentRegistry
    from ahp.transport.cache import ProtocolCache
    from ahp.transport.redis_bus import RedisBus

    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    reg = SkillRegistry()

    @reg.skill("acme", "support", "refunds", name="refund-skill")
    def f():
        return Skill(name="refund-skill", description="…")

    factory = AgentFactory(engine=engine, skills=reg)
    addr = AgentAddress.parse("acme.triage-bot.support.refunds.s.session.t1")
    profile = factory.profile_for(addr)
    assert any(s.name == "refund-skill" for s in profile.skills)


def test_factory_merges_inline_and_registry_skills(redis_client):
    """A skill on the CapabilityProvider AND one in the registry
    both end up on the profile; no collision because they have
    different names."""
    from ahp.core.compatibility import CompatibilityMatrix
    from ahp.engine.router import ProtocolEngine
    from ahp.engine.thread_manager import ThreadManager
    from ahp.registry.registry import AgentRegistry
    from ahp.transport.cache import ProtocolCache
    from ahp.transport.redis_bus import RedisBus

    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    caps = CapabilityRegistry()
    caps.register(
        AddressPattern.parse("acme.*.*.*.*.*.*"),
        skills=[Skill(name="inline-skill", description="…")],
    )

    skill_reg = SkillRegistry()

    @skill_reg.skill("acme", "support", "refunds", name="registry-skill")
    def f():
        return Skill(name="registry-skill", description="…")

    factory = AgentFactory(
        engine=engine, capabilities=caps, skills=skill_reg,
    )
    addr = AgentAddress.parse("acme.triage-bot.support.refunds.s.session.t1")
    profile = factory.profile_for(addr)
    names = {s.name for s in profile.skills}
    assert names == {"inline-skill", "registry-skill"}


def test_factory_raises_on_skill_name_collision(redis_client):
    """Two registry skills with the same short name addressed to one
    agent should raise — same wiring-bug story as tool collisions."""
    from ahp.core.compatibility import CompatibilityMatrix
    from ahp.engine.router import ProtocolEngine
    from ahp.engine.thread_manager import ThreadManager
    from ahp.registry.registry import AgentRegistry
    from ahp.transport.cache import ProtocolCache
    from ahp.transport.redis_bus import RedisBus

    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    skill_reg = SkillRegistry()

    @skill_reg.skill("acme", "support", "refunds", name="dup-a")
    def f1():
        return Skill(name="collide", description="first")

    @skill_reg.skill("acme", "support", "refunds", name="dup-b")
    def f2():
        return Skill(name="collide", description="second")

    factory = AgentFactory(engine=engine, skills=skill_reg)
    addr = AgentAddress.parse("acme.triage-bot.support.refunds.s.session.t1")
    with pytest.raises(ToolNameCollisionError, match="collide"):
        factory.profile_for(addr)


# ── CLI: list-skills ─────────────────────────────────────────────────


def _run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    rc = ahp.cli.main(list(argv), out=buf)
    return rc, buf.getvalue()


def test_cli_list_skills_empty():
    DEFAULT_SKILL_REGISTRY.clear()
    rc, out = _run_cli("list-skills")
    assert rc == 0
    assert "no skills registered" in out


def test_cli_list_skills_shows_registered():
    DEFAULT_SKILL_REGISTRY.clear()

    @skill("acme", "support", "refunds", name="refund-flow")
    def f():
        return Skill(
            name="refund-flow",
            description="Investigate and process a refund.",
            graph=_make_compiled_graph(),
            suggested_tools=(
                ToolAddress.parse("acme.api.*.orders.lookup_order"),
            ),
            suggested_information_sources=(
                ResourceAddress.parse("acme.data.policy.refunds.policy-text"),
                ResourceAddress.parse(
                    "legalcorp.data.regulation.consumer-protection.us-text"
                ),
            ),
        )

    try:
        rc, out = _run_cli("list-skills")
        assert rc == 0
        assert "acme.skill.support.refunds.refund-flow" in out
        assert "refund-flow" in out
        # Bundle sizes column should show t=1 and i=2.
        assert "t=1" in out
        assert "i=2" in out
        # Graph marker present.
        assert "yes" in out
    finally:
        DEFAULT_SKILL_REGISTRY.clear()


def test_cli_list_skills_filters_by_for_address():
    DEFAULT_SKILL_REGISTRY.clear()

    @skill("acme", "support", "refunds", name="visible-to-refunds")
    def f1():
        return Skill(name="visible-to-refunds", description="…")

    @skill("acme", "support", "billing", name="not-this-agent")
    def f2():
        return Skill(name="not-this-agent", description="…")

    try:
        rc, out = _run_cli(
            "list-skills",
            "--for", "acme.triage.support.refunds.s.session.t1",
        )
        assert rc == 0
        assert "visible-to-refunds" in out
        assert "not-this-agent" not in out
    finally:
        DEFAULT_SKILL_REGISTRY.clear()


def test_cli_list_skills_filters_by_tag():
    DEFAULT_SKILL_REGISTRY.clear()

    @skill("acme", "x", "y", name="primary-skill", tags=["primary"])
    def f1():
        return Skill(name="primary-skill", description="…")

    @skill("acme", "x", "y", name="secondary-skill", tags=["secondary"])
    def f2():
        return Skill(name="secondary-skill", description="…")

    try:
        rc, out = _run_cli("list-skills", "--tag", "primary")
        assert rc == 0
        assert "primary-skill" in out
        assert "secondary-skill" not in out
    finally:
        DEFAULT_SKILL_REGISTRY.clear()
