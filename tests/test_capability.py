"""Tests for CapabilityRegistry, AgentProfile composition, and factory wiring."""

from __future__ import annotations

import pytest

from ahp.adapters.base import AHPAgent
from ahp.adapters.capability import (
    AgentProfile,
    CapabilityRegistry,
    RagSource,
    Skill,
    Tool,
)
from ahp.adapters.factory import AgentFactory
from ahp.core.address import AgentAddress
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


def _tool(name: str) -> Tool:
    return Tool(name=name, description=f"{name} desc", handler=lambda **kw: name)


async def _retrieve(query: str) -> list[str]:
    return [f"doc-for-{query}"]


# ── basic resolution ────────────────────────────────────────────────────


def test_empty_registry_returns_default_profile():
    reg = CapabilityRegistry()
    addr = _addr("o.r.d.sd.s.session.i")
    profile = reg.resolve(addr)
    assert profile.address == addr
    assert profile.tools == ()
    assert profile.skills == ()
    assert profile.rag_sources == ()
    assert profile.prompt == ""
    assert profile.agent_kind == "react"


def test_register_and_resolve_tools():
    reg = CapabilityRegistry()
    finance_tools = (_tool("get_quote"), _tool("get_news"))
    reg.register("*.*.finance.*.*.*.*", tools=finance_tools)

    profile = reg.resolve(_addr("o.adversarial.finance.equities.s.session.frank"))
    assert profile.tools == finance_tools


def test_resolution_filters_by_pattern():
    reg = CapabilityRegistry()
    reg.register("*.*.finance.*.*.*.*", tools=(_tool("fin"),))
    reg.register("*.*.science.*.*.*.*", tools=(_tool("sci"),))

    fin = reg.resolve(_addr("o.r.finance.x.s.session.i"))
    sci = reg.resolve(_addr("o.r.science.x.s.session.i"))
    assert [t.name for t in fin.tools] == ["fin"]
    assert [t.name for t in sci.tools] == ["sci"]


def test_multiple_matching_providers_concatenate_tools():
    reg = CapabilityRegistry()
    reg.register("*.*.finance.*.*.*.*", tools=(_tool("base"),))
    reg.register("*.*.finance.equities.*.*.*", tools=(_tool("equities"),))

    profile = reg.resolve(_addr("o.r.finance.equities.s.session.i"))
    # Equal priorities → registration order (stable sort).
    assert [t.name for t in profile.tools] == ["base", "equities"]


def test_priority_ordering():
    reg = CapabilityRegistry()
    reg.register("*.*.finance.*.*.*.*", tools=(_tool("base"),), priority=0)
    reg.register("*.*.finance.equities.*.*.*", tools=(_tool("hi"),), priority=10)

    profile = reg.resolve(_addr("o.r.finance.equities.s.session.i"))
    assert [t.name for t in profile.tools] == ["hi", "base"]


# ── skills and prompts ──────────────────────────────────────────────────


def test_skills_and_all_tools():
    reg = CapabilityRegistry()
    valuation_skill = Skill(
        name="valuation",
        description="DCF and comparables",
        tools=(_tool("dcf"), _tool("comp")),
        prompt_fragment="Use DCF or comparables when valuing equities.",
    )
    reg.register("*.*.finance.equities.*.*.*", skills=(valuation_skill,))
    reg.register(
        "*.*.finance.*.*.*.*", tools=(_tool("get_quote"),),
    )

    profile = reg.resolve(_addr("o.r.finance.equities.s.session.i"))
    assert [s.name for s in profile.skills] == ["valuation"]
    tool_names = [t.name for t in profile.all_tools]
    assert "get_quote" in tool_names
    assert "dcf" in tool_names
    assert "comp" in tool_names


def test_prompt_composition():
    reg = CapabilityRegistry()
    reg.register("*.*.*.*.*.*.*", prompt="You are an AHP agent.")
    reg.register(
        "*.adversarial.*.*.*.*.*", prompt="Argue the bear case.", priority=5,
    )
    reg.register(
        "*.*.finance.*.*.*.*", prompt="You work in finance.", priority=3,
    )

    profile = reg.resolve(_addr("o.adversarial.finance.equities.s.session.i"))
    # Joined with blank line, in priority order.
    parts = profile.prompt.split("\n\n")
    assert parts == [
        "Argue the bear case.",
        "You work in finance.",
        "You are an AHP agent.",
    ]


# ── rag sources ────────────────────────────────────────────────────────


def test_rag_sources_compose():
    reg = CapabilityRegistry()
    rag = RagSource(name="sec-filings", retrieve=_retrieve)
    reg.register("*.*.finance.*.*.*.*", rag_sources=(rag,))

    profile = reg.resolve(_addr("o.r.finance.x.s.session.i"))
    assert [r.name for r in profile.rag_sources] == ["sec-filings"]


# ── agent kind ──────────────────────────────────────────────────────────


def test_agent_kind_uses_highest_priority_specifier():
    reg = CapabilityRegistry()
    # Default-priority provider sets "react".
    reg.register("*.*.*.*.*.*.*", agent_kind="react", priority=0)
    # Higher-priority provider says "deep" → wins.
    reg.register("*.collaborative.*.*.*.*.*", agent_kind="deep", priority=10)

    deep_profile = reg.resolve(_addr("o.collaborative.d.sd.s.session.i"))
    react_profile = reg.resolve(_addr("o.adversarial.d.sd.s.session.i"))
    assert deep_profile.agent_kind == "deep"
    assert react_profile.agent_kind == "react"


def test_agent_kind_default_when_none_specified():
    reg = CapabilityRegistry()
    reg.register("*.*.*.*.*.*.*", tools=(_tool("x"),))  # no agent_kind
    profile = reg.resolve(_addr("o.r.d.sd.s.session.i"))
    assert profile.agent_kind == "react"


# ── factory integration ────────────────────────────────────────────────


class _ProfileCapturingAgent(AHPAgent):
    """Agent that just records the profile it was built with."""

    def __init__(self, address, engine, profile):
        super().__init__(address, engine, heartbeat_interval=0)
        self.profile = profile

    async def handle_message(self, message: Message):
        return None


def test_factory_passes_profile_to_builder(stack):
    reg = CapabilityRegistry()
    reg.register("*.*.finance.*.*.*.*", tools=(_tool("get_quote"),))

    factory = AgentFactory(stack.engine, capabilities=reg)
    factory.register(
        "*.*.*.*.*.*.*",
        lambda a, e, p: _ProfileCapturingAgent(a, e, p),
    )

    agent: _ProfileCapturingAgent = factory.create(
        "o.adversarial.finance.equities.s.session.f",
    )
    assert isinstance(agent, _ProfileCapturingAgent)
    assert [t.name for t in agent.profile.tools] == ["get_quote"]


def test_factory_without_capabilities_supplies_empty_profile(stack):
    factory = AgentFactory(stack.engine)
    factory.register(
        "*.*.*.*.*.*.*",
        lambda a, e, p: _ProfileCapturingAgent(a, e, p),
    )
    agent: _ProfileCapturingAgent = factory.create("o.r.d.sd.s.session.i")
    assert agent.profile.tools == ()
    assert agent.profile.agent_kind == "react"


def test_factory_profile_for_helper(stack):
    reg = CapabilityRegistry()
    reg.register("*.*.science.*.*.*.*", tools=(_tool("simulate"),))
    factory = AgentFactory(stack.engine, capabilities=reg)
    profile = factory.profile_for("o.r.science.biology.s.session.f")
    assert [t.name for t in profile.tools] == ["simulate"]


async def test_spawn_passes_profiles(stack):
    """Provisioning spawn uses the same profile machinery."""
    reg = CapabilityRegistry()
    reg.register("*.adversarial.finance.*.*.*.*", tools=(_tool("debate"),))

    factory = AgentFactory(stack.engine, capabilities=reg)
    factory.register(
        "*.*.*.*.*.*.*",
        lambda a, e, p: _ProfileCapturingAgent(a, e, p),
    )

    result = await factory.spawn("3*.adversarial.finance.equities.s.session.*")
    assert len(result.new) == 3
    for agent in result.new:
        assert isinstance(agent, _ProfileCapturingAgent)
        assert [t.name for t in agent.profile.tools] == ["debate"]
