"""Tests for AgentFactory integration with ToolRegistry + ResourceRegistry."""

from __future__ import annotations

import pytest

from ahp.adapters.base import AHPAgent
from ahp.adapters.factory import AgentFactory
from ahp.adapters.resources import ResourceRegistry
from ahp.adapters.tool_registry import ToolRegistry
from ahp.core.address import AgentAddress
from ahp.core.message import Message


class _ProfileCapture(AHPAgent):
    """Captures the profile it was built with for inspection."""

    def __init__(self, address, engine, profile):
        super().__init__(address, engine, heartbeat_interval=0)
        self.profile = profile

    async def handle_message(self, message: Message):
        return None


def _agent(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── tools flow into the profile via tool registry ─────────────────────


def test_factory_pulls_tools_from_tool_registry(stack):
    tools = ToolRegistry()

    @tools.tool("tifin", "db", "adversarial", "crud")
    def upsert(table: str, row: dict) -> dict:
        """Upsert a row."""
        return {"ok": True}

    factory = AgentFactory(stack.engine, tools=tools)
    factory.register("*.*.*.*.*.*.*",
                     lambda a, e, p: _ProfileCapture(a, e, p))

    matching: _ProfileCapture = factory.create(
        "tifin.adversarial.finance.equities.s.session.frank",
    )
    other: _ProfileCapture = factory.create(
        "public.collaborative.finance.equities.s.session.alice",
    )
    assert [t.name for t in matching.profile.tools] == ["upsert"]
    assert other.profile.tools == ()


def test_tool_registry_and_capability_registry_compose(stack):
    """Inline (capability) tools and address-registered tools both end up on the profile."""
    from ahp.adapters import CapabilityRegistry, Tool

    caps = CapabilityRegistry()
    caps.register(
        "*.*.finance.*.*.*.*",
        tools=(Tool(name="inline_quote", description="inline", handler=lambda: 0),),
    )
    tools = ToolRegistry()

    @tools.tool("*", "db", "*", "crud")
    def upsert(): return None

    factory = AgentFactory(stack.engine, capabilities=caps, tools=tools)
    factory.register("*.*.*.*.*.*.*",
                     lambda a, e, p: _ProfileCapture(a, e, p))

    agent: _ProfileCapture = factory.create(
        "tifin.adversarial.finance.equities.s.session.frank",
    )
    names = sorted(t.name for t in agent.profile.tools)
    assert names == ["inline_quote", "upsert"]


# ── resources flow into the profile via resource registry ─────────────


def test_factory_pulls_resources_from_resource_registry(stack):
    resources = ResourceRegistry()

    @resources.resource("tifin", "fs", "finance", "documents", name="docs")
    def make_docs():
        return {"root": "/data/finance"}

    factory = AgentFactory(stack.engine, resources=resources)
    factory.register("*.*.*.*.*.*.*",
                     lambda a, e, p: _ProfileCapture(a, e, p))

    fin: _ProfileCapture = factory.create(
        "tifin.adversarial.finance.documents.s.session.f",
    )
    sci: _ProfileCapture = factory.create(
        "tifin.adversarial.science.papers.s.session.x",
    )
    assert "docs" in fin.profile.resources
    assert fin.profile.resources["docs"] == {"root": "/data/finance"}
    assert "docs" not in sci.profile.resources


def test_factory_resources_property_exposes_registry(stack):
    factory = AgentFactory(stack.engine)
    # Default registries created automatically.
    assert factory.tools is not None
    assert factory.resources is not None
    assert factory.capabilities is not None


# ── profile_for() lets you peek without building ──────────────────────


def test_profile_for_inspection(stack):
    tools = ToolRegistry()

    @tools.tool("tifin", "db", "*", "crud")
    def upsert(): return None

    factory = AgentFactory(stack.engine, tools=tools)
    profile = factory.profile_for(
        "tifin.adversarial.finance.equities.s.session.frank",
    )
    assert [t.name for t in profile.tools] == ["upsert"]
