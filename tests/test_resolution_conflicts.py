"""Tests for the network-mapping resolution-conflict surfacing.

Three classes of conflict that used to be silent footguns:

1. Two tools at different :class:`ToolAddress`-es with the same
   ``operation`` short name end up in one agent's profile. LangChain
   can't disambiguate by name → run-time crash. Now caught at
   profile-build time.
2. Two resources at different :class:`ResourceAddress`-es with the
   same ``name`` field end up in one agent's profile. The dict key
   would silently overwrite. Now raised.
3. Two :class:`AgentFactory` instances sharing one engine fight over
   ``engine.groups`` / ``engine.scope``. The clobber still happens
   (late-binding wins) but a logger warning surfaces it.
"""

from __future__ import annotations

import logging

import pytest

from ahp.adapters import (
    AgentFactory,
    CapabilityRegistry,
    GroupRegistry,
    ResourceNameCollisionError,
    ResourceRegistry,
    Tool,
    ToolNameCollisionError,
    ToolRegistry,
)
from ahp.adapters.errors import ResolutionConflictError
from ahp.core.address import AgentAddress
from ahp.engine.scope import ScopePolicy


def _agent(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── tool name collisions ───────────────────────────────────────────────


def test_tool_name_collision_across_registry_bindings(stack):
    """Two tools with different ToolAddresses but the same short name."""
    tools = ToolRegistry()

    # Both register under "lookup" but at different categories.
    @tools.tool("tifin", "db", "*", "crud", operation="lookup")
    def _crud_lookup(): return None

    @tools.tool("tifin", "api", "*", "search", operation="lookup")
    def _search_lookup(): return None

    factory = AgentFactory(stack.engine, tools=tools)

    with pytest.raises(ToolNameCollisionError, match=r"lookup"):
        factory.profile_for("tifin.adversarial.finance.equities.s.session.f")


def test_tool_name_collision_with_inline_capability_tool(stack):
    """An inline capability tool and a registry tool sharing a name."""
    caps = CapabilityRegistry()
    caps.register(
        "*.*.*.*.*.*.*",
        tools=(Tool(name="fetch", description="inline", handler=lambda: 0),),
    )
    tools = ToolRegistry()

    @tools.tool("tifin", "api", "*", "search", operation="fetch")
    def _fetch(): return None

    factory = AgentFactory(stack.engine, capabilities=caps, tools=tools)
    with pytest.raises(ToolNameCollisionError):
        factory.profile_for("tifin.adversarial.finance.equities.s.session.f")


def test_tool_name_no_collision_when_only_one_matches_agent(stack):
    """Tools at different addresses that don't both apply to one agent are fine."""
    tools = ToolRegistry()

    @tools.tool("tifin", "db", "adversarial", "crud", operation="lookup")
    def _adv_lookup(): return None

    @tools.tool("tifin", "db", "collaborative", "crud", operation="lookup")
    def _collab_lookup(): return None

    factory = AgentFactory(stack.engine, tools=tools)

    # An adversarial agent only sees _adv_lookup.
    p_adv = factory.profile_for("tifin.adversarial.finance.equities.s.session.f")
    assert [t.name for t in p_adv.tools] == ["lookup"]
    # A collaborative agent only sees _collab_lookup.
    p_collab = factory.profile_for("tifin.collaborative.finance.equities.s.session.a")
    assert [t.name for t in p_collab.tools] == ["lookup"]


def test_resolution_conflict_is_a_typed_error(stack):
    """ToolNameCollisionError is-a ResolutionConflictError for catch-all handlers."""
    tools = ToolRegistry()

    @tools.tool("tifin", "db", "*", "crud", operation="x")
    def _a(): return None

    @tools.tool("tifin", "api", "*", "search", operation="x")
    def _b(): return None

    factory = AgentFactory(stack.engine, tools=tools)
    with pytest.raises(ResolutionConflictError):
        factory.profile_for("tifin.adversarial.x.y.s.session.i")


# ── resource name collisions ──────────────────────────────────────────


def test_resource_name_collision(stack):
    """Two resources with the same `name` field matching one agent."""
    resources = ResourceRegistry()

    # Use an explicit broad allowed_for so both resources match the same
    # agent — the default convention narrows by domain+subdomain, which
    # otherwise prevents the overlap.
    @resources.resource(
        "tifin", "fs", "finance", "documents",
        name="docs", allowed_for="*.*.*.*.*.*.*",
    )
    def _docs_a(): return {"id": "a"}

    @resources.resource(
        "tifin", "fs", "finance", "filings",
        name="docs", allowed_for="*.*.*.*.*.*.*",
    )
    def _docs_b(): return {"id": "b"}

    factory = AgentFactory(stack.engine, resources=resources)
    with pytest.raises(ResourceNameCollisionError, match="docs"):
        factory.profile_for("tifin.adversarial.finance.equities.s.session.f")


def test_resource_name_no_collision_when_only_one_matches(stack):
    resources = ResourceRegistry()

    @resources.resource("tifin", "fs", "finance", "documents", name="store")
    def _fin_store(): return {"kind": "finance"}

    @resources.resource("tifin", "fs", "science", "papers", name="store")
    def _sci_store(): return {"kind": "science"}

    factory = AgentFactory(stack.engine, resources=resources)
    p_fin = factory.profile_for(
        "tifin.adversarial.finance.documents.s.session.f",
    )
    p_sci = factory.profile_for(
        "tifin.adversarial.science.papers.s.session.x",
    )
    assert p_fin.resources["store"]["kind"] == "finance"
    assert p_sci.resources["store"]["kind"] == "science"


# ── factory-on-engine clobber warning ─────────────────────────────────


def test_double_factory_warns_on_groups_clobber(stack, caplog):
    groups_a = GroupRegistry()
    groups_a.register("debaters", "*.adversarial.*.*.*.*.*")

    AgentFactory(stack.engine, groups=groups_a)
    # Second factory installs a different group registry. Should warn.
    groups_b = GroupRegistry()
    groups_b.register("readers", "*.interview.*.*.*.*.*")
    with caplog.at_level(logging.WARNING, logger="ahp.adapters.factory"):
        AgentFactory(stack.engine, groups=groups_b)

    messages = [r.getMessage() for r in caplog.records]
    assert any("engine.groups" in m and "overwrit" in m for m in messages), messages


def test_double_factory_warns_on_scope_clobber(stack, caplog):
    scope_a = ScopePolicy()
    scope_a.restrict(
        target="tifin.*.*.*.*.*.*",
        allow_sources="tifin.*.*.*.*.*.*",
    )
    AgentFactory(stack.engine, scope=scope_a)

    scope_b = ScopePolicy()
    scope_b.restrict(
        target="public.*.*.*.*.*.*",
        allow_sources="public.*.*.*.*.*.*",
    )
    with caplog.at_level(logging.WARNING, logger="ahp.adapters.factory"):
        AgentFactory(stack.engine, scope=scope_b)
    messages = [r.getMessage() for r in caplog.records]
    assert any("engine.scope" in m and "overwrit" in m for m in messages)


def test_no_warning_when_same_registry_reattached(stack, caplog):
    """Re-installing the SAME registry is idempotent; no warning."""
    groups = GroupRegistry()
    AgentFactory(stack.engine, groups=groups)
    with caplog.at_level(logging.WARNING, logger="ahp.adapters.factory"):
        AgentFactory(stack.engine, groups=groups)   # same instance
    messages = [r.getMessage() for r in caplog.records]
    assert not any("overwrit" in m for m in messages)
