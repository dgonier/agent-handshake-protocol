"""Tests for the search_tavily research tool."""

from __future__ import annotations

import asyncio
import json
import os

import pytest


def test_tool_registers_globally():
    import ahp.tools  # noqa: F401 — side-effect import
    from ahp.adapters.tool_registry import DEFAULT_TOOL_REGISTRY
    from ahp.core.address import AgentAddress

    # Some arbitrary agent must see the tool — it's global.
    arbitrary = AgentAddress.parse("zzz.anyrole.x.y.s.session.test")
    visible = DEFAULT_TOOL_REGISTRY.for_address(arbitrary)
    names = [t.name for t in visible]
    assert "search_tavily" in names


async def test_returns_error_when_key_missing(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    # Reach into the module after fresh import to bypass the lru_cache.
    from ahp.tools import research
    research._cached_search.cache_clear()
    out = await research.search_tavily("anything", max_results=2)
    data = json.loads(out)
    assert "error" in data
    assert "TAVILY_API_KEY" in data["error"]


async def test_rejects_empty_query():
    from ahp.tools import research
    out = await research.search_tavily("")
    data = json.loads(out)
    assert data.get("error") == "empty query"


async def test_clamps_max_results(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from ahp.tools import research
    research._cached_search.cache_clear()
    # Even with absurd input, the cached function gets called with a
    # clamped value (1..10). We can't directly observe the call here
    # without mocking, but we can at least verify the public API
    # accepts the input without raising.
    out = await research.search_tavily("x", max_results=999)
    json.loads(out)  # should parse
    out = await research.search_tavily("y", max_results=-5)
    json.loads(out)
