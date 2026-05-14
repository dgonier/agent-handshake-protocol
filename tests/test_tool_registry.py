"""Tests for the address-keyed ToolRegistry + @tool decorator."""

from __future__ import annotations

import pytest

from ahp.adapters.capability import Tool
from ahp.adapters.tool_address import ToolAddress
from ahp.adapters.tool_registry import ToolRegistry
from ahp.core.address import AgentAddress


def _agent(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── decorator-based registration ───────────────────────────────────────


def test_decorator_registers_at_address():
    reg = ToolRegistry()

    @reg.tool("tifin", "db", "adversarial", "crud")
    def update_record(table: str, row_id: str, fields: dict) -> dict:
        """Update a row in the table."""
        return {"ok": True}

    # Operation field auto-derived from function name.
    binding = reg.binding_at("tifin.db.adversarial.crud.update_record")
    assert binding.address.operation == "update_record"
    assert binding.tool.name == "update_record"
    assert "Update a row" in binding.tool.description


def test_decorator_keeps_function_callable():
    reg = ToolRegistry()

    @reg.tool("tifin", "db", "*", "crud")
    def upsert(x: int) -> int:
        return x + 1

    assert upsert(41) == 42


def test_explicit_operation_overrides_function_name():
    reg = ToolRegistry()

    @reg.tool("tifin", "db", "*", "crud", operation="merge_rows")
    def _fn(): return None

    assert "tifin.db.*.crud.merge_rows" in reg


def test_duplicate_address_rejected():
    reg = ToolRegistry()

    @reg.tool("tifin", "db", "*", "crud")
    def upsert(): return None

    with pytest.raises(ValueError, match="already registered"):
        @reg.tool("tifin", "db", "*", "crud")
        def upsert(): return None  # noqa: F811


# ── access scope: default convention ───────────────────────────────────


def test_default_allowed_for_matches_scope_role():
    reg = ToolRegistry()

    @reg.tool("tifin", "db", "adversarial", "crud")
    def update(): return None

    tools_match = reg.for_address(
        _agent("tifin.adversarial.finance.equities.s.session.f"),
    )
    tools_wrong_org = reg.for_address(
        _agent("public.adversarial.finance.equities.s.session.f"),
    )
    tools_wrong_role = reg.for_address(
        _agent("tifin.collaborative.finance.equities.s.session.f"),
    )
    assert [t.name for t in tools_match] == ["update"]
    assert tools_wrong_org == []
    assert tools_wrong_role == []


def test_wildcard_scope_allows_any_org():
    reg = ToolRegistry()

    @reg.tool("*", "tool", "*", "compute")
    def add_numbers(a: int, b: int) -> int:
        return a + b

    tools = reg.for_address(
        _agent("any.collaborative.x.y.s.session.i"),
    )
    assert [t.name for t in tools] == ["add_numbers"]


def test_explicit_allowed_for_overrides_convention():
    reg = ToolRegistry()

    @reg.tool("tifin", "api", "*", "search",
              allowed_for="*.adversarial.finance.*.*.*.*")
    def lookup(): return None

    fin_adv = _agent("public.adversarial.finance.equities.s.session.f")
    fin_collab = _agent("public.collaborative.finance.equities.s.session.f")
    assert [t.name for t in reg.for_address(fin_adv)] == ["lookup"]
    assert reg.for_address(fin_collab) == []


# ── tags ───────────────────────────────────────────────────────────────


def test_tag_filter_any_of_semantics():
    reg = ToolRegistry()

    @reg.tool("tifin", "db", "*", "crud", tags=["read-only"])
    def fetch(): return None

    @reg.tool("tifin", "db", "*", "crud", tags=["mutating", "slow"], operation="purge")
    def _purge(): return None

    addr = _agent("tifin.adversarial.x.y.s.session.f")

    # No filter → both.
    assert {t.name for t in reg.for_address(addr)} == {"fetch", "purge"}
    # Only read-only.
    ro = reg.for_address(addr, tags=["read-only"])
    assert {t.name for t in ro} == {"fetch"}
    # Any-of: 'slow' OR 'read-only' → both pass.
    both = reg.for_address(addr, tags=["read-only", "slow"])
    assert {t.name for t in both} == {"fetch", "purge"}


# ── unregister ────────────────────────────────────────────────────────


def test_unregister():
    reg = ToolRegistry()

    @reg.tool("tifin", "db", "*", "crud")
    def upsert(): return None

    assert "tifin.db.*.crud.upsert" in reg
    assert reg.unregister("tifin.db.*.crud.upsert")
    assert "tifin.db.*.crud.upsert" not in reg
    assert not reg.unregister("tifin.db.*.crud.upsert")  # already gone


# ── module-level @tool ────────────────────────────────────────────────


def test_module_level_tool_decorator_uses_default_registry():
    from ahp.adapters import DEFAULT_TOOL_REGISTRY, tool

    @tool("test-mod-level", "db", "*", "crud",
          operation="_unique_test_tool_xyz")
    def _fn(): return None

    try:
        assert "test-mod-level.db.*.crud._unique_test_tool_xyz" in DEFAULT_TOOL_REGISTRY
    finally:
        DEFAULT_TOOL_REGISTRY.unregister(
            "test-mod-level.db.*.crud._unique_test_tool_xyz"
        )
