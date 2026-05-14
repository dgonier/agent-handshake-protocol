"""Tests for ToolAddress + ResourceAddress (parsing, validation, conventions)."""

from __future__ import annotations

import pytest

from ahp.adapters.tool_address import ResourceAddress, ToolAddress
from ahp.core.address import AgentAddress


# ── ToolAddress ────────────────────────────────────────────────────────


def test_parse_round_trip():
    uri = "tifin.db.adversarial.crud.update"
    addr = ToolAddress.parse(uri)
    assert addr.scope == "tifin"
    assert addr.kind == "db"
    assert addr.role == "adversarial"
    assert addr.category == "crud"
    assert addr.operation == "update"
    assert str(addr) == uri


@pytest.mark.parametrize("bad", [
    "",
    "a.b.c.d",                # too few
    "a.b.c.d.e.f",            # too many
    "a..c.d.e",               # empty token
    "a.b.c d.e",              # space
    "a.b.c?d.e",              # ?
])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        ToolAddress.parse(bad)


def test_wildcard_fields_allowed():
    addr = ToolAddress.parse("*.db.*.crud.update")
    assert addr.scope == "*"
    assert addr.role == "*"


def test_derived_allowed_for_convention():
    """Tool address scope/role projects to agent address org/role."""
    addr = ToolAddress("tifin", "db", "adversarial", "crud", "update")
    pat = addr.derived_allowed_for()
    assert pat.org == "tifin"
    assert pat.role == "adversarial"
    # Other agent fields should not constrain.
    assert pat.domain == "*"
    assert pat.subdomain == "*"
    assert pat.accept == "*"
    assert pat.lifecycle == "*"
    assert pat.instance == "*"


def test_derived_allowed_for_matches_agent():
    addr = ToolAddress("tifin", "db", "adversarial", "crud", "update")
    pat = addr.derived_allowed_for()
    yes = AgentAddress.parse("tifin.adversarial.finance.equities.s.session.f")
    no_org = AgentAddress.parse("public.adversarial.finance.equities.s.session.f")
    no_role = AgentAddress.parse("tifin.collaborative.finance.equities.s.session.f")
    assert pat.matches(yes)
    assert not pat.matches(no_org)
    assert not pat.matches(no_role)


def test_wildcard_scope_matches_any_org():
    addr = ToolAddress("*", "db", "*", "crud", "update")
    pat = addr.derived_allowed_for()
    assert pat.matches(AgentAddress.parse("any.adversarial.x.y.s.session.i"))
    assert pat.matches(AgentAddress.parse("other.collaborative.x.y.s.session.i"))


# ── ResourceAddress ────────────────────────────────────────────────────


def test_resource_parse_round_trip():
    uri = "tifin.fs.finance.documents.docs-2024"
    addr = ResourceAddress.parse(uri)
    assert addr.scope == "tifin"
    assert addr.kind == "fs"
    assert addr.domain == "finance"
    assert addr.subdomain == "documents"
    assert addr.name == "docs-2024"
    assert str(addr) == uri


def test_resource_derived_allowed_for_uses_domain_subdomain():
    """Resources scope by domain/subdomain (shared across roles)."""
    addr = ResourceAddress("tifin", "fs", "finance", "documents", "docs-2024")
    pat = addr.derived_allowed_for()
    assert pat.org == "tifin"
    assert pat.role == "*"           # any role can read shared docs
    assert pat.domain == "finance"
    assert pat.subdomain == "documents"


def test_resource_allowed_for_filters_by_domain():
    addr = ResourceAddress("tifin", "fs", "finance", "documents", "docs-2024")
    pat = addr.derived_allowed_for()
    fin = AgentAddress.parse("tifin.adversarial.finance.documents.s.session.f")
    sci = AgentAddress.parse("tifin.adversarial.science.documents.s.session.f")
    assert pat.matches(fin)
    assert not pat.matches(sci)
