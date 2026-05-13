"""Tests for ProvisioningPattern — N* / *N spawn semantics."""

from __future__ import annotations

import pytest

from ahp.adapters.provisioning import (
    ProvisioningField,
    ProvisioningPattern,
    default_namer,
)


# ── parsing ─────────────────────────────────────────────────────────────


def test_concrete_only_pattern():
    p = ProvisioningPattern.parse("tifin.adversarial.finance.equities.s.session.frank")
    addrs = p.materialize()
    assert len(addrs) == 1
    assert str(addrs[0]) == "tifin.adversarial.finance.equities.s.session.frank"


def test_plain_wildcard_makes_one():
    p = ProvisioningPattern.parse("tifin.adversarial.finance.equities.s.session.*")
    assert p.total() == 1
    addrs = p.materialize()
    assert addrs[0].instance == "instance0"


def test_str_round_trip():
    spec = "4*.adversarial.finance.2*.s.session.*"
    p = ProvisioningPattern.parse(spec)
    assert str(p) == spec


def test_str_for_cross_join():
    spec = "*4.adversarial.finance.*2.s.session.*"
    p = ProvisioningPattern.parse(spec)
    assert str(p) == spec


# ── max-count semantics ────────────────────────────────────────────────


def test_user_example_4star_org_2star_subdomain():
    """4 orgs and 2 subdomains under prefix-N → 4 agents, subdomain cycles."""
    p = ProvisioningPattern.parse("4*.adversarial.finance.2*.s.session.*")
    assert p.total() == 4
    addrs = p.materialize()
    assert len(addrs) == 4

    orgs = [a.org for a in addrs]
    assert orgs == ["org0", "org1", "org2", "org3"]   # 4 distinct

    subs = [a.subdomain for a in addrs]
    assert subs == ["subdomain0", "subdomain1", "subdomain0", "subdomain1"]  # cycles


def test_max_uses_largest_count():
    """When prefix-N counts differ, total = max of them."""
    p = ProvisioningPattern.parse("3*.r.d.5*.s.session.*")
    assert p.total() == 5
    addrs = p.materialize()
    assert len(addrs) == 5
    # org has 3 values cycling, subdomain has 5 distinct
    assert [a.org for a in addrs] == ["org0", "org1", "org2", "org0", "org1"]
    assert [a.subdomain for a in addrs] == [
        "subdomain0", "subdomain1", "subdomain2", "subdomain3", "subdomain4",
    ]


def test_max_with_single_field():
    p = ProvisioningPattern.parse("4*.r.d.x.s.session.y")
    assert p.total() == 4
    addrs = p.materialize()
    orgs = [a.org for a in addrs]
    assert orgs == ["org0", "org1", "org2", "org3"]
    for a in addrs:
        assert a.subdomain == "x"
        assert a.instance == "y"


# ── cross-join semantics ────────────────────────────────────────────────


def test_suffix_N_is_cartesian():
    p = ProvisioningPattern.parse("*4.r.d.*2.s.session.*")
    assert p.total() == 8
    addrs = p.materialize()
    assert len(addrs) == 8
    # Every (org, subdomain) combination appears once.
    combos = {(a.org, a.subdomain) for a in addrs}
    assert combos == {
        (f"org{i}", f"subdomain{j}") for i in range(4) for j in range(2)
    }


def test_mixed_prefix_and_suffix():
    """`*4` org (cross) × `2*` subdomain (max-iter, =2) → 8 agents."""
    p = ProvisioningPattern.parse("*4.r.d.2*.s.session.*")
    assert p.total() == 8
    addrs = p.materialize()
    assert len(addrs) == 8
    # Each subdomain value appears with all 4 orgs.
    sd0 = [a for a in addrs if a.subdomain == "subdomain0"]
    sd1 = [a for a in addrs if a.subdomain == "subdomain1"]
    assert sorted(a.org for a in sd0) == ["org0", "org1", "org2", "org3"]
    assert sorted(a.org for a in sd1) == ["org0", "org1", "org2", "org3"]


def test_pure_cross_join_no_max_field():
    p = ProvisioningPattern.parse("*3.r.d.x.s.session.y")
    assert p.total() == 3
    orgs = [a.org for a in p.materialize()]
    assert orgs == ["org0", "org1", "org2"]


# ── custom namer ────────────────────────────────────────────────────────


def test_custom_namer():
    p = ProvisioningPattern.parse("4*.adversarial.finance.2*.s.session.*")
    pool = {
        "org": ["nike", "adidas", "coke", "pepsi"],
        "subdomain": ["sales", "marketing"],
        "instance": ["alpha"],
    }
    addrs = p.materialize(namer=lambda field, i: pool[field][i])
    orgs = [a.org for a in addrs]
    subs = [a.subdomain for a in addrs]
    assert orgs == ["nike", "adidas", "coke", "pepsi"]
    assert subs == ["sales", "marketing", "sales", "marketing"]


# ── field-level constraints ─────────────────────────────────────────────


def test_role_must_be_concrete():
    with pytest.raises(ValueError, match="role"):
        ProvisioningPattern.parse("o.*.d.sd.s.session.i")


def test_accept_default_for_star():
    p = ProvisioningPattern.parse("o.r.d.sd.*.session.i")
    addrs = p.materialize()
    assert addrs[0].accept == "s"


def test_lifecycle_default_for_star():
    p = ProvisioningPattern.parse("o.r.d.sd.s.*.i")
    addrs = p.materialize()
    assert addrs[0].lifecycle == "session"


def test_accept_rejects_count_syntax():
    with pytest.raises(ValueError, match="does not allow count syntax"):
        ProvisioningPattern.parse("o.r.d.sd.4*.session.i")


def test_lifecycle_rejects_count_syntax():
    with pytest.raises(ValueError, match="does not allow count syntax"):
        ProvisioningPattern.parse("o.r.d.sd.s.2*.i")


def test_accept_canonical_order_enforced():
    with pytest.raises(ValueError, match="canonical tier order"):
        ProvisioningPattern.parse("o.r.d.sd.js.session.i")


def test_lifecycle_value_validated():
    with pytest.raises(ValueError, match="invalid value"):
        ProvisioningPattern.parse("o.r.d.sd.s.forever.i")


# ── parsing errors ──────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", [
    "",
    "o.r.d.sd.s.session",            # too few fields
    "o.r.d.sd.s.session.i.x",        # too many
    "o.r..sd.s.session.i",           # empty field
    "o.r.d.sd.s.session.i?k=v",      # params not allowed
])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        ProvisioningPattern.parse(bad)


def test_field_zero_count_rejected():
    with pytest.raises(ValueError):
        ProvisioningField(name="org", kind="max", count=0)


# ── default namer ───────────────────────────────────────────────────────


def test_default_namer_format():
    assert default_namer("org", 0) == "org0"
    assert default_namer("subdomain", 42) == "subdomain42"


# ── dash syntax (reuse flag) ───────────────────────────────────────────


def test_no_dash_means_reuse_true():
    p = ProvisioningPattern.parse("4*.r.d.*2.s.session.*")
    assert p.org.reuse is True
    assert p.subdomain.reuse is True
    assert p.instance.reuse is True


def test_dash_prefix_means_fresh():
    p = ProvisioningPattern.parse("4-*.r.d.sd.s.session.*")
    assert p.org.kind == "max"
    assert p.org.count == 4
    assert p.org.reuse is False


def test_dash_suffix_means_fresh():
    p = ProvisioningPattern.parse("o.r.d.*-2.s.session.*")
    assert p.subdomain.kind == "cross"
    assert p.subdomain.count == 2
    assert p.subdomain.reuse is False


def test_str_round_trip_with_dash():
    spec = "4-*.adversarial.finance.*-2.s.session.*"
    p = ProvisioningPattern.parse(spec)
    assert str(p) == spec


def test_fixed_fields_have_reuse_false():
    p = ProvisioningPattern.parse("Nike.adversarial.finance.sales.s.session.alice")
    assert p.org.reuse is False
    assert p.role.reuse is False
    assert p.instance.reuse is False

