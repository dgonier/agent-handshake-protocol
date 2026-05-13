"""Tests for AddressPattern parsing and matching."""

from __future__ import annotations

import pytest

from ahp.core.address import AgentAddress
from ahp.core.pattern import WILDCARD, AddressPattern


# ── parsing ─────────────────────────────────────────────────────────────


def test_parse_full_pattern():
    pat = AddressPattern.parse("*.adversarial.science.*.s.*.*")
    assert pat.org == "*"
    assert pat.role == "adversarial"
    assert pat.domain == "science"
    assert pat.subdomain == "*"
    assert pat.accept == "s"
    assert pat.lifecycle == "*"
    assert pat.instance == "*"


def test_str_round_trip():
    s = "*.adversarial.science.*.s.*.*"
    assert str(AddressPattern.parse(s)) == s


def test_all_helper():
    pat = AddressPattern.all()
    assert all(f == WILDCARD for f in pat.fields)


@pytest.mark.parametrize("bad", [
    "",
    "a.b.c.d.s.session",                  # too few
    "a.b.c.d.s.session.i.extra",          # too many
    "a..c.d.s.session.i",                 # empty
    "a.b.c.d.s.session.i?x=1",            # params not allowed
])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        AddressPattern.parse(bad)


def test_patterns_are_hashable():
    p = AddressPattern.parse("*.*.*.*.*.*.*")
    assert hash(p) == hash(AddressPattern.parse("*.*.*.*.*.*.*"))


# ── matching: structural ────────────────────────────────────────────────


def _addr(s: str) -> AgentAddress:
    return AgentAddress.parse(s)


def test_full_wildcard_matches_anything():
    pat = AddressPattern.all()
    assert pat.matches(_addr("tifin.adversarial.finance.equities.j.longterm.frank"))
    assert pat.matches(_addr("public.human.general.x.s.ephemeral.alice"))


def test_exact_field_must_match():
    pat = AddressPattern.parse("tifin.*.*.*.*.*.*")
    assert pat.matches(_addr("tifin.adversarial.finance.equities.s.session.f"))
    assert not pat.matches(_addr("public.adversarial.finance.equities.s.session.f"))


def test_multiple_constraints():
    pat = AddressPattern.parse("*.adversarial.science.*.s.*.*")
    assert pat.matches(_addr("tifin.adversarial.science.biology.s.session.f"))
    assert pat.matches(_addr("public.adversarial.science.physics.sj.longterm.x"))
    # Wrong role
    assert not pat.matches(_addr("tifin.collaborative.science.biology.s.session.f"))
    # Wrong domain
    assert not pat.matches(_addr("tifin.adversarial.finance.equities.s.session.f"))


# ── matching: accept subset semantics ───────────────────────────────────


def test_accept_subset_match():
    """Pattern accept 's' matches any address whose accept INCLUDES 's'."""
    pat = AddressPattern.parse("*.*.*.*.s.*.*")
    assert pat.matches(_addr("o.r.d.sd.s.session.i"))
    assert pat.matches(_addr("o.r.d.sd.sj.session.i"))
    assert pat.matches(_addr("o.r.d.sd.sjbe.session.i"))
    # Address that doesn't accept 's'
    assert not pat.matches(_addr("o.r.d.sd.j.session.i"))
    assert not pat.matches(_addr("o.r.d.sd.be.session.i"))


def test_accept_subset_multiple_tiers():
    """Pattern accept 'sj' requires BOTH s and j."""
    pat = AddressPattern.parse("*.*.*.*.sj.*.*")
    assert pat.matches(_addr("o.r.d.sd.sj.session.i"))
    assert pat.matches(_addr("o.r.d.sd.sjbe.session.i"))
    assert not pat.matches(_addr("o.r.d.sd.s.session.i"))
    assert not pat.matches(_addr("o.r.d.sd.j.session.i"))


def test_accept_wildcard():
    pat = AddressPattern.parse("*.*.*.*.*.*.*")
    for accept in ["s", "j", "b", "e", "sj", "be", "sjbe"]:
        assert pat.matches(_addr(f"o.r.d.sd.{accept}.session.i"))


# ── address.matches delegates correctly ─────────────────────────────────


def test_address_matches_delegates():
    addr = _addr("tifin.adversarial.finance.equities.sj.session.f")
    pat = AddressPattern.parse("*.adversarial.*.*.s.*.*")
    assert addr.matches(pat)
    assert pat.matches(addr)


# ── accept-format helper ────────────────────────────────────────────────


def test_matches_accept_subset():
    assert AddressPattern.matches_accept("s", "s")
    assert AddressPattern.matches_accept("s", "sj")
    assert AddressPattern.matches_accept("s", "sjbe")
    assert not AddressPattern.matches_accept("j", "s")
    assert not AddressPattern.matches_accept("sj", "s")
    assert AddressPattern.matches_accept("sj", "sj")
    assert AddressPattern.matches_accept("sj", "sjbe")
