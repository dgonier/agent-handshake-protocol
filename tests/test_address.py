"""Tests for AgentAddress parsing, validation, and canonicalization."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from ahp.core.address import (
    ACCEPT_TIER_ORDER,
    VALID_ACCEPT_CHARS,
    VALID_LIFECYCLES,
    AgentAddress,
    _canonical_accept,
)


SAMPLE_URI = "tifin.adversarial.finance.projections.j.longterm.frank?stock=Tesla"


# ── parsing & round-trip ────────────────────────────────────────────────


def test_parse_full_uri():
    addr = AgentAddress.parse(SAMPLE_URI)
    assert addr.org == "tifin"
    assert addr.role == "adversarial"
    assert addr.domain == "finance"
    assert addr.subdomain == "projections"
    assert addr.accept == "j"
    assert addr.lifecycle == "longterm"
    assert addr.instance == "frank"
    assert addr.params == {"stock": "Tesla"}


def test_parse_no_params():
    addr = AgentAddress.parse("a.b.c.d.s.session.x")
    assert addr.params == {}
    assert str(addr) == "a.b.c.d.s.session.x"


def test_str_round_trip():
    addr = AgentAddress.parse(SAMPLE_URI)
    assert AgentAddress.parse(str(addr)) == addr


def test_param_order_irrelevant_for_equality():
    a = AgentAddress.parse("o.r.d.sd.s.session.i?a=1&b=2")
    b = AgentAddress.parse("o.r.d.sd.s.session.i?b=2&a=1")
    assert a == b
    assert hash(a) == hash(b)
    # str is canonical — sorted
    assert str(a) == str(b) == "o.r.d.sd.s.session.i?a=1&b=2"


def test_params_url_encoded():
    addr = AgentAddress(
        org="o", role="r", domain="d", subdomain="sd",
        accept="s", lifecycle="session", instance="i",
        params={"q": "hello world", "k": "a&b=c"},
    )
    # Should round-trip even with reserved chars.
    assert AgentAddress.parse(str(addr)) == addr


# ── validation: structural fields ───────────────────────────────────────


@pytest.mark.parametrize("bad_uri", [
    "",
    "a.b.c.d.s.session",                       # too few
    "a.b.c.d.s.session.i.extra",               # too many
    "a..c.d.s.session.i",                      # empty token
    "a.b.c.d.s.session.",                      # trailing empty
])
def test_parse_rejects_malformed(bad_uri):
    with pytest.raises(ValueError):
        AgentAddress.parse(bad_uri)


def test_parse_rejects_non_string():
    with pytest.raises(TypeError):
        AgentAddress.parse(12345)  # type: ignore[arg-type]


# ── validation: accept tier ─────────────────────────────────────────────


@pytest.mark.parametrize("accept", ["s", "j", "b", "e", "sj", "be", "sjbe"])
def test_accept_canonical_order_accepted(accept):
    addr = AgentAddress(
        org="o", role="r", domain="d", subdomain="sd",
        accept=accept, lifecycle="session", instance="i",
    )
    assert addr.accept == accept


@pytest.mark.parametrize("bad", ["js", "es", "ejbs", "bs", "ej"])
def test_accept_rejects_non_canonical_order(bad):
    with pytest.raises(ValueError, match="canonical tier order"):
        AgentAddress(
            org="o", role="r", domain="d", subdomain="sd",
            accept=bad, lifecycle="session", instance="i",
        )


def test_canonical_accept_helper():
    assert _canonical_accept("js") == "sj"
    assert _canonical_accept("ejbs") == "sjbe"
    assert _canonical_accept("eb") == "be"


def test_accept_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        AgentAddress(
            org="o", role="r", domain="d", subdomain="sd",
            accept="ss", lifecycle="session", instance="i",
        )


def test_accept_rejects_invalid_chars():
    with pytest.raises(ValueError, match="invalid characters"):
        AgentAddress(
            org="o", role="r", domain="d", subdomain="sd",
            accept="x", lifecycle="session", instance="i",
        )


def test_accept_rejects_empty():
    with pytest.raises(ValueError):
        AgentAddress(
            org="o", role="r", domain="d", subdomain="sd",
            accept="", lifecycle="session", instance="i",
        )


# ── validation: lifecycle ───────────────────────────────────────────────


@pytest.mark.parametrize("lc", sorted(VALID_LIFECYCLES))
def test_lifecycle_accepts_valid(lc):
    addr = AgentAddress(
        org="o", role="r", domain="d", subdomain="sd",
        accept="s", lifecycle=lc, instance="i",
    )
    assert addr.lifecycle == lc


def test_lifecycle_rejects_invalid():
    with pytest.raises(ValueError, match="invalid lifecycle"):
        AgentAddress(
            org="o", role="r", domain="d", subdomain="sd",
            accept="s", lifecycle="forever", instance="i",
        )


def test_stale_ok_lifecycle_parses():
    addr = AgentAddress.parse("o.r.d.sd.s.stale-ok.i")
    assert addr.lifecycle == "stale-ok"
    assert str(addr) == "o.r.d.sd.s.stale-ok.i"


# ── validation: tokens cannot contain delimiters ────────────────────────


@pytest.mark.parametrize("bad", ["with.dot", "with?question", "with space"])
def test_token_rejects_forbidden_chars(bad):
    with pytest.raises(ValueError, match="forbidden"):
        AgentAddress(
            org=bad, role="r", domain="d", subdomain="sd",
            accept="s", lifecycle="session", instance="i",
        )


def test_hyphen_allowed_in_tokens():
    addr = AgentAddress.parse("user-devin.human.finance.equities.s.session.dev-1")
    assert addr.org == "user-devin"
    assert addr.instance == "dev-1"


# ── helpers ─────────────────────────────────────────────────────────────


def test_accepts():
    addr = AgentAddress.parse("o.r.d.sd.sj.session.i")
    assert addr.accepts("s")
    assert addr.accepts("j")
    assert not addr.accepts("b")
    with pytest.raises(ValueError):
        addr.accepts("sj")


def test_accepts_any():
    addr = AgentAddress.parse("o.r.d.sd.sj.session.i")
    assert addr.accepts_any({"s", "b"})
    assert addr.accepts_any({"j"})
    assert not addr.accepts_any({"b", "e"})


def test_is_human():
    human = AgentAddress.parse("o.human.d.sd.s.session.i")
    bot = AgentAddress.parse("o.collaborative.d.sd.s.session.i")
    assert human.is_human
    assert not bot.is_human


def test_cache_key_stable():
    a = AgentAddress.parse("o.r.d.sd.s.session.i?x=1&y=2")
    b = AgentAddress.parse("o.r.d.sd.s.session.i?y=2&x=1")
    assert a.cache_key("interview.text") == b.cache_key("interview.text")
    assert a.cache_key("interview.text") != a.cache_key("interview.schema")


def test_cache_key_is_hex_sha256():
    addr = AgentAddress.parse("o.r.d.sd.s.session.i")
    key = addr.cache_key("c.x")
    assert len(key) == 64
    int(key, 16)  # parses as hex


def test_fields_property():
    addr = AgentAddress.parse(SAMPLE_URI)
    assert addr.fields == (
        "tifin", "adversarial", "finance", "projections",
        "j", "longterm", "frank",
    )


# ── property tests ──────────────────────────────────────────────────────


_token = st.text(
    alphabet=st.characters(
        whitelist_categories=("Ll", "Lu", "Nd"),
        whitelist_characters="-_",
    ),
    min_size=1,
    max_size=20,
)


def _accept_strategy():
    # Non-empty subset of ACCEPT_TIER_ORDER, in canonical tier order.
    return st.sets(
        st.sampled_from(list(ACCEPT_TIER_ORDER)),
        min_size=1,
        max_size=len(ACCEPT_TIER_ORDER),
    ).map(_canonical_accept)


_lifecycle = st.sampled_from(sorted(VALID_LIFECYCLES))


_param_key = st.text(min_size=1, max_size=8).filter(
    lambda s: "=" not in s and "&" not in s and "?" not in s
)
_param_val = st.text(min_size=0, max_size=20).filter(
    lambda s: "\x00" not in s
)


@given(
    org=_token, role=_token, domain=_token, subdomain=_token,
    accept=_accept_strategy(), lifecycle=_lifecycle, instance=_token,
    params=st.dictionaries(_param_key, _param_val, max_size=5),
)
def test_round_trip_property(org, role, domain, subdomain, accept, lifecycle, instance, params):
    """Any valid address survives str → parse → equal."""
    addr = AgentAddress(
        org=org, role=role, domain=domain, subdomain=subdomain,
        accept=accept, lifecycle=lifecycle, instance=instance,
        params=params,
    )
    parsed = AgentAddress.parse(str(addr))
    assert parsed == addr
    assert hash(parsed) == hash(addr)


@given(accept=_accept_strategy())
def test_accept_always_canonical(accept):
    assert accept == _canonical_accept(accept)
    assert len(set(accept)) == len(accept)
