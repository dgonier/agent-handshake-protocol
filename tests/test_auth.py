"""Tests for the registry's auth control plane.

Invariants:

1. Default policy is open — no behavior change when no policy is set.
2. ``AddressClaimPolicy`` denies registers/deregisters/heartbeats from
   principals whose claims don't cover the target address.
3. ``AddressClaimPolicy`` denies anonymous (``None``) principals.
4. ``OpenAuthPolicy`` accepts everything, including anonymous.
5. ``DenyAllPolicy`` refuses everything (useful for read-only mirrors).
6. The auth gate fires BEFORE Redis state changes — a denied
   register leaves no trace.
"""

from __future__ import annotations

import pytest

from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern
from ahp.registry import (
    AddressClaimPolicy,
    AgentMeta,
    AgentRegistry,
    DenyAllPolicy,
    OpenAuthPolicy,
    Principal,
    UnauthorizedRegistrationError,
)


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── Principal mechanics ────────────────────────────────────────────────


def test_principal_with_claims_parses_strings():
    p = Principal.with_claims(
        "node-a",
        "tifin.adversarial.finance.*.*.*.*",
        "tifin.collaborative.*.*.*.*.*",
    )
    assert len(p.claims) == 2
    assert all(isinstance(c, AddressPattern) for c in p.claims)


def test_principal_covers_checks_any_claim():
    p = Principal.with_claims(
        "node-a",
        "tifin.adversarial.finance.*.*.*.*",
        "tifin.collaborative.*.*.*.*.*",
    )
    assert p.covers(_addr("tifin.adversarial.finance.equities.s.session.f"))
    assert p.covers(_addr("tifin.collaborative.science.x.s.session.f"))
    assert not p.covers(_addr("public.adversarial.finance.x.s.session.f"))
    assert not p.covers(_addr("tifin.interview.science.x.s.session.f"))


def test_principal_with_no_claims_covers_nothing():
    p = Principal(id="anon", claims=())
    assert not p.covers(_addr("tifin.adversarial.finance.x.s.session.f"))


# ── Policy decisions in isolation ─────────────────────────────────────


def test_open_policy_allows_everything():
    pol = OpenAuthPolicy()
    addr = _addr("tifin.adversarial.finance.x.s.session.f")
    # Anonymous principal AND any other; both pass.
    assert pol.can_register(None, addr)
    assert pol.can_register(Principal(id="x", claims=()), addr)
    assert pol.can_deregister(None, addr)
    assert pol.can_heartbeat(None, addr)


def test_deny_all_policy_blocks_everything():
    pol = DenyAllPolicy()
    addr = _addr("tifin.adversarial.finance.x.s.session.f")
    p = Principal.with_claims("admin", "*.*.*.*.*.*.*")
    assert not pol.can_register(p, addr)
    assert not pol.can_deregister(p, addr)


def test_address_claim_policy_denies_anonymous():
    pol = AddressClaimPolicy()
    addr = _addr("tifin.adversarial.finance.x.s.session.f")
    assert not pol.can_register(None, addr)


def test_address_claim_policy_matches_against_claims():
    pol = AddressClaimPolicy()
    p = Principal.with_claims("node-a", "tifin.adversarial.*.*.*.*.*")
    in_scope = _addr("tifin.adversarial.finance.x.s.session.f")
    out_of_scope = _addr("tifin.collaborative.finance.x.s.session.f")
    assert pol.can_register(p, in_scope)
    assert not pol.can_register(p, out_of_scope)


# ── Registry-integration tests ────────────────────────────────────────


async def test_open_default_preserves_existing_behavior(redis_client):
    """No policy / no principal = current open-registration behavior."""
    registry = AgentRegistry(redis_client)
    addr = _addr("anyone.adversarial.x.y.s.session.i")
    await registry.register(addr)
    assert await registry.is_alive(addr)
    meta = await registry.get(addr)
    assert meta is not None


async def test_register_denied_for_uncovered_address(redis_client):
    principal = Principal.with_claims(
        "node-a", "tifin.adversarial.*.*.*.*.*",
    )
    registry = AgentRegistry(
        redis_client,
        principal=principal,
        policy=AddressClaimPolicy(),
    )
    foreign_addr = _addr("public.collaborative.x.y.s.session.f")
    with pytest.raises(UnauthorizedRegistrationError, match="node-a"):
        await registry.register(foreign_addr)
    # Confirm Redis never saw the write.
    assert await registry.get(foreign_addr) is None


async def test_register_allowed_for_covered_address(redis_client):
    principal = Principal.with_claims(
        "node-a", "tifin.adversarial.*.*.*.*.*",
    )
    registry = AgentRegistry(
        redis_client, principal=principal, policy=AddressClaimPolicy(),
    )
    addr = _addr("tifin.adversarial.finance.equities.s.session.bull")
    await registry.register(addr, AgentMeta(capabilities=["debate"]))
    assert await registry.is_alive(addr)


async def test_deregister_gated_by_policy(redis_client):
    # node-a sets it up; node-b can't tear it down.
    node_a = Principal.with_claims("node-a", "tifin.*.*.*.*.*.*")
    node_b = Principal.with_claims("node-b", "public.*.*.*.*.*.*")

    reg_a = AgentRegistry(
        redis_client, principal=node_a, policy=AddressClaimPolicy(),
    )
    reg_b = AgentRegistry(
        redis_client, principal=node_b, policy=AddressClaimPolicy(),
    )
    addr = _addr("tifin.adversarial.finance.equities.s.session.bull")
    await reg_a.register(addr)

    with pytest.raises(UnauthorizedRegistrationError):
        await reg_b.deregister(addr)
    assert await reg_a.is_alive(addr)   # still alive

    # The original owner can.
    await reg_a.deregister(addr)
    assert not await reg_a.is_alive(addr)


async def test_heartbeat_gated_by_policy(redis_client):
    node_a = Principal.with_claims("node-a", "tifin.*.*.*.*.*.*")
    foreign = Principal.with_claims("attacker", "public.*.*.*.*.*.*")

    reg_a = AgentRegistry(
        redis_client, principal=node_a, policy=AddressClaimPolicy(),
    )
    reg_foreign = AgentRegistry(
        redis_client, principal=foreign, policy=AddressClaimPolicy(),
    )
    addr = _addr("tifin.adversarial.finance.equities.s.session.bull")
    await reg_a.register(addr)

    with pytest.raises(UnauthorizedRegistrationError):
        await reg_foreign.heartbeat(addr)


async def test_anonymous_principal_under_claim_policy_is_denied(redis_client):
    registry = AgentRegistry(
        redis_client,
        principal=None,                  # anonymous
        policy=AddressClaimPolicy(),
    )
    with pytest.raises(UnauthorizedRegistrationError):
        await registry.register(_addr("tifin.adversarial.x.y.s.session.i"))


async def test_deny_all_blocks_even_with_principal(redis_client):
    p = Principal.with_claims("admin", "*.*.*.*.*.*.*")
    registry = AgentRegistry(
        redis_client, principal=p, policy=DenyAllPolicy(),
    )
    with pytest.raises(UnauthorizedRegistrationError):
        await registry.register(_addr("tifin.adversarial.x.y.s.session.i"))


# ── Read paths are NOT gated ──────────────────────────────────────────


async def test_resolve_and_discover_remain_open(redis_client):
    """Auth gates writes; resolution/discovery stays open.

    Adding read-side gates is a separate concern (and may not be
    desirable for federation, where every node needs to see who's on
    the network).
    """
    node_a = Principal.with_claims("node-a", "tifin.*.*.*.*.*.*")
    reg_a = AgentRegistry(
        redis_client, principal=node_a, policy=AddressClaimPolicy(),
    )
    await reg_a.register(
        _addr("tifin.adversarial.finance.equities.s.session.bull"),
    )

    # A different node with NO claims can still discover.
    reg_b = AgentRegistry(
        redis_client,
        principal=Principal.with_claims("node-b"),    # empty claims
        policy=AddressClaimPolicy(),
    )
    found = await reg_b.resolve(AddressPattern.parse("*.adversarial.*.*.*.*.*"))
    assert len(found) == 1
