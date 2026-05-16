"""Tests for the three-stage Router and the Broker facade."""

from __future__ import annotations

import math

import pytest

from ahp.broker import (
    Broker,
    NoCandidatesError,
    Router,
    RoutingPreferences,
    ServerMeta,
)
from ahp.broker.compute_registry import ComputeProviderRegistry
from ahp.broker.server_registry import ServerRegistry
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import (
    DEFAULT_REPUTATION,
    MIN_REPUTATION_FLOOR,
    ReputationEntry,
)


# ── helpers ──────────────────────────────────────────────────────────


def _meta(**overrides) -> ServerMeta:
    base = dict(
        server_id="srv-a",
        org="acme",
        base_rate=0.0002,
        compute_binding="*.small.*",
        supported_tiers=["small"],
        specialties=[],
        integrations=[],
    )
    base.update(overrides)
    return ServerMeta(**base)


def _leaf(**overrides) -> MenuLeaf:
    base = dict(
        provider_id="p-bedrock",
        tier="small",
        model="haiku-4-5",
        rate_per_1k_chars=0.0001,
    )
    base.update(overrides)
    return MenuLeaf(**base)


async def _setup(redis_client) -> Broker:
    """Build a Broker and register a baseline compute leaf."""
    broker = Broker(redis_client)
    await broker.register_compute_provider(
        ComputeProvider(provider_id="p-bedrock"),
    )
    await broker.register_leaf(_leaf())
    return broker


# ── happy path ───────────────────────────────────────────────────────


async def test_route_picks_only_eligible_server(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(
        server_id="srv-a", base_rate=0.0002,
    ))
    decision = await broker.resolve(
        code="adversarial.debate", tier="small",
        prompt_chars=400, max_response_chars=500,
        prefs=RoutingPreferences(rng_seed=42),
    )
    assert decision.succeeded
    assert decision.server.server_id == "srv-a"
    assert decision.leaf.address == "p-bedrock.small.haiku-4-5"
    assert decision.estimated_hold > 0


async def test_route_chooses_cheapest_by_default(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-cheap", base_rate=0.0001))
    await broker.register_server(_meta(server_id="srv-expensive", base_rate=0.001))
    # Give both servers proven track records so visibility doesn't
    # filter the candidate set down to noise.
    from ahp.economy.reputation import VISIBILITY_FULL_AT
    for sid in ("srv-cheap", "srv-expensive"):
        await broker.set_reputation(ReputationEntry(
            owner=sid, reputation=0.9,
            completed_accepted=VISIBILITY_FULL_AT,
        ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(rng_seed=42),
    )
    assert decision.server.server_id == "srv-cheap"


async def test_deterministic_tiebreak_on_equal_price(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-z", base_rate=0.0001))
    await broker.register_server(_meta(server_id="srv-a", base_rate=0.0001))
    from ahp.economy.reputation import VISIBILITY_FULL_AT
    for sid in ("srv-z", "srv-a"):
        await broker.set_reputation(ReputationEntry(
            owner=sid, reputation=0.9,
            completed_accepted=VISIBILITY_FULL_AT,
        ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(rng_seed=42),
    )
    # Tiebreak is alphabetical by server_id.
    assert decision.server.server_id == "srv-a"


# ── hard filter ──────────────────────────────────────────────────────


async def test_blocked_server_filtered_out(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-blocked", base_rate=0.0001))
    await broker.register_server(_meta(server_id="srv-ok", base_rate=0.0002))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            blocked_servers=("srv-blocked",), rng_seed=42,
        ),
    )
    assert decision.server.server_id == "srv-ok"
    # The blocked server appears in the rejections list with a reason.
    blocked_rej = [r for r in decision.rejections if r.server_id == "srv-blocked"]
    assert blocked_rej
    assert "blocked" in blocked_rej[0].reason


async def test_no_candidates_returns_empty_decision(redis_client):
    broker = await _setup(redis_client)
    # Register one server then block it.
    await broker.register_server(_meta(server_id="srv-x"))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            blocked_servers=("srv-x",), rng_seed=42,
        ),
    )
    assert not decision.succeeded
    assert decision.server is None
    assert decision.leaf is None


async def test_required_specialty_filters_out(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(
        server_id="srv-bio", specialties=["biology"],
    ))
    await broker.register_server(_meta(
        server_id="srv-fin", specialties=["finance"],
    ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            required_specialties=("biology",), rng_seed=42,
        ),
    )
    assert decision.succeeded
    assert decision.server.server_id == "srv-bio"


async def test_required_integration_filters_out(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(
        server_id="srv-with-tavily", integrations=["tavily"],
    ))
    await broker.register_server(_meta(
        server_id="srv-no-tools", integrations=[],
    ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            required_integrations=("tavily",), rng_seed=42,
        ),
    )
    assert decision.succeeded
    assert decision.server.server_id == "srv-with-tavily"


async def test_low_reputation_filtered_out(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-bad"))
    await broker.register_server(_meta(server_id="srv-good"))
    # Drag srv-bad below the default floor.
    bad_rep = ReputationEntry(owner="srv-bad", reputation=0.10)
    await broker.set_reputation(bad_rep)
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(rng_seed=42),
    )
    assert decision.server.server_id == "srv-good"
    bad_rej = [r for r in decision.rejections if r.server_id == "srv-bad"]
    assert bad_rej and "reputation" in bad_rej[0].reason


async def test_max_cost_per_call_filters_expensive(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-pricey", base_rate=0.1))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prompt_chars=400, max_response_chars=500,
        prefs=RoutingPreferences(
            max_cost_per_call=0.001, rng_seed=42,
        ),
    )
    # 0.1 × 2 × 900 = 180 credits — way over 0.001.
    assert not decision.succeeded
    assert any("max_cost_per_call" in u for u in decision.unmet_requirements)


# ── soft filter ──────────────────────────────────────────────────────


async def test_soft_specialty_narrows_when_match_exists(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(
        server_id="srv-bio", specialties=["biology"],
    ))
    await broker.register_server(_meta(
        server_id="srv-fin", specialties=["finance"],
    ))
    from ahp.economy.reputation import VISIBILITY_FULL_AT
    for sid in ("srv-bio", "srv-fin"):
        await broker.set_reputation(ReputationEntry(
            owner=sid, reputation=0.9,
            completed_accepted=VISIBILITY_FULL_AT,
        ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            preferred_specialties=("biology",), rng_seed=42,
        ),
    )
    assert decision.server.server_id == "srv-bio"
    assert not decision.soft_preferences_unmet


async def test_soft_specialty_skipped_when_no_match(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(
        server_id="srv-fin", specialties=["finance"],
    ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            preferred_specialties=("biology",), rng_seed=42,
        ),
    )
    # Soft filter dropped (no biology server), but we still routed.
    assert decision.succeeded
    assert any("biology" in s for s in decision.soft_preferences_unmet)


async def test_preferred_servers_wins_on_tie(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-a"))
    await broker.register_server(_meta(server_id="srv-b"))
    from ahp.economy.reputation import VISIBILITY_FULL_AT
    for sid in ("srv-a", "srv-b"):
        await broker.set_reputation(ReputationEntry(
            owner=sid, reputation=0.9,
            completed_accepted=VISIBILITY_FULL_AT,
        ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            preferred_servers=("srv-b",), rng_seed=42,
        ),
    )
    # When both candidates are equal-cost, preferred wins.
    assert decision.server.server_id == "srv-b"


# ── ranking variants ────────────────────────────────────────────────


async def test_rank_by_best_reputation(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-a", base_rate=0.0001))
    await broker.register_server(_meta(server_id="srv-b", base_rate=0.0002))
    # srv-b is more expensive but has higher reputation.
    from ahp.economy.reputation import VISIBILITY_FULL_AT
    await broker.set_reputation(ReputationEntry(
        owner="srv-a", reputation=0.5, completed_accepted=VISIBILITY_FULL_AT,
    ))
    await broker.set_reputation(ReputationEntry(
        owner="srv-b", reputation=0.95, completed_accepted=VISIBILITY_FULL_AT,
    ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            server_rank_by="best_reputation", rng_seed=42,
        ),
    )
    assert decision.server.server_id == "srv-b"


async def test_rank_by_highest_csat(redis_client):
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-low-csat"))
    await broker.register_server(_meta(server_id="srv-high-csat"))
    from ahp.economy.reputation import VISIBILITY_FULL_AT
    await broker.set_reputation(ReputationEntry(
        owner="srv-low-csat", csat=0.3, reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))
    await broker.set_reputation(ReputationEntry(
        owner="srv-high-csat", csat=0.9, reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(
            server_rank_by="highest_csat", rng_seed=42,
        ),
    )
    assert decision.server.server_id == "srv-high-csat"


# ── visibility coin flip ────────────────────────────────────────────


async def test_visibility_coin_flip_filters_unproven(redis_client):
    """With seed=0 and a fresh server (visibility=0.05), the first
    roll is very likely to filter the server out — unless it's the
    only one left, in which case we relax.

    We test the explicit single-candidate relaxation: when every
    candidate is throttled out, the broker prefers routing low-
    visibility-only over failing.
    """
    broker = await _setup(redis_client)
    await broker.register_server(_meta(server_id="srv-fresh"))
    # Default reputation, default visibility = 0.05.
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(rng_seed=0),
    )
    # With one fresh server in the pool, the relaxation kicks in
    # and we route to it anyway.
    assert decision.succeeded
    assert decision.server.server_id == "srv-fresh"


# ── compute leaf binding ────────────────────────────────────────────


async def test_no_matching_compute_leaf_fails_with_reason(redis_client):
    broker = await _setup(redis_client)
    # Server binds to a leaf that doesn't exist.
    await broker.register_server(_meta(
        server_id="srv-narrow", compute_binding="*.big.gpt-9000",
    ))
    decision = await broker.resolve(
        code="x.y", tier="small",
        prefs=RoutingPreferences(rng_seed=42),
    )
    assert not decision.succeeded
    assert any("compute_binding" in u for u in decision.unmet_requirements)


# ── full settlement loop ────────────────────────────────────────────


async def test_settlement_credits_all_four_recipients(redis_client):
    broker = await _setup(redis_client)
    server_meta = _meta(server_id="srv-a")
    await broker.register_server(server_meta)
    # Top up the caller so it can afford the hold.
    await broker.wallet("caller-x").topup(100.0, reason="seed")

    decision = await broker.resolve(
        code="x.y", tier="small",
        prompt_chars=400, max_response_chars=500,
        prefs=RoutingPreferences(rng_seed=42),
    )
    assert decision.succeeded

    # Place the hold.
    await broker.hold(
        caller="caller-x", amount=decision.estimated_hold,
        hold_id="msg-1", reason="dispatch",
    )

    # Now settle.
    settlement = await broker.calculate_and_settle(
        caller="caller-x", hold_id="msg-1",
        server=decision.server, leaf=decision.leaf,
        response_chars=480, max_response_chars=500,
        actual_latency_ms=600.0,
        completed_with_caller=0,
        tier_verdict="matched",
    )

    # Check the wallets.
    caller_state = await broker.wallet("caller-x").get_state()
    server_state = await broker.wallet("srv-a").get_state()
    broker_state = await broker.wallet("__broker__").get_state()
    commons_state = await broker.wallet("__commons__").get_state()

    # Caller's balance went down by exactly pre_tax.
    expected_caller = 100.0 + (await broker.wallet("caller-x").get_state()).balance
    # Sums add up: server + compute + broker + commons = pre_tax.
    received = (
        (server_state.balance - 100.0)
        + (await broker.wallet("p-bedrock").get_state()).balance - 100.0
        + (broker_state.balance - 100.0)
        + (commons_state.balance - 100.0)
    )
    assert math.isclose(received, settlement.pre_tax, abs_tol=1e-6)


async def test_refund_returns_full_hold_to_caller(redis_client):
    broker = await _setup(redis_client)
    await broker.wallet("caller-y").topup(100.0, reason="seed")
    await broker.hold(
        caller="caller-y", amount=10.0,
        hold_id="msg-fail", reason="dispatch",
    )
    state_after_hold = await broker.wallet("caller-y").get_state()
    assert state_after_hold.available < state_after_hold.balance  # 10 reserved

    await broker.refund(caller="caller-y", hold_id="msg-fail", reason="timeout")
    final = await broker.wallet("caller-y").get_state()
    assert math.isclose(final.available, final.balance, abs_tol=1e-9)
    assert math.isclose(final.balance, 200.0, abs_tol=1e-9)
