"""Tests for the settlement formula.

The formula is the contract — if these tests change, the economic
shape of the protocol has changed. Be deliberate.
"""

from __future__ import annotations

import math

import pytest

from ahp.economy.pricing import (
    BROKER_TAX_SHARE,
    COMMONS_TAX_SHARE,
    PROTOCOL_TAX_RATE,
    Settlement,
    SettlementInputs,
    estimate_hold,
    settle_payment,
    update_avg_overage,
)


# Default fixture: a neutral, honest call. Every multiplier is 1.0,
# so the math reduces to base_rate × tier_mult × chars minus the tax.
def _neutral_inputs(**overrides) -> SettlementInputs:
    base = dict(
        base_rate=0.0002,
        tier="small",
        response_chars=500,
        max_response_chars=500,
        actual_latency_ms=600.0,         # right at small-tier expected
        leaf_rate_per_1k_chars=0.00005,  # half the base — cheap provider
        completed_with_caller=0,
        server_reputation=0.5,           # 0.5 → reputation_mult = 1.0
        server_avg_overage=1.0,          # → verbosity_mult = 1.0
        tier_verdict="matched",
    )
    base.update(overrides)
    return SettlementInputs(**base)


# ── shape ──────────────────────────────────────────────────────────────


def test_settlement_amounts_split_correctly():
    s = settle_payment(_neutral_inputs())
    # All amounts sum to pre_tax + compute_cost (server is the residual
    # after compute is paid). Actually: server + compute + broker + commons
    # equals pre_tax IF compute is paid out of the residual. Let's verify
    # the documented invariant precisely.
    total_out = s.to_server + s.to_compute + s.to_broker + s.to_commons
    assert math.isclose(total_out, s.pre_tax, abs_tol=1e-9), (
        f"recipients sum to {total_out}, expected pre_tax {s.pre_tax}"
    )


def test_tax_split_matches_constants():
    s = settle_payment(_neutral_inputs())
    expected_tax = s.pre_tax * PROTOCOL_TAX_RATE
    assert math.isclose(s.total_tax, expected_tax, abs_tol=1e-9)
    assert math.isclose(
        s.to_broker, expected_tax * BROKER_TAX_SHARE, abs_tol=1e-9
    )
    assert math.isclose(
        s.to_commons, expected_tax * COMMONS_TAX_SHARE, abs_tol=1e-9
    )


def test_response_chars_clamped_to_budget():
    """A server that writes 5x the budget pays for the budget only."""
    over = settle_payment(_neutral_inputs(
        response_chars=2500, max_response_chars=500,
    ))
    on_budget = settle_payment(_neutral_inputs(
        response_chars=500, max_response_chars=500,
    ))
    assert over.pre_tax == on_budget.pre_tax
    assert over.effective_chars == 500


# ── multipliers in isolation ───────────────────────────────────────────


def test_tier_multiplier_doubles_per_step():
    """Tier scaling is 1/2/4/8 when all other multipliers are neutral.

    Each tier has a different expected latency, so the test fixture
    sets actual_latency_ms to that tier's expected to avoid triggering
    the anti-sandbagging too-fast penalty.
    """
    from ahp.economy.pricing import EXPECTED_LATENCY_MS
    tiny = settle_payment(_neutral_inputs(
        tier="tiny", actual_latency_ms=EXPECTED_LATENCY_MS["tiny"],
    ))
    small = settle_payment(_neutral_inputs(
        tier="small", actual_latency_ms=EXPECTED_LATENCY_MS["small"],
    ))
    medium = settle_payment(_neutral_inputs(
        tier="medium", actual_latency_ms=EXPECTED_LATENCY_MS["medium"],
    ))
    big = settle_payment(_neutral_inputs(
        tier="big", actual_latency_ms=EXPECTED_LATENCY_MS["big"],
    ))
    # tiny=1, small=2, medium=4, big=8
    assert math.isclose(small.pre_tax / tiny.pre_tax, 2.0, abs_tol=1e-9)
    assert math.isclose(medium.pre_tax / tiny.pre_tax, 4.0, abs_tol=1e-9)
    assert math.isclose(big.pre_tax / tiny.pre_tax, 8.0, abs_tol=1e-9)


def test_reputation_swings_payment_50pct_below_to_50pct_above():
    bad = settle_payment(_neutral_inputs(server_reputation=0.0))
    neutral = settle_payment(_neutral_inputs(server_reputation=0.5))
    perfect = settle_payment(_neutral_inputs(server_reputation=1.0))
    # reputation_mult = 0.5 + r, so 0.5x, 1.0x, 1.5x against neutral.
    assert math.isclose(bad.pre_tax / neutral.pre_tax, 0.5, abs_tol=1e-9)
    assert math.isclose(perfect.pre_tax / neutral.pre_tax, 1.5, abs_tol=1e-9)


def test_retention_grows_log_with_completed_calls():
    fresh = settle_payment(_neutral_inputs(completed_with_caller=0))
    bumpy = settle_payment(_neutral_inputs(completed_with_caller=9))
    deep = settle_payment(_neutral_inputs(completed_with_caller=999))
    # 1 + 0.05 * log10(1+N):  N=0→1.0  N=9→1.05  N=999→1.15
    assert math.isclose(bumpy.pre_tax / fresh.pre_tax, 1.05, abs_tol=1e-9)
    assert math.isclose(deep.pre_tax / fresh.pre_tax, 1.15, abs_tol=1e-9)


def test_verbosity_penalizes_chronic_bloat():
    concise = settle_payment(_neutral_inputs(server_avg_overage=0.9))
    neutral = settle_payment(_neutral_inputs(server_avg_overage=1.0))
    chatty = settle_payment(_neutral_inputs(server_avg_overage=1.3))
    bloat = settle_payment(_neutral_inputs(server_avg_overage=1.5))
    # verbosity_mult = clip(2 - avg_overage, 0.5, 1.1)
    assert math.isclose(concise.pre_tax / neutral.pre_tax, 1.1, abs_tol=1e-9)
    assert math.isclose(chatty.pre_tax / neutral.pre_tax, 0.7, abs_tol=1e-9)
    assert math.isclose(bloat.pre_tax / neutral.pre_tax, 0.5, abs_tol=1e-9)


def test_compute_mult_flags_sandbagging():
    # Caller flagged "sub_tier" → 0.5×.
    sandbag_verdict = settle_payment(_neutral_inputs(tier_verdict="sub_tier"))
    matched = settle_payment(_neutral_inputs(tier_verdict="matched"))
    assert math.isclose(sandbag_verdict.pre_tax / matched.pre_tax, 0.5, abs_tol=1e-9)

    # Too-fast latency at the claimed tier → another 0.5×.
    too_fast = settle_payment(_neutral_inputs(
        actual_latency_ms=100.0,  # << 0.3 × 600 = 180ms threshold
    ))
    assert math.isclose(too_fast.pre_tax / matched.pre_tax, 0.5, abs_tol=1e-9)

    # Both signals: 0.25×.
    both = settle_payment(_neutral_inputs(
        actual_latency_ms=100.0, tier_verdict="sub_tier",
    ))
    assert math.isclose(both.pre_tax / matched.pre_tax, 0.25, abs_tol=1e-9)


# ── compound scenarios ────────────────────────────────────────────────


def test_trusted_returning_concise_server_earns_premium():
    """The full positive-stack: high rep + relationship + concise + honest."""
    neutral = settle_payment(_neutral_inputs())
    premium = settle_payment(_neutral_inputs(
        server_reputation=1.0,
        completed_with_caller=999,
        server_avg_overage=0.9,
        # compute mult stays 1.0 — no sandbagging signal
    ))
    # 1.0 retention × 1.5 rep × 1.1 verbosity × 1.0 compute = 1.65 × 1.15
    # neutral = 1.0 × 1.0 × 1.0 × 1.0 = 1.0
    expected = 1.15 * 1.5 * 1.1 * 1.0
    assert math.isclose(premium.pre_tax / neutral.pre_tax, expected, abs_tol=1e-9)


def test_cheating_bloated_server_earns_pittance():
    """The full negative-stack: low rep + bloat + sandbagging."""
    neutral = settle_payment(_neutral_inputs())
    cheat = settle_payment(_neutral_inputs(
        server_reputation=0.0,
        server_avg_overage=1.5,
        tier_verdict="sub_tier",
        actual_latency_ms=50.0,
    ))
    # 1.0 retention × 0.5 rep × 0.5 verbosity × 0.25 compute = 0.0625
    expected = 1.0 * 0.5 * 0.5 * 0.25
    assert math.isclose(cheat.pre_tax / neutral.pre_tax, expected, abs_tol=1e-9)


def test_server_residual_negative_when_compute_too_expensive():
    """If leaf rate > what the formula yields, server's slice goes negative."""
    s = settle_payment(_neutral_inputs(
        base_rate=0.0001,
        leaf_rate_per_1k_chars=1.00,  # 5000x base — absurd
    ))
    # Pre-tax revenue = 0.0001 × 2 × 500 = 0.1 credits
    # Compute cost = 1.00 × 0.5 = 0.5 credits
    # → server gets paid negative, broker is expected to reject this dispatch
    assert s.to_server < 0


# ── hold estimation ───────────────────────────────────────────────────


def test_estimate_hold_covers_pre_tax_at_neutral_multipliers():
    """The hold should exactly equal a neutral-multiplier settlement."""
    hold = estimate_hold(
        base_rate=0.0002,
        tier="small",
        prompt_chars=400,
        max_response_chars=500,
        leaf_rate_per_1k_chars=0.0,
    )
    # neutral settlement: 0.0002 × 2 × (400+500) = 0.36
    assert math.isclose(hold.amount, 0.36, abs_tol=1e-9)
    assert hold.estimated_chars == 900


# ── rolling avg_overage ───────────────────────────────────────────────


def test_avg_overage_converges_to_steady_input():
    """Repeated samples at ratio X should converge ``current`` to X."""
    current = 1.0
    for _ in range(200):
        current = update_avg_overage(current, 600, 500)
    assert math.isclose(current, 1.2, abs_tol=1e-3)


def test_avg_overage_one_outlier_barely_moves_average():
    current = 1.0
    after_one_outlier = update_avg_overage(current, 2500, 500)
    # λ=0.05: new = 0.95 × 1.0 + 0.05 × 5.0 = 0.95 + 0.25 = 1.20
    assert math.isclose(after_one_outlier, 1.20, abs_tol=1e-9)


def test_avg_overage_no_change_on_zero_budget():
    """Defensive: max_response_chars=0 returns current unchanged."""
    current = 1.05
    assert update_avg_overage(current, 100, 0) == current
