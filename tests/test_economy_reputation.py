"""Tests for the reputation + visibility ledger."""

from __future__ import annotations

import math

import pytest

from ahp.economy.reputation import (
    DEFAULT_REPUTATION,
    DEFAULT_VISIBILITY,
    LAMBDA_CSAT,
    MIN_REPUTATION_FLOOR,
    REP_PENALTY_FAILURE,
    REP_REWARD_SUCCESS,
    VISIBILITY_FULL_AT,
    ReputationEntry,
    apply_csat,
    apply_outcome,
    visibility_factor,
)


def _fresh() -> ReputationEntry:
    return ReputationEntry(owner="s")


# ── outcomes ──────────────────────────────────────────────────────────


def test_accept_bumps_reputation_and_completion():
    e = _fresh()
    e = apply_outcome(e, "accepted", latency_ms=800)
    assert e.reputation == DEFAULT_REPUTATION + REP_REWARD_SUCCESS
    assert e.completed_accepted == 1
    assert e.completed_total == 1
    assert e.failed == 0
    assert math.isclose(e.avg_latency_ms, 800.0, abs_tol=1e-9)


def test_failure_is_10x_more_punitive_than_success_is_rewarding():
    e = _fresh()
    e = apply_outcome(e, "refunded")
    # Single failure drops rep by REP_PENALTY_FAILURE = 0.05
    assert math.isclose(e.reputation, DEFAULT_REPUTATION - REP_PENALTY_FAILURE)
    assert e.failed == 1


def test_repeated_cheating_drops_below_floor_quickly():
    e = _fresh()
    for _ in range(5):
        e = apply_outcome(e, "sub_tier")
    # 5 × 0.05 = 0.25 drop from 0.5 → 0.25, which IS below the floor (0.30)
    assert e.reputation < MIN_REPUTATION_FLOOR


def test_verbose_outcome_is_mild():
    """Verbose is a small rep hit but still counts as completed."""
    e = _fresh()
    e = apply_outcome(e, "verbose", latency_ms=500)
    assert e.completed_total == 1
    assert e.completed_accepted == 0
    # Verbose penalty = REP_REWARD_SUCCESS (much smaller than failure)
    assert math.isclose(e.reputation, DEFAULT_REPUTATION - REP_REWARD_SUCCESS)


# ── visibility ────────────────────────────────────────────────────────


def test_fresh_server_starts_at_default_visibility():
    assert math.isclose(visibility_factor(_fresh()), DEFAULT_VISIBILITY)


def test_visibility_grows_with_completions():
    fresh = _fresh()
    middling = _fresh()
    middling.completed_accepted = 50
    seasoned = _fresh()
    seasoned.completed_accepted = VISIBILITY_FULL_AT

    v_fresh = visibility_factor(fresh)
    v_middling = visibility_factor(middling)
    v_seasoned = visibility_factor(seasoned)

    assert v_fresh < v_middling < v_seasoned
    assert math.isclose(v_seasoned, 1.0, abs_tol=1e-9)


def test_visibility_zero_below_reputation_floor():
    e = _fresh()
    e.reputation = MIN_REPUTATION_FLOOR - 0.01
    e.completed_accepted = 1000  # would otherwise be max visibility
    assert visibility_factor(e) == 0.0


def test_visibility_caps_at_one():
    e = _fresh()
    e.completed_accepted = VISIBILITY_FULL_AT * 100
    assert visibility_factor(e) == 1.0


# ── CSAT ──────────────────────────────────────────────────────────────


def test_csat_neutral_default():
    e = _fresh()
    assert e.csat == 0.5
    assert e.csat_samples == 0


def test_first_csat_sample_replaces_default():
    """First survey response sets CSAT directly — no half-with-default."""
    e = apply_csat(_fresh(), score=0.9)
    assert math.isclose(e.csat, 0.9, abs_tol=1e-9)
    assert e.csat_samples == 1


def test_subsequent_csat_uses_ewma():
    e = apply_csat(_fresh(), score=0.9)
    e = apply_csat(e, score=0.5)
    # EWMA: 0.9 * (1 - LAMBDA_CSAT) + 0.5 * LAMBDA_CSAT
    expected = 0.9 * (1 - LAMBDA_CSAT) + 0.5 * LAMBDA_CSAT
    assert math.isclose(e.csat, expected, abs_tol=1e-9)
    assert e.csat_samples == 2


def test_csat_clamps_oob_scores():
    e = apply_csat(_fresh(), score=2.0)
    assert e.csat == 1.0
    e = apply_csat(_fresh(), score=-0.5)
    assert e.csat == 0.0


def test_csat_does_not_affect_reputation():
    """CSAT and reputation are independent dimensions."""
    e = _fresh()
    initial_rep = e.reputation
    e = apply_csat(e, score=0.0)  # awful CSAT
    assert e.reputation == initial_rep
    assert e.completed_accepted == 0
    assert e.failed == 0
