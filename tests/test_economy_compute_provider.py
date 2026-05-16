"""Tests for the compute provider menu + pattern resolver."""

from __future__ import annotations

import pytest

from ahp.economy.compute_provider import (
    ComputeProvider,
    MenuLeaf,
    best_leaf,
    matching_leaves,
    rank_leaves,
)


def _menu() -> list[MenuLeaf]:
    """A representative seven-leaf compute menu spanning all four tiers
    and three providers. Used as the test fixture for pattern matching
    and ranking.
    """
    return [
        MenuLeaf("bedrock-east", "big", "claude-opus-4-7",
                 rate_per_1k_chars=0.003, latency_p95_ms=4200, capacity=0.85),
        MenuLeaf("bedrock-east", "big", "claude-opus-4-1",
                 rate_per_1k_chars=0.0026, latency_p95_ms=3900, capacity=0.85),
        MenuLeaf("bedrock-east", "small", "haiku-4-5",
                 rate_per_1k_chars=0.0002, latency_p95_ms=600, capacity=0.95),
        MenuLeaf("groq-public", "medium", "llama-405b",
                 rate_per_1k_chars=0.0006, latency_p95_ms=1100, capacity=0.40),
        MenuLeaf("groq-public", "medium", "mistral-large-2",
                 rate_per_1k_chars=0.0004, latency_p95_ms=550, capacity=0.85),
        MenuLeaf("laptop-mlx", "small", "phi-4",
                 rate_per_1k_chars=0.00005, latency_p95_ms=900, capacity=0.30),
        MenuLeaf("laptop-mlx", "tiny", "qwen3-0.5b",
                 rate_per_1k_chars=0.00001, latency_p95_ms=200, capacity=0.50),
    ]


# ── leaf validation ──────────────────────────────────────────────────


def test_leaf_validation_rejects_bad_tier():
    with pytest.raises(ValueError):
        MenuLeaf("p", "frontier", "model", rate_per_1k_chars=0.01)


def test_leaf_validation_rejects_negative_rate():
    with pytest.raises(ValueError):
        MenuLeaf("p", "tiny", "model", rate_per_1k_chars=-0.001)


def test_leaf_validation_rejects_oob_capacity():
    with pytest.raises(ValueError):
        MenuLeaf("p", "tiny", "model", rate_per_1k_chars=0.01, capacity=1.5)


def test_leaf_address_is_three_field():
    leaf = MenuLeaf("bedrock-east", "big", "claude-opus-4-7", rate_per_1k_chars=0.003)
    assert leaf.address == "bedrock-east.big.claude-opus-4-7"


# ── pattern matching ─────────────────────────────────────────────────


def test_pattern_any_big_opus():
    leaves = _menu()
    matches = matching_leaves("*.big.opus*", leaves)
    # None of the models start with "opus" (they start with "claude-opus").
    # The pattern needs the full prefix.
    assert matches == []


def test_pattern_full_big_claude_opus():
    leaves = _menu()
    matches = matching_leaves("*.big.claude-opus*", leaves)
    assert len(matches) == 2
    assert all(m.tier == "big" for m in matches)


def test_pattern_provider_specific():
    leaves = _menu()
    matches = matching_leaves("groq-public.*.*", leaves)
    assert len(matches) == 2
    assert all(m.provider_id == "groq-public" for m in matches)


def test_pattern_no_match_returns_empty():
    leaves = _menu()
    matches = matching_leaves("nope.*.*", leaves)
    assert matches == []


# ── ranking ──────────────────────────────────────────────────────────


def test_rank_cheapest_picks_lowest_rate():
    leaves = matching_leaves("*.medium.*", _menu())
    ranked = rank_leaves(leaves, rank_by="cheapest")
    assert ranked[0][0].rate_per_1k_chars == min(l.rate_per_1k_chars for l in leaves)


def test_rank_lowest_latency_picks_fastest():
    leaves = matching_leaves("*.medium.*", _menu())
    ranked = rank_leaves(leaves, rank_by="lowest_latency")
    assert ranked[0][0].latency_p95_ms == min(l.latency_p95_ms for l in leaves)


def test_rank_filters_unhealthy_leaves():
    leaves = [
        MenuLeaf("p1", "small", "ok", rate_per_1k_chars=0.001, healthy=True),
        MenuLeaf("p2", "small", "broken", rate_per_1k_chars=0.0001, healthy=False),
    ]
    ranked = rank_leaves(leaves, rank_by="cheapest")
    assert len(ranked) == 1
    assert ranked[0][0].provider_id == "p1"


def test_rank_deterministic_tiebreak_by_address():
    """Identical leaves at the same score sort alphabetically."""
    leaves = [
        MenuLeaf("z-provider", "small", "model", rate_per_1k_chars=0.001,
                 latency_p95_ms=500, capacity=0.5),
        MenuLeaf("a-provider", "small", "model", rate_per_1k_chars=0.001,
                 latency_p95_ms=500, capacity=0.5),
    ]
    ranked = rank_leaves(leaves, rank_by="cheapest")
    # All signals identical → normalize_inverse returns 1.0 for both →
    # tie → alphabetical address wins.
    assert ranked[0][0].provider_id == "a-provider"


def test_best_leaf_end_to_end():
    leaves = _menu()
    leaf = best_leaf("*.big.claude-opus*", leaves, rank_by="cheapest")
    assert leaf is not None
    # Cheapest claude-opus is the 4-1 variant at 0.0026.
    assert leaf.model == "claude-opus-4-1"


def test_best_leaf_no_match_returns_none():
    leaves = _menu()
    assert best_leaf("*.big.gpt-*", leaves) is None
