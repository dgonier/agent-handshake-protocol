"""Compute providers and their menus.

A *compute provider* publishes a menu of ``(tier, model)`` leaves.
Each leaf is a single concrete model the provider offers at a stated
rate per 1k characters, with an advertised latency and a capacity
signal.

Servers don't bind to a specific leaf at registration; they bind to a
**pattern** (e.g. ``"*.frontier.opus*"``) and the broker resolves the
pattern against every compute provider's live menu at dispatch time,
ranks the matches, and picks the top one.

This module is pure data — dataclasses, validation, pattern matching.
The broker (:mod:`ahp.broker`) is responsible for persisting these to
Redis under ``ahp:compute_menu:<provider>.<tier>.<model>``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

from ahp.economy.tiers import Tier, parse_tier


RankBy = Literal[
    "cheapest", "lowest_latency", "best_uptime", "best_reputation",
]
"""How a server wants the broker to rank matching menu leaves."""


# Default composite-score weights when rank_by="cheapest". Each ranking
# strategy is just a reweighting of these four signals.
_DEFAULT_WEIGHTS: dict[RankBy, tuple[float, float, float, float]] = {
    # (price, latency_p95, capacity, reputation)
    "cheapest":         (0.60, 0.20, 0.15, 0.05),
    "lowest_latency":   (0.15, 0.60, 0.20, 0.05),
    "best_uptime":      (0.15, 0.20, 0.60, 0.05),
    "best_reputation":  (0.15, 0.10, 0.10, 0.65),
}


@dataclass(frozen=True)
class ComputeProvider:
    """Registry entry for a compute provider.

    The provider's *menu* lives in separate :class:`MenuLeaf` records.
    This dataclass is the entity-level metadata.
    """

    provider_id: str
    operator: str = ""                # human-readable owner
    endpoint: str = ""                # how servers actually call them
    region: str = ""
    reputation: float = 0.5           # broker-tracked, 0..1
    public_key: str = ""              # for signed handshake (future)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MenuLeaf:
    """One (provider, tier, model) entry on a compute menu.

    ``address`` is the canonical addressable string
    ``"{provider}.{tier}.{model}"`` used by server compute-binding
    patterns. ``rate_per_1k_chars`` is what the compute provider
    charges per 1000 characters of (prompt + response). Latency and
    capacity are advertised signals the provider self-reports; the
    broker uses them for ranking but truth is enforced via the
    settlement reputation loop.
    """

    provider_id: str
    tier: Tier
    model: str                        # e.g. "claude-opus-4-7"
    rate_per_1k_chars: float
    latency_p95_ms: float = 1000.0
    capacity: float = 1.0             # 0..1 hint: 1.0 = wide open
    healthy: bool = True

    def __post_init__(self) -> None:
        parse_tier(self.tier)
        if self.rate_per_1k_chars < 0:
            raise ValueError(
                f"rate_per_1k_chars must be non-negative, got {self.rate_per_1k_chars}"
            )
        if not (0.0 <= self.capacity <= 1.0):
            raise ValueError(
                f"capacity must be in [0,1], got {self.capacity}"
            )
        if self.latency_p95_ms < 0:
            raise ValueError(
                f"latency_p95_ms must be non-negative, got {self.latency_p95_ms}"
            )
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", self.provider_id):
            raise ValueError(
                f"provider_id must be lowercase kebab-case: {self.provider_id!r}"
            )
        if not re.match(r"^[a-z0-9][a-z0-9._-]*$", self.model):
            raise ValueError(
                f"model must be lowercase kebab/dot/underscore: {self.model!r}"
            )

    @property
    def address(self) -> str:
        """Canonical ``provider.tier.model`` string used by binding patterns."""
        return f"{self.provider_id}.{self.tier}.{self.model}"


# ── pattern matching ──────────────────────────────────────────────────


def _pattern_matches(pattern: str, address: str) -> bool:
    """Glob match the three-field address against a glob pattern.

    Field boundaries are dots; wildcards within a field use ``*``.
    So ``"*.frontier.opus*"`` matches
    ``"bedrock-east.frontier.claude-opus-4-7"`` and
    ``"groq.frontier.opus-mini"`` but not
    ``"bedrock-east.medium.haiku-4-5"``.
    """
    p_parts = pattern.split(".")
    a_parts = address.split(".")
    if len(p_parts) != 3 or len(a_parts) != 3:
        return False
    for pp, aa in zip(p_parts, a_parts):
        # Wildcard-aware fnmatch-style without importing fnmatch:
        # convert the glob to a regex once per call. Cheap at our scale.
        regex = "^" + re.escape(pp).replace(r"\*", ".*") + "$"
        if not re.match(regex, aa):
            return False
    return True


def matching_leaves(
    pattern: str, leaves: Iterable[MenuLeaf],
) -> list[MenuLeaf]:
    """Return only the leaves whose canonical address matches ``pattern``."""
    return [leaf for leaf in leaves if _pattern_matches(pattern, leaf.address)]


# ── ranking ───────────────────────────────────────────────────────────


def _normalize_inverse(values: list[float]) -> list[float]:
    """Map a list to [0,1] where the *lowest* original value scores 1.0.

    Used for price + latency where smaller is better. Constant inputs
    produce all 1.0.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [1.0 - (v - lo) / (hi - lo) for v in values]


def _normalize_direct(values: list[float]) -> list[float]:
    """Map a list to [0,1] where the *highest* original value scores 1.0."""
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_leaves(
    leaves: list[MenuLeaf],
    *,
    rank_by: RankBy = "cheapest",
    provider_reputations: dict[str, float] | None = None,
) -> list[tuple[MenuLeaf, float]]:
    """Rank a candidate set of menu leaves by the composite score.

    Returns the leaves paired with their score, sorted best-first.
    Unhealthy leaves are filtered out entirely. Ties are broken by
    leaf address (alphabetical) for deterministic routing.
    """
    healthy = [leaf for leaf in leaves if leaf.healthy]
    if not healthy:
        return []

    reps = provider_reputations or {}
    weights = _DEFAULT_WEIGHTS.get(rank_by, _DEFAULT_WEIGHTS["cheapest"])
    w_price, w_lat, w_cap, w_rep = weights

    prices = [l.rate_per_1k_chars for l in healthy]
    lats = [l.latency_p95_ms for l in healthy]
    caps = [l.capacity for l in healthy]
    rep_vals = [reps.get(l.provider_id, 0.5) for l in healthy]

    price_n = _normalize_inverse(prices)
    lat_n = _normalize_inverse(lats)
    cap_n = _normalize_direct(caps)
    rep_n = _normalize_direct(rep_vals)

    scored = [
        (
            leaf,
            w_price * price_n[i]
            + w_lat * lat_n[i]
            + w_cap * cap_n[i]
            + w_rep * rep_n[i],
        )
        for i, leaf in enumerate(healthy)
    ]
    # Best score first; tiebreak by address.
    scored.sort(key=lambda pair: (-pair[1], pair[0].address))
    return scored


def best_leaf(
    pattern: str,
    leaves: Iterable[MenuLeaf],
    *,
    rank_by: RankBy = "cheapest",
    provider_reputations: dict[str, float] | None = None,
) -> MenuLeaf | None:
    """Convenience: match + rank + return the top leaf, or None."""
    matches = matching_leaves(pattern, leaves)
    ranked = rank_leaves(
        matches, rank_by=rank_by, provider_reputations=provider_reputations,
    )
    return ranked[0][0] if ranked else None
