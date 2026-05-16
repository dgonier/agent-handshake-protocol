"""Model tier abstraction.

Agents declare which *tier* of model they need (``tiny`` / ``small`` /
``medium`` / ``big``). Each server maps the four tiers to whatever
concrete model id it has installed. Callers never see model ids; they
see tiers.

Tier multipliers are a fixed cost-shaped sequence ``1, 2, 4, 8`` so the
ratio between tiers is meaningful and predictable. The actual cost per
call is the multiplier × the server's ``exchange_rate`` × character
count of (prompt + response).

This module is pure data — no I/O, no Redis. The broker imports it.
"""

from __future__ import annotations

from typing import Final, Literal


Tier = Literal["tiny", "small", "medium", "big"]
"""Allowed tier names. Ordered from cheapest to priciest."""


TIER_MULTIPLIER: Final[dict[Tier, int]] = {
    "tiny":   1,
    "small":  2,
    "medium": 4,
    "big":    8,
}
"""Cost multipliers applied on top of a server's exchange rate.

Doubling per step keeps the ratios crisp without forcing servers to
fine-tune. Servers vary their absolute level via ``exchange_rate``;
the protocol fixes the relative slope.
"""


TIER_ORDER: Final[tuple[Tier, ...]] = ("tiny", "small", "medium", "big")
"""Canonical ordering, useful for sorting and 'next tier up' logic."""


DEFAULT_TIER: Final[Tier] = "small"
"""Tier requested when an agent doesn't specify one explicitly."""


def parse_tier(value: str) -> Tier:
    """Validate ``value`` as a known tier and return it typed.

    Raises ``ValueError`` if ``value`` isn't one of the canonical
    tier strings. Use this at every boundary that takes tier from a
    string source (CLI args, env vars, JSON over the wire).
    """
    if value not in TIER_MULTIPLIER:
        raise ValueError(
            f"unknown tier {value!r}; valid: {sorted(TIER_MULTIPLIER.keys())}"
        )
    return value  # type: ignore[return-value]


def tier_multiplier(tier: Tier | str) -> int:
    """Look up the cost multiplier for a tier. Raises on unknown."""
    t = parse_tier(tier) if isinstance(tier, str) and tier not in TIER_MULTIPLIER else tier
    return TIER_MULTIPLIER[t]  # type: ignore[index]
