"""Settlement formula and rolling-statistics helpers.

All functions in this module are pure data. The broker calls them
when it needs to compute a hold, settle a payment, or update a
server's rolling stats.

Formula
-------

For each call:

    chars = min(actual_response_chars, max_response_chars)

    pre_tax = base_rate × tier_mult × chars
            × retention_mult         # 1.0 - 1.2
            × reputation_mult        # 0.5 - 1.5
            × verbosity_mult         # 0.5 - 1.1
            × compute_mult           # 0.25 - 1.0

    compute_cost  = leaf_menu_rate × (chars / 1000)
    protocol_tax  = pre_tax × 0.05
    to_broker     = protocol_tax × 0.60
    to_commons    = protocol_tax × 0.40
    to_server     = pre_tax − compute_cost − protocol_tax
    caller_pays   = pre_tax    (server is the residual claimant)

The server is the residual claimant: their take equals pre-tax minus
the compute cost (paid to the compute provider) and the protocol tax
(split broker/commons). If ``to_server`` would go negative, the leaf
is too expensive for this server's posted ``base_rate`` and the call
should be rejected at pattern-resolution time rather than dispatched.

Behavioral multipliers
----------------------

* **retention** rewards relationships. ``1 + α × log10(1 + N)`` where
  ``N`` is the number of completed calls between this exact
  (caller, server) pair. Caps at ~1.2× around 10k completed calls.
* **reputation** is ``0.5 + r`` where ``r ∈ [0, 1]`` is the broker's
  rolling completion-quality score for the server. Range 0.5..1.5.
* **verbosity** rewards rolling-average response discipline. The
  broker tracks an EWMA of ``response_chars / max_response_chars``
  per server. A server whose average is at 1.0 gets multiplier 1.0;
  consistently under-budget gets up to 1.1×; consistently bloated
  gets down to 0.5×.
* **compute_mult** is the protocol's anti-sandbagging term. It's the
  product of two 0.5/1.0 flags:
    - latency_too_fast: response came back in < 30% of expected
      latency for the claimed tier → 0.5×
    - tier_verdict == "sub_tier": caller flagged the response as not
      matching the claimed tier → 0.5×
  Range 0.25..1.0; honest servers always get 1.0.

All knobs (tax rate, weight shares, multiplier ranges, EWMA λ) are
module-level constants so the broker operator can tune them once at
deploy time without forking the formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

from ahp.economy.tiers import Tier, tier_multiplier


# ── tunable constants ─────────────────────────────────────────────────


DEFAULT_BASE_RATE: Final[float] = 0.0002
"""Sensible default ``base_rate`` for a server that doesn't pick one.

Translates to: at tier=small (mult=2), 1000 chars = 0.0002 × 2 × 1000
= 0.4 credits per call. A fresh wallet of 100 credits buys ~250
small-tier calls before the multipliers ever kick in.
"""


DEFAULT_EXPECTED_RESPONSE_CHARS: Final[int] = 800
"""Default ``max_response_chars`` for callers that don't specify."""


PROTOCOL_TAX_RATE: Final[float] = 0.05
"""5% of every settlement goes to broker + commons."""


BROKER_TAX_SHARE: Final[float] = 0.60
"""Fraction of the protocol tax that goes to the broker wallet."""


COMMONS_TAX_SHARE: Final[float] = 1.0 - BROKER_TAX_SHARE
"""Fraction of the protocol tax that goes to the commons pool."""


# Behavioral multipliers — ranges and the rates of change.
RETENTION_ALPHA: Final[float] = 0.05
"""``log10`` coefficient for the retention bonus."""

REPUTATION_FLOOR_MULT: Final[float] = 0.5
REPUTATION_CEIL_MULT: Final[float] = 1.5

VERBOSITY_FLOOR_MULT: Final[float] = 0.5
VERBOSITY_CEIL_MULT: Final[float] = 1.1

LAMBDA_AVG_OVERAGE: Final[float] = 0.05
"""EWMA decay for ``avg_overage``. Full memory ~ 1/λ = 20 calls.

Quick enough to react to a server changing behavior, slow enough that
single-call outliers don't whipsaw the multiplier.
"""


# Per-tier expected latency, used by the compute_mult sandbagging
# check. A response that comes back in less than 30% of expected
# latency for its tier is flagged as suspiciously fast and the
# server's compute_mult halves.
EXPECTED_LATENCY_MS: Final[dict[Tier, float]] = {
    "tiny":   200.0,
    "small":  600.0,
    "medium": 1500.0,
    "big":    4000.0,
}


# ── data types ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SettlementInputs:
    """Everything pricing needs to settle one call.

    Kept as a single struct rather than a long parameter list to make
    test fixtures and broker integration cleaner.
    """

    # Posted by the server.
    base_rate: float
    tier: Tier

    # Observed at dispatch / response.
    response_chars: int
    max_response_chars: int
    actual_latency_ms: float

    # Leaf binding (chosen by the broker's pattern resolver).
    leaf_rate_per_1k_chars: float

    # Pulled from the broker's per-pair / per-server ledgers.
    completed_with_caller: int
    server_reputation: float
    server_avg_overage: float

    # Caller-supplied verdict at settlement time.
    tier_verdict: Literal["matched", "sub_tier"] = "matched"


@dataclass(frozen=True)
class Settlement:
    """Outcome of one settled call.

    The four recipient amounts always sum to ``pre_tax`` — the caller's
    total cost. ``effective_chars`` is what the formula actually used
    after the ``max_response_chars`` cap.
    """

    pre_tax: float
    to_server: float
    to_compute: float
    to_broker: float
    to_commons: float
    effective_chars: int

    @property
    def total_tax(self) -> float:
        return self.to_broker + self.to_commons


@dataclass(frozen=True)
class Hold:
    """A pre-dispatch hold on the caller's wallet.

    The broker estimates the call's cost (with neutral 1.0× multipliers,
    so the hold is conservative) and reserves that amount. On
    settlement the actual cost may differ; any over-hold is released
    back to the caller, any shortfall is debited.
    """

    amount: float
    estimated_chars: int


# ── core calculations ─────────────────────────────────────────────────


def character_count(*parts: str | None) -> int:
    """Sum ``len(p)`` across non-None string parts."""
    return sum(len(p) for p in parts if p is not None)


def estimate_hold(
    *,
    base_rate: float,
    tier: Tier,
    prompt_chars: int,
    max_response_chars: int = DEFAULT_EXPECTED_RESPONSE_CHARS,
    leaf_rate_per_1k_chars: float = 0.0,
) -> Hold:
    """Conservative dispatch-time hold.

    Uses neutral 1.0× multipliers and the worst-case (full budget)
    response size. The actual settled amount will usually be lower.
    The hold also covers the compute_cost so the caller's wallet has
    enough to pay the compute provider even in the worst case.
    """
    chars = max(0, prompt_chars) + max(0, max_response_chars)
    tm = tier_multiplier(tier)
    pre_tax = float(base_rate) * tm * chars  # all behavior mults = 1.0
    compute_cost = float(leaf_rate_per_1k_chars) * (chars / 1000.0)
    # The hold is the caller's exposure = pre_tax. (Compute cost and
    # tax both come out of the server's residual; the caller's bill is
    # pre_tax flat.)
    return Hold(amount=pre_tax, estimated_chars=chars)


def settle_payment(inp: SettlementInputs) -> Settlement:
    """Compute the final settlement for one call.

    Pure: no I/O, no Redis. The broker passes in everything it has
    measured. Returns the four-way split.

    If the formula produces ``to_server < 0`` — meaning the compute
    cost plus the protocol tax exceed pre-tax revenue — the call was
    routed to a leaf the server can't afford. We return the math
    honestly (negative ``to_server``); the broker is responsible for
    rejecting such dispatches at pattern-resolution time, not for
    silently truncating them here.
    """
    # 1. Effective characters: cap by the caller's budget.
    chars = min(int(inp.response_chars), int(inp.max_response_chars))
    chars = max(0, chars)

    # 2. Multipliers.
    tm = tier_multiplier(inp.tier)

    retention = 1.0 + RETENTION_ALPHA * math.log10(
        1 + max(0, inp.completed_with_caller)
    )

    rep = max(0.0, min(1.0, inp.server_reputation))
    rep_mult = REPUTATION_FLOOR_MULT + rep
    # Defensive clamp: keep within declared range even if reputation
    # is ever stored outside [0,1].
    rep_mult = max(REPUTATION_FLOOR_MULT, min(REPUTATION_CEIL_MULT, rep_mult))

    verbosity_mult = _verbosity_multiplier(inp.server_avg_overage)

    compute_mult = _compute_multiplier(
        actual_latency_ms=inp.actual_latency_ms,
        tier=inp.tier,
        tier_verdict=inp.tier_verdict,
    )

    # 3. Pre-tax revenue.
    pre_tax = (
        float(inp.base_rate) * tm * chars
        * retention * rep_mult * verbosity_mult * compute_mult
    )

    # 4. Split.
    compute_cost = float(inp.leaf_rate_per_1k_chars) * (chars / 1000.0)
    protocol_tax = pre_tax * PROTOCOL_TAX_RATE
    to_broker = protocol_tax * BROKER_TAX_SHARE
    to_commons = protocol_tax * COMMONS_TAX_SHARE
    to_server = pre_tax - compute_cost - protocol_tax

    return Settlement(
        pre_tax=pre_tax,
        to_server=to_server,
        to_compute=compute_cost,
        to_broker=to_broker,
        to_commons=to_commons,
        effective_chars=chars,
    )


def _verbosity_multiplier(avg_overage: float) -> float:
    """Map a server's rolling avg_overage to a payment multiplier.

    ``avg_overage`` is the ratio of response_chars to max_response_chars
    averaged over recent calls. A value of 1.0 means "on budget."
    """
    raw = 2.0 - max(0.0, float(avg_overage))
    return max(VERBOSITY_FLOOR_MULT, min(VERBOSITY_CEIL_MULT, raw))


def _compute_multiplier(
    *,
    actual_latency_ms: float,
    tier: Tier,
    tier_verdict: Literal["matched", "sub_tier"],
) -> float:
    """The anti-sandbagging multiplier."""
    expected = EXPECTED_LATENCY_MS.get(tier, 1000.0)
    too_fast = actual_latency_ms < 0.3 * expected
    latency_mult = 0.5 if too_fast else 1.0
    verdict_mult = 0.5 if tier_verdict == "sub_tier" else 1.0
    return latency_mult * verdict_mult


# ── rolling stats ─────────────────────────────────────────────────────


def update_avg_overage(
    current: float, new_response_chars: int, max_response_chars: int,
    *, lmbd: float = LAMBDA_AVG_OVERAGE,
) -> float:
    """Update the server's exponentially-weighted avg_overage.

    ``new_ratio = response_chars / max_response_chars``. The update is
    ``current * (1 - λ) + new_ratio * λ``. A fresh server with no
    prior samples should be initialized at 1.0 (assumed neutral).
    """
    if max_response_chars <= 0:
        return current
    new_ratio = max(0.0, new_response_chars) / float(max_response_chars)
    return (1.0 - lmbd) * current + lmbd * new_ratio
