"""ahp.economy — money primitives for the broker.

One currency (``credit``). Every actor has a wallet in the same unit;
servers, humans, compute providers, the broker itself, and the
commons pool look identical to the protocol.

Settlement formula:

    chars   = min(actual_response_chars, max_response_chars)
    pre_tax = base_rate × tier_mult × chars
            × retention_mult     # 1.0 - 1.2  (repeat-customer premium)
            × reputation_mult    # 0.5 - 1.5  (earned quality)
            × verbosity_mult     # 0.5 - 1.1  (rolling response discipline)
            × compute_mult       # 0.25 - 1.0 (anti-sandbagging)

    compute_cost  = leaf.rate_per_1k_chars × (chars / 1000)
    protocol_tax  = pre_tax × 0.05
    to_broker     = protocol_tax × 0.60
    to_commons    = protocol_tax × 0.40
    to_server     = pre_tax - compute_cost - protocol_tax
    caller_pays   = pre_tax     # server is the residual claimant

* :mod:`ahp.economy.tiers` — tiny/small/medium/big and their fixed
  cost multipliers (1/2/4/8).
* :mod:`ahp.economy.compute_provider` — compute provider directory
  and menu leaves; address-pattern resolver.
* :mod:`ahp.economy.pricing` — the pure-data settlement formula and
  the rolling-stats helpers.
* :mod:`ahp.economy.reputation` — completion-rate + visibility cap.
* :mod:`ahp.economy.wallet` — atomic hold/settle/refund (Redis CAS).
"""

from ahp.economy.agent_banker import (
    BROKER_WALLET,
    COMMONS_WALLET,
    STARTING_FUND_BY_LIFECYCLE,
    AgentBanker,
    funding_source_for,
    refund_destination_for,
)
from ahp.economy.compute_provider import (
    ComputeProvider,
    MenuLeaf,
    RankBy,
    best_leaf,
    matching_leaves,
    rank_leaves,
)
from ahp.economy.pricing import (
    BROKER_TAX_SHARE,
    COMMONS_TAX_SHARE,
    DEFAULT_BASE_RATE,
    DEFAULT_EXPECTED_RESPONSE_CHARS,
    EXPECTED_LATENCY_MS,
    LAMBDA_AVG_OVERAGE,
    PROTOCOL_TAX_RATE,
    Hold,
    Settlement,
    SettlementInputs,
    character_count,
    estimate_hold,
    settle_payment,
    update_avg_overage,
)
from ahp.economy.reputation import (
    DEFAULT_REPUTATION,
    DEFAULT_VISIBILITY,
    LAMBDA_CSAT,
    MIN_REPUTATION_FLOOR,
    REP_PENALTY_FAILURE,
    REP_REWARD_SUCCESS,
    VISIBILITY_FULL_AT,
    ReputationEntry,
    SettlementOutcome,
    apply_csat,
    apply_outcome,
    visibility_factor,
)
from ahp.economy.tiers import (
    DEFAULT_TIER,
    TIER_MULTIPLIER,
    TIER_ORDER,
    Tier,
    parse_tier,
    tier_multiplier,
)
from ahp.economy.wallet import (
    HOLD_TTL_SECONDS,
    INITIAL_FUND,
    HoldExpiredError,
    InsufficientFundsError,
    UnknownHoldError,
    Wallet,
    WalletState,
    apply_credit,
    apply_hold,
    apply_refund,
    apply_release,
)


__all__ = [
    "AgentBanker",
    "BROKER_TAX_SHARE",
    "BROKER_WALLET",
    "COMMONS_TAX_SHARE",
    "COMMONS_WALLET",
    "ComputeProvider",
    "DEFAULT_BASE_RATE",
    "DEFAULT_EXPECTED_RESPONSE_CHARS",
    "DEFAULT_REPUTATION",
    "DEFAULT_TIER",
    "DEFAULT_VISIBILITY",
    "EXPECTED_LATENCY_MS",
    "HOLD_TTL_SECONDS",
    "Hold",
    "HoldExpiredError",
    "INITIAL_FUND",
    "InsufficientFundsError",
    "LAMBDA_AVG_OVERAGE",
    "LAMBDA_CSAT",
    "MIN_REPUTATION_FLOOR",
    "MenuLeaf",
    "PROTOCOL_TAX_RATE",
    "REP_PENALTY_FAILURE",
    "REP_REWARD_SUCCESS",
    "RankBy",
    "ReputationEntry",
    "STARTING_FUND_BY_LIFECYCLE",
    "Settlement",
    "SettlementInputs",
    "SettlementOutcome",
    "TIER_MULTIPLIER",
    "TIER_ORDER",
    "Tier",
    "UnknownHoldError",
    "VISIBILITY_FULL_AT",
    "Wallet",
    "WalletState",
    "apply_credit",
    "apply_csat",
    "apply_hold",
    "apply_outcome",
    "apply_refund",
    "apply_release",
    "best_leaf",
    "character_count",
    "estimate_hold",
    "funding_source_for",
    "matching_leaves",
    "parse_tier",
    "refund_destination_for",
    "rank_leaves",
    "settle_payment",
    "tier_multiplier",
    "update_avg_overage",
    "visibility_factor",
]
