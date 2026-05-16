"""Reputation + visibility ledger.

Pure data: the broker persists :class:`ReputationEntry` records per
server, calls :func:`apply_outcome` on each settled call to update
them, and uses :func:`visibility_factor` at routing time to throttle
how often a low-experience server is considered.

The interesting moves:

* **Reputation is asymmetric.** Successes nudge it up by ε; failures
  drag it down by N×ε. Sandbagging surfaces fast.
* **Visibility is a cap, not a sort key.** A fresh server has
  ``visibility=0.05`` regardless of how attractive their rate card
  is — the router rolls a weighted coin to decide whether a low-
  visibility server is in the candidate set on this call. Visibility
  grows with cumulative *completed-and-accepted* calls, not raw
  attempts.
* **Reputation floor as hard filter.** Below ``MIN_REPUTATION_FLOOR``
  a server is filtered out of routing entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final, Literal


DEFAULT_REPUTATION: Final[float] = 0.5
"""Starting reputation for a freshly registered server (0.5 on [0,1])."""


DEFAULT_VISIBILITY: Final[float] = 0.05
"""Starting visibility cap. A fresh server is considered for ~5% of
matching dispatches until it has built up experience.
"""


MIN_REPUTATION_FLOOR: Final[float] = 0.30
"""Hard filter: servers below this are not routable at all.

Set deliberately below ``DEFAULT_REPUTATION`` so a fresh server
isn't gated out before it has a chance to prove itself. Cheaters
who repeatedly fail or get refunded drop below this line in a
handful of failures (rep_decay per failure is large).
"""


REP_REWARD_SUCCESS: Final[float] = 0.005
"""Per-success reputation increment."""


REP_PENALTY_FAILURE: Final[float] = 0.05
"""Per-failure reputation decrement. 10× the success reward —
asymmetric on purpose.
"""


VISIBILITY_FULL_AT: Final[int] = 500
"""Number of completed-and-accepted calls at which visibility hits 1.0.

The curve is ``visibility = min(1, base + log10(1 + N) / log10(VISIBILITY_FULL_AT))``
so growth is fast early and tapers as the server proves itself.
"""


# ── data ──────────────────────────────────────────────────────────────


SettlementOutcome = Literal["accepted", "verbose", "sub_tier", "refunded", "timeout"]
"""How a single call was settled, from the caller's perspective.

``accepted`` is the only outcome that grants reputation reward.
``verbose`` is a soft penalty (half-pay + tiny rep hit), counted as a
'completed-but-suboptimal' call for visibility purposes.
``sub_tier``, ``refunded``, ``timeout`` are full reputation hits.
"""


@dataclass
class ReputationEntry:
    """Reputation state for an addressable economic actor.

    ``owner`` is any opaque identity string — a server id, a full
    :class:`~ahp.core.AgentAddress`, a human's address, the broker, or
    the commons pool. The reputation primitive is identity-agnostic;
    the broker decides how it scopes ownership.

    ``reputation`` is the 0..1 quality score that feeds the settlement
    formula's ``reputation_mult``. ``completed_accepted`` is the
    cumulative count of successful calls — drives ``visibility``.
    Latency stats are tracked here too so the broker can rank
    candidates by latency without a separate ledger.

    ``csat`` is the survey-driven score (post-hoc usefulness rating).
    It's a separate dimension from ``reputation``: reputation is
    system-observed (settlement verdicts, latency), CSAT is
    consumer-observed. Both are routing inputs; surveys aren't yet
    wired (see :mod:`ahp.broker.surveys` stub).
    """

    owner: str
    reputation: float = DEFAULT_REPUTATION
    completed_accepted: int = 0
    completed_total: int = 0          # accepted + verbose
    failed: int = 0                   # sub_tier + refunded + timeout
    sum_latency_ms: float = 0.0       # for avg_latency
    avg_overage: float = 1.0          # rolling EWMA of response/budget ratio
    csat: float = 0.5                 # rolling EWMA of survey scores; neutral default
    csat_samples: int = 0             # how many survey responses contributed

    @property
    def avg_latency_ms(self) -> float:
        denom = self.completed_total
        return (self.sum_latency_ms / denom) if denom else 0.0


# ── updates ───────────────────────────────────────────────────────────


def apply_outcome(
    entry: ReputationEntry,
    outcome: SettlementOutcome,
    *,
    latency_ms: float = 0.0,
) -> ReputationEntry:
    """Return a new :class:`ReputationEntry` with the outcome applied.

    Frozen-style: we return a new entry rather than mutating in place.
    Lets the broker store the result transactionally with the wallet
    write.
    """
    new = ReputationEntry(
        owner=entry.owner,
        reputation=entry.reputation,
        completed_accepted=entry.completed_accepted,
        completed_total=entry.completed_total,
        failed=entry.failed,
        sum_latency_ms=entry.sum_latency_ms,
        avg_overage=entry.avg_overage,
        csat=entry.csat,
        csat_samples=entry.csat_samples,
    )

    if outcome == "accepted":
        new.reputation = min(1.0, new.reputation + REP_REWARD_SUCCESS)
        new.completed_accepted += 1
        new.completed_total += 1
        new.sum_latency_ms += max(0.0, latency_ms)
    elif outcome == "verbose":
        # Mild — completed the work but ignored the budget. Half a hit.
        new.reputation = max(0.0, new.reputation - REP_REWARD_SUCCESS)
        new.completed_total += 1
        new.sum_latency_ms += max(0.0, latency_ms)
    else:
        # sub_tier / refunded / timeout
        new.reputation = max(0.0, new.reputation - REP_PENALTY_FAILURE)
        new.failed += 1
    return new


LAMBDA_CSAT: Final[float] = 0.1
"""EWMA decay for ``csat`` updates.

Higher than the verbosity EWMA (0.05) because CSAT samples are scarcer
— each individual response is more informative than each settlement
verdict, and we want the rolling score to reflect recent surveys
without waiting for hundreds of samples to converge.
"""


def apply_csat(entry: ReputationEntry, score: float) -> ReputationEntry:
    """Return a new entry with one survey score folded into CSAT.

    ``score`` is the survey response in [0, 1]. We clamp defensively
    in case a survey backend returns out-of-range values. The first
    sample replaces the neutral default outright; subsequent samples
    apply EWMA so a single outlier doesn't whipsaw the score.
    """
    score = max(0.0, min(1.0, float(score)))
    new = ReputationEntry(
        owner=entry.owner,
        reputation=entry.reputation,
        completed_accepted=entry.completed_accepted,
        completed_total=entry.completed_total,
        failed=entry.failed,
        sum_latency_ms=entry.sum_latency_ms,
        avg_overage=entry.avg_overage,
        csat=entry.csat,
        csat_samples=entry.csat_samples,
    )
    if new.csat_samples == 0:
        new.csat = score
    else:
        new.csat = (1.0 - LAMBDA_CSAT) * new.csat + LAMBDA_CSAT * score
    new.csat_samples += 1
    return new


def visibility_factor(entry: ReputationEntry) -> float:
    """Probability (0..1) that this server is considered by the router.

    Starts at :data:`DEFAULT_VISIBILITY` for a brand new server and
    grows logarithmically with ``completed_accepted`` until it hits
    1.0 at :data:`VISIBILITY_FULL_AT` completions.

    ``reputation`` below :data:`MIN_REPUTATION_FLOOR` zeros visibility
    so the router can't pick the server even by coin flip.
    """
    if entry.reputation < MIN_REPUTATION_FLOOR:
        return 0.0
    n = max(0, entry.completed_accepted)
    if n >= VISIBILITY_FULL_AT:
        return 1.0
    # Anchor: visibility(0) = DEFAULT_VISIBILITY, visibility(FULL) = 1.0.
    # Use log10(1+n) / log10(1+FULL) for a curve that's fast early.
    fraction = math.log10(1 + n) / math.log10(1 + VISIBILITY_FULL_AT)
    return DEFAULT_VISIBILITY + (1.0 - DEFAULT_VISIBILITY) * fraction
