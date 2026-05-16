"""Three-stage router: pick one server + one compute leaf per dispatch.

Pipeline:

1. **Hard filter** — eliminate candidates that can't be routed under
   any circumstance: insufficient reputation, missing required
   integration / specialty, alive=False, wallet underfunded, in the
   caller's blocklist, exceeds max_cost_per_call.
   *Failing the hard filter → NoCandidatesError, no dispatch.*

2. **Soft filter** — narrow by preferences: preferred specialty,
   preferred integration, preferred-server-first. If applying a soft
   filter would empty the set, the filter is *skipped* and the
   decision is annotated with which preferences couldn't be honored.

3. **Sort** — by ``rank_by``. Default is ``cheapest``. Ties broken
   alphabetically by server id for determinism (predictable cache
   keys, repeatable behavior).

A **visibility coin flip** is layered after stage 1: a server with
visibility=0.05 is considered for ~5% of routes. This throttles
fresh / unproven servers regardless of how attractive their price
or capability is. They earn visibility over time as reputation
accrues.

The router returns a :class:`RoutingDecision` carrying:
* chosen server + compute leaf (or None on failure)
* the locked-in price (used for the wallet hold)
* a list of (server_id, why_filtered) entries for the candidates
  that didn't make it — auditable, helpful for debugging "why didn't
  my agent get this call?"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Literal

from ahp.broker.compute_registry import ComputeProviderRegistry
from ahp.broker.server_registry import ServerMeta, ServerRegistry
from ahp.economy.compute_provider import MenuLeaf, RankBy, best_leaf
from ahp.economy.pricing import (
    DEFAULT_EXPECTED_RESPONSE_CHARS,
    estimate_hold,
)
from ahp.economy.reputation import (
    MIN_REPUTATION_FLOOR,
    ReputationEntry,
    visibility_factor,
)
from ahp.economy.tiers import Tier, parse_tier


# ── inputs ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RoutingPreferences:
    """The caller's preferences for a single routing decision.

    Hard requirements raise :class:`NoCandidatesError` if unsatisfied.
    Soft preferences degrade gracefully and are reported in the
    decision's annotations.
    """

    # Hard requirements — must be satisfied or no dispatch.
    min_reputation: float = MIN_REPUTATION_FLOOR
    min_csat: float = 0.0                  # neutral default; ignore until surveys ship
    required_specialties: tuple[str, ...] = ()
    required_integrations: tuple[str, ...] = ()
    max_cost_per_call: float | None = None  # estimated hold ceiling
    blocked_servers: tuple[str, ...] = ()

    # Soft preferences — drop gracefully if they'd empty the set.
    preferred_specialties: tuple[str, ...] = ()
    preferred_integrations: tuple[str, ...] = ()
    preferred_servers: tuple[str, ...] = ()

    # Ranking.
    server_rank_by: Literal[
        "cheapest", "best_reputation", "highest_csat", "lowest_latency",
    ] = "cheapest"
    compute_rank_by: RankBy = "cheapest"

    # Random source for visibility coin-flips. Override in tests for
    # determinism; default is the system random.
    rng_seed: int | None = None


# ── outputs ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FilterReason:
    """One server that didn't make the cut, with why."""

    server_id: str
    stage: Literal["hard", "soft", "visibility"]
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    """Result of one routing attempt.

    On success, ``server`` and ``leaf`` are both populated and
    ``estimated_hold`` is the dispatch-time wallet hold. On failure,
    they're None and ``unmet_requirements`` / ``rejections`` explain
    why.
    """

    server: ServerMeta | None
    leaf: MenuLeaf | None
    estimated_hold: float = 0.0
    rejections: tuple[FilterReason, ...] = ()
    soft_preferences_unmet: tuple[str, ...] = ()
    unmet_requirements: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return self.server is not None and self.leaf is not None


class NoCandidatesError(Exception):
    """No server (or no compute leaf) made it through hard filtering.

    Carries the :class:`RoutingDecision` so callers can inspect why.
    """

    def __init__(self, decision: RoutingDecision):
        super().__init__("no routable candidates")
        self.decision = decision


# ── the router ───────────────────────────────────────────────────────


class Router:
    """Stateless: composes registries + reputation into a decision.

    The router doesn't write to Redis — it reads, decides, returns.
    Wallet holds and reputation updates are the broker's job after
    a routing decision is acted on.
    """

    def __init__(
        self,
        *,
        server_registry: ServerRegistry,
        compute_registry: ComputeProviderRegistry,
        reputation_lookup,  # async (owner: str) -> ReputationEntry | None
    ) -> None:
        self._servers = server_registry
        self._compute = compute_registry
        self._rep = reputation_lookup

    async def resolve(
        self,
        *,
        code: str,
        tier: Tier,
        domain: str | None = None,
        prompt_chars: int = 0,
        max_response_chars: int = DEFAULT_EXPECTED_RESPONSE_CHARS,
        prefs: RoutingPreferences | None = None,
    ) -> RoutingDecision:
        """Route one dispatch.

        Sequence:
            1. Discover live servers matching capability shape.
            2. Hard-filter (reputation, must-haves, block list, alive).
            3. Visibility coin-flip throttle.
            4. Soft-filter (preferences), degrading on empty.
            5. Rank.
            6. For the winning server, resolve compute leaf.
            7. Estimate hold, check max_cost_per_call.
        """
        parse_tier(tier)
        prefs = prefs or RoutingPreferences()
        rng = random.Random(prefs.rng_seed)
        rejections: list[FilterReason] = []
        soft_unmet: list[str] = []
        unmet_required: list[str] = []

        # 1. Capability filter via the server registry.
        candidates = await self._servers.discover(
            code=code, tier=tier, alive_only=True,
        )

        # 2. Hard filter
        kept: list[tuple[ServerMeta, ReputationEntry]] = []
        for srv in candidates:
            rep = await self._rep(srv.server_id) or ReputationEntry(owner=srv.server_id)
            rejection = self._hard_filter_one(srv, rep, prefs)
            if rejection:
                rejections.append(rejection)
                continue
            kept.append((srv, rep))

        if not kept:
            unmet_required = self._summarize_required(rejections, prefs)
            return RoutingDecision(
                server=None, leaf=None,
                rejections=tuple(rejections),
                unmet_requirements=tuple(unmet_required),
            )

        # 3. Visibility coin-flip
        gated: list[tuple[ServerMeta, ReputationEntry]] = []
        for srv, rep in kept:
            v = visibility_factor(rep)
            if rng.random() < v:
                gated.append((srv, rep))
            else:
                rejections.append(FilterReason(
                    srv.server_id, "visibility",
                    f"v={v:.2f} not in roll",
                ))
        # If every candidate was throttled, relax — better to route to
        # the lowest-visibility server than to fail entirely. The
        # broker logs this as a "low-visibility-only" route.
        if not gated:
            gated = kept

        # 4. Soft filter (degrade gracefully)
        narrowed = self._apply_soft_filter(gated, prefs, soft_unmet)

        # 5. Sort
        ranked = self._rank(narrowed, prefs)
        chosen_server, _ = ranked[0]

        # 6. Compute leaf resolution
        leaves = await self._compute.list_leaves(only_alive_providers=True)
        provider_reps = {l.provider_id: 0.5 for l in leaves}  # TODO when provider rep lands
        chosen_leaf = best_leaf(
            chosen_server.compute_binding,
            leaves,
            rank_by=prefs.compute_rank_by,
            provider_reputations=provider_reps,
        )
        if chosen_leaf is None:
            unmet_required.append(
                f"server {chosen_server.server_id!r} compute_binding "
                f"{chosen_server.compute_binding!r} matched no live leaves"
            )
            return RoutingDecision(
                server=None, leaf=None,
                rejections=tuple(rejections),
                unmet_requirements=tuple(unmet_required),
                soft_preferences_unmet=tuple(soft_unmet),
            )

        # 7. Estimate hold
        hold = estimate_hold(
            base_rate=chosen_server.base_rate,
            tier=tier,
            prompt_chars=prompt_chars,
            max_response_chars=max_response_chars,
            leaf_rate_per_1k_chars=chosen_leaf.rate_per_1k_chars,
        )
        if (
            prefs.max_cost_per_call is not None
            and hold.amount > prefs.max_cost_per_call
        ):
            return RoutingDecision(
                server=None, leaf=None,
                rejections=tuple(rejections),
                unmet_requirements=tuple(unmet_required) + (
                    f"estimated hold {hold.amount:.4f} > "
                    f"max_cost_per_call {prefs.max_cost_per_call:.4f}",
                ),
                soft_preferences_unmet=tuple(soft_unmet),
            )

        return RoutingDecision(
            server=chosen_server,
            leaf=chosen_leaf,
            estimated_hold=hold.amount,
            rejections=tuple(rejections),
            soft_preferences_unmet=tuple(soft_unmet),
        )

    # ── stage helpers ─────────────────────────────────────────────────

    def _hard_filter_one(
        self,
        srv: ServerMeta,
        rep: ReputationEntry,
        prefs: RoutingPreferences,
    ) -> FilterReason | None:
        if srv.server_id in prefs.blocked_servers:
            return FilterReason(srv.server_id, "hard", "blocked by caller")
        if rep.reputation < prefs.min_reputation:
            return FilterReason(
                srv.server_id, "hard",
                f"reputation {rep.reputation:.2f} < min {prefs.min_reputation:.2f}",
            )
        if rep.csat < prefs.min_csat:
            return FilterReason(
                srv.server_id, "hard",
                f"csat {rep.csat:.2f} < min {prefs.min_csat:.2f}",
            )
        for sp in prefs.required_specialties:
            if sp not in srv.specialties:
                return FilterReason(
                    srv.server_id, "hard",
                    f"missing required specialty {sp!r}",
                )
        for ig in prefs.required_integrations:
            if ig not in srv.integrations:
                return FilterReason(
                    srv.server_id, "hard",
                    f"missing required integration {ig!r}",
                )
        return None

    def _apply_soft_filter(
        self,
        kept: list[tuple[ServerMeta, ReputationEntry]],
        prefs: RoutingPreferences,
        soft_unmet: list[str],
    ) -> list[tuple[ServerMeta, ReputationEntry]]:
        """Apply each soft filter in turn; revert if it empties the set."""
        result = kept

        for sp in prefs.preferred_specialties:
            after = [(s, r) for s, r in result if sp in s.specialties]
            if after:
                result = after
            else:
                soft_unmet.append(f"preferred specialty {sp!r} unmet")

        for ig in prefs.preferred_integrations:
            after = [(s, r) for s, r in result if ig in s.integrations]
            if after:
                result = after
            else:
                soft_unmet.append(f"preferred integration {ig!r} unmet")

        if prefs.preferred_servers:
            after = [
                (s, r) for s, r in result if s.server_id in prefs.preferred_servers
            ]
            if after:
                result = after
            else:
                soft_unmet.append("none of preferred_servers were candidates")

        return result

    def _rank(
        self,
        candidates: list[tuple[ServerMeta, ReputationEntry]],
        prefs: RoutingPreferences,
    ) -> list[tuple[ServerMeta, ReputationEntry]]:
        """Sort the surviving candidates by the requested strategy.

        Tiebreak is alphabetical by server_id so repeated routing
        decisions with the same input produce the same output — good
        for cache stability.
        """
        rank_by = prefs.server_rank_by

        def key(pair: tuple[ServerMeta, ReputationEntry]):
            srv, rep = pair
            if rank_by == "cheapest":
                return (srv.base_rate, srv.server_id)
            if rank_by == "best_reputation":
                return (-rep.reputation, srv.server_id)
            if rank_by == "highest_csat":
                return (-rep.csat, srv.server_id)
            if rank_by == "lowest_latency":
                return (rep.avg_latency_ms or float("inf"), srv.server_id)
            return (srv.server_id,)

        return sorted(candidates, key=key)

    def _summarize_required(
        self,
        rejections: list[FilterReason],
        prefs: RoutingPreferences,
    ) -> list[str]:
        out: list[str] = []
        if prefs.required_specialties:
            out.append(f"required specialties: {list(prefs.required_specialties)}")
        if prefs.required_integrations:
            out.append(f"required integrations: {list(prefs.required_integrations)}")
        if prefs.min_reputation > 0:
            out.append(f"min_reputation: {prefs.min_reputation}")
        if prefs.min_csat > 0:
            out.append(f"min_csat: {prefs.min_csat}")
        return out
