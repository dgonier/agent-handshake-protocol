"""Broker facade — the top-level entry point.

Composes:

* :class:`ServerRegistry` — who's online
* :class:`ComputeProviderRegistry` — what compute is available
* :class:`Router` — pick one server + leaf
* :class:`Wallet`-s for each economic actor
* A :class:`ReputationStore` shim that reads/writes
  :class:`ReputationEntry` records under ``ahp:reputation:<owner>``

The :class:`Broker` is the single object the engine, runner, and
viewer talk to. It exposes:

* ``broker.register_server(meta)`` — server lifecycle
* ``broker.register_compute_provider(provider)`` + ``register_leaf(leaf)``
* ``broker.resolve(...)`` — routing decision
* ``broker.hold(...)`` / ``broker.settle(...)`` / ``broker.refund(...)``
  — wallet operations bound to a settlement
* ``broker.wallet(owner)`` — escape hatch for inspecting balances

The broker doesn't enforce *who* can call these methods. That's the
host application's job; this is the protocol-level mechanism.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from typing import Any

from ahp.audit import AuditEvent, AuditSink
from ahp.broker.compute_registry import ComputeProviderRegistry
from ahp.broker.router import (
    NoCandidatesError,
    RoutingDecision,
    RoutingPreferences,
    Router,
)
from ahp.broker.server_registry import ServerMeta, ServerRegistry
from ahp.broker.surveys import (
    SurveyKind,
    SurveyQueue,
    SurveyRequest,
    SurveyResponse,
)
from ahp.economy.agent_banker import BROKER_WALLET, COMMONS_WALLET, AgentBanker
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.pricing import (
    DEFAULT_EXPECTED_RESPONSE_CHARS,
    Settlement,
    SettlementInputs,
    settle_payment,
    update_avg_overage,
)
from ahp.economy.reputation import (
    DEFAULT_REPUTATION,
    ReputationEntry,
    SettlementOutcome,
    apply_outcome,
)
from ahp.economy.tiers import Tier
from ahp.economy.wallet import Wallet


log = logging.getLogger("ahp.broker")

REPUTATION_KEY = "ahp:reputation:{owner}"

DEFAULT_SURVEY_STAKES_THRESHOLD: float = 5.0
"""Settlement ``pre_tax`` above which the broker auto-enqueues a survey.

Survey reward defaults to 5% of the settled ``pre_tax`` so a 10-credit
interaction yields a 0.5-credit survey reward. Both knobs are
overridable via env (``AHP_SURVEY_STAKES_THRESHOLD``,
``AHP_SURVEY_REWARD_RATE``) so operators can tune without touching code.
"""

DEFAULT_SURVEY_REWARD_RATE: float = 0.05


class Broker:
    """Top-level facade for the routing + settlement broker."""

    def __init__(
        self,
        redis_client: Any,
        *,
        audit: AuditSink | None = None,
        registry: Any = None,
    ) -> None:
        """Construct the broker facade.

        ``registry`` is an optional :class:`~ahp.registry.AgentRegistry`
        reference. When wired, the broker feeds settlement signal back
        into per-agent :attr:`AgentMeta.reputation` after each
        successful settlement — accepted outcomes nudge the responder's
        reputation up, failures (sub_tier, timeout, refund) nudge it
        down. Same asymmetric magnitudes as the server-level
        :class:`ReputationEntry`.

        Without a registry the broker still functions; the per-agent
        ``AgentMeta.reputation`` field just stays at its default. The
        server-level reputation (``ahp:reputation:<server_id>``) is
        always tracked because it's the routing signal.
        """
        self._redis = redis_client
        self._audit = audit
        self._registry = registry
        self.servers = ServerRegistry(redis_client)
        self.compute = ComputeProviderRegistry(redis_client)
        self.banker = AgentBanker(redis_client)
        self.surveys = SurveyQueue(redis_client, audit=audit)
        self.router = Router(
            server_registry=self.servers,
            compute_registry=self.compute,
            reputation_lookup=self.get_reputation,
        )

    async def _emit(
        self,
        op: str,
        *,
        target: str | None = None,
        success: bool = True,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Best-effort audit emission. Never raises."""
        if self._audit is None:
            return
        try:
            await self._audit.emit(AuditEvent(
                op=op, target=target,
                success=success, error=error,
                extra=extra or {},
            ))
        except Exception:
            log.exception("audit emit failed for op=%s", op)

    # ── server lifecycle ──────────────────────────────────────────────

    async def register_server(self, meta: ServerMeta) -> None:
        await self.servers.register(meta)
        await self._emit(
            "broker.server.register",
            target=meta.server_id,
            extra={
                "org": meta.org, "base_rate": meta.base_rate,
                "binding": meta.compute_binding,
            },
        )

    async def deregister_server(self, server_id: str) -> None:
        await self.servers.deregister(server_id)
        await self._emit("broker.server.deregister", target=server_id)

    async def heartbeat_server(self, server_id: str) -> bool:
        ok = await self.servers.heartbeat(server_id)
        await self._emit(
            "broker.server.heartbeat",
            target=server_id,
            success=ok,
            error=None if ok else "NotRegistered",
        )
        return ok

    # ── compute provider lifecycle ────────────────────────────────────

    async def register_compute_provider(
        self,
        provider: ComputeProvider,
        *,
        prove_alive: bool = True,
    ) -> None:
        """Persist a compute provider's metadata.

        ``prove_alive=False`` requires the provider to send an explicit
        ``heartbeat_compute_provider`` call before its leaves become
        visible — the "health must be proven before you're on the menu"
        path. Default ``True`` matches the self-hosted case where
        registration is also the first heartbeat.
        """
        await self.compute.register_provider(provider, prove_alive=prove_alive)
        await self._emit(
            "broker.provider.register",
            target=provider.provider_id,
            extra={"prove_alive": prove_alive},
        )

    async def deregister_compute_provider(
        self,
        provider_id: str,
        *,
        graceful: bool = True,
    ) -> None:
        """Drop a provider. ``graceful=False`` arms an outage credit
        on the next ``check_compute_outages()`` sweep, in case the
        watchdog has detected the provider failed without saying so."""
        await self.compute.deregister_provider(provider_id, graceful=graceful)
        await self._emit(
            "broker.provider.deregister",
            target=provider_id,
            extra={"graceful": graceful},
        )

    async def heartbeat_compute_provider(self, provider_id: str) -> bool:
        ok = await self.compute.heartbeat_provider(provider_id)
        await self._emit(
            "broker.provider.heartbeat",
            target=provider_id,
            success=ok,
            error=None if ok else "NotRegistered",
        )
        return ok

    async def register_leaf(self, leaf: MenuLeaf) -> None:
        await self.compute.register_leaf(leaf)
        await self._emit(
            "broker.leaf.register",
            target=leaf.address,
            extra={
                "provider": leaf.provider_id, "tier": leaf.tier,
                "model": leaf.model,
                "rate_per_1k_chars": leaf.rate_per_1k_chars,
                "latency_p95_ms": leaf.latency_p95_ms,
            },
        )

    async def check_compute_outages(self) -> list[str]:
        """Detect and credit any unplanned-outage events for compute
        providers since the last sweep.

        For each provider whose last-seen sentinel survived past its
        heartbeat TTL without a graceful deregister, this folds a
        ``timeout`` outcome into that provider's reputation record and
        returns the provider id. Idempotent — the same outage is only
        credited once because ``detect_outage`` clears the sentinel.
        """
        hit: list[str] = []
        for provider in await self.compute.list_providers():
            if await self.compute.detect_outage(provider.provider_id):
                await self.record_outcome(
                    provider.provider_id,
                    "timeout",
                    latency_ms=0.0,
                    response_chars=0,
                    max_response_chars=0,
                )
                hit.append(provider.provider_id)
                await self._emit(
                    "broker.provider.outage",
                    target=provider.provider_id,
                    success=False, error="UnplannedDeregister",
                )
        return hit

    # ── routing ──────────────────────────────────────────────────────

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
        return await self.router.resolve(
            code=code, tier=tier, domain=domain,
            prompt_chars=prompt_chars,
            max_response_chars=max_response_chars,
            prefs=prefs,
        )

    # ── wallet operations ────────────────────────────────────────────

    def wallet(self, owner: str) -> Wallet:
        """Get a :class:`Wallet` handle for any owner address."""
        return Wallet(self._redis, owner=owner)

    async def hold(
        self,
        *,
        caller: str,
        amount: float,
        hold_id: str,
        reason: str = "",
    ) -> None:
        """Place a dispatch-time hold on the caller's wallet."""
        await self.wallet(caller).hold(
            hold_id=hold_id, amount=amount, reason=reason,
        )

    async def settle(
        self,
        *,
        caller: str,
        hold_id: str,
        settlement: Settlement,
        server_owner: str,
        compute_provider_id: str | None,
    ) -> None:
        """Apply a settlement, paying out to all four recipients atomically.

        Caller's hold is released and debited by ``settlement.pre_tax``.
        The four shares are credited to their respective wallets. If
        the caller's compute provider is None (self-hosted), the
        compute slice flows back to the server's own wallet.
        """
        caller_w = self.wallet(caller)
        server_w = self.wallet(server_owner)
        broker_w = self.wallet(BROKER_WALLET)
        commons_w = self.wallet(COMMONS_WALLET)

        # 1. Release the hold by debiting the actual pre_tax.
        await caller_w.settle_against_hold(
            hold_id=hold_id, debit=settlement.pre_tax,
            reason=f"settle hold {hold_id}",
        )

        # 2. Credit the server (residual).
        if settlement.to_server > 0:
            await server_w.topup(
                settlement.to_server,
                reason=f"server share of {hold_id}",
            )

        # 3. Credit compute provider OR the server itself if self-hosted.
        if settlement.to_compute > 0:
            target = compute_provider_id or server_owner
            await self.wallet(target).topup(
                settlement.to_compute,
                reason=f"compute share of {hold_id}",
            )

        # 4. Tax.
        if settlement.to_broker > 0:
            await broker_w.topup(
                settlement.to_broker,
                reason=f"broker tax {hold_id}",
            )
        if settlement.to_commons > 0:
            await commons_w.topup(
                settlement.to_commons,
                reason=f"commons tax {hold_id}",
            )

    async def refund(
        self,
        *,
        caller: str,
        hold_id: str,
        reason: str = "",
    ) -> None:
        """Release a hold back to the caller (no debit). For dispatch
        failures and timeouts.
        """
        await self.wallet(caller).refund(hold_id, reason=reason)

    # ── reputation ───────────────────────────────────────────────────

    async def get_reputation(self, owner: str) -> ReputationEntry | None:
        raw = await self._redis.get(REPUTATION_KEY.format(owner=owner))
        if raw is None:
            return None
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        fields = ReputationEntry.__dataclass_fields__
        return ReputationEntry(**{k: v for k, v in data.items() if k in fields})

    async def set_reputation(self, entry: ReputationEntry) -> None:
        payload = json.dumps(asdict(entry))
        await self._redis.set(REPUTATION_KEY.format(owner=entry.owner), payload)

    async def record_outcome(
        self,
        owner: str,
        outcome: SettlementOutcome,
        *,
        latency_ms: float = 0.0,
        response_chars: int = 0,
        max_response_chars: int = 0,
    ) -> ReputationEntry:
        """Update a reputation record from one settled call.

        Loads the current entry (or initializes from default), applies
        the outcome, also folds the response/budget ratio into
        ``avg_overage``, and persists.
        """
        current = await self.get_reputation(owner)
        if current is None:
            current = ReputationEntry(owner=owner)
        updated = apply_outcome(current, outcome, latency_ms=latency_ms)
        if max_response_chars > 0:
            updated.avg_overage = update_avg_overage(
                updated.avg_overage, response_chars, max_response_chars,
            )
        await self.set_reputation(updated)
        return updated

    # ── full settlement pipeline ─────────────────────────────────────

    async def calculate_and_settle(
        self,
        *,
        caller: str,
        hold_id: str,
        server: ServerMeta,
        leaf: MenuLeaf | None,
        response_chars: int,
        max_response_chars: int,
        actual_latency_ms: float,
        completed_with_caller: int,
        tier_verdict: str = "matched",
        responder: Any = None,
    ) -> Settlement:
        """Compute the settlement, apply it, and update reputation.

        This is the one-stop method the engine will call after a
        successful response. It:
            1. Reads the server's current reputation + verbosity stats.
            2. Calls :func:`settle_payment` to get the four-way split.
            3. Atomically credits all four wallets via :meth:`settle`.
            4. Updates the server's reputation record.
            5. If a registry was wired at construction and ``responder``
               is supplied, nudges the responding agent's
               :attr:`AgentMeta.reputation` by the same asymmetric
               magnitudes used at the server tier.

        Returns the :class:`Settlement` so the engine / audit can
        record it.
        """
        rep = await self.get_reputation(server.server_id) or ReputationEntry(
            owner=server.server_id,
        )
        leaf_rate = leaf.rate_per_1k_chars if leaf is not None else 0.0
        inputs = SettlementInputs(
            base_rate=server.base_rate,
            tier=leaf.tier if leaf is not None else "small",
            response_chars=response_chars,
            max_response_chars=max_response_chars,
            actual_latency_ms=actual_latency_ms,
            leaf_rate_per_1k_chars=leaf_rate,
            completed_with_caller=completed_with_caller,
            server_reputation=rep.reputation,
            server_avg_overage=rep.avg_overage,
            tier_verdict=tier_verdict,  # type: ignore[arg-type]
        )
        settlement = settle_payment(inputs)

        await self.settle(
            caller=caller,
            hold_id=hold_id,
            settlement=settlement,
            server_owner=server.server_id,
            compute_provider_id=leaf.provider_id if leaf is not None else None,
        )

        outcome: SettlementOutcome
        if tier_verdict == "sub_tier":
            outcome = "sub_tier"
        else:
            outcome = "accepted"
        await self.record_outcome(
            server.server_id, outcome,
            latency_ms=actual_latency_ms,
            response_chars=response_chars,
            max_response_chars=max_response_chars,
        )

        # Per-agent reputation feedback. The responder is the agent
        # whose response triggered this settlement; nudge AgentMeta.
        # reputation in the registry so per-agent rep evolves over
        # time alongside the server tier.
        if responder is not None and self._registry is not None:
            from ahp.economy.reputation import (
                REP_PENALTY_FAILURE, REP_REWARD_SUCCESS,
            )
            delta = (
                REP_REWARD_SUCCESS if outcome == "accepted"
                else -REP_PENALTY_FAILURE
            )
            try:
                await self._registry.update_reputation(responder, delta)
            except Exception:
                log.exception(
                    "agent reputation nudge failed; settlement stands "
                    "(responder=%s)", responder,
                )

        await self._emit(
            "broker.settlement",
            target=server.server_id,
            extra={
                "caller": caller,
                "hold_id": hold_id,
                "leaf": leaf.address if leaf is not None else None,
                "pre_tax": round(float(settlement.pre_tax), 6),
                "to_server": round(float(settlement.to_server), 6),
                "to_compute": round(float(settlement.to_compute), 6),
                "to_broker": round(float(settlement.to_broker), 6),
                "to_commons": round(float(settlement.to_commons), 6),
                "effective_chars": int(settlement.effective_chars),
                "actual_latency_ms": round(float(actual_latency_ms), 2),
                "tier_verdict": tier_verdict,
            },
        )

        # Stakes-gated auto-survey. Only fires when the settlement
        # cleared the threshold AND the target server has consented to
        # surveys. Caller is the surveyed actor (they're the one who
        # got the response and can judge usefulness).
        await self._maybe_request_survey(
            caller=caller,
            server=server,
            settlement=settlement,
            hold_id=hold_id,
        )

        return settlement

    # ── surveys ──────────────────────────────────────────────────────

    async def request_survey(
        self,
        *,
        kind: SurveyKind,
        target_server: str,
        surveyed_actor: str,
        recipe: str,
        settlement_id: str,
        reward: float,
        delay_seconds: float = 300.0,
        ttl_seconds: float = 86_400.0,
    ) -> SurveyRequest | None:
        """Public entry point to enqueue a survey.

        Consults the target server's ``survey_opt_in`` flag — surveys
        targeting an opted-out server are NOT enqueued, and this
        returns ``None``. Returns the :class:`SurveyRequest` on
        successful enqueue.

        Idempotent re-enqueue (same ``settlement_id`` + same actor)
        still produces a fresh ``survey_id`` because consent state may
        have changed; callers wanting idempotency should pass a
        pre-built request through :meth:`SurveyQueue.enqueue` directly.
        """
        meta = await self.servers.get(target_server)
        if meta is not None and not meta.survey_opt_in:
            log.info(
                "skipping survey: target server %s opted out (survey_opt_in=False)",
                target_server,
            )
            return None
        # Pull consent snapshot for queue-time hint. Defaults match the
        # ServerMeta defaults (csat_routing on, training off).
        csat_at_queue = bool(meta.csat_routing_opt_in) if meta else True
        train_at_queue = bool(meta.training_data_opt_in) if meta else False
        request = SurveyRequest.new(
            kind=kind,
            target_server=target_server,
            surveyed_actor=surveyed_actor,
            recipe=recipe,
            settlement_id=settlement_id,
            reward=reward,
            delay_seconds=delay_seconds,
            ttl_seconds=ttl_seconds,
        )
        # Stamp the consent snapshot.
        request = SurveyRequest(
            survey_id=request.survey_id,
            kind=request.kind,
            target_server=request.target_server,
            surveyed_actor=request.surveyed_actor,
            recipe=request.recipe,
            settlement_id=request.settlement_id,
            reward=request.reward,
            dispatch_at=request.dispatch_at,
            expires_at=request.expires_at,
            consent_csat_routing_at_queue=csat_at_queue,
            consent_training_export_at_queue=train_at_queue,
        )
        await self.surveys.enqueue(request)
        return request

    async def submit_survey_response(
        self,
        response: SurveyResponse,
    ) -> bool:
        """Wallet + CSAT side effects fold into ``self``."""
        return await self.surveys.submit_response(response, broker=self)

    async def _maybe_request_survey(
        self,
        *,
        caller: str,
        server: ServerMeta,
        settlement: Settlement,
        hold_id: str,
    ) -> None:
        """Auto-enqueue a survey when settlement stakes warrant it.

        Threshold + reward rate live in env so operators tune without
        code changes. Errors are logged but never bubble — survey
        enqueue failures must not undo a completed settlement.
        """
        try:
            threshold = float(os.environ.get(
                "AHP_SURVEY_STAKES_THRESHOLD",
                str(DEFAULT_SURVEY_STAKES_THRESHOLD),
            ))
            reward_rate = float(os.environ.get(
                "AHP_SURVEY_REWARD_RATE",
                str(DEFAULT_SURVEY_REWARD_RATE),
            ))
        except ValueError:
            threshold, reward_rate = (
                DEFAULT_SURVEY_STAKES_THRESHOLD,
                DEFAULT_SURVEY_REWARD_RATE,
            )
        if settlement.pre_tax < threshold:
            return
        reward = round(settlement.pre_tax * reward_rate, 6)
        try:
            await self.request_survey(
                kind="post_settlement",
                target_server=server.server_id,
                surveyed_actor=caller,
                recipe="post_settlement:csat",
                settlement_id=hold_id,
                reward=reward,
            )
        except Exception:
            log.exception(
                "auto-enqueue survey failed; settlement stands "
                "(server=%s caller=%s hold_id=%s)",
                server.server_id, caller, hold_id,
            )
