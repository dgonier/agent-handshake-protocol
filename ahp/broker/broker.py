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
import time
from dataclasses import asdict
from typing import Any

from ahp.broker.compute_registry import ComputeProviderRegistry
from ahp.broker.router import (
    NoCandidatesError,
    RoutingDecision,
    RoutingPreferences,
    Router,
)
from ahp.broker.server_registry import ServerMeta, ServerRegistry
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


REPUTATION_KEY = "ahp:reputation:{owner}"


class Broker:
    """Top-level facade for the routing + settlement broker."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self.servers = ServerRegistry(redis_client)
        self.compute = ComputeProviderRegistry(redis_client)
        self.banker = AgentBanker(redis_client)
        self.router = Router(
            server_registry=self.servers,
            compute_registry=self.compute,
            reputation_lookup=self.get_reputation,
        )

    # ── server lifecycle ──────────────────────────────────────────────

    async def register_server(self, meta: ServerMeta) -> None:
        await self.servers.register(meta)

    async def deregister_server(self, server_id: str) -> None:
        await self.servers.deregister(server_id)

    async def heartbeat_server(self, server_id: str) -> bool:
        return await self.servers.heartbeat(server_id)

    # ── compute provider lifecycle ────────────────────────────────────

    async def register_compute_provider(self, provider: ComputeProvider) -> None:
        await self.compute.register_provider(provider)

    async def register_leaf(self, leaf: MenuLeaf) -> None:
        await self.compute.register_leaf(leaf)

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
    ) -> Settlement:
        """Compute the settlement, apply it, and update reputation.

        This is the one-stop method the engine will call after a
        successful response. It:
            1. Reads the server's current reputation + verbosity stats.
            2. Calls :func:`settle_payment` to get the four-way split.
            3. Atomically credits all four wallets via :meth:`settle`.
            4. Updates the server's reputation record.

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
        return settlement
