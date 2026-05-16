"""Compute provider registry.

Two Redis keyspaces:

* ``ahp:compute_provider`` — hash of ``provider_id`` →
  :class:`~ahp.economy.ComputeProvider` JSON.
* ``ahp:compute_menu:<provider>.<tier>.<model>`` — one key per
  :class:`~ahp.economy.MenuLeaf`. Pattern resolution scans this
  keyspace.

The split is deliberate: provider metadata is small and rarely changes
(`HGETALL` is fine), while menu leaves change more often (capacity,
latency, healthy flag) and benefit from per-leaf updates without
rewriting the whole provider blob.

Liveness:

* Providers heartbeat themselves; absence of a heartbeat after the TTL
  means the broker treats their leaves as unhealthy when ranking, even
  if the JSON still claims healthy=True.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, AsyncIterator

from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.tiers import parse_tier


PROVIDER_HASH_KEY = "ahp:compute_provider"
PROVIDER_ALIVE_KEY = "ahp:compute_provider:alive:{provider_id}"
MENU_LEAF_KEY = "ahp:compute_menu:{address}"
MENU_LEAF_INDEX = "ahp:compute_menu:index"
"""SET of all live MenuLeaf addresses, kept in sync with the per-leaf keys.

Avoids a `KEYS ahp:compute_menu:*` scan on every pattern resolve.
"""

DEFAULT_HEARTBEAT_TTL: int = 30


class ComputeProviderRegistry:
    """Directory of compute providers and their menu leaves."""

    def __init__(
        self,
        redis_client: Any,
        *,
        heartbeat_ttl: int = DEFAULT_HEARTBEAT_TTL,
    ) -> None:
        if heartbeat_ttl <= 0:
            raise ValueError(f"heartbeat_ttl must be positive, got {heartbeat_ttl}")
        self._redis = redis_client
        self._ttl = heartbeat_ttl

    @property
    def heartbeat_ttl(self) -> int:
        return self._ttl

    # ── providers ──────────────────────────────────────────────────────

    async def register_provider(self, provider: ComputeProvider) -> None:
        payload = json.dumps(asdict(provider))
        await self._redis.hset(PROVIDER_HASH_KEY, provider.provider_id, payload)
        await self._mark_alive(provider.provider_id)

    async def deregister_provider(self, provider_id: str) -> None:
        # Cascade: drop the provider's leaves too.
        leaves = await self.menu_for_provider(provider_id)
        for leaf in leaves:
            await self.deregister_leaf(leaf.address)
        await self._redis.hdel(PROVIDER_HASH_KEY, provider_id)
        await self._redis.delete(PROVIDER_ALIVE_KEY.format(provider_id=provider_id))

    async def heartbeat_provider(self, provider_id: str) -> bool:
        exists = await self._redis.hexists(PROVIDER_HASH_KEY, provider_id)
        if not exists:
            return False
        await self._mark_alive(provider_id)
        return True

    async def get_provider(self, provider_id: str) -> ComputeProvider | None:
        raw = await self._redis.hget(PROVIDER_HASH_KEY, provider_id)
        if raw is None:
            return None
        return _provider_from_raw(raw)

    async def is_provider_alive(self, provider_id: str) -> bool:
        return bool(await self._redis.exists(
            PROVIDER_ALIVE_KEY.format(provider_id=provider_id)
        ))

    async def list_providers(self) -> list[ComputeProvider]:
        out: list[ComputeProvider] = []
        async for p in self._scan_providers():
            out.append(p)
        return out

    # ── menu leaves ───────────────────────────────────────────────────

    async def register_leaf(self, leaf: MenuLeaf) -> None:
        # Validate the provider exists. We don't *require* it (some
        # bootstrap orderings register leaves before the provider),
        # but we'll create an empty provider record so the leaf has
        # a parent.
        if not await self._redis.hexists(PROVIDER_HASH_KEY, leaf.provider_id):
            placeholder = ComputeProvider(provider_id=leaf.provider_id)
            await self.register_provider(placeholder)

        payload = json.dumps({
            "provider_id": leaf.provider_id,
            "tier": leaf.tier,
            "model": leaf.model,
            "rate_per_1k_chars": leaf.rate_per_1k_chars,
            "latency_p95_ms": leaf.latency_p95_ms,
            "capacity": leaf.capacity,
            "healthy": leaf.healthy,
        })
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.set(MENU_LEAF_KEY.format(address=leaf.address), payload)
            pipe.sadd(MENU_LEAF_INDEX, leaf.address)
            await pipe.execute()

    async def deregister_leaf(self, address: str) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.delete(MENU_LEAF_KEY.format(address=address))
            pipe.srem(MENU_LEAF_INDEX, address)
            await pipe.execute()

    async def get_leaf(self, address: str) -> MenuLeaf | None:
        raw = await self._redis.get(MENU_LEAF_KEY.format(address=address))
        if raw is None:
            return None
        return _leaf_from_raw(raw)

    async def list_leaves(
        self,
        *,
        only_alive_providers: bool = True,
    ) -> list[MenuLeaf]:
        """Return every registered leaf.

        ``only_alive_providers=True`` filters out leaves whose provider
        has missed heartbeats. Pattern resolution should pass this so
        dead providers don't get routed traffic.
        """
        addresses = await self._redis.smembers(MENU_LEAF_INDEX)
        if not addresses:
            return []
        out: list[MenuLeaf] = []
        for addr in addresses:
            if isinstance(addr, (bytes, bytearray)):
                addr = addr.decode("utf-8")
            leaf = await self.get_leaf(addr)
            if leaf is None:
                # Stale index entry — clean it up opportunistically.
                await self._redis.srem(MENU_LEAF_INDEX, addr)
                continue
            if only_alive_providers and not await self.is_provider_alive(
                leaf.provider_id,
            ):
                continue
            out.append(leaf)
        return out

    async def menu_for_provider(self, provider_id: str) -> list[MenuLeaf]:
        all_leaves = await self.list_leaves(only_alive_providers=False)
        return [l for l in all_leaves if l.provider_id == provider_id]

    # ── internals ─────────────────────────────────────────────────────

    async def _mark_alive(self, provider_id: str) -> None:
        await self._redis.set(
            PROVIDER_ALIVE_KEY.format(provider_id=provider_id), "1",
            ex=self._ttl,
        )

    async def _scan_providers(self) -> AsyncIterator[ComputeProvider]:
        entries = await self._redis.hgetall(PROVIDER_HASH_KEY)
        for _, raw in entries.items():
            try:
                yield _provider_from_raw(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue


def _provider_from_raw(raw: str | bytes) -> ComputeProvider:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    # Keep only fields the dataclass actually has, so older payloads
    # don't choke on new fields and new payloads can extend.
    fields = ComputeProvider.__dataclass_fields__
    return ComputeProvider(**{k: v for k, v in data.items() if k in fields})


def _leaf_from_raw(raw: str | bytes) -> MenuLeaf:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    parse_tier(data["tier"])  # validate
    return MenuLeaf(
        provider_id=data["provider_id"],
        tier=data["tier"],
        model=data["model"],
        rate_per_1k_chars=float(data.get("rate_per_1k_chars", 0.0)),
        latency_p95_ms=float(data.get("latency_p95_ms", 1000.0)),
        capacity=float(data.get("capacity", 1.0)),
        healthy=bool(data.get("healthy", True)),
    )
