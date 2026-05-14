"""Response cache keyed by (target address, code).

Cache TTL is derived from the *target's* lifecycle field: a query against
a ``longterm`` agent caches for 24h, against an ``ephemeral`` agent for
zero (skipped). Invalidation supports pattern + params filtering by
scanning entries — Phase 2's storage layout favors simplicity over
write-time index maintenance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ahp.core.address import AgentAddress
from ahp.core.message import LIFECYCLE_TTL, Message
from ahp.core.pattern import AddressPattern
from ahp.transport.keys import Keys


@dataclass(frozen=True)
class CachedEntry:
    """A stored cache entry, including the metadata needed for invalidation."""

    target_uri: str
    code: str
    response: Message

    def matches(self, pattern: AddressPattern, params: dict[str, str] | None) -> bool:
        target = AgentAddress.parse(self.target_uri)
        if not pattern.matches(target):
            return False
        if params:
            for k, v in params.items():
                if target.params.get(k) != v:
                    return False
        return True


class ProtocolCache:
    """Read-through cache for GET-style verb results."""

    LIFECYCLE_TTL: dict[str, int] = dict(LIFECYCLE_TTL)
    """Public copy of the lifecycle → TTL table."""

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client

    # ── key derivation ──────────────────────────────────────────────────

    @staticmethod
    def derive_key(request: Message) -> str:
        """Cache key digest for a request. Stable across runs.

        Combines the target address, the interaction code, and a digest
        of the request body — so two queries to the same ``(target,
        code)`` with different bodies don't collide on a cache slot.
        """
        return request.cache_key()

    @classmethod
    def ttl_for(cls, target: AgentAddress) -> int:
        return cls.LIFECYCLE_TTL.get(target.lifecycle, 0)

    # ── read / write ────────────────────────────────────────────────────

    async def get(self, request: Message) -> Message | None:
        """Return the cached response for ``request`` or None."""
        key = Keys.cache_key(self.derive_key(request))
        raw = await self._redis.get(key)
        if raw is None:
            return None
        return self._decode(raw).response

    async def put(self, request: Message, response: Message) -> bool:
        """Cache ``response`` keyed to ``request``. No-op for TTL=0. Returns whether stored."""
        target = request.target
        if not isinstance(target, AgentAddress):
            return False
        ttl = self.ttl_for(target)
        if ttl <= 0:
            return False
        entry = CachedEntry(
            target_uri=str(target),
            code=request.code,
            response=response,
        )
        key = Keys.cache_key(self.derive_key(request))
        await self._redis.set(key, self._encode(entry), ex=ttl)
        return True

    async def invalidate(
        self,
        pattern: AddressPattern,
        params: dict[str, str] | None = None,
    ) -> int:
        """Delete cached entries whose target matches ``pattern`` and includes ``params``."""
        deleted = 0
        async for key in self._redis.scan_iter(match=Keys.cache_scan_pattern()):
            raw = await self._redis.get(key)
            if raw is None:
                continue
            try:
                entry = self._decode(raw)
            except (json.JSONDecodeError, KeyError, ValueError):
                # Skip malformed entries — could be from a future schema version.
                continue
            if entry.matches(pattern, params):
                await self._redis.delete(key)
                deleted += 1
        return deleted

    async def clear(self) -> int:
        """Delete every AHP cache entry. Returns count deleted."""
        deleted = 0
        async for key in self._redis.scan_iter(match=Keys.cache_scan_pattern()):
            await self._redis.delete(key)
            deleted += 1
        return deleted

    # ── (de)serialization ───────────────────────────────────────────────

    @staticmethod
    def _encode(entry: CachedEntry) -> str:
        return json.dumps(
            {
                "target_uri": entry.target_uri,
                "code": entry.code,
                "response": entry.response.to_dict(),
            }
        )

    @staticmethod
    def _decode(raw: Any) -> CachedEntry:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        return CachedEntry(
            target_uri=data["target_uri"],
            code=data["code"],
            response=Message.from_dict(data["response"]),
        )
