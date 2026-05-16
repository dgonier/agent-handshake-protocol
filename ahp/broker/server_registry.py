"""Server registry: the directory of processes that host agents.

A *server* is a process running a :class:`ProtocolEngine`. It claims
some set of agent addresses (typically all addresses where
``org == its_org``), and earns credits when other servers route to
those agents.

Storage layout (under :class:`Keys`):

* ``ahp:server`` — Redis hash mapping ``server_id`` → JSON
  :class:`ServerMeta`.
* ``ahp:server:alive:<server_id>`` — TTL'd liveness marker, refreshed
  by heartbeat.

The registry only stores **declared** server metadata. The behavioral
ledger (reputation, wallet, rolling stats) lives in adjacent
keyspaces under :mod:`ahp.economy`.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator

from ahp.economy.tiers import Tier, parse_tier


SERVER_HASH_KEY = "ahp:server"
SERVER_ALIVE_KEY = "ahp:server:alive:{server_id}"

DEFAULT_HEARTBEAT_TTL: int = 30
"""Seconds a server's liveness marker survives without a heartbeat refresh.

Matches :data:`ahp.registry.DEFAULT_HEARTBEAT_TTL` so the two layers
fail at roughly the same time.
"""


@dataclass
class ServerMeta:
    """Declared metadata for one server.

    Fields are split into three groups:

    * **identity** — who you are
    * **capability** — what you offer
    * **economics** — how you charge and bind to compute
    """

    # identity
    server_id: str
    org: str = ""                          # claimed org-namespace prefix
    operator: str = ""                     # human-readable owner
    public_key: str = ""                   # for signed registration (future)

    # capability
    specialties: list[str] = field(default_factory=list)
    integrations: list[str] = field(default_factory=list)
    supported_codes: list[str] = field(default_factory=list)
    supported_tiers: list[Tier] = field(default_factory=list)

    # economics
    base_rate: float = 0.0002              # credits per char per tier-unit
    compute_binding: str = "*.*.*"         # pattern over MenuLeaf addresses
    compute_ranking: str = "cheapest"      # passed to MenuLeaf ranker

    # consent — surveys + training data
    # Defaults: respond to surveys (yes), use CSAT for routing (yes),
    # contribute to open training-data corpus (NO — explicit opt-in).
    survey_opt_in: bool = True
    csat_routing_opt_in: bool = True
    training_data_opt_in: bool = False

    # housekeeping
    registered_at: float = field(default_factory=time.time)
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Cheap validation at construction; catches bad values before
        # they're persisted.
        if not self.server_id or not isinstance(self.server_id, str):
            raise ValueError(f"server_id must be a non-empty string: {self.server_id!r}")
        if self.base_rate < 0:
            raise ValueError(f"base_rate must be non-negative, got {self.base_rate}")
        for t in self.supported_tiers:
            parse_tier(t)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "ServerMeta":
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        # Tolerant of older payloads that don't have every field.
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class ServerRegistry:
    """Redis-backed directory of servers."""

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

    # ── lifecycle ──────────────────────────────────────────────────────

    async def register(self, meta: ServerMeta) -> None:
        """Register or update a server and mark it alive."""
        await self._redis.hset(SERVER_HASH_KEY, meta.server_id, meta.to_json())
        await self._mark_alive(meta.server_id)

    async def deregister(self, server_id: str) -> None:
        await self._redis.hdel(SERVER_HASH_KEY, server_id)
        await self._redis.delete(SERVER_ALIVE_KEY.format(server_id=server_id))

    async def heartbeat(self, server_id: str) -> bool:
        """Refresh liveness. Returns False if not registered."""
        exists = await self._redis.hexists(SERVER_HASH_KEY, server_id)
        if not exists:
            return False
        await self._mark_alive(server_id)
        return True

    async def is_alive(self, server_id: str) -> bool:
        return bool(await self._redis.exists(
            SERVER_ALIVE_KEY.format(server_id=server_id)
        ))

    async def get(self, server_id: str) -> ServerMeta | None:
        raw = await self._redis.hget(SERVER_HASH_KEY, server_id)
        if raw is None:
            return None
        return ServerMeta.from_json(raw)

    # ── discovery ──────────────────────────────────────────────────────

    async def list_all(self, *, alive_only: bool = False) -> list[ServerMeta]:
        """Return every registered server."""
        out: list[ServerMeta] = []
        async for meta in self._scan(alive_only=alive_only):
            out.append(meta)
        return out

    async def count(self, *, alive_only: bool = False) -> int:
        if not alive_only:
            return await self._redis.hlen(SERVER_HASH_KEY)
        n = 0
        async for _ in self._scan(alive_only=True):
            n += 1
        return n

    async def discover(
        self,
        *,
        specialty: str | None = None,
        integration: str | None = None,
        code: str | None = None,
        tier: Tier | None = None,
        alive_only: bool = True,
    ) -> list[ServerMeta]:
        """Filter the directory by declared capability.

        These are *capability* filters — the router applies further
        filters for reputation, wallet balance, and preferences.
        """
        if tier is not None:
            parse_tier(tier)

        out: list[ServerMeta] = []
        async for meta in self._scan(alive_only=alive_only):
            if specialty and specialty not in meta.specialties:
                continue
            if integration and integration not in meta.integrations:
                continue
            if code and meta.supported_codes and code not in meta.supported_codes:
                continue
            if tier and meta.supported_tiers and tier not in meta.supported_tiers:
                continue
            out.append(meta)
        return out

    # ── internals ─────────────────────────────────────────────────────

    async def _mark_alive(self, server_id: str) -> None:
        await self._redis.set(
            SERVER_ALIVE_KEY.format(server_id=server_id), "1", ex=self._ttl,
        )

    async def _scan(
        self, *, alive_only: bool,
    ) -> AsyncIterator[ServerMeta]:
        entries = await self._redis.hgetall(SERVER_HASH_KEY)
        for server_id, raw in entries.items():
            if isinstance(server_id, (bytes, bytearray)):
                server_id = server_id.decode("utf-8")
            try:
                meta = ServerMeta.from_json(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue  # corrupt entry — skip
            if alive_only and not await self.is_alive(server_id):
                continue
            yield meta
