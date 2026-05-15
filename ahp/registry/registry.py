"""Agent registry: registration, pattern resolution, discovery, liveness.

Storage layout (under :class:`Keys`):

* ``ahp:registry`` — Redis hash mapping ``str(AgentAddress)`` →
  JSON-encoded :class:`AgentMeta`.
* ``ahp:alive:<addr>`` — TTL'd marker key; an agent is considered live
  while this key exists.

Pattern resolution scans the registry hash and filters in Python. This is
adequate up to thousands of agents; a search index can replace it later
without changing the public API.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncIterator

from ahp.audit import AuditEvent, AuditSink
from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern
from ahp.registry.auth import (
    AuthPolicy,
    OpenAuthPolicy,
    Principal,
    UnauthorizedRegistrationError,
)
from ahp.transport.keys import Keys


DEFAULT_HEARTBEAT_TTL: int = 30
"""Seconds an agent's liveness marker survives without a heartbeat refresh."""


@dataclass
class AgentMeta:
    """Registry metadata attached to a registered agent."""

    capabilities: list[str] = field(default_factory=list)
    reputation: float = 0.0
    health_endpoint: str | None = None
    description: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    # Wall-clock seconds since epoch when this entry was last written.
    registered_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> "AgentMeta":
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        return cls(**data)


class AgentRegistry:
    """Redis-backed agent registry."""

    def __init__(
        self,
        redis_client: Any,
        *,
        heartbeat_ttl: int = DEFAULT_HEARTBEAT_TTL,
        principal: Principal | None = None,
        policy: AuthPolicy | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        if heartbeat_ttl <= 0:
            raise ValueError(f"heartbeat_ttl must be positive, got {heartbeat_ttl}")
        self._redis = redis_client
        self._heartbeat_ttl = heartbeat_ttl
        # Auth: default policy is permissive (open). Wire a concrete
        # AddressClaimPolicy + Principal to restrict who may register
        # which addresses; see ahp.registry.auth.
        self._principal = principal
        self._policy: AuthPolicy = policy or OpenAuthPolicy()
        # Audit: optional sink; emit-and-forget for register/heartbeat/
        # deregister outcomes. Read-side ops stay silent.
        self._audit = audit

    @property
    def principal(self) -> Principal | None:
        return self._principal

    @property
    def policy(self) -> AuthPolicy:
        return self._policy

    @property
    def heartbeat_ttl(self) -> int:
        return self._heartbeat_ttl

    # ── lifecycle ───────────────────────────────────────────────────────

    async def register(
        self,
        address: AgentAddress,
        metadata: AgentMeta | None = None,
    ) -> None:
        """Register or update an agent. Also marks it live for ``heartbeat_ttl`` seconds.

        Consults the active :class:`AuthPolicy`; raises
        :class:`UnauthorizedRegistrationError` if the principal isn't
        allowed to claim this address.
        """
        if not self._policy.can_register(self._principal, address):
            await self._emit(
                "registry.register", address, success=False,
                error="UnauthorizedRegistrationError",
            )
            raise UnauthorizedRegistrationError(
                f"principal {self._principal.id if self._principal else '<anonymous>'!r} "
                f"is not authorized to register {address}"
            )
        meta = metadata or AgentMeta()
        await self._redis.hset(
            Keys.registry_hash(), str(address), meta.to_json()
        )
        await self._mark_alive(address)
        await self._emit("registry.register", address)

    async def deregister(self, address: AgentAddress) -> None:
        """Remove an agent and its liveness marker.

        Consults the active :class:`AuthPolicy`.
        """
        if not self._policy.can_deregister(self._principal, address):
            await self._emit(
                "registry.deregister", address, success=False,
                error="UnauthorizedRegistrationError",
            )
            raise UnauthorizedRegistrationError(
                f"principal {self._principal.id if self._principal else '<anonymous>'!r} "
                f"is not authorized to deregister {address}"
            )
        await self._redis.hdel(Keys.registry_hash(), str(address))
        await self._redis.delete(Keys.alive_key(address))
        await self._emit("registry.deregister", address)

    async def heartbeat(self, address: AgentAddress) -> bool:
        """Refresh an agent's liveness marker.

        Returns False if not registered. Raises
        :class:`UnauthorizedRegistrationError` if the active policy
        denies the heartbeat — heartbeating extends a registration,
        which is a privileged op.
        """
        if not self._policy.can_heartbeat(self._principal, address):
            await self._emit(
                "registry.heartbeat", address, success=False,
                error="UnauthorizedRegistrationError",
            )
            raise UnauthorizedRegistrationError(
                f"principal {self._principal.id if self._principal else '<anonymous>'!r} "
                f"is not authorized to heartbeat {address}"
            )
        exists = await self._redis.hexists(Keys.registry_hash(), str(address))
        if not exists:
            await self._emit(
                "registry.heartbeat", address, success=False,
                error="NotRegistered",
            )
            return False
        await self._mark_alive(address)
        await self._emit("registry.heartbeat", address)
        return True

    async def is_alive(self, address: AgentAddress) -> bool:
        return bool(await self._redis.exists(Keys.alive_key(address)))

    async def get(self, address: AgentAddress) -> AgentMeta | None:
        raw = await self._redis.hget(Keys.registry_hash(), str(address))
        if raw is None:
            return None
        return AgentMeta.from_json(raw)

    async def count(self, *, alive_only: bool = False) -> int:
        if not alive_only:
            return await self._redis.hlen(Keys.registry_hash())
        n = 0
        async for _ in self._iter_alive():
            n += 1
        return n

    # ── discovery ───────────────────────────────────────────────────────

    async def resolve(
        self,
        pattern: AddressPattern,
        *,
        alive_only: bool = True,
    ) -> list[AgentAddress]:
        """Return all registered agents matching ``pattern``."""
        results: list[AgentAddress] = []
        async for addr, _meta in self._scan(alive_only=alive_only):
            if pattern.matches(addr):
                results.append(addr)
        return results

    async def discover(
        self,
        *,
        org: str = "*",
        role: str = "*",
        domain: str = "*",
        subdomain: str = "*",
        accept: str = "*",
        lifecycle: str = "*",
        instance: str = "*",
        min_reputation: float = 0.0,
        capability: str | None = None,
        alive_only: bool = True,
    ) -> list[tuple[AgentAddress, AgentMeta]]:
        """Rich discovery query. Returns ``(address, metadata)`` pairs."""
        pattern = AddressPattern(
            org=org, role=role, domain=domain, subdomain=subdomain,
            accept=accept, lifecycle=lifecycle, instance=instance,
        )
        out: list[tuple[AgentAddress, AgentMeta]] = []
        async for addr, meta in self._scan(alive_only=alive_only):
            if not pattern.matches(addr):
                continue
            if meta.reputation < min_reputation:
                continue
            if capability is not None and capability not in meta.capabilities:
                continue
            out.append((addr, meta))
        return out

    async def list_all(self, *, alive_only: bool = False) -> list[AgentAddress]:
        out: list[AgentAddress] = []
        async for addr, _ in self._scan(alive_only=alive_only):
            out.append(addr)
        return out

    # ── internals ───────────────────────────────────────────────────────

    async def _mark_alive(self, address: AgentAddress) -> None:
        await self._redis.set(
            Keys.alive_key(address), "1", ex=self._heartbeat_ttl
        )

    async def _emit(
        self,
        op: str,
        address: AgentAddress,
        *,
        success: bool = True,
        error: str | None = None,
    ) -> None:
        if self._audit is None:
            return
        principal_id = self._principal.id if self._principal else None
        await self._audit.emit(AuditEvent(
            op=op,
            principal=principal_id,
            target=str(address),
            success=success,
            error=error,
        ))

    async def _scan(
        self, *, alive_only: bool,
    ) -> AsyncIterator[tuple[AgentAddress, AgentMeta]]:
        entries = await self._redis.hgetall(Keys.registry_hash())
        for addr_str, meta_raw in entries.items():
            if isinstance(addr_str, (bytes, bytearray)):
                addr_str = addr_str.decode("utf-8")
            try:
                addr = AgentAddress.parse(addr_str)
            except ValueError:
                continue  # corrupt entry — skip
            if alive_only and not await self.is_alive(addr):
                continue
            try:
                meta = AgentMeta.from_json(meta_raw)
            except (json.JSONDecodeError, TypeError):
                continue
            yield addr, meta

    async def _iter_alive(self) -> AsyncIterator[AgentAddress]:
        async for addr, _ in self._scan(alive_only=True):
            yield addr
