"""Auth control plane — who may register at which addresses.

Default is **open**: with no :class:`Principal` / :class:`AuthPolicy`
attached, any process with Redis access can register at any address.
That matches the protocol's "open beyond normal filtering" stance and
keeps single-tenant deployments ergonomic.

Opt in by constructing the registry with both a ``principal`` (the
identity of the current process) and a ``policy`` that decides
whether that principal is allowed to mutate a given address:

::

    from ahp.registry.auth import AddressClaimPolicy, Principal

    node_a = Principal(
        id="node-a",
        claims=(AddressPattern.parse("tifin.adversarial.finance.*.*.*.*"),),
    )

    registry = AgentRegistry(
        redis_client,
        principal=node_a,
        policy=AddressClaimPolicy(),
    )

    # Allowed:
    await registry.register(AgentAddress.parse(
        "tifin.adversarial.finance.equities.s.session.bull",
    ))

    # Denied — node-a's claims don't cover collaborative agents:
    await registry.register(AgentAddress.parse(
        "tifin.collaborative.finance.equities.s.session.alice",
    ))   # → UnauthorizedRegistrationError

The :class:`Principal` itself carries the claim list directly. In a
real deployment those claims would arrive as a signed JWT or other
verifiable credential; this module is intentionally agnostic about
where the proof of identity comes from — it just consumes the parsed
claims. Add a signature-verification layer on top when you ship.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable

from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern


class UnauthorizedRegistrationError(Exception):
    """Raised when a principal tries to mutate the registry beyond its claims."""


@dataclass(frozen=True)
class Principal:
    """Identity of the current process for registry operations.

    ``claims`` is the set of :class:`AddressPattern`-s this principal
    is allowed to register / deregister / heartbeat. Anonymous /
    unauthenticated principals would carry an empty claim tuple — the
    default ``AddressClaimPolicy`` denies those entirely (use
    :class:`OpenAuthPolicy` instead for genuinely open operation).
    """

    id: str
    claims: tuple[AddressPattern, ...] = ()
    metadata: dict = field(default_factory=dict)

    @classmethod
    def with_claims(cls, id: str, *patterns: AddressPattern | str) -> "Principal":
        parsed = tuple(
            AddressPattern.parse(p) if isinstance(p, str) else p
            for p in patterns
        )
        return cls(id=id, claims=parsed)

    def covers(self, address: AgentAddress) -> bool:
        return any(pat.matches(address) for pat in self.claims)


# ── policies ───────────────────────────────────────────────────────────


class AuthPolicy(ABC):
    """Decides whether a principal may mutate a given registry entry."""

    @abstractmethod
    def can_register(
        self, principal: Principal | None, address: AgentAddress,
    ) -> bool:
        ...

    def can_deregister(
        self, principal: Principal | None, address: AgentAddress,
    ) -> bool:
        # Same semantics by default — owners can take down what they put up.
        return self.can_register(principal, address)

    def can_heartbeat(
        self, principal: Principal | None, address: AgentAddress,
    ) -> bool:
        # Heartbeat extends a registration — same gate.
        return self.can_register(principal, address)


class OpenAuthPolicy(AuthPolicy):
    """Default. Any principal (including ``None``) can do anything."""

    def can_register(self, principal, address):  # type: ignore[override]
        return True


class AddressClaimPolicy(AuthPolicy):
    """A principal may mutate an address iff one of their claims matches.

    A ``None`` principal is denied — this policy presumes the caller
    has identified themselves. Use :class:`OpenAuthPolicy` if you
    want anonymity.
    """

    def can_register(self, principal, address):  # type: ignore[override]
        if principal is None:
            return False
        return principal.covers(address)


class DenyAllPolicy(AuthPolicy):
    """Refuses everything. Useful for read-only mirrors of a registry."""

    def can_register(self, principal, address):  # type: ignore[override]
        return False
