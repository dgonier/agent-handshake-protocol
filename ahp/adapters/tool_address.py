"""Structured addresses for tools and resources.

Mirrors the agent address scheme — every callable or resource has a
canonical, dotted address that drives discovery, access control, and
auto-binding to agents.

* :class:`ToolAddress` — ``scope.kind.role.category.operation``
* :class:`ResourceAddress` — ``scope.kind.domain.subdomain.name``

Each field accepts ``*`` as a wildcard at *registration* time (e.g.
``scope="*"`` means "any org") and at *matching* time. The
``derived_allowed_for`` helpers translate a tool / resource address
into an :class:`AddressPattern` over agent addresses, encoding the
default convention: a tool's ``scope`` must match the agent's ``org``
and its ``role`` must match the agent's ``role`` (other fields don't
gate access).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

from ahp.core.pattern import AddressPattern


_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-]+|\*$")

_FIELD_FORBIDDEN = (".", "?", " ", "\t", "\n")


def _validate_field(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name!r} must be a non-empty string, got {value!r}")
    for forbidden in _FIELD_FORBIDDEN:
        if forbidden in value:
            raise ValueError(
                f"{name!r} contains forbidden character {forbidden!r}: {value!r}"
            )


@dataclass(frozen=True)
class ToolAddress:
    """Five-field tool address.

    ``scope.kind.role.category.operation``
    """

    scope: str       # org / tenant (``"tifin"``, ``"public"``, ``"*"``)
    kind: str        # ``"db"``, ``"fs"``, ``"api"``, ``"compute"``, ``"tool"`` ...
    role: str        # agent role this serves (``"adversarial"``, ``"*"``)
    category: str    # ``"crud"``, ``"search"``, ``"read"``, ``"write"``, ``"compute"``
    operation: str   # specific action (usually the function name)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "scope", "kind", "role", "category", "operation",
    )

    def __post_init__(self) -> None:
        for name in self.FIELDS:
            _validate_field(name, getattr(self, name))

    @classmethod
    def parse(cls, uri: str) -> "ToolAddress":
        parts = uri.strip().split(".")
        if len(parts) != 5:
            raise ValueError(
                f"tool address must have exactly 5 dot-separated fields, "
                f"got {len(parts)}: {uri!r}"
            )
        return cls(*parts)

    def __str__(self) -> str:
        return ".".join(getattr(self, f) for f in self.FIELDS)

    def __repr__(self) -> str:
        return f"ToolAddress({str(self)!r})"

    # ── default agent-access convention ────────────────────────────────

    def derived_allowed_for(self) -> AddressPattern:
        """Return the default :class:`AddressPattern` over agent addresses.

        Convention: a tool ``s.k.r.c.o`` may be used by any agent whose
        ``org`` field matches ``s`` and whose ``role`` field matches
        ``r``. Other agent fields are not constrained. Wildcards on the
        tool side translate to wildcards on the agent side.

        Override this via ``allowed_for=`` on registration when the
        convention isn't a fit.
        """
        return AddressPattern(
            org=self.scope, role=self.role,
            domain="*", subdomain="*", accept="*",
            lifecycle="*", instance="*",
        )


@dataclass(frozen=True)
class ResourceAddress:
    """Five-field resource address.

    ``scope.kind.domain.subdomain.name``
    """

    scope: str       # org / tenant
    kind: str        # ``"fs"``, ``"db"``, ``"vector"``, ``"kv"``, ``"api"``
    domain: str      # ``"finance"``, ``"science"``, ``"any"``
    subdomain: str   # ``"equities"``, ``"biology"``, ``"any"``
    name: str        # specific instance (``"sec-edgar"``, ``"docs-2024"``)

    FIELDS: ClassVar[tuple[str, ...]] = (
        "scope", "kind", "domain", "subdomain", "name",
    )

    def __post_init__(self) -> None:
        for name in self.FIELDS:
            _validate_field(name, getattr(self, name))

    @classmethod
    def parse(cls, uri: str) -> "ResourceAddress":
        parts = uri.strip().split(".")
        if len(parts) != 5:
            raise ValueError(
                f"resource address must have exactly 5 dot-separated "
                f"fields, got {len(parts)}: {uri!r}"
            )
        return cls(*parts)

    def __str__(self) -> str:
        return ".".join(getattr(self, f) for f in self.FIELDS)

    def __repr__(self) -> str:
        return f"ResourceAddress({str(self)!r})"

    def derived_allowed_for(self) -> AddressPattern:
        """Default access convention for resources.

        A resource ``s.k.d.sd.n`` is visible to agents whose ``org``
        matches ``s`` AND whose ``domain`` matches ``d`` AND whose
        ``subdomain`` matches ``sd``. Role / accept / lifecycle /
        instance are not constrained — a resource is shared across
        roles within its domain by default.
        """
        return AddressPattern(
            org=self.scope, role="*",
            domain=self.domain, subdomain=self.subdomain,
            accept="*", lifecycle="*", instance="*",
        )
