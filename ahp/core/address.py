"""Agent addresses.

An address is a dot-delimited 7-tuple, optionally followed by `?key=value`
query parameters:

    {org}.{role}.{domain}.{subdomain}.{accept}.{lifecycle}.{instance}?{params}

Example: ``tifin.adversarial.finance.projections.j.longterm.frank?stock=Tesla``
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl, quote, unquote

if TYPE_CHECKING:
    from ahp.core.pattern import AddressPattern


ACCEPT_TIER_ORDER: str = "sjbe"
"""Canonical order of accept-tier characters: string < JSON < bytes < embeddings.

Examples from the spec: ``"s"``, ``"j"``, ``"sj"``, ``"be"``, ``"sjbe"``.
"""

VALID_ACCEPT_CHARS: frozenset[str] = frozenset(ACCEPT_TIER_ORDER)
"""Accept tier characters: s=string, j=JSON, b=bytes, e=embeddings."""


class AcceptTier:
    """Named constants for the four accept tiers.

    Use ``AcceptTier.STRING`` instead of ``"s"`` at call sites where
    readability matters. Each constant is the canonical single-char
    tier code; multi-tier accept strings (e.g. ``"sj"``) still use the
    raw chars.
    """

    STRING: str = "s"
    JSON: str = "j"
    BYTES: str = "b"
    EMBEDDINGS: str = "e"

    ALL: str = ACCEPT_TIER_ORDER
    """The full ``"sjbe"`` string, in canonical tier order."""


_ACCEPT_RANK: dict[str, int] = {c: i for i, c in enumerate(ACCEPT_TIER_ORDER)}


def _canonical_accept(accept: str) -> str:
    """Sort accept characters into canonical tier order."""
    return "".join(sorted(accept, key=_ACCEPT_RANK.__getitem__))

VALID_LIFECYCLES: frozenset[str] = frozenset(
    {"longterm", "session", "ephemeral", "stale-ok"}
)

_FIELD_NAMES: tuple[str, ...] = (
    "org",
    "role",
    "domain",
    "subdomain",
    "accept",
    "lifecycle",
    "instance",
)


def _validate_token(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name!r} must be a non-empty string, got {value!r}")
    for forbidden in (".", "?", " ", "\t", "\n"):
        if forbidden in value:
            raise ValueError(
                f"{name!r} contains forbidden character {forbidden!r}: {value!r}"
            )


@dataclass
class AgentAddress:
    """A fully-qualified address for a single agent instance.

    Two addresses compare equal when their canonical URI form is identical,
    so ``params`` order is irrelevant.
    """

    org: str
    role: str
    domain: str
    subdomain: str
    accept: str
    lifecycle: str
    instance: str
    params: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("org", "role", "domain", "subdomain", "instance"):
            _validate_token(name, getattr(self, name))

        if not self.accept:
            raise ValueError("accept tier must be non-empty")
        if any(c not in VALID_ACCEPT_CHARS for c in self.accept):
            raise ValueError(
                f"accept contains invalid characters: {self.accept!r} "
                f"(valid: {''.join(sorted(VALID_ACCEPT_CHARS))})"
            )
        if len(set(self.accept)) != len(self.accept):
            raise ValueError(f"accept has duplicate characters: {self.accept!r}")
        if self.accept != _canonical_accept(self.accept):
            raise ValueError(
                f"accept must be in canonical tier order (s,j,b,e): "
                f"got {self.accept!r}, expected {_canonical_accept(self.accept)!r}"
            )

        if self.lifecycle not in VALID_LIFECYCLES:
            raise ValueError(
                f"invalid lifecycle {self.lifecycle!r}; "
                f"valid: {sorted(VALID_LIFECYCLES)}"
            )

        if not isinstance(self.params, dict):
            raise TypeError(f"params must be a dict, got {type(self.params).__name__}")
        for k, v in self.params.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise TypeError(
                    f"params keys and values must be strings; "
                    f"got {k!r}={v!r}"
                )

    @classmethod
    def parse(cls, uri: str) -> "AgentAddress":
        """Parse a URI like ``org.role.domain.subdomain.accept.lifecycle.instance?k=v``."""
        if not isinstance(uri, str):
            raise TypeError(f"uri must be str, got {type(uri).__name__}")
        uri = uri.strip()
        if not uri:
            raise ValueError("cannot parse empty address")

        if "?" in uri:
            addr_part, _, params_part = uri.partition("?")
            params = {k: v for k, v in parse_qsl(params_part, keep_blank_values=True)}
        else:
            addr_part = uri
            params = {}

        parts = addr_part.split(".")
        if len(parts) != 7:
            raise ValueError(
                f"address must have exactly 7 dot-separated fields, got "
                f"{len(parts)}: {uri!r}"
            )

        return cls(
            org=parts[0],
            role=parts[1],
            domain=parts[2],
            subdomain=parts[3],
            accept=parts[4],
            lifecycle=parts[5],
            instance=parts[6],
            params=params,
        )

    def __str__(self) -> str:
        base = ".".join(getattr(self, name) for name in _FIELD_NAMES)
        if not self.params:
            return base
        # Canonical params: sorted by key, percent-encoded
        encoded = "&".join(
            f"{quote(k, safe='')}={quote(v, safe='')}"
            for k, v in sorted(self.params.items())
        )
        return f"{base}?{encoded}"

    def __repr__(self) -> str:
        return f"AgentAddress({str(self)!r})"

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AgentAddress):
            return NotImplemented
        return str(self) == str(other)

    # ── public helpers ──────────────────────────────────────────────────

    def matches(self, pattern: "AddressPattern") -> bool:
        """True if this address matches the given wildcard pattern."""
        return pattern.matches(self)

    def cache_key(self, code: str) -> str:
        """Deterministic SHA-256 cache key for ``(address, code)``.

        Params are included via the canonical URI form, so the same logical
        address always produces the same key regardless of dict ordering.
        """
        canonical = json.dumps(
            {"addr": str(self), "code": code},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def accepts(self, tier: str) -> bool:
        """True if this agent accepts the given single-character tier."""
        if len(tier) != 1:
            raise ValueError(f"tier must be a single character, got {tier!r}")
        return tier in self.accept

    def accepts_any(self, tiers: set[str]) -> bool:
        """True if this agent accepts at least one of the given tiers."""
        return bool(set(self.accept) & tiers)

    @property
    def is_human(self) -> bool:
        return self.role == "human"

    @property
    def fields(self) -> tuple[str, ...]:
        """The seven structural fields, in canonical order (no params)."""
        return tuple(getattr(self, name) for name in _FIELD_NAMES)
