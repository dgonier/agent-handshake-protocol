"""Wildcard-capable address patterns for routing and broadcast.

A pattern has the same 7 structural fields as :class:`AgentAddress`, but
each field may be the literal ``*`` to match anything. The accept field is
special: a non-wildcard pattern accept like ``"sj"`` matches any address
whose accept *includes both* ``s`` and ``j`` (subset semantics).

Patterns do not carry query params — those are matched against the
concrete agent by other layers (e.g. the engine matches request params
against agent capabilities).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ahp.core.address import AgentAddress


WILDCARD: str = "*"

_FIELD_NAMES: tuple[str, ...] = (
    "org",
    "role",
    "domain",
    "subdomain",
    "accept",
    "lifecycle",
    "instance",
)


@dataclass(frozen=True)
class AddressPattern:
    """A 7-field pattern with per-field wildcard support."""

    org: str
    role: str
    domain: str
    subdomain: str
    accept: str
    lifecycle: str
    instance: str

    def __post_init__(self) -> None:
        for name in _FIELD_NAMES:
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"pattern field {name!r} must be non-empty string, got {value!r}"
                )
            if "." in value or "?" in value:
                raise ValueError(
                    f"pattern field {name!r} contains forbidden character: {value!r}"
                )

    @classmethod
    def parse(cls, pattern: str) -> "AddressPattern":
        """Parse a 7-field dotted pattern string."""
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be str, got {type(pattern).__name__}")
        pattern = pattern.strip()
        if "?" in pattern:
            raise ValueError(f"patterns do not carry query params: {pattern!r}")
        parts = pattern.split(".")
        if len(parts) != 7:
            raise ValueError(
                f"pattern must have exactly 7 dot-separated fields, got "
                f"{len(parts)}: {pattern!r}"
            )
        return cls(*parts)

    @classmethod
    def all(cls) -> "AddressPattern":
        """The fully-wildcarded pattern that matches every address."""
        return cls(*([WILDCARD] * 7))

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(getattr(self, name) for name in _FIELD_NAMES)

    def __str__(self) -> str:
        return ".".join(self.fields)

    def __repr__(self) -> str:
        return f"AddressPattern({str(self)!r})"

    # ── matching ────────────────────────────────────────────────────────

    def matches(self, address: "AgentAddress") -> bool:
        """True if ``address`` satisfies this pattern.

        Each non-wildcard structural field must match exactly. The accept
        field uses subset semantics: pattern accept ``"sj"`` requires the
        address to accept at least ``s`` and ``j``.
        """
        # Structural fields (exact match, except wildcard)
        for name in ("org", "role", "domain", "subdomain", "lifecycle", "instance"):
            pat = getattr(self, name)
            if pat == WILDCARD:
                continue
            if pat != getattr(address, name):
                return False

        # Accept: subset semantics
        if self.accept != WILDCARD:
            if not set(self.accept).issubset(set(address.accept)):
                return False

        return True

    @staticmethod
    def matches_accept(sender_accept: str, receiver_accept: str) -> bool:
        """True if the sender's output formats are all acceptable to the receiver.

        A sender emitting only strings (``"s"``) can talk to any receiver
        whose accept set contains ``s``. A sender emitting ``"sj"`` requires
        the receiver to accept both.
        """
        return set(sender_accept).issubset(set(receiver_accept))
