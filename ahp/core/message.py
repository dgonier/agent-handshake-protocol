"""Protocol message envelope.

A :class:`Message` carries a single protocol-level action between agents.
The envelope is transport-agnostic — it's serialized as JSON for Redis
streams, but the same struct could ride any transport.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Union

from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern


Verb = Literal[
    "SEND",
    "SEND-GET",
    "CAST",
    "CAST-GET",
    "CAST-SUB",
    "INVALIDATE",
]

VALID_VERBS: frozenset[str] = frozenset(
    {"SEND", "SEND-GET", "CAST", "CAST-GET", "CAST-SUB", "INVALIDATE"}
)

# Lifecycle → TTL in seconds. 0 means "do not cache".
LIFECYCLE_TTL: dict[str, int] = {
    "longterm": 86_400,    # 24h
    "session": 3_600,      # 1h
    "ephemeral": 0,        # no cache
    "stale-ok": 604_800,   # 7d
}

# Verbs that target a single agent vs. a pattern.
_POINT_TO_POINT_VERBS: frozenset[str] = frozenset({"SEND", "SEND-GET"})
_BROADCAST_VERBS: frozenset[str] = frozenset({"CAST", "CAST-GET", "CAST-SUB", "INVALIDATE"})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


def body_digest(body: Any) -> str:
    """Stable hex digest of a message body for cache-key derivation.

    ``None`` and missing bodies hash to the empty string so that two
    requests with no body share a key. Bytes hash directly. Anything else
    is JSON-encoded with ``sort_keys=True``; non-JSON-serializable values
    fall back to ``repr()`` (best-effort — caller-defined types should
    implement ``__repr__`` deterministically if they want stable keys).
    """
    if body is None:
        return ""
    if isinstance(body, (bytes, bytearray)):
        return hashlib.sha256(bytes(body)).hexdigest()
    try:
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    except TypeError:
        encoded = repr(body)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class Message:
    """A single protocol message.

    The ``target`` is an :class:`AgentAddress` for point-to-point verbs
    (``SEND``, ``SEND-GET``) and an :class:`AddressPattern` for broadcast
    verbs (``CAST``, ``CAST-GET``, ``CAST-SUB``, ``INVALIDATE``).

    ``ttl`` defaults to the value implied by the source agent's lifecycle.
    """

    source: AgentAddress
    target: Union[AgentAddress, AddressPattern]
    verb: str
    code: str
    thread: str
    body: Any = None
    timestamp: datetime = field(default_factory=_utcnow)
    ttl: int | None = None
    reply_to: str | None = None
    message_id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if not isinstance(self.source, AgentAddress):
            raise TypeError(
                f"source must be AgentAddress, got {type(self.source).__name__}"
            )
        if not isinstance(self.target, (AgentAddress, AddressPattern)):
            raise TypeError(
                f"target must be AgentAddress or AddressPattern, "
                f"got {type(self.target).__name__}"
            )

        if self.verb not in VALID_VERBS:
            raise ValueError(
                f"invalid verb {self.verb!r}; valid: {sorted(VALID_VERBS)}"
            )

        if self.verb in _POINT_TO_POINT_VERBS and isinstance(self.target, AddressPattern):
            raise ValueError(
                f"verb {self.verb!r} requires an AgentAddress target, "
                f"not an AddressPattern"
            )
        if self.verb in _BROADCAST_VERBS and isinstance(self.target, AgentAddress):
            # Auto-broadcast to a concrete address is technically a degenerate
            # case; we permit it but normalize nothing — the engine can decide.
            pass

        if not isinstance(self.code, str) or not self.code:
            raise ValueError(f"code must be a non-empty string, got {self.code!r}")
        if not isinstance(self.thread, str) or not self.thread:
            raise ValueError(f"thread must be a non-empty string, got {self.thread!r}")

        if self.ttl is None:
            self.ttl = LIFECYCLE_TTL.get(self.source.lifecycle, 0)
        elif not isinstance(self.ttl, int) or self.ttl < 0:
            raise ValueError(f"ttl must be a non-negative int, got {self.ttl!r}")

        if self.timestamp.tzinfo is None:
            # Normalize naive timestamps to UTC rather than reject them; callers
            # building from raw dicts often forget the tz.
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)

    # ── helpers ─────────────────────────────────────────────────────────

    @property
    def is_broadcast(self) -> bool:
        return isinstance(self.target, AddressPattern)

    def cache_key(self) -> str:
        """SHA-256 cache key over (target address, code, body).

        Only valid for point-to-point requests — broadcast targets aren't
        cached. The body digest makes distinct queries to the same
        ``(target, code)`` collide-free, while same-body retries reuse
        the cached response.
        """
        if not isinstance(self.target, AgentAddress):
            raise ValueError(
                "cache keys require a concrete AgentAddress target, "
                "not an AddressPattern"
            )
        canonical = json.dumps(
            {
                "addr": str(self.target),
                "code": self.code,
                "body": body_digest(self.body),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def expects_response(self) -> bool:
        return self.verb in {"SEND-GET", "CAST-GET"}

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict form for transport."""
        return {
            "message_id": self.message_id,
            "source": str(self.source),
            "target": str(self.target),
            "target_kind": "pattern" if self.is_broadcast else "address",
            "verb": self.verb,
            "code": self.code,
            "thread": self.thread,
            "body": self.body,
            "timestamp": self.timestamp.isoformat(),
            "ttl": self.ttl,
            "reply_to": self.reply_to,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Inverse of :meth:`to_dict`."""
        source = AgentAddress.parse(data["source"])
        target_str = data["target"]
        kind = data.get("target_kind")
        if kind == "pattern":
            target: Union[AgentAddress, AddressPattern] = AddressPattern.parse(target_str)
        elif kind == "address":
            target = AgentAddress.parse(target_str)
        else:
            # Infer from verb if kind is missing.
            verb = data["verb"]
            if verb in _BROADCAST_VERBS:
                target = AddressPattern.parse(target_str)
            else:
                target = AgentAddress.parse(target_str)

        ts_raw = data["timestamp"]
        timestamp = (
            ts_raw
            if isinstance(ts_raw, datetime)
            else datetime.fromisoformat(ts_raw)
        )

        return cls(
            source=source,
            target=target,
            verb=data["verb"],
            code=data["code"],
            thread=data["thread"],
            body=data.get("body"),
            timestamp=timestamp,
            ttl=data.get("ttl"),
            reply_to=data.get("reply_to"),
            message_id=data.get("message_id", _new_id()),
        )
