"""Typed audit event.

Every emission point in the protocol produces one of these. The shape
is deliberately flat so it serializes cleanly to JSON for downstream
log/stream sinks (CloudWatch, OpenSearch, stdout, etc.).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """One observed operation on the AHP fabric.

    Fields:

    * ``op`` — short, namespaced op name (``"registry.register"``,
      ``"engine.send"``, ``"engine.cast"``, ``"cache.invalidate"``, ...).
    * ``timestamp`` — wall-clock seconds since epoch. Defaulted at
      construction so emit sites don't have to think about it.
    * ``principal`` — opaque principal id of whoever initiated the op,
      or ``None`` for anonymous/system actions.
    * ``source`` — string form of the originating :class:`AgentAddress`
      when the op is a dispatched message. ``None`` for registry ops.
    * ``target`` — string form of the resolved target address or
      pattern, when applicable.
    * ``code`` — protocol code on the message, when applicable.
    * ``verb`` — protocol verb on the message, when applicable.
    * ``success`` — outcome flag. ``False`` is paired with ``error``.
    * ``error`` — short error class name + message when ``success`` is
      false. Kept compact — full stack traces belong in logs, not the
      audit stream.
    * ``extra`` — sink-specific structured detail (counts, latencies,
      cache hit flags). Keys should be flat and stable.
    """

    op: str
    timestamp: float = field(default_factory=time.time)
    principal: str | None = None
    source: str | None = None
    target: str | None = None
    code: str | None = None
    verb: str | None = None
    success: bool = True
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict suitable for JSON serialization."""
        return asdict(self)

    def to_json(self) -> str:
        """Compact JSON encoding for log/stream sinks."""
        return json.dumps(self.to_dict(), separators=(",", ":"), default=str)
