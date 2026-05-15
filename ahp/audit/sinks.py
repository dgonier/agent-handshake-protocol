"""Audit sinks.

Sinks consume :class:`AuditEvent` instances. They are intentionally
async — even the in-memory sink — so callers can treat every sink the
same way without branching on type.

Three lightweight sinks live here:

* :class:`NullAuditSink` — drops everything. The default when no sink
  is wired so the audit path is a single ``if`` check.
* :class:`InMemoryAuditSink` — keeps a bounded ring of recent events.
  Useful for tests and debugging.
* :class:`LoggingAuditSink` — forwards JSON to a stdlib
  :class:`logging.Logger`.

A combinator:

* :class:`MultiSink` — fans an event out to multiple downstream sinks.
  Errors in one sink do not stop delivery to the others.

The CloudWatch sink lives in :mod:`ahp.audit.cloudwatch` to keep the
``boto3`` import optional.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Iterable, Protocol, runtime_checkable

from ahp.audit.event import AuditEvent


log = logging.getLogger(__name__)


@runtime_checkable
class AuditSink(Protocol):
    """Anything that can absorb an audit event."""

    async def emit(self, event: AuditEvent) -> None: ...


class NullAuditSink:
    """Drops every event. Default for "audit off"."""

    async def emit(self, event: AuditEvent) -> None:  # noqa: D401
        return None


class InMemoryAuditSink:
    """Keeps the last ``capacity`` events in a deque.

    Intended for tests and short-lived debug sessions. Not thread-safe
    beyond what ``deque`` guarantees — but the protocol is single-event
    appendright so concurrent emits are fine in practice.
    """

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._events: deque[AuditEvent] = deque(maxlen=capacity)

    async def emit(self, event: AuditEvent) -> None:
        self._events.append(event)

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def clear(self) -> None:
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


class LoggingAuditSink:
    """Forwards events to a stdlib logger as compact JSON.

    The logger is captured by name so library users can route the
    audit stream through their own logging config (formatters,
    handlers, filters) without changing AHP itself.
    """

    def __init__(
        self,
        logger_name: str = "ahp.audit",
        level: int = logging.INFO,
    ) -> None:
        self._logger = logging.getLogger(logger_name)
        self._level = level

    async def emit(self, event: AuditEvent) -> None:
        self._logger.log(self._level, event.to_json())


class MultiSink:
    """Fan-out combinator.

    Emits to each downstream sink in order. A failure in one sink is
    logged and swallowed — the audit path must not surface errors back
    into the protocol hot path.
    """

    def __init__(self, sinks: Iterable[AuditSink]) -> None:
        self._sinks: list[AuditSink] = list(sinks)

    async def emit(self, event: AuditEvent) -> None:
        for sink in self._sinks:
            try:
                await sink.emit(event)
            except Exception:
                log.exception("audit sink %r raised; continuing", sink)
