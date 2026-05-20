"""ahp.audit — typed audit events and pluggable sinks.

The audit layer is *strictly observational*. Nothing in the protocol
core depends on it; it is an opt-in side channel that records the
outcome of registry mutations, message dispatch, and cache control.

Concrete sinks ship in :mod:`ahp.audit.sinks`. The CloudWatch sink
(:class:`CloudWatchLogsSink`) needs ``boto3`` (the ``[aws]`` extra);
the in-memory and logging sinks have zero extra dependencies.

The sink protocol is async to keep network-bound sinks honest, but the
typical emit site does not ``await`` long — sinks should buffer/batch
internally rather than blocking the caller.
"""

from typing import TYPE_CHECKING, Any

from ahp.audit.event import AuditEvent
from ahp.audit.sinks import (
    DEFAULT_REDIS_AUDIT_STREAM,
    AuditSink,
    InMemoryAuditSink,
    LoggingAuditSink,
    MultiSink,
    NullAuditSink,
    RedisStreamAuditSink,
)

if TYPE_CHECKING:  # pragma: no cover
    from ahp.audit.cloudwatch import CloudWatchLogsSink  # noqa: F401


def __getattr__(name: str) -> Any:
    # Lazy import so callers without boto3 can still use the rest of
    # the audit package.
    if name == "CloudWatchLogsSink":
        from ahp.audit.cloudwatch import CloudWatchLogsSink
        return CloudWatchLogsSink
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AuditEvent",
    "AuditSink",
    "DEFAULT_REDIS_AUDIT_STREAM",
    "InMemoryAuditSink",
    "LoggingAuditSink",
    "MultiSink",
    "NullAuditSink",
    "RedisStreamAuditSink",
    "CloudWatchLogsSink",
]
