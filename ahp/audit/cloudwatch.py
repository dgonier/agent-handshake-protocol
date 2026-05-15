"""CloudWatch Logs sink for AuditEvents.

Lives in its own module so the rest of :mod:`ahp.audit` stays free of
the ``boto3`` import. Requires the ``[aws]`` extra.

Design notes
------------

* CloudWatch's ``PutLogEvents`` takes a batch with millisecond
  timestamps and string messages. Events must be in ascending
  timestamp order within a batch and must not span > 24 hours.
* Sequence tokens are no longer required (deprecated 2023). This
  sink omits them.
* The log group / log stream are created on first emit if they do
  not already exist. Subsequent ``ResourceAlreadyExistsException``
  is treated as success.
* Emits are buffered. By default, a batch is flushed when it reaches
  ``batch_size`` events OR when ``flush_interval`` seconds have
  elapsed since the oldest pending event. Call :meth:`flush`
  explicitly on shutdown.
* The sink degrades gracefully: if a flush fails the events are
  re-buffered (capped at ``max_buffer``) and a warning is logged.
  The protocol hot path never sees the failure.

This module deliberately uses :mod:`asyncio.to_thread` for boto3
calls — the ``logs`` client is sync, and we don't want to block the
event loop on a network round-trip.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

from ahp.audit.event import AuditEvent


if TYPE_CHECKING:  # pragma: no cover
    pass


log = logging.getLogger(__name__)


DEFAULT_LOG_GROUP: str = "/ahp/audit"
DEFAULT_BATCH_SIZE: int = 100
DEFAULT_FLUSH_INTERVAL: float = 5.0
DEFAULT_MAX_BUFFER: int = 10_000


class CloudWatchLogsSink:
    """Batched, async-friendly CloudWatch Logs sink.

    Parameters
    ----------
    log_group:
        Name of the CloudWatch log group. Created on first use if
        absent.
    log_stream:
        Name of the log stream within the group. Defaults to a
        wall-clock-stamped name so concurrent processes don't collide.
    region:
        AWS region. Falls back to the default boto3 chain (env,
        config, instance role) when ``None``.
    client:
        Optional pre-built ``boto3.client('logs')``. Primarily for
        tests; production callers leave this ``None`` and let the
        sink build its own.
    batch_size, flush_interval, max_buffer:
        Tuning knobs for the buffering behavior described in the
        module docstring.
    """

    def __init__(
        self,
        *,
        log_group: str = DEFAULT_LOG_GROUP,
        log_stream: str | None = None,
        region: str | None = None,
        client: Any = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_buffer: int = DEFAULT_MAX_BUFFER,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if flush_interval <= 0:
            raise ValueError(f"flush_interval must be positive, got {flush_interval}")
        if max_buffer < batch_size:
            raise ValueError(
                f"max_buffer ({max_buffer}) must be >= batch_size ({batch_size})"
            )
        self._log_group = log_group
        self._log_stream = log_stream or _default_stream_name()
        self._region = region
        self._client = client
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._max_buffer = max_buffer
        self._buffer: list[dict[str, Any]] = []
        self._oldest_pending_at: float | None = None
        self._initialized = False
        self._lock = asyncio.Lock()

    @property
    def log_group(self) -> str:
        return self._log_group

    @property
    def log_stream(self) -> str:
        return self._log_stream

    # ── sink protocol ─────────────────────────────────────────────────

    async def emit(self, event: AuditEvent) -> None:
        async with self._lock:
            self._buffer.append({
                "timestamp": int(event.timestamp * 1000),
                "message": event.to_json(),
            })
            if self._oldest_pending_at is None:
                self._oldest_pending_at = event.timestamp
            should_flush = (
                len(self._buffer) >= self._batch_size
                or (
                    self._oldest_pending_at is not None
                    and time.time() - self._oldest_pending_at >= self._flush_interval
                )
            )
        if should_flush:
            await self.flush()

    async def flush(self) -> None:
        """Send any buffered events to CloudWatch. Safe to call any time."""
        async with self._lock:
            if not self._buffer:
                return
            pending = self._buffer
            self._buffer = []
            self._oldest_pending_at = None
        await self._send_batch(pending)

    async def aclose(self) -> None:
        """Flush remaining events. Call on shutdown."""
        await self.flush()

    # ── internals ─────────────────────────────────────────────────────

    async def _send_batch(self, pending: list[dict[str, Any]]) -> None:
        try:
            await asyncio.to_thread(self._sync_send_batch, pending)
        except Exception:
            log.exception(
                "CloudWatch flush failed; re-buffering %d events", len(pending)
            )
            async with self._lock:
                # Re-add to front, then trim from the head if we exceed
                # max_buffer. Keeps the freshest data when overflowing.
                merged = pending + self._buffer
                if len(merged) > self._max_buffer:
                    merged = merged[-self._max_buffer:]
                self._buffer = merged
                if merged:
                    self._oldest_pending_at = merged[0]["timestamp"] / 1000.0

    def _sync_send_batch(self, pending: list[dict[str, Any]]) -> None:
        client = self._ensure_client()
        if not self._initialized:
            self._ensure_group_and_stream(client)
            self._initialized = True
        # CloudWatch requires ascending timestamps within a batch.
        pending.sort(key=lambda e: e["timestamp"])
        client.put_log_events(
            logGroupName=self._log_group,
            logStreamName=self._log_stream,
            logEvents=pending,
        )

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover — exercised by env, not test
            raise RuntimeError(
                "CloudWatchLogsSink needs boto3; install with: pip install 'ahp[aws]'"
            ) from e
        self._client = boto3.client("logs", region_name=self._region)
        return self._client

    def _ensure_group_and_stream(self, client: Any) -> None:
        # Idempotent: ResourceAlreadyExistsException is the success path
        # for the second-and-subsequent caller in any region.
        already_exists = _AlreadyExistsMatcher(client)
        try:
            client.create_log_group(logGroupName=self._log_group)
        except Exception as e:
            if not already_exists(e):
                raise
        try:
            client.create_log_stream(
                logGroupName=self._log_group, logStreamName=self._log_stream
            )
        except Exception as e:
            if not already_exists(e):
                raise


def _default_stream_name() -> str:
    # Wall-clock + pid keeps concurrent processes apart without needing
    # external coordination. CloudWatch tolerates colons in names.
    import os
    return f"ahp-{int(time.time())}-{os.getpid()}"


class _AlreadyExistsMatcher:
    """Detects ResourceAlreadyExistsException across boto3 versions.

    boto3 sometimes raises a generic ``ClientError`` with a code in
    ``response['Error']['Code']`` and sometimes a typed exception on
    the client. Match either.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def __call__(self, exc: BaseException) -> bool:
        # Typed-exception path.
        exceptions = getattr(self._client, "exceptions", None)
        typed = getattr(exceptions, "ResourceAlreadyExistsException", None)
        if typed is not None and isinstance(exc, typed):
            return True
        # Generic ClientError path.
        response = getattr(exc, "response", None)
        if isinstance(response, dict):
            error = response.get("Error", {})
            return error.get("Code") == "ResourceAlreadyExistsException"
        return False
