"""Canonical Redis key/channel names.

Centralized so the bus, cache, and registry agree on layout. All keys are
prefixed with ``ahp:`` to keep AHP state segregated from other Redis users.
"""

from __future__ import annotations

from ahp.core.address import AgentAddress


PREFIX = "ahp"


class Keys:
    """Pure key/channel name builders. No I/O."""

    # ── pub/sub channels ─────────────────────────────────────────────
    @staticmethod
    def agent_channel(address: AgentAddress | str) -> str:
        """Pub/sub channel for direct delivery to a specific agent."""
        return f"{PREFIX}:agent:{address}"

    @staticmethod
    def reply_channel(message_id: str) -> str:
        """Ephemeral channel used to collect responses for SEND-GET / CAST-GET."""
        return f"{PREFIX}:reply:{message_id}"

    # ── streams ──────────────────────────────────────────────────────
    @staticmethod
    def thread_stream(thread_id: str) -> str:
        """Append-only stream of all messages on a given thread."""
        return f"{PREFIX}:thread:{thread_id}"

    @staticmethod
    def thread_meta(thread_id: str) -> str:
        """Hash storing a thread's topic/initiator/timestamps/status."""
        return f"{PREFIX}:thread-meta:{thread_id}"

    @staticmethod
    def thread_participants(thread_id: str) -> str:
        """Set of canonical agent URIs participating in a thread."""
        return f"{PREFIX}:thread-parts:{thread_id}"

    # ── registry ─────────────────────────────────────────────────────
    @staticmethod
    def registry_hash() -> str:
        """Hash mapping canonical address URI → JSON-encoded AgentMeta."""
        return f"{PREFIX}:registry"

    @staticmethod
    def alive_key(address: AgentAddress | str) -> str:
        """TTL key used as a liveness marker for an agent."""
        return f"{PREFIX}:alive:{address}"

    # ── cache ────────────────────────────────────────────────────────
    @staticmethod
    def cache_key(digest: str) -> str:
        """Cache entry keyed by SHA-256 digest of (address, code)."""
        return f"{PREFIX}:cache:{digest}"

    @staticmethod
    def cache_scan_pattern() -> str:
        """SCAN pattern matching all cache entries."""
        return f"{PREFIX}:cache:*"
