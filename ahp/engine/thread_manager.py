"""Thread lifecycle, participation, and history.

A *thread* is the unit of multi-message conversation. Its message log
lives in the same Redis stream that :class:`RedisBus` writes to; the
manager adds metadata (topic, initiator, status) and a participant set,
and provides tier-filtered history reads for observers.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from ahp.core.address import AgentAddress
from ahp.core.compatibility import CompatibilityMatrix
from ahp.core.message import Message
from ahp.transport.keys import Keys
from ahp.transport.redis_bus import RedisBus


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, *, max_len: int = 40) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    return slug[:max_len] or "thread"


@dataclass
class Thread:
    """Lightweight value object for thread metadata."""

    thread_id: str
    topic: str
    initiator: AgentAddress
    created_at: float
    closed_at: float | None = None

    @property
    def is_closed(self) -> bool:
        return self.closed_at is not None


class ThreadManager:
    """Manage thread metadata, participation, and history reads.

    Construct with a Redis client and a :class:`RedisBus` — the bus owns
    the underlying message stream; this manager layers metadata + reads
    on top.
    """

    def __init__(self, redis_client: Any, bus: RedisBus) -> None:
        self._redis = redis_client
        self._bus = bus
        self._matrix = CompatibilityMatrix()

    # ── lifecycle ───────────────────────────────────────────────────────

    async def create(
        self,
        topic: str,
        initiator: AgentAddress,
        *,
        thread_id: str | None = None,
    ) -> Thread:
        """Create a new thread. ``thread_id`` is auto-generated when omitted."""
        if not topic:
            raise ValueError("thread topic must be non-empty")
        thread_id = thread_id or f"thread::{_slugify(topic)}::{uuid.uuid4().hex[:8]}"
        created_at = time.time()
        await self._redis.hset(
            Keys.thread_meta(thread_id),
            mapping={
                "topic": topic,
                "initiator": str(initiator),
                "created_at": str(created_at),
            },
        )
        await self._redis.sadd(Keys.thread_participants(thread_id), str(initiator))
        return Thread(
            thread_id=thread_id,
            topic=topic,
            initiator=initiator,
            created_at=created_at,
        )

    async def get(self, thread_id: str) -> Thread | None:
        data = await self._redis.hgetall(Keys.thread_meta(thread_id))
        if not data:
            return None
        return Thread(
            thread_id=thread_id,
            topic=data["topic"],
            initiator=AgentAddress.parse(data["initiator"]),
            created_at=float(data["created_at"]),
            closed_at=float(data["closed_at"]) if data.get("closed_at") else None,
        )

    async def close(self, thread_id: str) -> bool:
        """Mark a thread closed. Returns False if it doesn't exist."""
        if not await self._redis.exists(Keys.thread_meta(thread_id)):
            return False
        await self._redis.hset(
            Keys.thread_meta(thread_id), "closed_at", str(time.time())
        )
        return True

    async def is_closed(self, thread_id: str) -> bool:
        thread = await self.get(thread_id)
        return bool(thread and thread.is_closed)

    # ── participation ───────────────────────────────────────────────────

    async def join(self, thread_id: str, agent: AgentAddress) -> None:
        await self._redis.sadd(Keys.thread_participants(thread_id), str(agent))

    async def leave(self, thread_id: str, agent: AgentAddress) -> None:
        await self._redis.srem(Keys.thread_participants(thread_id), str(agent))

    async def participants(self, thread_id: str) -> list[AgentAddress]:
        members = await self._redis.smembers(Keys.thread_participants(thread_id))
        return [AgentAddress.parse(m) for m in members]

    async def is_participant(self, thread_id: str, agent: AgentAddress) -> bool:
        return bool(
            await self._redis.sismember(
                Keys.thread_participants(thread_id), str(agent)
            )
        )

    # ── messages ────────────────────────────────────────────────────────

    async def append(self, thread_id: str, message: Message) -> str:
        """Append a message to the thread, auto-joining its source as participant.

        Note this writes to the *thread* given in the call, not necessarily
        the one in ``message.thread`` — the engine uses this when routing
        messages whose envelope thread is the authoritative one. Mismatch
        is allowed but logged via the participant set.
        """
        await self.join(thread_id, message.source)
        return await self._bus.append_thread(message)

    async def get_history(
        self,
        thread_id: str,
        *,
        tier_filter: str | None = None,
        min_id: str = "-",
        max_id: str = "+",
        count: int | None = None,
    ) -> list[Message]:
        """Read thread history. With ``tier_filter`` set, return only messages
        whose code is at least partially renderable at one of the requested
        tiers (used to feed human observers an `s`-only view, for example).
        """
        history = await self._bus.get_thread(
            thread_id, min_id=min_id, max_id=max_id, count=count,
        )
        if tier_filter is None:
            return history
        filter_set = set(tier_filter)
        if not filter_set:
            return history
        return [
            m for m in history
            if self._matrix.required_tiers(m.code) & filter_set
        ]

    async def length(self, thread_id: str) -> int:
        return await self._bus.thread_length(thread_id)
