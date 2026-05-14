"""Redis-backed message bus.

The bus is the wire layer: it serializes :class:`Message` envelopes and
moves them between agents using Redis pub/sub for delivery and Redis
streams for thread history. Pattern resolution is **not** the bus's
job — callers pass already-resolved target lists. The engine (Phase 3)
handles registry lookups before invoking the bus.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable

from ahp.core.address import AgentAddress
from ahp.core.message import Message
from ahp.transport.keys import Keys


log = logging.getLogger(__name__)

MessageHandler = Callable[[Message], Awaitable[None]]


def _encode(message: Message) -> str:
    """JSON-encode a Message for wire transport."""
    return json.dumps(message.to_dict(), default=_json_default)


def _json_default(obj: Any) -> Any:
    # Bytes bodies are not JSON-serializable; callers should encode upstream.
    # We surface a clear error here rather than silently corrupt data.
    if isinstance(obj, (bytes, bytearray)):
        raise TypeError(
            "raw bytes bodies are not JSON-serializable; base64-encode "
            "or use a string/dict body for transport"
        )
    raise TypeError(f"object of type {type(obj).__name__} is not JSON-serializable")


def _decode_payload(raw: Any) -> Message:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return Message.from_dict(json.loads(raw))


MessagePredicate = Callable[[Message], bool]
"""Filter callable applied to each inbound Message inside a Subscription."""


class Subscription:
    """An active pub/sub subscription. Use as an async context manager.

    A ``predicate`` may be supplied to drop messages that don't match —
    useful for tap subscriptions where the channel carries every message
    on the bus and the subscriber only wants a slice.
    """

    def __init__(
        self,
        pubsub: Any,
        channels: tuple[str, ...],
        *,
        predicate: MessagePredicate | None = None,
    ) -> None:
        self._pubsub = pubsub
        self._channels = channels
        self._predicate = predicate
        self._closed = False

    async def __aenter__(self) -> "Subscription":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def get_one(self, timeout: float | None = None) -> Message | None:
        """Wait up to ``timeout`` seconds for a single message that passes the predicate."""
        if self._closed:
            return None
        loop_deadline = (
            asyncio.get_event_loop().time() + timeout
            if timeout is not None
            else None
        )
        while True:
            remaining = (
                None
                if loop_deadline is None
                else max(0.0, loop_deadline - asyncio.get_event_loop().time())
            )
            raw = await self._pubsub.get_message(
                timeout=remaining, ignore_subscribe_messages=True
            )
            if raw is None or raw.get("type") != "message":
                return None
            msg = _decode_payload(raw["data"])
            if self._predicate is None or self._predicate(msg):
                return msg
            if loop_deadline is not None and asyncio.get_event_loop().time() >= loop_deadline:
                return None

    async def messages(
        self,
        *,
        idle_timeout: float = 0.5,
    ) -> AsyncIterator[Message]:
        """Async iterator over inbound messages. Stops when ``close()`` is called."""
        while not self._closed:
            raw = await self._pubsub.get_message(
                timeout=idle_timeout, ignore_subscribe_messages=True
            )
            if raw is None:
                continue
            if raw.get("type") != "message":
                continue
            msg = _decode_payload(raw["data"])
            if self._predicate is None or self._predicate(msg):
                yield msg

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._pubsub.unsubscribe(*self._channels)
        with contextlib.suppress(Exception):
            await self._pubsub.aclose()


class RedisBus:
    """Async message bus over Redis pub/sub + streams.

    Construct from a connected ``redis.asyncio.Redis`` (or fakeredis) client,
    or from a URL via :meth:`from_url`. All client interactions use
    ``decode_responses=True`` semantics — string payloads in and out.
    """

    def __init__(self, redis_client: Any) -> None:
        self._redis = redis_client
        self._consumers: list[asyncio.Task[None]] = []

    @classmethod
    async def from_url(cls, url: str) -> "RedisBus":
        import redis.asyncio as aioredis  # local import — only when used
        client = aioredis.from_url(url, decode_responses=True)
        return cls(client)

    @property
    def redis(self) -> Any:
        return self._redis

    async def close(self) -> None:
        """Cancel consumer tasks and close the Redis connection."""
        for task in self._consumers:
            task.cancel()
        for task in self._consumers:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._consumers.clear()
        with contextlib.suppress(Exception):
            await self._redis.aclose()

    # ── thread history ──────────────────────────────────────────────────

    async def append_thread(self, message: Message) -> str:
        """Append a message to its thread's stream. Returns the stream entry id."""
        return await self._redis.xadd(
            Keys.thread_stream(message.thread),
            {"data": _encode(message)},
        )

    async def get_thread(
        self,
        thread_id: str,
        *,
        min_id: str = "-",
        max_id: str = "+",
        count: int | None = None,
    ) -> list[Message]:
        """Read thread history in append order. ``min_id``/``max_id`` are stream IDs."""
        kwargs: dict[str, Any] = {"min": min_id, "max": max_id}
        if count is not None:
            kwargs["count"] = count
        entries = await self._redis.xrange(Keys.thread_stream(thread_id), **kwargs)
        return [_decode_payload(fields["data"]) for _, fields in entries]

    async def thread_length(self, thread_id: str) -> int:
        return await self._redis.xlen(Keys.thread_stream(thread_id))

    # ── point-to-point ──────────────────────────────────────────────────

    async def send(self, message: Message) -> int:
        """Deliver a point-to-point message. Returns # of subscribers reached."""
        if not isinstance(message.target, AgentAddress):
            raise ValueError("send() requires an AgentAddress target")
        await self.append_thread(message)
        return await self._publish(Keys.agent_channel(message.target), message)

    async def send_get(self, message: Message, timeout: float) -> Message | None:
        """Send and wait up to ``timeout`` seconds for exactly one reply."""
        if not isinstance(message.target, AgentAddress):
            raise ValueError("send_get() requires an AgentAddress target")
        replies = await self._collect_replies(
            message, [message.target], timeout=timeout, max_responses=1,
        )
        return replies[0] if replies else None

    # ── broadcast ───────────────────────────────────────────────────────

    async def cast(self, message: Message, targets: Iterable[AgentAddress]) -> int:
        """Fan out a broadcast to pre-resolved targets. Returns total deliveries."""
        await self.append_thread(message)
        delivered = 0
        for target in targets:
            delivered += await self._publish(Keys.agent_channel(target), message)
        return delivered

    async def cast_get(
        self,
        message: Message,
        targets: list[AgentAddress],
        timeout: float,
        max_responses: int | None = None,
    ) -> list[Message]:
        """Fan out and collect responses, bounded by ``timeout`` and ``max_responses``.

        If ``max_responses`` is None, collects up to ``len(targets)`` responses.
        """
        limit = max_responses if max_responses is not None else len(targets)
        return await self._collect_replies(
            message, targets, timeout=timeout, max_responses=limit,
        )

    # ── replies ─────────────────────────────────────────────────────────

    async def send_reply(self, original: Message, response: Message) -> int:
        """Publish a response on the reply channel established by the requester."""
        if not original.reply_to:
            raise ValueError(
                f"original message {original.message_id} has no reply_to channel"
            )
        await self.append_thread(response)
        return await self._publish(original.reply_to, response)

    # ── subscriptions ───────────────────────────────────────────────────

    async def listen(self, address: AgentAddress) -> Subscription:
        """Subscribe to an agent's inbox channel. Caller owns the Subscription."""
        channel = Keys.agent_channel(address)
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        # Drain the initial confirmation frame so the first real message is clean.
        await pubsub.get_message(timeout=0.05, ignore_subscribe_messages=True)
        return Subscription(pubsub, (channel,))

    def consume(
        self,
        address: AgentAddress,
        handler: MessageHandler,
    ) -> asyncio.Task[None]:
        """Spawn a background task that delivers inbox messages to ``handler``.

        The returned task is tracked by the bus and cancelled on :meth:`close`.
        """
        task = asyncio.create_task(self._consume_loop(address, handler))
        self._consumers.append(task)
        return task

    async def _consume_loop(
        self,
        address: AgentAddress,
        handler: MessageHandler,
    ) -> None:
        sub = await self.listen(address)
        try:
            async for msg in sub.messages():
                try:
                    await handler(msg)
                except Exception:  # pragma: no cover — defensive
                    log.exception("handler for %s raised", address)
        except asyncio.CancelledError:
            raise
        finally:
            await sub.close()

    # ── internals ───────────────────────────────────────────────────────

    async def _publish(self, channel: str, message: Message) -> int:
        payload = _encode(message)
        delivered = await self._redis.publish(channel, payload)
        # Mirror to the tap channel so CAST-SUB subscribers see every message
        # without coupling to per-agent channels. Tap is best-effort — failures
        # never break primary delivery.
        with contextlib.suppress(Exception):
            await self._redis.publish(Keys.tap_channel(), payload)
        return delivered

    async def tap_subscribe(
        self,
        *,
        predicate: MessagePredicate | None = None,
    ) -> Subscription:
        """Subscribe to the tap channel — every message ever published.

        Pass ``predicate`` to filter the stream client-side. The engine's
        CAST-SUB handler builds a predicate from the verb's pattern target
        and code glob.
        """
        channel = Keys.tap_channel()
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        await pubsub.get_message(timeout=0.05, ignore_subscribe_messages=True)
        return Subscription(pubsub, (channel,), predicate=predicate)

    async def _collect_replies(
        self,
        message: Message,
        targets: Iterable[AgentAddress],
        *,
        timeout: float,
        max_responses: int,
    ) -> list[Message]:
        reply_channel = Keys.reply_channel(message.message_id)
        # Mutate reply_to so responders know where to deliver.
        message.reply_to = reply_channel

        pubsub = self._redis.pubsub()
        await pubsub.subscribe(reply_channel)
        # Drain subscribe confirmation.
        await pubsub.get_message(timeout=0.05, ignore_subscribe_messages=True)

        try:
            await self.append_thread(message)
            for target in targets:
                await self._publish(Keys.agent_channel(target), message)

            responses: list[Message] = []
            loop = asyncio.get_event_loop()
            deadline = loop.time() + timeout
            while len(responses) < max_responses:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                raw = await pubsub.get_message(
                    timeout=remaining, ignore_subscribe_messages=True
                )
                if raw is None:
                    break
                if raw.get("type") != "message":
                    continue
                responses.append(_decode_payload(raw["data"]))
            return responses
        finally:
            with contextlib.suppress(Exception):
                await pubsub.unsubscribe(reply_channel)
            with contextlib.suppress(Exception):
                await pubsub.aclose()
