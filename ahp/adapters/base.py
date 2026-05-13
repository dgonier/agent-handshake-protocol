"""AHPAgent — abstract base class for protocol participants.

Lifecycle::

    agent = MyAgent(address, engine, ...)
    await agent.register()       # add to registry
    await agent.start()          # subscribe inbox + start heartbeat
    ... do work ...
    await agent.stop()           # cancel tasks
    await agent.deregister()     # remove from registry

Subclasses override :meth:`handle_message`. The base handles:

* inbox subscription via the bus consumer
* optional periodic heartbeat to keep liveness fresh
* auto-reply: if ``handle_message`` returns a Message and the request
  has ``reply_to`` set, the base sends it back as a reply
* exception isolation: handler errors are wrapped into an
  ``error.internal`` Message returned to the requester (if any)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from typing import Any

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


log = logging.getLogger(__name__)


class AHPAgent(ABC):
    """Base class for any AHP agent."""

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        *,
        metadata: AgentMeta | None = None,
        heartbeat_interval: float = 10.0,
        default_thread: str | None = None,
    ) -> None:
        self.address = address
        self.engine = engine
        self.metadata = metadata or AgentMeta()
        self.heartbeat_interval = heartbeat_interval
        self.default_thread = default_thread or f"thread::{address}::default"
        self._consumer_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

    # ── must override ───────────────────────────────────────────────────

    @abstractmethod
    async def handle_message(self, message: Message) -> Message | None:
        """Process an inbound message. Return a reply, or None to drop.

        If the inbound message has ``reply_to`` set and this returns a
        Message, the base class will publish it on the reply channel.
        """

    # ── registry lifecycle ──────────────────────────────────────────────

    async def register(self) -> None:
        await self.engine.registry.register(self.address, self.metadata)

    async def deregister(self) -> None:
        await self.engine.registry.deregister(self.address)

    async def heartbeat(self) -> bool:
        return await self.engine.registry.heartbeat(self.address)

    # ── inbox lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        """Begin consuming inbox messages and (optionally) heartbeating."""
        if self._consumer_task is not None:
            return
        self._consumer_task = self.engine.bus.consume(self.address, self._dispatch)
        if self.heartbeat_interval > 0:
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def stop(self) -> None:
        for task in (self._consumer_task, self._heartbeat_task):
            if task is None:
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._consumer_task = None
        self._heartbeat_task = None

    # ── outbound convenience ────────────────────────────────────────────

    async def send(
        self,
        target: AgentAddress | AddressPattern,
        code: str,
        body: Any,
        *,
        verb: str = "SEND",
        thread: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Construct an envelope and route it through the engine."""
        message = Message(
            source=self.address,
            target=target,
            verb=verb,
            code=code,
            thread=thread or self.default_thread,
            body=body,
        )
        return await self.engine.handle(message, **kwargs)

    async def broadcast(
        self,
        pattern: AddressPattern,
        code: str,
        body: Any,
        *,
        verb: str = "CAST-GET",
        thread: str | None = None,
        **kwargs: Any,
    ) -> Any:
        return await self.send(
            pattern, code, body, verb=verb, thread=thread, **kwargs,
        )

    # ── internals ───────────────────────────────────────────────────────

    async def _dispatch(self, message: Message) -> None:
        """Bus consumer callback: route an inbound message into handle_message."""
        try:
            response = await self.handle_message(message)
        except Exception as exc:
            log.exception("agent %s raised in handle_message", self.address)
            response = self._build_error_response(message, exc)
        if response is not None and message.reply_to:
            with contextlib.suppress(Exception):
                await self.engine.bus.send_reply(message, response)

    def _build_error_response(self, request: Message, exc: Exception) -> Message:
        return Message(
            source=self.address,
            target=request.source,
            verb="SEND",
            code=Code.ERROR_INTERNAL,
            thread=request.thread,
            body=f"{type(exc).__name__}: {exc}",
        )

    async def _heartbeat_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                with contextlib.suppress(Exception):
                    await self.engine.registry.heartbeat(self.address)
        except asyncio.CancelledError:
            raise
