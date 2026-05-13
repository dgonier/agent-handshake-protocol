"""HumanAgent — protocol participant with a natural-language frontend.

The human's accept tier is typically ``s`` (string). Incoming messages
are filtered by observation level, rendered to text, and pushed to the
``on_message`` callback. When a message expects a reply, the agent
optionally calls ``input_provider`` to obtain the human's response.

Observation levels:

================  =========================================================
``L0``            silent — display nothing (useful for paused/lurking)
``L1``            summary — code + sender + first 200 chars of body
``L2``            full body included
``L3``            everything, including internal errors and debug codes
================  =========================================================
"""

from __future__ import annotations

from typing import Awaitable, Callable, Literal

from ahp.adapters.base import AHPAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


ObservationLevel = Literal["L0", "L1", "L2", "L3"]

DisplayCallback = Callable[[str], Awaitable[None]]
"""Async callable that takes rendered text and shows it to the human."""

InputProvider = Callable[[Message], Awaitable[str]]
"""Async callable that returns the human's text response to an inbound message."""


class HumanAgent(AHPAgent):
    """An AHP participant backed by a human via two async callbacks.

    ``on_message`` is invoked with rendered text for any message the
    current observation level allows. If the message expects a response
    and ``input_provider`` is set, the human's text is wrapped into a
    reply Message.
    """

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        *,
        on_message: DisplayCallback | None = None,
        input_provider: InputProvider | None = None,
        observation_level: ObservationLevel = "L1",
        metadata: AgentMeta | None = None,
        **kwargs,
    ) -> None:
        super().__init__(address, engine, metadata=metadata, **kwargs)
        self.on_message = on_message
        self.input_provider = input_provider
        self.observation_level: ObservationLevel = observation_level

    # ── core handler ────────────────────────────────────────────────────

    async def handle_message(self, message: Message) -> Message | None:
        if not self._should_display(message):
            return None
        if self.on_message is not None:
            await self.on_message(self._render(message))
        if message.expects_response and self.input_provider is not None:
            text = await self.input_provider(message)
            return Message(
                source=self.address,
                target=message.source,
                verb="SEND",
                code=Code.HUMAN_QUERY,
                thread=message.thread,
                body=text,
            )
        return None

    # ── filtering ───────────────────────────────────────────────────────

    def _should_display(self, msg: Message) -> bool:
        level = self.observation_level
        if level == "L0":
            return False
        if level == "L3":
            return True
        # L1 hides internal/error chatter; L2 shows everything except L3-only.
        if level == "L1" and Code.is_error(msg.code):
            return False
        return True

    def _render(self, msg: Message) -> str:
        if self.observation_level == "L1":
            preview = self._preview(msg.body)
            return f"[{msg.source.instance}] {msg.code}: {preview}"
        full = self._format_body(msg.body)
        suffix = ""
        if self.observation_level == "L3":
            suffix = f"\n  message_id={msg.message_id} thread={msg.thread} verb={msg.verb}"
        return f"[{msg.source}] {msg.code}\n{full}{suffix}"

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _preview(body: object, *, limit: int = 200) -> str:
        text = HumanAgent._format_body(body)
        if len(text) <= limit:
            return text
        return text[: limit - 1] + "…"

    @staticmethod
    def _format_body(body: object) -> str:
        if body is None:
            return "(empty)"
        if isinstance(body, str):
            return body
        return repr(body)
