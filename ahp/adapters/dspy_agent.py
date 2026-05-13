"""DSPy adapter — wraps a :class:`dspy.Module` as an AHP agent.

DSPy modules are synchronous and signature-driven. The adapter:

1. Builds the module's kwargs from the inbound message via
   ``input_mapper`` (defaults to ``{input_field: message.body}``).
2. Runs the module in a worker thread (DSPy is sync).
3. Extracts the response from the returned ``dspy.Prediction`` via
   ``output_mapper`` (defaults to reading the ``output_field``
   attribute).
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

import dspy

from ahp.adapters.base import AHPAgent
from ahp.core.address import AgentAddress
from ahp.core.message import Message
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


InputMapper = Callable[[Message], dict]
"""Build dspy.Module kwargs from an inbound Message."""

OutputMapper = Callable[[Any, Message], Optional[Message]]
"""Build a reply Message from a dspy.Prediction + the request."""


def default_input_mapper(input_field: str) -> InputMapper:
    def mapper(message: Message) -> dict:
        return {input_field: message.body}

    return mapper


def default_output_mapper(
    address: AgentAddress,
    output_field: str,
) -> OutputMapper:
    def mapper(prediction: Any, request: Message) -> Message | None:
        if not request.expects_response:
            return None
        body = getattr(prediction, output_field, None)
        return Message(
            source=address,
            target=request.source,
            verb="SEND",
            code=request.code,
            thread=request.thread,
            body=body,
        )

    return mapper


class DSPyAgent(AHPAgent):
    """An AHP agent backed by a :class:`dspy.Module`."""

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        module: dspy.Module,
        *,
        input_field: str = "text",
        output_field: str = "answer",
        input_mapper: InputMapper | None = None,
        output_mapper: OutputMapper | None = None,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(address, engine, metadata=metadata, **kwargs)
        self.module = module
        self.input_field = input_field
        self.output_field = output_field
        self._input_mapper = input_mapper or default_input_mapper(input_field)
        self._output_mapper = output_mapper or default_output_mapper(
            address, output_field,
        )

    async def handle_message(self, message: Message) -> Message | None:
        module_kwargs = self._input_mapper(message)
        prediction = await asyncio.to_thread(self.module, **module_kwargs)
        return self._output_mapper(prediction, message)
