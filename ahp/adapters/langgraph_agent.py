"""LangGraph adapter — wraps a compiled :class:`StateGraph` as an AHP agent.

The agent's state schema is up to you; the adapter just calls
``graph.ainvoke(state, config=...)`` with state derived from the inbound
message. Customize via ``input_mapper`` and ``output_mapper``.

:class:`DeepAgentDAG` is the recursive variant: the engine and the
agent's address are passed through LangGraph's ``configurable`` dict
so graph nodes can call back into the protocol (e.g. fetch from another
AHP agent during a multi-step plan).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from langgraph.graph.state import CompiledStateGraph

from ahp.adapters.base import AHPAgent
from ahp.core.address import AgentAddress
from ahp.core.message import Message
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


InputMapper = Callable[[Message], dict]
"""Translate an inbound Message into a graph input state."""

OutputMapper = Callable[[dict, Message], Optional[Message]]
"""Translate a graph result + the request into a reply Message (or None to drop)."""


def default_input_mapper(message: Message) -> dict:
    """Default: pack message fields into a flat state dict."""
    return {
        "input": message.body,
        "code": message.code,
        "source": str(message.source),
        "thread": message.thread,
    }


def default_output_mapper_for(address: AgentAddress) -> OutputMapper:
    """Default output mapper that grabs ``output``/``result``/``answer`` from state."""

    def mapper(state: dict, request: Message) -> Message | None:
        if not request.expects_response:
            return None
        body = state.get("output", state.get("result", state.get("answer")))
        return Message(
            source=address,
            target=request.source,
            verb="SEND",
            code=request.code,
            thread=request.thread,
            body=body,
        )

    return mapper


class LangGraphAgent(AHPAgent):
    """An AHP agent that processes incoming messages by invoking a LangGraph."""

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        graph: CompiledStateGraph,
        *,
        input_mapper: InputMapper | None = None,
        output_mapper: OutputMapper | None = None,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(address, engine, metadata=metadata, **kwargs)
        self.graph = graph
        self._input_mapper = input_mapper or default_input_mapper
        self._output_mapper = output_mapper or default_output_mapper_for(address)

    async def handle_message(self, message: Message) -> Message | None:
        state = self._input_mapper(message)
        result = await self.graph.ainvoke(state, config=self._invoke_config(message))
        return self._output_mapper(result, message)

    def _invoke_config(self, message: Message) -> dict:
        """Hook for subclasses to add config keys passed to graph nodes."""
        return {}


class DeepAgentDAG(LangGraphAgent):
    """LangGraph agent whose nodes can recurse back into the AHP engine.

    The engine, this agent's address, and the inbound message are
    threaded through LangGraph's ``configurable`` dict. Nodes that need
    to make AHP calls accept ``config`` and read::

        engine: ProtocolEngine = config["configurable"]["ahp_engine"]
        sender: AgentAddress   = config["configurable"]["ahp_address"]
        inbound: Message       = config["configurable"]["ahp_message"]

    The convention keeps the LangGraph state schema free of
    transport-layer plumbing.
    """

    def _invoke_config(self, message: Message) -> dict:
        return {
            "configurable": {
                "ahp_engine": self.engine,
                "ahp_address": self.address,
                "ahp_message": message,
            }
        }
