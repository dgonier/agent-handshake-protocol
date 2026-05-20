"""ReAct agent adapter.

Wraps ``langchain.agents.create_agent`` (the LangChain v1 ReAct loop
that superseded ``langgraph.prebuilt.create_react_agent``) so an
:class:`AgentProfile` + a chat model produces an :class:`AHPAgent`
ready to plug into the protocol.

Profile → ReAct mapping:

* ``profile.all_tools`` are converted to LangChain ``StructuredTool``
  instances (the protocol's :class:`Tool` is intentionally minimal;
  this adapter performs the translation).
* ``profile.prompt`` becomes the ``system_prompt`` passed to
  ``create_agent``.
* The agent's inbox messages enter the graph as a single
  ``HumanMessage``; the last AI message in the resulting state becomes
  the reply body.

History note: this adapter previously used
``langgraph.prebuilt.create_react_agent`` (deprecated in LangGraph v1
in favor of LangChain's ``create_agent``). The output state shape is
the same — the keyword that names the system prompt changed
(``prompt=`` → ``system_prompt=``) and the import moved from
``langgraph.prebuilt`` to ``langchain.agents``.
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool

from ahp.adapters.capability import AgentProfile, Tool
from ahp.adapters.langgraph_agent import LangGraphAgent
from ahp.core.address import AgentAddress
from ahp.core.message import Message
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


def _to_langchain_tool(tool: Tool) -> StructuredTool:
    """Convert an AHP :class:`Tool` to a LangChain ``StructuredTool``.

    Detects coroutine handlers and wires them as the tool's async path so
    LangChain doesn't try to ``asyncio.run`` them inside an already-running
    event loop.
    """
    import asyncio

    if asyncio.iscoroutinefunction(tool.handler):
        return StructuredTool.from_function(
            coroutine=tool.handler,
            name=tool.name,
            description=tool.description,
        )
    return StructuredTool.from_function(
        func=tool.handler,
        name=tool.name,
        description=tool.description,
    )


def _react_input_mapper(message: Message) -> dict:
    """Wrap the inbound body as a single HumanMessage on the graph state."""
    return {"messages": [HumanMessage(content=str(message.body))]}


def _react_output_mapper_for(address: AgentAddress):
    def mapper(state: dict, request: Message) -> Message | None:
        if not request.expects_response:
            return None
        messages = state.get("messages") or []
        # Find the last AIMessage; fall back to last message of any kind.
        body: str = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage):
                body = _coerce_content(m.content)
                break
        if not body and messages:
            body = _coerce_content(messages[-1].content)
        return Message(
            source=address,
            target=request.source,
            verb="SEND",
            code=request.code,
            thread=request.thread,
            body=body,
        )

    return mapper


def _coerce_content(content: Any) -> str:
    """LangChain content may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class ReactAgent(LangGraphAgent):
    """An AHP agent backed by a LangGraph ReAct loop driven by a chat model."""

    @classmethod
    def from_profile(
        cls,
        address: AgentAddress,
        engine: ProtocolEngine,
        profile: AgentProfile,
        model: Any,
        *,
        metadata: AgentMeta | None = None,
        extra_tools: list[Tool] | None = None,
        **kwargs: Any,
    ) -> "ReactAgent":
        """Build a ReactAgent from an :class:`AgentProfile` + chat ``model``.

        ``extra_tools`` are appended to whatever the profile already
        supplies — useful when a builder wants to inject AHP-aware tools
        (e.g. a tool that calls back into the engine) that aren't in the
        capability registry.
        """
        tools_in: list[Tool] = list(profile.all_tools)
        if extra_tools:
            tools_in.extend(extra_tools)
        lc_tools = [_to_langchain_tool(t) for t in tools_in]
        graph = create_agent(
            model=model,
            tools=lc_tools,
            system_prompt=profile.prompt or None,
        )
        return cls(
            address=address,
            engine=engine,
            graph=graph,
            input_mapper=_react_input_mapper,
            output_mapper=_react_output_mapper_for(address),
            metadata=metadata,
            **kwargs,
        )
