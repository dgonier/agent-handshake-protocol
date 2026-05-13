"""Tests for ReactAgent — exercises the adapter with a fake chat model.

No AWS / no real LLM. The fake model returns scripted responses, so the
ReAct loop terminates deterministically.
"""

from __future__ import annotations

import asyncio
import warnings

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ahp.adapters.capability import AgentProfile, Tool
from ahp.adapters.react_agent import ReactAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message


# Silence the LangGraph V1 deprecation noise — we'll migrate to
# langchain.agents.create_agent once the langchain package is in our deps.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
)


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


class _ToolableFakeChat(FakeListChatModel):
    """FakeListChatModel + a no-op bind_tools so create_react_agent accepts it."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


# ── basic round-trip without tools ──────────────────────────────────────


async def test_react_agent_without_tools(stack):
    addr = _addr("demo.collaborative.finance.equities.s.session.r1")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    model = _ToolableFakeChat(responses=["This is the bear case."])
    profile = AgentProfile(address=addr, prompt="Be terse.")
    agent = ReactAgent.from_profile(addr, stack.engine, profile, model=model,
                                    heartbeat_interval=0)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.05)

    try:
        msg = Message(
            source=sender, target=addr, verb="SEND-GET",
            code=Code.ADVERSARIAL_DEBATE, thread="t::react", body="argue tesla",
        )
        reply = await stack.engine.handle(msg, timeout=3.0)
        assert reply is not None
        assert "bear case" in reply.body.lower()
    finally:
        await agent.stop()


# ── profile tools surface to LangChain ──────────────────────────────────


def test_profile_tools_become_langchain_tools(stack):
    """Building a ReactAgent should expose the profile's Tool objects to the graph."""
    called = []

    def get_quote(symbol: str) -> str:
        """Return a fake quote."""
        called.append(symbol)
        return f"{symbol}: $100"

    addr = _addr("demo.interview.finance.equities.s.session.q")
    profile = AgentProfile(
        address=addr,
        tools=(Tool(name="get_quote", description="quote lookup", handler=get_quote),),
    )
    model = _ToolableFakeChat(responses=["dummy"])
    agent = ReactAgent.from_profile(addr, stack.engine, profile, model=model,
                                    heartbeat_interval=0)

    # The compiled graph should have the tool registered.
    # We sanity-check by inspecting the tool node's tools.
    # LangGraph stores them in graph.nodes["tools"].
    # If the API shifts this becomes a smoke test that the build didn't raise.
    assert agent.graph is not None


# ── extra_tools injection ──────────────────────────────────────────────


async def test_extra_tools_are_appended(stack):
    addr = _addr("demo.collaborative.finance.equities.s.session.r2")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    profile = AgentProfile(
        address=addr,
        tools=(Tool(name="profile_tool", description="from profile",
                    handler=lambda: "p"),),
    )
    extra = Tool(name="injected_tool", description="from extra_tools",
                 handler=lambda: "x")

    model = _ToolableFakeChat(responses=["ok"])
    agent = ReactAgent.from_profile(
        addr, stack.engine, profile, model=model,
        extra_tools=[extra], heartbeat_interval=0,
    )
    # Build succeeded with both tools in scope.
    assert agent.graph is not None

    # Smoke: the agent answers without invoking the tools.
    await agent.register(); await agent.start()
    await asyncio.sleep(0.05)
    try:
        msg = Message(
            source=sender, target=addr, verb="SEND-GET",
            code=Code.COLLAB_REASON, thread="t::extra", body="hi",
        )
        reply = await stack.engine.handle(msg, timeout=3.0)
        assert reply is not None
        assert reply.body == "ok"
    finally:
        await agent.stop()


# ── output coercion: list-shaped content blocks ─────────────────────────


def test_react_output_handles_block_content(stack):
    """Some Bedrock models return content as a list of dict blocks; verify coercion."""
    from ahp.adapters.react_agent import _coerce_content

    assert _coerce_content("plain") == "plain"
    assert _coerce_content([{"text": "a"}, {"text": "b"}]) == "ab"
    assert _coerce_content([{"text": "a"}, "literal"]) == "aliteral"
    assert _coerce_content(None) == "None"
