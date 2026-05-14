"""Tests for DeepAgent — wraps deepagents.create_deep_agent.

Uses a fake chat model so the tests don't hit any LLM.
"""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ahp.adapters.capability import AgentProfile, Skill, Tool
from ahp.adapters.deep_agent import DeepAgent, _skill_to_subagent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message


pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
)


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


class _ToolableFakeChat(FakeListChatModel):
    """FakeListChatModel + a no-op bind_tools so create_deep_agent accepts it."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


# ── skill → SubAgent mapping ───────────────────────────────────────────


def test_skill_to_subagent_maps_minimal():
    sk = Skill(name="valuation", description="DCF + comparables")
    out = _skill_to_subagent(sk)
    assert out["name"] == "valuation"
    assert out["description"] == "DCF + comparables"
    # Falls back to description when no prompt_fragment.
    assert out["system_prompt"] == "DCF + comparables"
    # No tools = key absent (TypedDict NotRequired).
    assert "tools" not in out


def test_skill_to_subagent_uses_prompt_fragment_when_set():
    sk = Skill(
        name="valuation",
        description="DCF + comparables",
        prompt_fragment="When asked to value, prefer DCF unless the asset is non-cash-flowing.",
    )
    out = _skill_to_subagent(sk)
    assert "DCF unless" in out["system_prompt"]


def test_skill_to_subagent_translates_tools():
    sk = Skill(
        name="ts",
        description="time-series fetch",
        tools=(Tool(name="fetch_ohlc", description="fetch OHLC",
                    handler=lambda symbol: [1, 2, 3]),),
    )
    out = _skill_to_subagent(sk)
    assert "tools" in out
    assert len(out["tools"]) == 1
    assert out["tools"][0].name == "fetch_ohlc"


# ── from_profile round-trip ────────────────────────────────────────────


async def test_deep_agent_round_trip_no_tools(stack):
    """A trivial deep agent with no tools / no subagents still answers."""
    addr = _addr("demo.collaborative.finance.equities.s.session.deepy")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    model = _ToolableFakeChat(responses=["here is the brief"])
    profile = AgentProfile(address=addr, prompt="Be terse.")
    agent = DeepAgent.from_profile(
        addr, stack.engine, profile, model=model, heartbeat_interval=0,
    )
    await agent.register(); await agent.start()
    await asyncio.sleep(0.05)
    try:
        reply = await stack.engine.handle(
            Message(
                source=sender, target=addr, verb="SEND-GET",
                code=Code.COLLAB_REASON, thread="t::deep", body="question",
            ),
            timeout=5.0,
        )
        assert reply is not None
        assert "here is the brief" in reply.body
    finally:
        await agent.stop()


async def test_deep_agent_builds_with_skills(stack):
    """Profile skills produce subagents the planner knows about."""
    addr = _addr("demo.collaborative.finance.equities.s.session.coord")

    model = _ToolableFakeChat(responses=["coordinated answer"])
    profile = AgentProfile(
        address=addr,
        prompt="Coordinate.",
        skills=(
            Skill(name="valuation", description="DCF",
                  prompt_fragment="Use DCF for valuation."),
            Skill(name="risk", description="risk analysis",
                  prompt_fragment="List downside risks."),
        ),
    )
    # Just verify the build completes; the fake model won't invoke any
    # subagents but the compilation exercises that codepath.
    agent = DeepAgent.from_profile(
        addr, stack.engine, profile, model=model, heartbeat_interval=0,
    )
    assert agent.graph is not None


# ── async tool wiring ──────────────────────────────────────────────────


def test_async_tool_handler_wired_as_coroutine():
    """The to-langchain translator should recognize coroutine handlers."""
    from ahp.adapters.react_agent import _to_langchain_tool

    async def async_handler(x: int) -> int:
        return x * 2

    def sync_handler(x: int) -> int:
        return x + 1

    async_tool = Tool(name="async", description="async", handler=async_handler)
    sync_tool = Tool(name="sync", description="sync", handler=sync_handler)

    lc_async = _to_langchain_tool(async_tool)
    lc_sync = _to_langchain_tool(sync_tool)

    # StructuredTool exposes `func` and `coroutine`. Async tool should have
    # coroutine set and func empty (or vice versa for sync).
    assert lc_async.coroutine is not None
    assert lc_async.func is None
    assert lc_sync.func is not None
    assert lc_sync.coroutine is None
