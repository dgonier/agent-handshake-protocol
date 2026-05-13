"""Tests for LangGraphAgent + DeepAgentDAG.

Uses a deliberately tiny LangGraph that doesn't call any LLM — just
deterministic state transformations.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

import pytest
from langgraph.graph import END, START, StateGraph

from ahp.adapters.langgraph_agent import DeepAgentDAG, LangGraphAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── tiny LangGraph: uppercase ──────────────────────────────────────────


class UpperState(TypedDict, total=False):
    input: str
    output: str


def _build_upper_graph():
    def upper(state):
        return {"output": state["input"].upper()}

    g = StateGraph(UpperState)
    g.add_node("up", upper)
    g.add_edge(START, "up")
    g.add_edge("up", END)
    return g.compile()


async def test_langgraph_agent_round_trip(stack):
    addr = _addr("demo.collaborative.finance.equities.s.session.lg")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    agent = LangGraphAgent(
        addr, stack.engine, _build_upper_graph(), heartbeat_interval=0,
    )
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.05)
    try:
        msg = Message(
            source=sender, target=addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::lg", body="hello",
        )
        reply = await stack.engine.handle(msg, timeout=2.0)
        assert reply is not None
        assert reply.body == "HELLO"
    finally:
        await agent.stop()


async def test_langgraph_agent_no_reply_when_not_expected(stack):
    addr = _addr("demo.collaborative.finance.equities.s.session.lg2")
    agent = LangGraphAgent(
        addr, stack.engine, _build_upper_graph(), heartbeat_interval=0,
    )
    # Direct handle_message call: verb=SEND, no reply_to.
    msg = Message(
        source=_addr("demo.collaborative.finance.equities.s.session.alice"),
        target=addr, verb="SEND",
        code=Code.INTERVIEW_TEXT, thread="thread::lg2", body="hi",
    )
    reply = await agent.handle_message(msg)
    assert reply is None


# ── custom mappers ─────────────────────────────────────────────────────


async def test_custom_input_and_output_mappers(stack):
    addr = _addr("demo.collaborative.finance.equities.s.session.lg3")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    class S(TypedDict, total=False):
        prompt: str
        answer: str

    def runner(state):
        return {"answer": f"the answer to '{state['prompt']}' is 42"}

    g = StateGraph(S)
    g.add_node("run", runner)
    g.add_edge(START, "run")
    g.add_edge("run", END)
    graph = g.compile()

    def into_state(msg):
        return {"prompt": msg.body}

    def from_state(state, request):
        if not request.expects_response:
            return None
        return Message(
            source=addr, target=request.source, verb="SEND",
            code=request.code, thread=request.thread, body=state["answer"],
        )

    agent = LangGraphAgent(
        addr, stack.engine, graph,
        input_mapper=into_state, output_mapper=from_state,
        heartbeat_interval=0,
    )
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.05)
    try:
        msg = Message(
            source=sender, target=addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::lg3", body="life",
        )
        reply = await stack.engine.handle(msg, timeout=2.0)
        assert reply is not None
        assert reply.body == "the answer to 'life' is 42"
    finally:
        await agent.stop()


# ── DeepAgentDAG: node recurses through engine ─────────────────────────


async def test_deep_agent_recurses_via_engine(stack):
    """The DAG node uses config['configurable']['ahp_engine'] to call out."""
    inner_addr = _addr("demo.interview.finance.equities.s.session.inner")
    deep_addr = _addr("demo.collaborative.finance.equities.s.session.deep")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    # Inner agent: responds with "inner-said:" prefix.
    from ahp.adapters.base import AHPAgent

    class _Inner(AHPAgent):
        async def handle_message(self, message: Message):
            if not message.expects_response:
                return None
            return Message(
                source=self.address, target=message.source, verb="SEND",
                code=message.code, thread=message.thread,
                body=f"inner-said:{message.body}",
            )

    inner = _Inner(inner_addr, stack.engine, heartbeat_interval=0)
    await inner.register()
    await inner.start()
    await asyncio.sleep(0.05)

    # Deep agent's graph node calls the engine to ask the inner agent.
    class DeepState(TypedDict, total=False):
        question: str
        output: str

    async def ask_inner(state, config):
        engine = config["configurable"]["ahp_engine"]
        self_addr = config["configurable"]["ahp_address"]
        msg = Message(
            source=self_addr, target=inner_addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::deep-recursion",
            body=state["question"],
        )
        inner_reply = await engine.handle(msg, timeout=2.0)
        return {"output": f"deep:{inner_reply.body}"}

    g = StateGraph(DeepState)
    g.add_node("ask", ask_inner)
    g.add_edge(START, "ask")
    g.add_edge("ask", END)
    deep_graph = g.compile()

    def into_state(msg):
        return {"question": msg.body}

    deep = DeepAgentDAG(
        deep_addr, stack.engine, deep_graph,
        input_mapper=into_state, heartbeat_interval=0,
    )
    await deep.register()
    await deep.start()
    await asyncio.sleep(0.05)

    try:
        msg = Message(
            source=sender, target=deep_addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::deep", body="ping",
        )
        reply = await stack.engine.handle(msg, timeout=3.0)
        assert reply is not None
        assert reply.body == "deep:inner-said:ping"
    finally:
        await deep.stop()
        await inner.stop()
