"""Tests for DSPyAgent.

Uses a stubbed :class:`dspy.Module` whose ``forward`` doesn't call any
language model — keeps tests deterministic and dep-light.
"""

from __future__ import annotations

import asyncio

import dspy
import pytest

from ahp.adapters.dspy_agent import DSPyAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


class _PrefixModule(dspy.Module):
    """Returns ``Prediction(answer=f'prefix:{text}')`` with no LM call."""

    def __init__(self, prefix: str = "prefix"):
        super().__init__()
        self.prefix = prefix

    def forward(self, text: str):
        return dspy.Prediction(answer=f"{self.prefix}:{text}")


async def test_dspy_agent_round_trip(stack):
    addr = _addr("demo.interview.finance.equities.s.session.dspy")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    agent = DSPyAgent(addr, stack.engine, _PrefixModule(), heartbeat_interval=0)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.05)

    try:
        msg = Message(
            source=sender, target=addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::dspy", body="hi",
        )
        reply = await stack.engine.handle(msg, timeout=2.0)
        assert reply is not None
        assert reply.body == "prefix:hi"
    finally:
        await agent.stop()


async def test_dspy_agent_custom_field_names(stack):
    """A module whose signature uses ``query``/``response`` rather than the defaults."""
    class CustomModule(dspy.Module):
        def forward(self, query: str):
            return dspy.Prediction(response=query[::-1])

    addr = _addr("demo.interview.finance.equities.s.session.dspy2")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    agent = DSPyAgent(
        addr, stack.engine, CustomModule(),
        input_field="query", output_field="response",
        heartbeat_interval=0,
    )
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.05)

    try:
        msg = Message(
            source=sender, target=addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::dspy2", body="hello",
        )
        reply = await stack.engine.handle(msg, timeout=2.0)
        assert reply is not None
        assert reply.body == "olleh"
    finally:
        await agent.stop()


async def test_dspy_agent_does_not_reply_for_send(stack):
    """SEND (no reply_to) should not auto-reply even if module returns something."""
    addr = _addr("demo.interview.finance.equities.s.session.dspy3")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    agent = DSPyAgent(addr, stack.engine, _PrefixModule(), heartbeat_interval=0)

    msg = Message(
        source=sender, target=addr, verb="SEND",
        code=Code.INTERVIEW_TEXT, thread="thread::dspy3", body="hi",
    )
    # SEND should still build a reply Message from the prediction, but the
    # base class will skip publishing because reply_to is None.
    reply = await agent.handle_message(msg)
    assert reply is None  # expects_response is False
