"""Tests for HumanAgent."""

from __future__ import annotations

import asyncio

import pytest

from ahp.adapters.human import HumanAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


HUMAN = _addr("demo.human.general.cli.s.session.devin")
SENDER = _addr("demo.adversarial.finance.equities.s.session.frank")


def _make(stack, **kwargs):
    return HumanAgent(HUMAN, stack.engine, heartbeat_interval=0, **kwargs)


def _msg(code=Code.HUMAN_QUERY, body="What's the outlook?", *, expects_response=False):
    verb = "SEND-GET" if expects_response else "SEND"
    return Message(
        source=SENDER, target=HUMAN, verb=verb,
        code=code, thread="thread::human", body=body,
        reply_to="ahp:reply:fake" if expects_response else None,
    )


# ── observation levels ─────────────────────────────────────────────────


async def test_L0_silent(stack):
    seen = []
    h = _make(stack, observation_level="L0", on_message=lambda s: _capture(seen, s))
    await h.handle_message(_msg(body="hi"))
    assert seen == []


async def test_L1_summary_skips_errors(stack):
    seen = []
    h = _make(stack, observation_level="L1", on_message=lambda s: _capture(seen, s))
    await h.handle_message(_msg(body="ok"))
    await h.handle_message(_msg(code=Code.ERROR_INTERNAL, body="oops"))
    assert len(seen) == 1
    assert "human.query" in seen[0]
    assert "frank" in seen[0]


async def test_L2_includes_full_body(stack):
    seen = []
    h = _make(stack, observation_level="L2", on_message=lambda s: _capture(seen, s))
    await h.handle_message(_msg(body="long body content"))
    assert "long body content" in seen[0]


async def test_L3_shows_errors_and_internal(stack):
    seen = []
    h = _make(stack, observation_level="L3", on_message=lambda s: _capture(seen, s))
    await h.handle_message(_msg(code=Code.ERROR_INTERNAL, body="oops"))
    assert len(seen) == 1
    assert "message_id=" in seen[0]


async def test_long_body_truncated_at_L1(stack):
    seen = []
    h = _make(stack, observation_level="L1", on_message=lambda s: _capture(seen, s))
    body = "x" * 500
    await h.handle_message(_msg(body=body))
    assert seen[0].endswith("…")
    assert len(seen[0]) < 500


# ── reply via input_provider ───────────────────────────────────────────


async def test_replies_when_response_expected(stack):
    seen = []

    async def provider(msg: Message) -> str:
        return f"user-reply-to:{msg.body}"

    h = _make(stack,
              observation_level="L1",
              on_message=lambda s: _capture(seen, s),
              input_provider=provider)

    reply = await h.handle_message(_msg(body="what?", expects_response=True))
    assert reply is not None
    assert reply.body == "user-reply-to:what?"
    assert reply.code == Code.HUMAN_QUERY
    assert reply.source == HUMAN
    assert reply.target == SENDER


async def test_no_reply_when_no_provider(stack):
    h = _make(stack)
    reply = await h.handle_message(_msg(body="what?", expects_response=True))
    assert reply is None


async def test_no_reply_when_response_not_expected(stack):
    async def provider(msg):
        raise AssertionError("provider should not be called for non-response messages")

    h = _make(stack, input_provider=provider)
    reply = await h.handle_message(_msg(body="just observe"))
    assert reply is None


# ── helper ─────────────────────────────────────────────────────────────


async def _capture(into: list, s: str) -> None:
    into.append(s)
