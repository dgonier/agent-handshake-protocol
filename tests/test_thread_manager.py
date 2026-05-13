"""Tests for ThreadManager."""

from __future__ import annotations

import pytest

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.engine.thread_manager import Thread, ThreadManager


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


ALICE = _addr("demo.collaborative.finance.equities.s.session.alice")
BOB = _addr("demo.adversarial.finance.equities.s.session.bob")


def _msg(thread_id: str, *, source=ALICE, target=BOB, code=Code.INTERVIEW_TEXT, body="x"):
    return Message(
        source=source, target=target, verb="SEND",
        code=code, thread=thread_id, body=body,
    )


# ── lifecycle ───────────────────────────────────────────────────────────


async def test_create_returns_thread_with_slug(stack):
    tm: ThreadManager = stack.threads
    t = await tm.create("Tesla 12-Month Outlook", ALICE)
    assert isinstance(t, Thread)
    assert "tesla-12-month-outlook" in t.thread_id
    assert t.topic == "Tesla 12-Month Outlook"
    assert t.initiator == ALICE
    assert not t.is_closed


async def test_get_returns_none_for_missing(stack):
    assert await stack.threads.get("thread::nope") is None


async def test_get_round_trips(stack):
    created = await stack.threads.create("topic", ALICE)
    fetched = await stack.threads.get(created.thread_id)
    assert fetched is not None
    assert fetched.thread_id == created.thread_id
    assert fetched.topic == "topic"
    assert fetched.initiator == ALICE
    assert abs(fetched.created_at - created.created_at) < 1e-3


async def test_create_rejects_empty_topic(stack):
    with pytest.raises(ValueError):
        await stack.threads.create("", ALICE)


async def test_custom_thread_id(stack):
    t = await stack.threads.create("topic", ALICE, thread_id="thread::manual")
    assert t.thread_id == "thread::manual"


async def test_close_marks_closed(stack):
    t = await stack.threads.create("topic", ALICE)
    assert await stack.threads.close(t.thread_id)
    fetched = await stack.threads.get(t.thread_id)
    assert fetched is not None
    assert fetched.is_closed


async def test_close_missing_returns_false(stack):
    assert not await stack.threads.close("thread::nope")


# ── participation ──────────────────────────────────────────────────────


async def test_initiator_is_initial_participant(stack):
    t = await stack.threads.create("topic", ALICE)
    parts = await stack.threads.participants(t.thread_id)
    assert parts == [ALICE]


async def test_join_and_leave(stack):
    t = await stack.threads.create("topic", ALICE)
    await stack.threads.join(t.thread_id, BOB)
    parts = sorted(map(str, await stack.threads.participants(t.thread_id)))
    assert parts == sorted([str(ALICE), str(BOB)])

    await stack.threads.leave(t.thread_id, BOB)
    parts2 = await stack.threads.participants(t.thread_id)
    assert parts2 == [ALICE]


async def test_is_participant(stack):
    t = await stack.threads.create("topic", ALICE)
    assert await stack.threads.is_participant(t.thread_id, ALICE)
    assert not await stack.threads.is_participant(t.thread_id, BOB)


# ── append / history ──────────────────────────────────────────────────


async def test_append_records_and_auto_joins(stack):
    t = await stack.threads.create("topic", ALICE)
    await stack.threads.append(t.thread_id, _msg(t.thread_id, source=BOB, body="hello"))
    parts = sorted(map(str, await stack.threads.participants(t.thread_id)))
    assert str(BOB) in parts


async def test_history_returns_messages_in_order(stack):
    t = await stack.threads.create("topic", ALICE)
    for i in range(3):
        await stack.threads.append(
            t.thread_id, _msg(t.thread_id, body=f"hi-{i}"),
        )
    history = await stack.threads.get_history(t.thread_id)
    assert [m.body for m in history] == ["hi-0", "hi-1", "hi-2"]
    assert await stack.threads.length(t.thread_id) == 3


async def test_tier_filter_drops_incompatible_codes(stack):
    t = await stack.threads.create("topic", ALICE)
    # s-only and j-required messages mixed
    await stack.threads.append(
        t.thread_id, _msg(t.thread_id, code=Code.INTERVIEW_TEXT, body="text"),
    )
    await stack.threads.append(
        t.thread_id, _msg(t.thread_id, code=Code.INTERVIEW_SCHEMA, body={"k": "v"}),
    )
    await stack.threads.append(
        t.thread_id, _msg(t.thread_id, code=Code.INTERVIEW_EMBEDDINGS, body="emb-ref"),
    )

    s_view = await stack.threads.get_history(t.thread_id, tier_filter="s")
    codes = {m.code for m in s_view}
    assert Code.INTERVIEW_TEXT in codes
    assert Code.INTERVIEW_SCHEMA not in codes        # needs j
    assert Code.INTERVIEW_EMBEDDINGS not in codes    # needs b or e


async def test_empty_tier_filter_returns_all(stack):
    t = await stack.threads.create("topic", ALICE)
    await stack.threads.append(t.thread_id, _msg(t.thread_id, code=Code.INTERVIEW_TEXT))
    await stack.threads.append(t.thread_id, _msg(t.thread_id, code=Code.INTERVIEW_SCHEMA))
    all_ = await stack.threads.get_history(t.thread_id, tier_filter="")
    assert len(all_) == 2
