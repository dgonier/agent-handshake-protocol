"""Tests for the AHPAgent base class behavior."""

from __future__ import annotations

import asyncio

import pytest

from ahp.adapters.base import AHPAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


class _EchoAgent(AHPAgent):
    """Returns ``f'echo:{body}'`` as a reply."""

    async def handle_message(self, message):
        if not message.expects_response:
            return None
        return Message(
            source=self.address, target=message.source, verb="SEND",
            code=Code.INTERVIEW_TEXT, thread=message.thread,
            body=f"echo:{message.body}",
        )


class _NoiseAgent(AHPAgent):
    """Raises so we can test error wrapping."""

    async def handle_message(self, message):
        raise RuntimeError("boom")


async def test_register_appears_in_registry(stack):
    addr = _addr("demo.collaborative.finance.equities.s.session.echo")
    agent = _EchoAgent(addr, stack.engine, heartbeat_interval=0)
    await agent.register()
    assert await stack.registry.get(addr) is not None
    assert await stack.registry.is_alive(addr)


async def test_deregister_removes(stack):
    addr = _addr("demo.collaborative.finance.equities.s.session.echo")
    agent = _EchoAgent(addr, stack.engine, heartbeat_interval=0)
    await agent.register()
    await agent.deregister()
    assert await stack.registry.get(addr) is None


async def test_start_consumes_and_auto_replies(stack):
    addr = _addr("demo.adversarial.finance.equities.s.session.echo")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    agent = _EchoAgent(addr, stack.engine, heartbeat_interval=0)
    await agent.register()
    await agent.start()
    # Give the consumer task a moment to subscribe.
    await asyncio.sleep(0.05)

    try:
        # Use the engine to drive a SEND-GET; the agent should auto-reply.
        msg = Message(
            source=sender, target=addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::echo", body="hi",
        )
        reply = await stack.engine.handle(msg, timeout=2.0)
        assert reply is not None
        assert reply.body == "echo:hi"
    finally:
        await agent.stop()


async def test_handler_exception_returns_error_message(stack):
    addr = _addr("demo.adversarial.finance.equities.s.session.noisy")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    agent = _NoiseAgent(addr, stack.engine, heartbeat_interval=0)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.05)

    try:
        msg = Message(
            source=sender, target=addr, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="thread::noisy", body="hi",
        )
        reply = await stack.engine.handle(msg, timeout=2.0)
        assert reply is not None
        assert reply.code == Code.ERROR_INTERNAL
        assert "RuntimeError: boom" in reply.body
    finally:
        await agent.stop()


async def test_no_reply_when_request_has_no_reply_to(stack):
    """Plain SEND with expects_response=False should not generate a reply."""
    addr = _addr("demo.adversarial.finance.equities.s.session.silent")
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")

    received_reply_count = 0

    class CountReturn(AHPAgent):
        async def handle_message(self, message):
            nonlocal received_reply_count
            # Return a message even though no reply_to is set.
            return Message(
                source=self.address, target=message.source, verb="SEND",
                code=Code.INTERVIEW_TEXT, thread=message.thread, body="x",
            )

    agent = CountReturn(addr, stack.engine, heartbeat_interval=0)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.05)

    # Subscribe to the sender's inbox so we'd see any unexpected reply.
    sender_sub = await stack.bus.listen(sender)
    await asyncio.sleep(0.01)
    try:
        msg = Message(
            source=sender, target=addr, verb="SEND",   # no reply expected
            code=Code.INTERVIEW_TEXT, thread="thread::silent", body="hi",
        )
        # Need sender registered to send/be alive — but message routing only
        # requires the TARGET registered, which it is. SEND returns delivery count.
        delivered = await stack.engine.handle(msg)
        assert delivered >= 1

        late = await sender_sub.get_one(timeout=0.3)
        assert late is None  # no unsolicited reply
    finally:
        await sender_sub.close()
        await agent.stop()


async def test_send_helper_routes_through_engine(stack):
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")
    target = _addr("demo.adversarial.finance.equities.s.session.bob")

    sender_agent = _EchoAgent(sender, stack.engine, heartbeat_interval=0)
    # Register target so SEND has somewhere to go.
    await stack.registry.register(target)
    sub = await stack.bus.listen(target)
    await asyncio.sleep(0.01)
    try:
        await sender_agent.send(target, Code.INTERVIEW_TEXT, "hi-helper")
        received = await sub.get_one(timeout=1.0)
        assert received is not None
        assert received.body == "hi-helper"
        assert received.source == sender
    finally:
        await sub.close()


async def test_broadcast_helper_returns_responses(stack):
    sender = _addr("demo.collaborative.finance.equities.s.session.alice")
    bob = _addr("demo.adversarial.finance.equities.s.session.bob")

    sender_agent = _EchoAgent(sender, stack.engine, heartbeat_interval=0)

    bob_agent = _EchoAgent(bob, stack.engine, heartbeat_interval=0)
    await bob_agent.register()
    await bob_agent.start()
    await asyncio.sleep(0.05)

    try:
        replies = await sender_agent.broadcast(
            AddressPattern.parse("*.adversarial.*.*.s.*.*"),
            Code.INTERVIEW_TEXT,
            "probe",
            timeout=2.0,
        )
        assert len(replies) == 1
        assert replies[0].body == "echo:probe"
    finally:
        await bob_agent.stop()


async def test_stop_is_idempotent(stack):
    addr = _addr("demo.adversarial.finance.equities.s.session.iddy")
    agent = _EchoAgent(addr, stack.engine, heartbeat_interval=0)
    await agent.start()
    await agent.stop()
    await agent.stop()  # should not raise
