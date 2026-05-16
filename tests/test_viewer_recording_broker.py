"""``RecordingBroker`` smoke test.

The viewer's :class:`RecordingBroker` wraps the real broker and
captures each ``calculate_and_settle`` call into an in-process list.
The settlements panel in the economy page renders that list. If a
future refactor of the broker API silently breaks the recording, the
panel goes empty without an error — this test guards against that.
"""

from __future__ import annotations

import asyncio

import pytest

from ahp.adapters.base import AHPAgent
from ahp.broker import Broker, ServerMeta
from ahp.core import AgentAddress, Code, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import ReputationEntry, VISIBILITY_FULL_AT
from ahp.engine.router import ProtocolEngine
from ahp.engine.thread_manager import ThreadManager
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus

from examples.viewer.runner import RecordingBroker


class _EchoAgent(AHPAgent):
    async def handle_message(self, message: Message) -> Message | None:
        return Message(
            source=self.address, target=message.source,
            verb="SEND", code=message.code,
            body=f"echo: {message.body}", thread=message.thread,
        )


async def test_recording_broker_captures_send_get_settlement(redis_client):
    inner = Broker(redis_client)
    broker = RecordingBroker(inner)

    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0, broker=broker,
    )

    # Wire one self-hosted server + leaf.
    await broker.register_server(ServerMeta(
        server_id="acme", org="acme", base_rate=0.0002,
        compute_binding="acme.small.echo",
        supported_tiers=["small"],
    ))
    await broker.register_compute_provider(ComputeProvider(provider_id="acme"))
    await broker.register_leaf(MenuLeaf(
        provider_id="acme", tier="small", model="echo",
        rate_per_1k_chars=0.0,
    ))
    await broker.set_reputation(ReputationEntry(
        owner="acme", reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))

    addr = AgentAddress.parse("acme.researcher.x.y.s.session.echo-0")
    agent = _EchoAgent(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.1)

    caller = AgentAddress.parse("you.human.x.y.s.session.caller")
    await broker.wallet(str(caller)).topup(50.0, reason="seed")
    await registry.register(caller)

    assert broker.settlements == []

    msg = Message(
        source=caller, target=addr,
        code=Code.INTERVIEW_TEXT, verb="SEND-GET",
        body="hello", thread="t::rec",
    )
    response = await engine.handle(msg, timeout=2.0)
    assert response is not None

    # One settlement got recorded with the expected shape.
    assert len(broker.settlements) == 1
    s = broker.settlements[0]
    assert s["caller"] == str(caller)
    assert s["server_id"] == "acme"
    assert s["server_org"] == "acme"
    assert s["compute_leaf"] == "acme.small.echo"
    assert s["compute_provider"] == "acme"
    assert s["effective_chars"] > 0
    assert s["pre_tax"] > 0.0
    assert s["to_server"] >= 0.0
    # Tax flowed.
    assert s["to_broker"] >= 0.0
    assert s["to_commons"] >= 0.0
    assert s["at"] > 0.0

    await agent.stop()
    await bus.close()
