"""Tests for AgentFactory.invite — SLM-driven slate creation."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ahp.adapters import AgentFactory, AgentProfile, CapabilityRegistry
from ahp.adapters.base import AHPAgent
from ahp.core import AgentAddress, AddressPattern, Message
from ahp.engine.router import ProtocolEngine


@dataclass
class _Resp:
    content: str


class _ScriptedModel:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, prompt):
        return _Resp(content=self._content)


class _NoopAgent(AHPAgent):
    async def handle_message(self, message: Message):
        return None


def _builder(addr, engine, profile):
    return _NoopAgent(address=addr, engine=engine)


GOOD_4 = '''
{
  "agents": [
    {"slug": "inflation", "system": "You hold inflation."},
    {"slug": "cyclic", "system": "You hold cyclic."},
    {"slug": "quantum", "system": "You hold quantum."},
    {"slug": "simulation", "system": "You hold simulation."}
  ]
}
'''


async def test_invite_without_slm_raises(stack):
    factory = AgentFactory(stack.engine, capabilities=CapabilityRegistry())
    with pytest.raises(RuntimeError, match="requires an SLM"):
        await factory.invite(
            org="t", role="adversarial", domain="science",
            subdomain="astrophysics", topic="x", count=2,
        )


async def test_invite_materializes_addresses_and_personas(stack):
    factory = AgentFactory(
        stack.engine, capabilities=CapabilityRegistry(),
        slm=_ScriptedModel(GOOD_4),
    )
    factory.register(
        AddressPattern(
            org="*", role="adversarial", domain="science",
            subdomain="astrophysics", accept="*", lifecycle="*", instance="*",
        ),
        _builder,
    )
    result = await factory.invite(
        org="tifin", role="adversarial",
        domain="science", subdomain="astrophysics",
        topic="What caused the Big Bang?", count=4,
        mode_hint="adversarial debate",
    )
    assert len(result.new) == 4
    instances = sorted(a.address.instance for a in result.new)
    assert instances == ["cyclic", "inflation", "quantum", "simulation"]
    # Personas are keyed by full address string.
    for agent in result.new:
        assert factory.persona_for(agent.address) is not None
        assert factory.persona_for(agent.address).startswith("You hold")


async def test_set_slm_after_construction(stack):
    factory = AgentFactory(stack.engine, capabilities=CapabilityRegistry())
    factory.register(
        AddressPattern.parse("*.adversarial.*.*.*.*.*"),
        _builder,
    )
    factory.set_slm(_ScriptedModel(GOOD_4))
    result = await factory.invite(
        org="t", role="adversarial",
        domain="science", subdomain="astrophysics",
        topic="x", count=4,
    )
    assert len(result.new) == 4


async def test_invite_reuses_alive_agents(stack):
    """If an address is already alive, invite() reuses it (no new builder)."""
    factory = AgentFactory(
        stack.engine, capabilities=CapabilityRegistry(),
        slm=_ScriptedModel(GOOD_4),
    )
    factory.register(
        AddressPattern.parse("*.adversarial.*.*.*.*.*"),
        _builder,
    )
    # Pre-register one of the addresses the SLM is about to propose.
    pre = AgentAddress.parse(
        "tifin.adversarial.science.astrophysics.s.session.cyclic"
    )
    await stack.registry.register(pre)
    result = await factory.invite(
        org="tifin", role="adversarial",
        domain="science", subdomain="astrophysics",
        topic="x", count=4,
    )
    assert len(result.new) == 3
    assert pre in result.reused


async def test_invite_and_start_registers_and_starts(stack):
    factory = AgentFactory(
        stack.engine, capabilities=CapabilityRegistry(),
        slm=_ScriptedModel(GOOD_4),
    )
    factory.register(
        AddressPattern.parse("*.adversarial.*.*.*.*.*"),
        _builder,
    )
    result = await factory.invite_and_start(
        org="tifin", role="adversarial",
        domain="science", subdomain="astrophysics",
        topic="x", count=4,
    )
    # Each new agent is alive in the registry.
    for agent in result.new:
        assert await stack.registry.is_alive(agent.address)
    # Stop them for clean shutdown.
    for agent in result.new:
        await agent.stop()
