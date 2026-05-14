"""Tests for GroupRegistry + AHPAgent.broadcast_to (the simple-string broadcast).

Covers the end-to-end walkthrough: register a group, register two
agents matching its pattern, then broadcast by group name from a
third agent and verify both responded.
"""

from __future__ import annotations

import asyncio

import pytest

from ahp.adapters.base import AHPAgent
from ahp.adapters.factory import AgentFactory
from ahp.adapters.groups import GroupRegistry
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── registry mechanics ────────────────────────────────────────────────


def test_register_and_resolve_by_name():
    groups = GroupRegistry()
    groups.register("debaters", "*.adversarial.*.*.*.*.*")
    pat = groups.resolve("debaters")
    assert isinstance(pat, AddressPattern)
    assert pat.role == "adversarial"


def test_resolve_falls_back_to_raw_pattern():
    """Unknown names that look like patterns get parsed."""
    groups = GroupRegistry()
    pat = groups.resolve("*.collaborative.*.*.*.*.*")
    assert pat.role == "collaborative"


def test_resolve_address_pattern_passes_through():
    groups = GroupRegistry()
    raw = AddressPattern.parse("*.*.*.*.*.*.*")
    assert groups.resolve(raw) is raw


def test_duplicate_name_rejected():
    groups = GroupRegistry()
    groups.register("debaters", "*.adversarial.*.*.*.*.*")
    with pytest.raises(ValueError, match="already registered"):
        groups.register("debaters", "*.*.*.*.*.*.*")


def test_name_with_dot_rejected():
    groups = GroupRegistry()
    with pytest.raises(ValueError, match="must not contain '\\.'"):
        groups.register("a.b", "*.*.*.*.*.*.*")


def test_empty_name_rejected():
    groups = GroupRegistry()
    with pytest.raises(ValueError):
        groups.register("", "*.*.*.*.*.*.*")


# ── factory wires groups onto the engine ──────────────────────────────


def test_factory_exposes_groups_on_engine(stack):
    groups = GroupRegistry()
    groups.register("debaters", "*.adversarial.*.*.*.*.*")
    factory = AgentFactory(stack.engine, groups=groups)
    # Engine now carries the registry, no factory reference needed.
    assert stack.engine.groups is groups


# ── end-to-end: broadcast_to by group name ────────────────────────────


class _LabeledEcho(AHPAgent):
    """Replies with ``label:body``."""

    def __init__(self, address, engine, label):
        super().__init__(address, engine, heartbeat_interval=0)
        self.label = label

    async def handle_message(self, message: Message):
        if not message.expects_response:
            return None
        return Message(
            source=self.address, target=message.source, verb="SEND",
            code=message.code, thread=message.thread,
            body=f"{self.label}:{message.body}",
        )


async def test_broadcast_to_group_name_routes_to_pattern(stack):
    """Walkthrough: name a group, register two matching agents, broadcast by name."""
    groups = GroupRegistry()
    groups.register(
        "debaters", "*.adversarial.finance.*.s.*.*",
        description="Finance bull/bear pool",
    )
    AgentFactory(stack.engine, groups=groups)  # wires engine.groups

    bull = _LabeledEcho(
        _addr("demo.adversarial.finance.equities.s.session.bull"),
        stack.engine, "bull",
    )
    bear = _LabeledEcho(
        _addr("demo.adversarial.finance.equities.s.session.bear"),
        stack.engine, "bear",
    )
    bystander = _LabeledEcho(
        _addr("demo.collaborative.finance.equities.s.session.alice"),
        stack.engine, "alice",
    )
    for a in (bull, bear, bystander):
        await a.register()
        await a.start()
    await asyncio.sleep(0.05)

    try:
        replies = await bystander.broadcast_to(
            "debaters",                                # <-- the simple string
            code=Code.ADVERSARIAL_DEBATE,
            body="argue Tesla",
            timeout=2.0,
        )
        bodies = sorted(r.body for r in replies)
        assert bodies == ["bear:argue Tesla", "bull:argue Tesla"]
    finally:
        for a in (bull, bear, bystander):
            await a.stop()


async def test_broadcast_to_works_with_raw_pattern_string(stack):
    """If you don't pre-register a group, a raw pattern string still works."""
    AgentFactory(stack.engine, groups=GroupRegistry())  # empty registry

    bull = _LabeledEcho(
        _addr("demo.adversarial.finance.equities.s.session.bull"),
        stack.engine, "bull",
    )
    caller = _LabeledEcho(
        _addr("demo.collaborative.finance.equities.s.session.alice"),
        stack.engine, "alice",
    )
    for a in (bull, caller):
        await a.register()
        await a.start()
    await asyncio.sleep(0.05)

    try:
        replies = await caller.broadcast_to(
            "*.adversarial.finance.*.s.*.*",          # raw pattern string
            code=Code.ADVERSARIAL_DEBATE,
            body="hi",
            timeout=2.0,
        )
        assert [r.body for r in replies] == ["bull:hi"]
    finally:
        for a in (bull, caller):
            await a.stop()


async def test_broadcast_to_works_without_engine_groups(stack):
    """Agents with no GroupRegistry wired in still accept raw pattern strings."""
    # Don't wire a factory at all — engine.groups stays None.
    assert getattr(stack.engine, "groups", None) is None

    bull = _LabeledEcho(
        _addr("demo.adversarial.finance.equities.s.session.bull"),
        stack.engine, "bull",
    )
    caller = _LabeledEcho(
        _addr("demo.collaborative.finance.equities.s.session.alice"),
        stack.engine, "alice",
    )
    for a in (bull, caller):
        await a.register()
        await a.start()
    await asyncio.sleep(0.05)

    try:
        replies = await caller.broadcast_to(
            "*.adversarial.finance.*.s.*.*",
            code=Code.ADVERSARIAL_DEBATE, body="x", timeout=2.0,
        )
        assert [r.body for r in replies] == ["bull:x"]
    finally:
        for a in (bull, caller):
            await a.stop()
