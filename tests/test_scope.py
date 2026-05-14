"""Tests for ScopePolicy — open-default access control over addresses.

Verifies the core invariants:

1. No rules → everything passes (existing routing behavior preserved).
2. Adding a rule narrows the matching target's accessibility.
3. Multiple rules on the same target UNION (any matching source passes).
4. Tighter rules at deeper address levels coexist with looser rules at
   shallower levels — sources matching either are allowed.
5. Point-to-point verbs raise ``UnauthorizedError`` on denial; broadcast
   verbs silently drop disallowed targets.
6. ``INVALIDATE`` is not currently gated by scope (cache control is a
   separate plane).
"""

from __future__ import annotations

import asyncio

import pytest

from ahp.adapters.base import AHPAgent
from ahp.adapters.factory import AgentFactory
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern
from ahp.engine.errors import UnauthorizedError
from ahp.engine.scope import ScopePolicy


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── unit tests for the policy data structure ──────────────────────────


def test_empty_policy_allows_everything():
    p = ScopePolicy()
    src = _addr("a.adversarial.x.y.s.session.i")
    tgt = _addr("b.collaborative.x.y.s.session.i")
    assert p.is_allowed(src, tgt, Code.INTERVIEW_TEXT)


def test_rule_narrows_matching_target():
    p = ScopePolicy()
    p.restrict(
        target="tifin.*.*.*.*.*.*",
        allow_sources="tifin.*.*.*.*.*.*",
    )
    tifin_src = _addr("tifin.adversarial.finance.equities.s.session.f")
    public_src = _addr("public.collaborative.x.y.s.session.f")
    tifin_tgt = _addr("tifin.adversarial.finance.equities.s.session.g")
    public_tgt = _addr("public.adversarial.x.y.s.session.g")

    assert p.is_allowed(tifin_src, tifin_tgt, Code.INTERVIEW_TEXT)
    assert not p.is_allowed(public_src, tifin_tgt, Code.INTERVIEW_TEXT)
    # public_tgt is NOT covered by any rule → still open.
    assert p.is_allowed(public_src, public_tgt, Code.INTERVIEW_TEXT)
    assert p.is_allowed(tifin_src, public_tgt, Code.INTERVIEW_TEXT)


def test_multiple_rules_on_same_target_union():
    p = ScopePolicy()
    p.restrict(target="tifin.db.*.*.*.*.*", allow_sources="tifin.adversarial.*.*.*.*.*")
    p.restrict(target="tifin.db.*.*.*.*.*", allow_sources="tifin.collaborative.*.*.*.*.*")

    db_tgt = _addr("tifin.db.x.y.s.session.i")
    assert p.is_allowed(
        _addr("tifin.adversarial.x.y.s.session.a"), db_tgt, Code.INTERVIEW_TEXT,
    )
    assert p.is_allowed(
        _addr("tifin.collaborative.x.y.s.session.b"), db_tgt, Code.INTERVIEW_TEXT,
    )
    # Neither rule's source matches: blocked.
    assert not p.is_allowed(
        _addr("tifin.interview.x.y.s.session.c"), db_tgt, Code.INTERVIEW_TEXT,
    )


def test_progressive_tightening_by_address_depth():
    """Looser rules at shallow address levels + tighter rules at deeper ones."""
    p = ScopePolicy()
    # Anyone in tifin can reach tifin.*
    p.restrict(target="tifin.*.*.*.*.*.*", allow_sources="tifin.*.*.*.*.*.*")
    # But only finance agents can reach tifin.*.finance.*
    p.restrict(
        target="tifin.*.finance.*.*.*.*",
        allow_sources="tifin.*.finance.*.*.*.*",
    )

    fin_src = _addr("tifin.adversarial.finance.equities.s.session.f")
    sci_src = _addr("tifin.adversarial.science.biology.s.session.g")
    fin_tgt = _addr("tifin.collaborative.finance.equities.s.session.h")

    # Finance agent can reach finance target (both rules accept).
    assert p.is_allowed(fin_src, fin_tgt, Code.INTERVIEW_TEXT)
    # Science agent: rule 1 ALLOWS (tifin → tifin), rule 2 DENIES
    # (not finance → finance). Union semantics: rule 1's allow passes.
    # → Allowed (the looser rule wins by union).
    assert p.is_allowed(sci_src, fin_tgt, Code.INTERVIEW_TEXT)


def test_code_glob_filters_rule_applicability():
    p = ScopePolicy()
    # Only adversarial agents can perform mutating CRUD on the DB.
    p.restrict(
        target="tifin.db.*.*.*.*.*",
        allow_sources="tifin.adversarial.*.*.*.*.*",
        code="collaborative.delegate",
    )
    db_tgt = _addr("tifin.db.x.y.s.session.i")
    src_anyone = _addr("public.collaborative.x.y.s.session.a")

    # Code doesn't match the rule → rule doesn't cover this call → open.
    assert p.is_allowed(src_anyone, db_tgt, Code.INTERVIEW_TEXT)
    # Code DOES match the rule → rule covers → denied for non-adversarial.
    assert not p.is_allowed(src_anyone, db_tgt, Code.COLLAB_DELEGATE)


def test_clear_restores_open_default():
    p = ScopePolicy()
    p.restrict(target="*.*.*.*.*.*.*", allow_sources="tifin.*.*.*.*.*.*")
    src = _addr("public.x.y.z.s.session.i")
    tgt = _addr("public.a.b.c.s.session.j")
    assert not p.is_allowed(src, tgt, Code.INTERVIEW_TEXT)
    p.clear()
    assert p.is_allowed(src, tgt, Code.INTERVIEW_TEXT)


# ── engine integration ───────────────────────────────────────────────


class _Echo(AHPAgent):
    async def handle_message(self, message: Message):
        if not message.expects_response:
            return None
        return Message(
            source=self.address, target=message.source, verb="SEND",
            code=message.code, thread=message.thread,
            body=f"echo:{message.body}",
        )


async def test_send_unauthorized_raises(stack):
    """Point-to-point SEND on a blocked route raises UnauthorizedError."""
    scope = ScopePolicy()
    scope.restrict(
        target="tifin.adversarial.*.*.*.*.*",
        allow_sources="tifin.*.*.*.*.*.*",
    )
    AgentFactory(stack.engine, scope=scope)   # wires engine.scope

    bull = _Echo(
        _addr("tifin.adversarial.finance.equities.s.session.bull"),
        stack.engine, heartbeat_interval=0,
    )
    await bull.register(); await bull.start()
    await asyncio.sleep(0.05)

    outsider = _addr("public.collaborative.x.y.s.session.alice")

    try:
        msg = Message(
            source=outsider, target=bull.address, verb="SEND",
            code=Code.INTERVIEW_TEXT, thread="t::scope::send", body="hi",
        )
        with pytest.raises(UnauthorizedError):
            await stack.engine.handle(msg)
    finally:
        await bull.stop()


async def test_send_get_unauthorized_raises(stack):
    scope = ScopePolicy()
    scope.restrict(
        target="tifin.adversarial.*.*.*.*.*",
        allow_sources="tifin.*.*.*.*.*.*",
    )
    AgentFactory(stack.engine, scope=scope)

    bull = _Echo(
        _addr("tifin.adversarial.finance.equities.s.session.bull"),
        stack.engine, heartbeat_interval=0,
    )
    await bull.register(); await bull.start()
    await asyncio.sleep(0.05)

    try:
        msg = Message(
            source=_addr("public.collaborative.x.y.s.session.alice"),
            target=bull.address, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="t::scope::send-get", body="hi",
        )
        with pytest.raises(UnauthorizedError):
            await stack.engine.handle(msg, timeout=0.5)
    finally:
        await bull.stop()


async def test_cast_get_silently_drops_unauthorized_targets(stack):
    """Broadcast filters disallowed targets — no exception, fewer replies."""
    scope = ScopePolicy()
    # Only tifin agents may reach tifin.adversarial.*.
    scope.restrict(
        target="tifin.adversarial.*.*.*.*.*",
        allow_sources="tifin.*.*.*.*.*.*",
    )
    AgentFactory(stack.engine, scope=scope)

    bull = _Echo(
        _addr("tifin.adversarial.finance.equities.s.session.bull"),
        stack.engine, heartbeat_interval=0,
    )
    public_bear = _Echo(
        _addr("public.adversarial.finance.equities.s.session.bear"),
        stack.engine, heartbeat_interval=0,
    )
    for a in (bull, public_bear):
        await a.register(); await a.start()
    await asyncio.sleep(0.05)

    outsider_src = _addr("public.collaborative.x.y.s.session.alice")

    try:
        # Outsider broadcasts to *.adversarial.*. The bull on tifin is
        # gated (scope rule covers tifin.adversarial.* and outsider
        # doesn't match). public.adversarial.* is NOT covered → open.
        msg = Message(
            source=outsider_src,
            target=AddressPattern.parse("*.adversarial.*.*.*.*.*"),
            verb="CAST-GET", code=Code.ADVERSARIAL_DEBATE,
            thread="t::scope::cast", body="Tesla",
        )
        replies = await stack.engine.handle(msg, timeout=2.0)
        bodies = [r.body for r in replies]
        # Only public bear answered; tifin bull was scope-filtered.
        assert bodies == ["echo:Tesla"]
        assert all("bear" in r.source.instance for r in replies)
    finally:
        await bull.stop()
        await public_bear.stop()


async def test_allowed_source_passes(stack):
    """Same scope, but the source DOES match → message goes through."""
    scope = ScopePolicy()
    scope.restrict(
        target="tifin.adversarial.*.*.*.*.*",
        allow_sources="tifin.*.*.*.*.*.*",
    )
    AgentFactory(stack.engine, scope=scope)

    bull = _Echo(
        _addr("tifin.adversarial.finance.equities.s.session.bull"),
        stack.engine, heartbeat_interval=0,
    )
    await bull.register(); await bull.start()
    await asyncio.sleep(0.05)

    try:
        msg = Message(
            source=_addr("tifin.collaborative.finance.equities.s.session.alice"),
            target=bull.address, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="t::scope::ok", body="hi",
        )
        reply = await stack.engine.handle(msg, timeout=2.0)
        assert reply is not None
        assert reply.body == "echo:hi"
    finally:
        await bull.stop()


async def test_no_scope_means_open_default(stack):
    """Without a ScopePolicy, the engine behaves exactly as before."""
    assert stack.engine.scope is None
    bull = _Echo(
        _addr("tifin.adversarial.finance.equities.s.session.bull"),
        stack.engine, heartbeat_interval=0,
    )
    await bull.register(); await bull.start()
    await asyncio.sleep(0.05)

    try:
        msg = Message(
            source=_addr("public.anyone.x.y.s.session.outsider"),
            target=bull.address, verb="SEND-GET",
            code=Code.INTERVIEW_TEXT, thread="t::open", body="hi",
        )
        reply = await stack.engine.handle(msg, timeout=2.0)
        assert reply is not None
        assert reply.body == "echo:hi"
    finally:
        await bull.stop()


async def test_invalidate_not_gated_by_scope(stack):
    """Cache control is a separate plane; INVALIDATE shouldn't trip on scope."""
    scope = ScopePolicy()
    scope.restrict(
        target="*.*.*.*.*.*.*",
        allow_sources="nobody.*.*.*.*.*.*",   # effectively blocks everyone
    )
    AgentFactory(stack.engine, scope=scope)

    msg = Message(
        source=_addr("public.x.y.z.s.session.i"),
        target=AddressPattern.parse("*.*.*.*.*.*.*"),
        verb="INVALIDATE", code=Code.ERROR_INTERNAL,
        thread="t::scope::inv", body={},
    )
    # Should not raise — INVALIDATE goes straight to cache.invalidate
    # without consulting scope.
    n = await stack.engine.handle(msg)
    assert n == 0
