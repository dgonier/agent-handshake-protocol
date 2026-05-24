"""Format taxonomy tests — the protocol-level enforcement + the
FormatAgent wrapper, exercised against the 24 game-mode formats.

Five clusters:

1. Format dataclass — validation rules for turn_sequence vs
   legacy_session; permission helpers.
2. Engine enforcement — Message.format triggers _check_format;
   bad format name, bad turn, bad role each raise
   FormatViolationError; format=None bypasses entirely.
3. The 24 game modes — every one builds + every graph invokes;
   every one has the metadata the rest of the system relies on.
4. FormatAgent — contract check rejects missing hooks; dispatch
   routes by turn primitive; non-format-tagged messages drop.
5. End-to-end — a FormatAgent receives a real SEND-GET through
   ProtocolEngine and the right hook fires.
"""

from __future__ import annotations

import asyncio

import pytest

from ahp.adapters import (
    FORMATS,
    Format,
    FormatAgent,
    TerminationRule,
    get_format,
    list_formats,
)
from ahp.adapters.game_modes import GAME_MODE_FORMATS
from ahp.core import AgentAddress, Code, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.engine.errors import FormatViolationError
from ahp.engine.router import ProtocolEngine
from ahp.engine.thread_manager import ThreadManager
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


# ── Format dataclass validation ──────────────────────────────────────


def test_turn_sequence_requires_turn_primitives():
    with pytest.raises(ValueError, match="turn_primitives"):
        Format(
            name="bad", description="d",
            recipe_kind="turn_sequence",
            role_set=("x",),  # turn_primitives missing
        )


def test_turn_sequence_requires_role_set():
    with pytest.raises(ValueError, match="role_set"):
        Format(
            name="bad", description="d",
            recipe_kind="turn_sequence",
            turn_primitives=(Code.TURN_ASK,),
            # role_set missing
        )


def test_role_turn_permissions_keys_must_be_in_role_set():
    with pytest.raises(ValueError, match="not in role_set"):
        Format(
            name="bad", description="d",
            recipe_kind="turn_sequence",
            turn_primitives=(Code.TURN_ASK,),
            role_set=("questioner",),
            role_turn_permissions={
                "questioner": frozenset({Code.TURN_ASK}),
                "responder": frozenset({Code.TURN_ASK}),  # not in role_set
            },
        )


def test_permitted_turns_must_be_in_turn_primitives():
    with pytest.raises(ValueError, match="not in turn_primitives"):
        Format(
            name="bad", description="d",
            recipe_kind="turn_sequence",
            turn_primitives=(Code.TURN_ASK,),
            role_set=("questioner",),
            role_turn_permissions={
                "questioner": frozenset({Code.TURN_VOICE}),  # not in primitives
            },
        )


def test_legacy_session_requires_code_and_recipe():
    with pytest.raises(ValueError, match="code"):
        Format(
            name="bad", description="d",
            recipe_kind="legacy_session",
            # code + round1_recipe both missing
        )


def test_is_turn_legal_with_no_permissions_allows_all():
    """A format that declares roles but no role_turn_permissions
    map doesn't gate by role — the engine skips the check."""
    fmt = Format(
        name="open", description="d",
        recipe_kind="turn_sequence",
        turn_primitives=(Code.TURN_ASK,),
        role_set=("anyone",),
        # no role_turn_permissions
    )
    assert fmt.is_turn_legal("anyone", Code.TURN_ASK)
    assert fmt.is_turn_legal("stranger", Code.TURN_ASK)


def test_is_turn_legal_respects_role_gate():
    fmt = get_format("information-exchange")
    assert fmt.is_turn_legal("questioner", Code.TURN_ASK)
    assert not fmt.is_turn_legal("responder", Code.TURN_ASK)
    assert fmt.is_turn_legal("responder", Code.TURN_ANSWER)


def test_is_turn_in_format_legacy_returns_true():
    """Legacy session formats don't constrain turns — is_turn_in_format
    always returns True for them."""
    debate = get_format("debate")
    assert debate.is_turn_in_format(Code.TURN_VOICE)  # nonsense for debate


# ── Engine enforcement ──────────────────────────────────────────────


def _engine(redis_client) -> ProtocolEngine:
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    return ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )


async def test_check_format_unknown_name_raises(redis_client):
    engine = _engine(redis_client)
    msg = Message(
        source=AgentAddress.parse("acme.questioner.x.y.s.session.q1"),
        target=AgentAddress.parse("acme.responder.x.y.s.session.r1"),
        verb="SEND", code=Code.TURN_ASK, body="hi",
        thread="t::1", format="does-not-exist",
    )
    with pytest.raises(FormatViolationError, match="no such format"):
        await engine.handle(msg)
    await engine.bus.close()


async def test_check_format_code_not_in_turn_primitives_raises(redis_client):
    engine = _engine(redis_client)
    # information-exchange permits ASK/ANSWER/CLARIFY/CONFIRM — VOICE
    # is from polyphonic.
    msg = Message(
        source=AgentAddress.parse("acme.questioner.x.y.s.session.q1"),
        target=AgentAddress.parse("acme.responder.x.y.s.session.r1"),
        verb="SEND", code=Code.TURN_VOICE, body="hi",
        thread="t::2", format="information-exchange",
    )
    with pytest.raises(FormatViolationError, match="not a turn primitive"):
        await engine.handle(msg)
    await engine.bus.close()


async def test_check_format_wrong_role_raises(redis_client):
    engine = _engine(redis_client)
    # Source role 'responder' trying to send ASK in information-exchange
    # — only questioner can ASK.
    msg = Message(
        source=AgentAddress.parse("acme.responder.x.y.s.session.r1"),
        target=AgentAddress.parse("acme.questioner.x.y.s.session.q1"),
        verb="SEND", code=Code.TURN_ASK, body="hi",
        thread="t::3", format="information-exchange",
    )
    with pytest.raises(FormatViolationError, match="not permitted"):
        await engine.handle(msg)
    await engine.bus.close()


async def test_format_none_bypasses_check(redis_client):
    """The default Message(format=None) skips format enforcement
    entirely — backwards compat."""
    engine = _engine(redis_client)
    msg = Message(
        source=AgentAddress.parse("acme.responder.x.y.s.session.r1"),
        target=AgentAddress.parse("acme.questioner.x.y.s.session.q1"),
        verb="SEND", code=Code.TURN_ASK, body="hi",
        thread="t::4",
        # format=None (default) — engine doesn't check turn-vocab or role
    )
    # Will fail at compatibility / registry checks (target not alive,
    # no engine in place), but won't raise FormatViolationError.
    try:
        await engine.handle(msg)
    except FormatViolationError:
        pytest.fail("format=None should bypass format enforcement")
    except Exception:
        pass  # any other error is fine — we only care about format check
    await engine.bus.close()


# ── The 24 game modes ────────────────────────────────────────────────


def test_24_game_modes_registered():
    """All 24 game-mode formats end up in FORMATS."""
    turn_seq = [f for f in list_formats() if f.recipe_kind == "turn_sequence"]
    assert len(turn_seq) == 24


def test_game_mode_canonical_tuple_matches_registry():
    """GAME_MODE_FORMATS and the registry agree on the same 24."""
    canonical_names = {f.name for f in GAME_MODE_FORMATS}
    registered = {f.name for f in list_formats() if f.recipe_kind == "turn_sequence"}
    assert canonical_names == registered


@pytest.mark.parametrize("fmt", GAME_MODE_FORMATS, ids=lambda f: f.name)
def test_every_game_mode_has_required_metadata(fmt: Format):
    """Every game mode declares turn_primitives, role_set,
    termination_rule, invariants_prompt, graph_builder."""
    assert fmt.turn_primitives, f"{fmt.name} missing turn_primitives"
    assert fmt.role_set, f"{fmt.name} missing role_set"
    assert fmt.termination_rule is not None
    assert fmt.invariants_prompt, f"{fmt.name} missing invariants_prompt"
    assert fmt.graph_builder is not None, f"{fmt.name} missing graph_builder"


@pytest.mark.parametrize("fmt", GAME_MODE_FORMATS, ids=lambda f: f.name)
def test_every_game_mode_graph_compiles_and_invokes(fmt: Format):
    graph = fmt.graph_builder()
    result = graph.invoke({"turns_so_far": []})
    assert result["terminated"] is True, (
        f"{fmt.name} graph didn't terminate"
    )
    assert len(result["turns_so_far"]) == len(fmt.turn_primitives), (
        f"{fmt.name}: ran {len(result['turns_so_far'])} turns, "
        f"expected {len(fmt.turn_primitives)}"
    )


# ── FormatAgent contract check ───────────────────────────────────────


async def test_format_agent_unknown_format_in_supported_raises(redis_client):
    engine = _engine(redis_client)

    class BadAgent(FormatAgent):
        supported_formats = ("does-not-exist",)

    addr = AgentAddress.parse("acme.responder.x.y.s.session.r1")
    with pytest.raises(TypeError, match="no such format"):
        BadAgent(address=addr, engine=engine)
    await engine.bus.close()


async def test_format_agent_missing_hook_raises(redis_client):
    """A subclass that declares info-exchange support but doesn't
    override on_ask should fail loudly at construction."""
    engine = _engine(redis_client)

    class _IncompleteAgent(FormatAgent):
        supported_formats = ("information-exchange",)
        # No on_ask / on_answer / on_clarify / on_confirm overrides

    addr = AgentAddress.parse("acme.responder.x.y.s.session.r1")
    with pytest.raises(TypeError, match="contract violation"):
        _IncompleteAgent(address=addr, engine=engine)
    await engine.bus.close()


async def test_format_agent_all_hooks_present_constructs(redis_client):
    """A subclass that overrides every required hook passes the
    contract check."""
    engine = _engine(redis_client)

    class _CompleteAgent(FormatAgent):
        supported_formats = ("information-exchange",)

        async def on_ask(self, message):
            return None

        async def on_answer(self, message):
            return None

        async def on_clarify(self, message):
            return None

        async def on_confirm(self, message):
            return None

    addr = AgentAddress.parse("acme.responder.x.y.s.session.r1")
    agent = _CompleteAgent(address=addr, engine=engine)
    assert "format-agent" in agent.metadata.capabilities
    assert agent.metadata.extra["formats"]["supported"] == ["information-exchange"]
    await engine.bus.close()


async def test_format_agent_drops_messages_without_format(redis_client):
    """A message without a format declaration shouldn't be routed
    to a FormatAgent's turn hooks."""
    engine = _engine(redis_client)

    fired = []

    class _Agent(FormatAgent):
        supported_formats = ("information-exchange",)

        async def on_ask(self, message):
            fired.append("ask")
            return None

        async def on_answer(self, message):
            return None

        async def on_clarify(self, message):
            return None

        async def on_confirm(self, message):
            return None

    addr = AgentAddress.parse("acme.responder.x.y.s.session.r1")
    agent = _Agent(address=addr, engine=engine)
    msg = Message(
        source=AgentAddress.parse("you.h.x.y.s.session.caller"),
        target=addr, code=Code.TURN_ASK,
        verb="SEND-GET", body="hi", thread="t::drop",
        format=None,  # explicitly no format
    )
    result = await agent.handle_message(msg)
    assert result is None
    assert fired == []
    await engine.bus.close()


async def test_format_agent_dispatches_to_correct_hook(redis_client):
    engine = _engine(redis_client)

    fired = []

    class _Agent(FormatAgent):
        supported_formats = ("information-exchange",)

        async def on_ask(self, message):
            fired.append("ask")
            return Message(
                source=self.address, target=message.source,
                verb="SEND", code=Code.TURN_ANSWER,
                body="answered", thread=message.thread,
            )

        async def on_answer(self, message):
            fired.append("answer")
            return None

        async def on_clarify(self, message):
            return None

        async def on_confirm(self, message):
            return None

    addr = AgentAddress.parse("acme.responder.x.y.s.session.r1")
    agent = _Agent(address=addr, engine=engine)
    msg = Message(
        source=AgentAddress.parse("you.questioner.x.y.s.session.q1"),
        target=addr, code=Code.TURN_ASK,
        verb="SEND-GET", body="hi", thread="t::dispatch",
        format="information-exchange",
    )
    result = await agent.handle_message(msg)
    assert fired == ["ask"]
    assert result is not None
    assert result.body == "answered"
    await engine.bus.close()


# ── End-to-end through ProtocolEngine ────────────────────────────────


async def test_format_agent_end_to_end_send_get(redis_client):
    """A real SEND-GET with format=information-exchange routes
    through the engine's format check and into the FormatAgent's
    on_ask hook."""
    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(), threads,
        default_timeout=2.0,
    )

    class _Responder(FormatAgent):
        supported_formats = ("information-exchange",)

        async def on_ask(self, message):
            return Message(
                source=self.address, target=message.source,
                verb="SEND", code=Code.TURN_ANSWER,
                body=f"echo: {message.body}", thread=message.thread,
            )

        async def on_answer(self, message): return None
        async def on_clarify(self, message): return None
        async def on_confirm(self, message): return None

    addr = AgentAddress.parse("acme.responder.x.y.s.session.r-e2e")
    agent = _Responder(address=addr, engine=engine)
    await agent.register()
    await agent.start()
    await asyncio.sleep(0.1)

    caller = AgentAddress.parse("you.questioner.x.y.s.session.q-e2e")
    await registry.register(caller)
    try:
        msg = Message(
            source=caller, target=addr,
            code=Code.TURN_ASK, verb="SEND-GET",
            body="what is X?", thread="t::e2e",
            format="information-exchange",
        )
        response = await engine.handle(msg, timeout=2.0)
        assert response is not None
        assert "echo: what is X?" in response.body
    finally:
        await agent.stop()
        await bus.close()


# ── Soft invariants metadata ─────────────────────────────────────────


def test_anti_synthesis_invariant_appears_in_agonistic_prompt():
    """The Agonistic format's invariants_prompt should explicitly
    say no synthesis — this is the doc's load-bearing claim."""
    fmt = get_format("agonistic")
    assert "synthesis" in fmt.invariants_prompt.lower()
    assert "no" in fmt.invariants_prompt.lower() or \
           "premature consensus" in fmt.invariants_prompt.lower()


def test_no_winner_invariant_appears_in_hermeneutic_prompt():
    fmt = get_format("hermeneutic")
    assert "winner" in fmt.invariants_prompt.lower()


def test_questioner_doesnt_assert_in_socratic_prompt():
    fmt = get_format("socratic")
    assert "not assert" in fmt.invariants_prompt.lower() or \
           "only ask" in fmt.invariants_prompt.lower()
