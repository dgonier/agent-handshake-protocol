"""Session formats — bind a protocol code, role, recipe slate, and turn
pattern into a single named interaction shape.

Two flavors share the :class:`Format` dataclass via the
``recipe_kind`` discriminator:

1. ``recipe_kind="legacy_session"`` (default; backwards compat) — the
   original session-shaped format. Fields ``code``, ``role``,
   ``round1_recipe``, plus the round-kind metadata, drive the
   moderator-led multi-round runner in ``examples/viewer/runner.py``.

2. ``recipe_kind="turn_sequence"`` (the format taxonomy) — describes
   a dyadic or n-adic conversation in terms of:
   * ``turn_primitives`` — the set of ``turn.*`` :class:`Code`-s that
     are legal moves in this format.
   * ``role_set`` — the conversational roles participants take
     (``"questioner"``, ``"responder"``, ``"rhetor"``, ``"audience"``,
     ``"mediator"``, etc).
   * ``role_turn_permissions`` — ``{role -> {allowed_turn_primitive}}``
     map. Engine-level enforcement rejects messages whose role isn't
     permitted to send that turn.
   * ``termination_rule`` — when the conversation ends (counter,
     signal, never).
   * ``invariants_prompt`` — text describing the format's soft
     invariants (anti-synthesis pressure for Agonistic, "no
     rebuttal until reflection accepted" for Rogerian). The agent
     loads this into the LLM's system prompt; not engine-enforced.
   * ``graph_builder`` — a callable returning a compiled LangGraph
     DAG implementing the format's rhythm. ``None`` if the format
     ships only as a spec (allowed but uncommon).

Legacy session fields:

* ``code`` — protocol :class:`Code` for every round.
* ``role`` — the address ``role`` the SLM materializes agents at.
* ``round1_recipe`` — recipe key for the opening round
  (``"<role>:<mode>"``).
* ``round2_recipe`` — recipe key for the middle round, or ``None``.
* ``closing_recipe`` — recipe key for the closing round, or ``None``.
* ``count_strategy`` — ``"as_requested"`` or ``"force_one"``.
* ``round2_kind`` / ``closing_kind`` — ``"broadcast"`` /
  ``"sequential_probes"`` / ``"skip"``.
* ``probe_count`` — turns when ``*_kind == "sequential_probes"``.
* ``mode_hint`` — plain-English nudge for the :class:`Inviter`.

Adding a format is one entry in :data:`FORMATS` — no runner changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from ahp.core.codes import Code


CountStrategy = Literal["as_requested", "force_one"]
RoundKind = Literal["broadcast", "sequential_probes", "skip"]
RecipeKind = Literal["legacy_session", "turn_sequence"]
TerminationKind = Literal["counter", "signal", "never", "convergence"]


@dataclass(frozen=True)
class TerminationRule:
    """How a turn-sequence format ends.

    * ``"counter"`` — terminate after ``max_turns`` turns.
    * ``"signal"`` — terminate when a specific turn primitive fires
      (e.g. ``Code.TURN_COMMIT`` for Negotiation, ``Code.TURN_DECIDE``
      for Deliberative).
    * ``"convergence"`` — terminate when a per-format measurement
      flatlines (e.g. Dialectical synthesis stops adding info). The
      format author plugs in the measurement via the graph; this
      field documents the intent.
    * ``"never"`` — does not formally terminate. Used by Polyphonic,
      Levinasian, Agonistic. Callers extract state at any point.
    """

    kind: TerminationKind = "counter"
    max_turns: int = 16
    """Used when kind=='counter'. Hard cap regardless of other rules."""

    signal_codes: tuple[str, ...] = ()
    """Used when kind=='signal'. Any of these turn codes terminates."""

    description: str = ""
    """Human-readable summary for docs / list-formats output."""


@dataclass(frozen=True)
class Format:
    """One named interaction shape — either a legacy session
    template or a turn-sequence game mode (see module docstring)."""

    name: str
    description: str

    # ── legacy session fields (recipe_kind="legacy_session") ─────────
    # All default to None / "*" so a turn_sequence format can omit
    # them. __post_init__ validates required combos.
    code: str | None = None
    role: str = "*"
    round1_recipe: str | None = None
    round2_recipe: str | None = None
    closing_recipe: str | None = None
    count_strategy: CountStrategy = "as_requested"
    round2_kind: RoundKind = "broadcast"
    closing_kind: RoundKind = "broadcast"
    probe_count: int = 3
    mode_hint: str = ""

    # ── turn-sequence fields (recipe_kind="turn_sequence") ────────────
    recipe_kind: RecipeKind = "legacy_session"

    turn_primitives: tuple[str, ...] = ()
    """The ``turn.*`` codes legal in this format. Engine checks
    ``message.code in turn_primitives`` for turn_sequence formats."""

    role_set: tuple[str, ...] = ()
    """The conversational roles participants take (e.g. "questioner",
    "responder"). Set semantics: the union of legal address.role
    fields participants may carry. Empty tuple means the format
    doesn't constrain role."""

    role_turn_permissions: dict[str, frozenset[str]] = field(
        default_factory=dict
    )
    """Map ``role -> {allowed turn primitives}``. Engine enforces:
    sender's address.role must be in this map, and message.code must
    be in the permitted set for that role. Empty dict means no
    role-based gating."""

    termination_rule: TerminationRule = field(
        default_factory=TerminationRule
    )
    """How the conversation ends. Default: 16-turn counter."""

    invariants_prompt: str = ""
    """Soft invariants the LLM should read in its system prompt.
    Anti-synthesis pressure for Agonistic, "reflection before
    rebuttal" for Rogerian, etc. Not engine-enforced — the format
    author is trusting the model to follow the guidance."""

    graph_builder: Callable[..., Any] | None = None
    """A factory returning a compiled LangGraph DAG implementing the
    format's rhythm. Called with format-specific kwargs (typically
    a participants list and a thread id). ``None`` is allowed —
    means the format ships as a spec only, no executable graph."""

    measurement_hooks: tuple[Callable[..., Any], ...] = ()
    """Optional callables that score a thread post-hoc for soft
    invariants (e.g. "did this conversation prematurely synthesize?").
    Each takes a thread history; returns a float. Scores flow into
    audit + reputation. Empty tuple = no post-hoc measurement."""

    def __post_init__(self) -> None:
        # Validate the recipe_kind + required fields combo.
        if self.recipe_kind == "legacy_session":
            if self.code is None:
                raise ValueError(
                    f"legacy_session format {self.name!r} requires "
                    f"a code"
                )
            if self.round1_recipe is None:
                raise ValueError(
                    f"legacy_session format {self.name!r} requires "
                    f"round1_recipe"
                )
        elif self.recipe_kind == "turn_sequence":
            if not self.turn_primitives:
                raise ValueError(
                    f"turn_sequence format {self.name!r} requires "
                    f"non-empty turn_primitives"
                )
            if not self.role_set:
                raise ValueError(
                    f"turn_sequence format {self.name!r} requires "
                    f"non-empty role_set"
                )
            # role_turn_permissions keys must be a subset of role_set.
            for role in self.role_turn_permissions:
                if role not in self.role_set:
                    raise ValueError(
                        f"format {self.name!r} role_turn_permissions "
                        f"references role {role!r} not in role_set "
                        f"{self.role_set}"
                    )
            # Every permitted turn must be in turn_primitives.
            for role, turns in self.role_turn_permissions.items():
                extras = set(turns) - set(self.turn_primitives)
                if extras:
                    raise ValueError(
                        f"format {self.name!r} role_turn_permissions"
                        f"[{role!r}] permits turns not in "
                        f"turn_primitives: {sorted(extras)}"
                    )
        else:  # pragma: no cover — Literal restricts
            raise ValueError(
                f"unknown recipe_kind: {self.recipe_kind!r}"
            )

    # ── helpers ──────────────────────────────────────────────────────

    def is_turn_legal(self, role: str, turn_code: str) -> bool:
        """True when an agent with the given role may send a message
        with this turn primitive. Used by engine-level enforcement.

        When ``role_turn_permissions`` is empty the check passes —
        callers haven't declared a role gate, so the engine doesn't
        impose one. When the role is in the map but its permitted
        set doesn't include the turn, returns False.
        """
        if not self.role_turn_permissions:
            return True
        permitted = self.role_turn_permissions.get(role)
        if permitted is None:
            return False
        return turn_code in permitted

    def is_turn_in_format(self, turn_code: str) -> bool:
        """True when ``turn_code`` is among this format's primitives.

        Always True for legacy_session formats (they don't declare a
        turn vocabulary)."""
        if self.recipe_kind == "legacy_session":
            return True
        return turn_code in self.turn_primitives


# ── format registry ────────────────────────────────────────────────────


FORMATS: dict[str, Format] = {
    f.name: f for f in [

        # ── DEBATE ──────────────────────────────────────────────────────
        Format(
            name="debate",
            description="Adversarial: stake, attack, rebut, close.",
            code=Code.ADVERSARIAL_DEBATE,
            role="adversarial",
            round1_recipe="adversarial:debate-me",
            round2_recipe="adversarial:debate-others",
            closing_recipe="adversarial:closing",
            mode_hint="adversarial debate with critique and closing rounds",
        ),

        # ── INTERVIEW ──────────────────────────────────────────────────
        Format(
            name="interview-me",
            description="Solo long-form: one expert, moderator probes.",
            code=Code.HUMAN_QUERY,
            role="interview",
            round1_recipe="interview:me",
            round2_recipe="interview:me-probe",
            closing_recipe="interview:me-summary",
            count_strategy="force_one",
            round2_kind="sequential_probes",
            closing_kind="sequential_probes",
            probe_count=3,
            mode_hint="long-form one-on-one expert interview",
        ),
        Format(
            name="interview-yall",
            description="Panel openings + a synthesis closing.",
            code=Code.HUMAN_QUERY,
            role="interview",
            round1_recipe="interview:yall",
            round2_recipe=None,
            closing_recipe="interview:yall-synthesize",
            round2_kind="skip",
            mode_hint="panel of expert perspectives, parallel openings",
        ),
        Format(
            name="interview-eachother",
            description="Panel openings; peers probe each other.",
            code=Code.HUMAN_QUERY,
            role="interview",
            round1_recipe="interview:yall",
            round2_recipe="interview:eachother",
            closing_recipe="interview:eachother-answer",
            mode_hint="panel of experts who will ask each other follow-ups",
        ),

        # ── COLLABORATE ────────────────────────────────────────────────
        Format(
            name="collaborate-joint",
            description="Joint problem-solving: propose, build, synthesize.",
            code=Code.COLLAB_REASON,
            role="collaborate",
            round1_recipe="collaborate:joint-propose",
            round2_recipe="collaborate:joint-build-on",
            closing_recipe="collaborate:joint-synthesize",
            mode_hint="team of collaborators solving a shared problem together",
        ),
        Format(
            name="collaborate-role",
            description="Role-divided planning: plan, conflict, consolidate.",
            code=Code.COLLAB_REASON,
            role="collaborate",
            round1_recipe="collaborate:role-plan",
            round2_recipe="collaborate:role-conflicts",
            closing_recipe="collaborate:role-consolidate",
            mode_hint="team where each member owns a distinct role in one plan",
        ),
        Format(
            name="collaborate-brainstorm",
            description="Divergence + clustering + top-3 commitment.",
            code=Code.COLLAB_REASON,
            role="collaborate",
            round1_recipe="collaborate:brainstorm-diverge",
            round2_recipe="collaborate:brainstorm-cluster",
            closing_recipe="collaborate:brainstorm-top3",
            mode_hint="brainstorming session with diverse idea generators",
        ),

        # ── CONVERSE ───────────────────────────────────────────────────
        Format(
            name="converse-free",
            description="Free-flowing chat: open, respond, reflect.",
            code=Code.COLLAB_REASON,
            role="converse",
            round1_recipe="converse:free-open",
            round2_recipe="converse:free-respond",
            closing_recipe="converse:free-reflect",
            mode_hint="open-ended conversation, no goal, just exchange",
        ),
        Format(
            name="converse-socratic",
            description="Pose-and-answer: question, answer, unanswered.",
            code=Code.COLLAB_REASON,
            role="converse",
            round1_recipe="converse:socratic-question",
            round2_recipe="converse:socratic-answer",
            closing_recipe="converse:socratic-unanswered",
            mode_hint="socratic dialogue where participants pose pointed questions",
        ),
        Format(
            name="converse-devils",
            description="Devil's advocate: position, flip, final.",
            code=Code.COLLAB_REASON,
            role="converse",
            round1_recipe="converse:devils-position",
            round2_recipe="converse:devils-flip",
            closing_recipe="converse:devils-final",
            mode_hint="participants prepared to argue both sides; opinionated but honest",
        ),

        # ── FICTION ────────────────────────────────────────────────────
        Format(
            name="fiction-theatre",
            description="Character-driven scene: enter, respond, final beat.",
            code=Code.COLLAB_REASON,
            role="fiction",
            round1_recipe="fiction:theatre-enter",
            round2_recipe="fiction:theatre-respond",
            closing_recipe="fiction:theatre-final",
            mode_hint="cast of characters for a short dramatic scene",
        ),
        Format(
            name="fiction-authors",
            description="Co-authors writing one narration: setup, complication, resolution.",
            code=Code.COLLAB_REASON,
            role="fiction",
            round1_recipe="fiction:authors-setup",
            round2_recipe="fiction:authors-complication",
            closing_recipe="fiction:authors-resolution",
            mode_hint="co-authors in a writers' room building one short story together",
        ),

        # ── DELIBERATE ─────────────────────────────────────────────────
        Format(
            name="deliberate",
            description="Decision panel: position, negotiate, vote.",
            code=Code.COLLAB_CONSENSUS,
            role="deliberate",
            round1_recipe="deliberate:position",
            round2_recipe="deliberate:negotiate",
            closing_recipe="deliberate:vote",
            mode_hint="stakeholders deliberating to reach a decision",
        ),

        # ── TEACH ──────────────────────────────────────────────────────
        Format(
            name="teach",
            description="Single teacher + 3 misconception questions.",
            code=Code.HUMAN_EXPLAIN,
            role="teach",
            round1_recipe="teach:explain",
            round2_recipe="teach:misconception",
            closing_recipe="teach:remember",
            count_strategy="force_one",
            round2_kind="sequential_probes",
            closing_kind="sequential_probes",
            probe_count=3,
            mode_hint="domain expert chosen specifically for clear teaching ability",
        ),

        # ── ESTIMATE ───────────────────────────────────────────────────
        Format(
            name="estimate",
            description="Forecasting panel: propose, update, commit.",
            code=Code.COLLAB_CONSENSUS,
            role="estimate",
            round1_recipe="estimate:propose",
            round2_recipe="estimate:update",
            closing_recipe="estimate:commit",
            mode_hint="diverse forecasters with different priors on this question",
        ),

        # ── INTERROGATE ────────────────────────────────────────────────
        Format(
            name="interrogate",
            description="Hostile cross-examination of one witness.",
            code=Code.ADVERSARIAL_CHALLENGE,
            role="interrogate",
            round1_recipe="interrogate:open",
            round2_recipe="interrogate:cross",
            closing_recipe="interrogate:hold",
            count_strategy="force_one",
            round2_kind="sequential_probes",
            closing_kind="sequential_probes",
            probe_count=3,
            mode_hint="single witness with a defensible position to cross-examine",
        ),
    ]
}


# ── register the 24 turn-sequence game modes ────────────────────────
# The taxonomy doc's 24 formats live in ahp/adapters/game_modes.py.
# We import the canonical tuple and fold it into FORMATS so all
# discovery (`ahp list-formats`, get_format(...), the engine's
# _check_format) sees both legacy session formats and the new
# turn-sequence formats through one registry.
from ahp.adapters.game_modes import GAME_MODE_FORMATS as _GAME_MODE_FORMATS

for _fmt in _GAME_MODE_FORMATS:
    if _fmt.name in FORMATS:
        raise RuntimeError(
            f"format name collision at import time: {_fmt.name!r} "
            f"appears in both legacy FORMATS and the game_modes module"
        )
    FORMATS[_fmt.name] = _fmt


class FormatNotFoundError(LookupError):
    """No format registered with the given name."""


def get_format(name: str) -> Format:
    if name not in FORMATS:
        raise FormatNotFoundError(
            f"no format named {name!r}; available: {sorted(FORMATS.keys())}"
        )
    return FORMATS[name]


def list_formats() -> list[Format]:
    return sorted(FORMATS.values(), key=lambda f: f.name)
