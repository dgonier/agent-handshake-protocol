"""Session formats — bind a protocol code, role, recipe slate, and turn
pattern into a single named interaction shape.

A *format* describes one end-to-end run:

* ``code`` — protocol :class:`Code` for every round.
* ``role`` — the address ``role`` the SLM materializes agents at.
* ``round1_recipe`` — recipe key for the opening round
  (``"<role>:<mode>"``).
* ``round2_recipe`` — recipe key for the middle round, or ``None`` to
  skip.
* ``closing_recipe`` — recipe key for the closing round, or ``None`` to
  skip. Every format that ships defines one.
* ``count_strategy`` — ``"as_requested"`` or ``"force_one"`` (used by
  one-on-one formats like ``interview-me``, ``teach``, ``interrogate``).
* ``round2_kind`` — how round 2 is dispatched:
    * ``"broadcast"`` — moderator broadcasts to every agent.
    * ``"sequential_probes"`` — N moderator probes to the single agent.
    * ``"skip"`` — no round 2.
* ``closing_kind`` — same vocabulary, applied to the closing round.
* ``probe_count`` — turns when ``*_kind == "sequential_probes"``.
* ``mode_hint`` — plain-English nudge passed to the :class:`Inviter` so
  the SLM picks *kind*-appropriate perspectives.

Adding a format is one entry in :data:`FORMATS` — no runner changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ahp.core.codes import Code


CountStrategy = Literal["as_requested", "force_one"]
RoundKind = Literal["broadcast", "sequential_probes", "skip"]


@dataclass(frozen=True)
class Format:
    """One named interaction shape."""

    name: str
    description: str
    code: str
    role: str
    round1_recipe: str
    round2_recipe: str | None
    closing_recipe: str | None
    count_strategy: CountStrategy = "as_requested"
    round2_kind: RoundKind = "broadcast"
    closing_kind: RoundKind = "broadcast"
    probe_count: int = 3
    mode_hint: str = ""


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
