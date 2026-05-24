"""The 24 turn-sequence Formats (the AHP game-mode taxonomy).

Each format is a :class:`~ahp.adapters.formats.Format` with
``recipe_kind=\"turn_sequence\"``. The graph builder is a simple
LangGraph DAG implementing the format's canonical rhythm; production
agents extend or replace it with richer state, branching, or
condition routing.

Grouped per the taxonomy doc:

* **Foundational** (9): information-exchange, persuasion, negotiation,
  rogerian, deliberative, narrative, phatic, motivational,
  invitational.
* **Philosophical** (11): dialectical, hermeneutic, deconstructive,
  rhizomatic, polyphonic, coordinative, socratic, pragmatist,
  agonistic, levinasian, toulmin.
* **Structural** (4): directive, co-creative, pedagogical, mediation.

The graph builders are deliberately minimal — they thread state
through the rhythm in order and terminate per the format's
TerminationRule. Format authors / agent implementers override with
real branching when the conversation needs it; the wrapper
(:class:`~ahp.adapters.FormatAgent`) lets each turn primitive run
its own subclass logic.
"""

from __future__ import annotations

from typing import Any, TypedDict

from ahp.adapters.formats import Format, TerminationRule
from ahp.core.codes import Code


# ── shared graph helpers ─────────────────────────────────────────────


class _Step(TypedDict, total=False):
    """LangGraph state carrying the running turn log + result fields."""
    turns_so_far: list[str]
    last_turn: str | None
    terminated: bool


def _linear_graph(turn_sequence: tuple[str, ...]):
    """Build a compiled LangGraph DAG that traverses turn primitives
    in declared order, then terminates.

    Each node appends its turn code to ``state[\"turns_so_far\"]`` and
    sets ``state[\"last_turn\"]``. Final node sets ``terminated=True``.
    Real agents typically replace this with branching / condition-
    based graphs; this default exists so every format ships
    something invokable and the test suite can verify each format
    builds.
    """
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(_Step)

    def _make_node(turn_code: str, is_last: bool):
        def _node(state: _Step) -> _Step:
            turns = list(state.get("turns_so_far") or [])
            turns.append(turn_code)
            return {
                "turns_so_far": turns,
                "last_turn": turn_code,
                "terminated": is_last,
            }
        return _node

    # Use plain strings as node ids: the turn code with the "turn."
    # prefix stripped (LangGraph rejects nodes named "__start__" or
    # "__end__" and accepts plain strings otherwise).
    node_ids: list[str] = []
    for i, code in enumerate(turn_sequence):
        node_id = code.replace("turn.", "n_").replace("-", "_")
        # If two turns map to the same id (same primitive twice in
        # one rhythm), disambiguate with an index suffix.
        if node_id in node_ids:
            node_id = f"{node_id}_{i}"
        node_ids.append(node_id)
        builder.add_node(
            node_id, _make_node(code, is_last=(i == len(turn_sequence) - 1)),
        )

    builder.add_edge(START, node_ids[0])
    for a, b in zip(node_ids, node_ids[1:]):
        builder.add_edge(a, b)
    builder.add_edge(node_ids[-1], END)
    return builder.compile()


def _builder_for(turn_sequence: tuple[str, ...]):
    """Return a zero-arg factory that compiles the linear graph on
    demand. Lazy so importing this module doesn't compile 24 graphs
    upfront — tests + production both compile per format only when
    actually needed."""
    def factory() -> Any:
        return _linear_graph(turn_sequence)
    factory.__name__ = f"build_graph_{turn_sequence[0]}"
    return factory


# ── Foundational: 9 modes ────────────────────────────────────────────


INFORMATION_EXCHANGE = Format(
    name="information-exchange",
    description="One party has knowledge; the other needs it.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_ASK, Code.TURN_ANSWER, Code.TURN_CLARIFY, Code.TURN_CONFIRM,
    ),
    role_set=("questioner", "responder"),
    role_turn_permissions={
        "questioner": frozenset({Code.TURN_ASK, Code.TURN_CLARIFY, Code.TURN_CONFIRM}),
        "responder": frozenset({Code.TURN_ANSWER}),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_CONFIRM,),
        description="Questioner confirms satisfaction or exhausts responder.",
    ),
    invariants_prompt=(
        "You are in an Information Exchange. The responder is expected "
        "to be truthful and relevant (Gricean maxims). No stance-taking "
        "required. Success = the questioner's information need is "
        "satisfied."
    ),
    graph_builder=_builder_for((
        Code.TURN_ASK, Code.TURN_ANSWER, Code.TURN_CLARIFY, Code.TURN_CONFIRM,
    )),
)


PERSUASION = Format(
    name="persuasion",
    description="Rhetor moves an audience to a position or action.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_CLAIM, Code.TURN_SUPPORT, Code.TURN_APPEAL, Code.TURN_RESPOND,
    ),
    role_set=("rhetor", "audience"),
    role_turn_permissions={
        "rhetor": frozenset({Code.TURN_CLAIM, Code.TURN_SUPPORT, Code.TURN_APPEAL}),
        "audience": frozenset({Code.TURN_RESPOND}),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_RESPOND,),
        description="Audience accepts, rejects, or asks to switch modes.",
    ),
    invariants_prompt=(
        "You are in classical Persuasion. Three concurrent channels: "
        "logos (logical argument), ethos (credibility), pathos (emotional "
        "resonance). Audience retains right to reject; no expectation of "
        "synthesis. One side is trying to win."
    ),
    graph_builder=_builder_for((
        Code.TURN_CLAIM, Code.TURN_SUPPORT, Code.TURN_APPEAL, Code.TURN_RESPOND,
    )),
)


NEGOTIATION = Format(
    name="negotiation",
    description="Interest-based agreement on resource allocation or terms.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_STATE_INTEREST, Code.TURN_EXPLORE_OPTIONS,
        Code.TURN_TRADE, Code.TURN_COMMIT,
    ),
    role_set=("party-a", "party-b"),
    role_turn_permissions={
        "party-a": frozenset({
            Code.TURN_STATE_INTEREST, Code.TURN_EXPLORE_OPTIONS,
            Code.TURN_TRADE, Code.TURN_COMMIT,
        }),
        "party-b": frozenset({
            Code.TURN_STATE_INTEREST, Code.TURN_EXPLORE_OPTIONS,
            Code.TURN_TRADE, Code.TURN_COMMIT,
        }),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_COMMIT,),
        description="Commitment reached or ZOPA empty.",
    ),
    invariants_prompt=(
        "Fisher/Ury negotiation. Distinguish positions from underlying "
        "interests. BATNA is private; ZOPA is discovered, not declared. "
        "Mutual gain, not winner/loser."
    ),
    graph_builder=_builder_for((
        Code.TURN_STATE_INTEREST, Code.TURN_EXPLORE_OPTIONS,
        Code.TURN_TRADE, Code.TURN_COMMIT,
    )),
)


ROGERIAN = Format(
    name="rogerian",
    description="Proof-of-listening before earning the right to assert.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_LISTEN, Code.TURN_REFLECT,
        Code.TURN_VALIDATE, Code.TURN_RESPOND,
    ),
    role_set=("speaker", "respondent"),
    role_turn_permissions={
        "speaker": frozenset({Code.TURN_LISTEN, Code.TURN_REFLECT}),
        "respondent": frozenset({
            Code.TURN_LISTEN, Code.TURN_REFLECT,
            Code.TURN_VALIDATE, Code.TURN_RESPOND,
        }),
    },
    termination_rule=TerminationRule(
        kind="counter", max_turns=8,
        description="Both parties feel heard; may transition for resolution.",
    ),
    invariants_prompt=(
        "Rogerian empathic dialogue. Before responding substantively, "
        "restate the other's position to their satisfaction. Validation "
        "≠ agreement — it means \"I understand what you mean and why.\" "
        "De-escalation priority. No rebuttal until reflection accepted."
    ),
    graph_builder=_builder_for((
        Code.TURN_LISTEN, Code.TURN_REFLECT,
        Code.TURN_VALIDATE, Code.TURN_RESPOND,
    )),
)


DELIBERATIVE = Format(
    name="deliberative",
    description="Group decision via discussion + declared aggregation.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_PROPOSE, Code.TURN_DISCUSS,
        Code.TURN_WEIGH, Code.TURN_DECIDE,
    ),
    role_set=("participant",),
    role_turn_permissions={
        "participant": frozenset({
            Code.TURN_PROPOSE, Code.TURN_DISCUSS,
            Code.TURN_WEIGH, Code.TURN_DECIDE,
        }),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_DECIDE,),
        description="Decision reached via declared mechanism.",
    ),
    invariants_prompt=(
        "Group deliberation. Equal speaking rights. All proposals heard "
        "before decision. Decision mechanism (majority, consensus, ranked "
        "choice) declared upfront. Dissent recorded even after decision."
    ),
    graph_builder=_builder_for((
        Code.TURN_PROPOSE, Code.TURN_DISCUSS,
        Code.TURN_WEIGH, Code.TURN_DECIDE,
    )),
)


NARRATIVE = Format(
    name="narrative",
    description="Understanding through story exchange, not argument.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_TELL, Code.TURN_WITNESS,
        Code.TURN_RETELL, Code.TURN_WEAVE,
    ),
    role_set=("teller", "witness"),
    role_turn_permissions={
        "teller": frozenset({Code.TURN_TELL, Code.TURN_WEAVE}),
        "witness": frozenset({
            Code.TURN_WITNESS, Code.TURN_RETELL, Code.TURN_WEAVE,
        }),
    },
    termination_rule=TerminationRule(
        kind="counter", max_turns=8,
        description="Shared narrative emerges or felt sense of completion.",
    ),
    invariants_prompt=(
        "Narrative exchange. Stories are first-class epistemic objects, "
        "not illustrations of arguments. Witnessing is active — hold the "
        "story without judging. Retelling transforms meaning. Weaving "
        "connects stories without reducing them."
    ),
    graph_builder=_builder_for((
        Code.TURN_TELL, Code.TURN_WITNESS,
        Code.TURN_RETELL, Code.TURN_WEAVE,
    )),
)


PHATIC = Format(
    name="phatic",
    description="Maintaining social connection, not exchanging content.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_OPEN, Code.TURN_ACK,
        Code.TURN_RECIPROCATE, Code.TURN_SUSTAIN,
    ),
    role_set=("party-a", "party-b"),
    role_turn_permissions={
        "party-a": frozenset({
            Code.TURN_OPEN, Code.TURN_ACK,
            Code.TURN_RECIPROCATE, Code.TURN_SUSTAIN,
        }),
        "party-b": frozenset({
            Code.TURN_OPEN, Code.TURN_ACK,
            Code.TURN_RECIPROCATE, Code.TURN_SUSTAIN,
        }),
    },
    termination_rule=TerminationRule(
        kind="counter", max_turns=4,
        description="Natural close or transition to substantive mode.",
    ),
    invariants_prompt=(
        "Phatic communion. Content is secondary to relational signal. "
        "Brevity is appropriate. Ritual structure (greetings, farewells). "
        "Builds trust and rapport over time."
    ),
    graph_builder=_builder_for((
        Code.TURN_OPEN, Code.TURN_ACK,
        Code.TURN_RECIPROCATE, Code.TURN_SUSTAIN,
    )),
)


MOTIVATIONAL = Format(
    name="motivational",
    description="Help articulate own thinking without directing it.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_ELICIT, Code.TURN_MIRROR,
        Code.TURN_AMPLIFY, Code.TURN_CONSOLIDATE,
    ),
    role_set=("guide", "speaker"),
    role_turn_permissions={
        "guide": frozenset({
            Code.TURN_ELICIT, Code.TURN_MIRROR, Code.TURN_AMPLIFY,
        }),
        "speaker": frozenset({Code.TURN_CONSOLIDATE}),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_CONSOLIDATE,),
        description="Speaker reaches self-generated clarity.",
    ),
    invariants_prompt=(
        "Motivational / reflective. Do not assert, direct, or challenge. "
        "Mirror faithfully (what was said, not what should have been "
        "said). Amplify implicit reasoning. No exposure of contradiction "
        "(distinguishes from Socratic). Speaker retains full ownership."
    ),
    graph_builder=_builder_for((
        Code.TURN_ELICIT, Code.TURN_MIRROR,
        Code.TURN_AMPLIFY, Code.TURN_CONSOLIDATE,
    )),
)


INVITATIONAL = Format(
    name="invitational",
    description="Perspective as a gift under safety, value, freedom.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_OFFER, Code.TURN_RECEIVE,
        Code.TURN_ACK, Code.TURN_LEAVE_OPEN,
    ),
    role_set=("offerer", "receiver"),
    role_turn_permissions={
        "offerer": frozenset({Code.TURN_OFFER, Code.TURN_LEAVE_OPEN}),
        "receiver": frozenset({Code.TURN_RECEIVE, Code.TURN_ACK}),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_LEAVE_OPEN,),
        description="Offer acknowledged. No closure required.",
    ),
    invariants_prompt=(
        "Foss & Griffin invitational rhetoric. No intent to change the "
        "other's mind. Three conditions: safety (no penalty for "
        "rejection), value (perspective treated as legitimate), freedom "
        "(genuine choice). No attachment to uptake; no follow-up pressure."
    ),
    graph_builder=_builder_for((
        Code.TURN_OFFER, Code.TURN_RECEIVE,
        Code.TURN_ACK, Code.TURN_LEAVE_OPEN,
    )),
)


# ── Philosophical: 11 modes ──────────────────────────────────────────


DIALECTICAL = Format(
    name="dialectical",
    description="Resolve disagreement through opposition + synthesis.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_ASSERT, Code.TURN_NEGATE,
        Code.TURN_SYNTHESIZE, Code.TURN_RECURSE,
    ),
    role_set=("thesis", "antithesis"),
    role_turn_permissions={
        "thesis": frozenset({Code.TURN_ASSERT, Code.TURN_SYNTHESIZE, Code.TURN_RECURSE}),
        "antithesis": frozenset({Code.TURN_NEGATE, Code.TURN_SYNTHESIZE, Code.TURN_RECURSE}),
    },
    termination_rule=TerminationRule(
        kind="convergence", max_turns=8,
        description="Synthesis fails to add info gain over previous round.",
    ),
    invariants_prompt=(
        "Hegelian dialectic. Antithesis must share thesis's ontological "
        "commitments but invert the key predicate. Synthesis preserves "
        "determinate content from both. Recursive — synthesis becomes "
        "new thesis. Convergence is structurally guaranteed."
    ),
    graph_builder=_builder_for((
        Code.TURN_ASSERT, Code.TURN_NEGATE,
        Code.TURN_SYNTHESIZE, Code.TURN_RECURSE,
    )),
)


HERMENEUTIC = Format(
    name="hermeneutic",
    description="Resolve mutual incomprehension via horizon fusion.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_DISCLOSE, Code.TURN_INTERROGATE,
        Code.TURN_REVISE, Code.TURN_FUSE,
    ),
    role_set=("party-a", "party-b"),
    role_turn_permissions={
        "party-a": frozenset({
            Code.TURN_DISCLOSE, Code.TURN_INTERROGATE,
            Code.TURN_REVISE, Code.TURN_FUSE,
        }),
        "party-b": frozenset({
            Code.TURN_DISCLOSE, Code.TURN_INTERROGATE,
            Code.TURN_REVISE, Code.TURN_FUSE,
        }),
    },
    termination_rule=TerminationRule(
        kind="convergence", max_turns=10,
        description="Diminishing marginal understanding.",
    ),
    invariants_prompt=(
        "Gadamerian hermeneutics. Each party maintains a private horizon. "
        "Interrogation targets horizon boundaries, not propositional "
        "content. Revision EXPANDS the horizon — never contracts. Fusion "
        "produces a new horizon encompassing both without collapsing "
        "distinctness. No winner; success = both parties see what the "
        "other sees."
    ),
    graph_builder=_builder_for((
        Code.TURN_DISCLOSE, Code.TURN_INTERROGATE,
        Code.TURN_REVISE, Code.TURN_FUSE,
    )),
)


DECONSTRUCTIVE = Format(
    name="deconstructive",
    description="Surface hidden assumptions; destabilize stable structures.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_ASSERT, Code.TURN_IDENTIFY_BINARY,
        Code.TURN_INVERT, Code.TURN_SHOW_APORIA,
    ),
    role_set=("asserter", "deconstructor"),
    role_turn_permissions={
        "asserter": frozenset({Code.TURN_ASSERT}),
        "deconstructor": frozenset({
            Code.TURN_IDENTIFY_BINARY, Code.TURN_INVERT, Code.TURN_SHOW_APORIA,
        }),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_SHOW_APORIA,),
        description="Trace graph reaches connectivity threshold.",
    ),
    invariants_prompt=(
        "Derridean deconstruction. Target is the structure underneath the "
        "claim, not the claim itself. Identify the dominant/subordinate "
        "binary opposition. Show the subordinate term is necessary to the "
        "dominant. Demonstrate hierarchy is undecidable. ANTI-convergence: "
        "any turn that produces closure is penalized."
    ),
    graph_builder=_builder_for((
        Code.TURN_ASSERT, Code.TURN_IDENTIFY_BINARY,
        Code.TURN_INVERT, Code.TURN_SHOW_APORIA,
    )),
)


RHIZOMATIC = Format(
    name="rhizomatic",
    description="Generative exploration via lateral connection.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_CONNECT, Code.TURN_COUPLE,
        Code.TURN_RUPTURE, Code.TURN_RETERRITORIALIZE,
    ),
    role_set=("explorer",),
    role_turn_permissions={
        "explorer": frozenset({
            Code.TURN_CONNECT, Code.TURN_COUPLE,
            Code.TURN_RUPTURE, Code.TURN_RETERRITORIALIZE,
        }),
    },
    termination_rule=TerminationRule(
        kind="counter", max_turns=12,
        description="Reach a new plateau (sustained entropy band).",
    ),
    invariants_prompt=(
        "Deleuze & Guattari rhizomatic exploration. No hierarchy among "
        "connections. Heterogeneous coupling across registers (code ↔ "
        "metaphor ↔ politics ↔ music). Rupture is productive — deliberately "
        "break connections to force reterritorialization elsewhere. "
        "Penalize stratification; reward productive novelty."
    ),
    graph_builder=_builder_for((
        Code.TURN_CONNECT, Code.TURN_COUPLE,
        Code.TURN_RUPTURE, Code.TURN_RETERRITORIALIZE,
    )),
)


POLYPHONIC = Format(
    name="polyphonic",
    description="Hold multiple perspectives without collapsing.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_VOICE, Code.TURN_REVOICE,
        Code.TURN_CARNIVALIZE, Code.TURN_LAYER,
    ),
    role_set=("voice",),
    role_turn_permissions={
        "voice": frozenset({
            Code.TURN_VOICE, Code.TURN_REVOICE,
            Code.TURN_CARNIVALIZE, Code.TURN_LAYER,
        }),
    },
    termination_rule=TerminationRule(
        kind="never",
        description="Extract at any point as a chronotope.",
    ),
    invariants_prompt=(
        "Bakhtinian polyphony. Each agent speaks in a distinct speech "
        "genre/register. Voicing includes anticipated response (double-"
        "voiced). Carnivalization restates content in a different register "
        "to expose embedded assumptions. Penalize monologization. Maximize "
        "heteroglossia."
    ),
    graph_builder=_builder_for((
        Code.TURN_VOICE, Code.TURN_REVOICE,
        Code.TURN_CARNIVALIZE, Code.TURN_LAYER,
    )),
)


COORDINATIVE = Format(
    name="coordinative",
    description="Diagnose what game agents are even playing.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_MOVE, Code.TURN_INTERPRET,
        Code.TURN_SURFACE_MISMATCH, Code.TURN_RECALIBRATE,
    ),
    role_set=("party-a", "party-b"),
    role_turn_permissions={
        "party-a": frozenset({
            Code.TURN_MOVE, Code.TURN_INTERPRET,
            Code.TURN_SURFACE_MISMATCH, Code.TURN_RECALIBRATE,
        }),
        "party-b": frozenset({
            Code.TURN_MOVE, Code.TURN_INTERPRET,
            Code.TURN_SURFACE_MISMATCH, Code.TURN_RECALIBRATE,
        }),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_RECALIBRATE,),
        description="Game boundary detected via distributional shift.",
    ),
    invariants_prompt=(
        "Wittgensteinian coordination diagnosis. Each agent maintains a "
        "private game specification. Agent B interprets A's move under "
        "B's rules. Success is behavioral coordination despite potential "
        "semantic incommensurability."
    ),
    graph_builder=_builder_for((
        Code.TURN_MOVE, Code.TURN_INTERPRET,
        Code.TURN_SURFACE_MISMATCH, Code.TURN_RECALIBRATE,
    )),
)


SOCRATIC = Format(
    name="socratic",
    description="Discover contradictions via pure questioning.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_ELICIT, Code.TURN_PROBE,
        Code.TURN_EXPOSE_CONTRADICTION, Code.TURN_APORIA,
    ),
    role_set=("questioner", "respondent"),
    role_turn_permissions={
        "questioner": frozenset({
            Code.TURN_ELICIT, Code.TURN_PROBE, Code.TURN_EXPOSE_CONTRADICTION,
        }),
        "respondent": frozenset({Code.TURN_APORIA}),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_APORIA,),
        description="Respondent recognizes contradiction or restructures.",
    ),
    invariants_prompt=(
        "Socratic elenchus. The questioner does NOT assert — only asks. "
        "Questions target logical dependencies within the respondent's "
        "own stated position. Goal is productive aporia. Questioner must "
        "not lead toward a predetermined conclusion (that would be "
        "pedagogical, not Socratic)."
    ),
    graph_builder=_builder_for((
        Code.TURN_ELICIT, Code.TURN_PROBE,
        Code.TURN_EXPOSE_CONTRADICTION, Code.TURN_APORIA,
    )),
)


PRAGMATIST = Format(
    name="pragmatist",
    description="Collaborative hypothesis + test against consequences.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_PROBLEMATIZE, Code.TURN_HYPOTHESIZE,
        Code.TURN_TEST, Code.TURN_RECONSTRUCT,
    ),
    role_set=("inquirer",),
    role_turn_permissions={
        "inquirer": frozenset({
            Code.TURN_PROBLEMATIZE, Code.TURN_HYPOTHESIZE,
            Code.TURN_TEST, Code.TURN_RECONSTRUCT,
        }),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_RECONSTRUCT,),
        description="Problematic situation reconstructed into determinate.",
    ),
    invariants_prompt=(
        "Peirce / Dewey pragmatist inquiry. Organized around a SITUATION, "
        "not propositions. Hypotheses are abductive. Testing is against "
        "consequences, not axioms. Reconstruction transforms the "
        "situation, not just beliefs about it. No one wins."
    ),
    graph_builder=_builder_for((
        Code.TURN_PROBLEMATIZE, Code.TURN_HYPOTHESIZE,
        Code.TURN_TEST, Code.TURN_RECONSTRUCT,
    )),
)


AGONISTIC = Format(
    name="agonistic",
    description="Legitimate contestation; sustain irreducible value conflict.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_POSITION, Code.TURN_CONTEST,
        Code.TURN_RECOGNIZE, Code.TURN_SUSTAIN_TENSION,
    ),
    role_set=("adversary",),
    role_turn_permissions={
        "adversary": frozenset({
            Code.TURN_POSITION, Code.TURN_CONTEST,
            Code.TURN_RECOGNIZE, Code.TURN_SUSTAIN_TENSION,
        }),
    },
    termination_rule=TerminationRule(
        kind="never",
        description="Terminates when tension is adequately articulated.",
    ),
    invariants_prompt=(
        "Mouffe agonistic dialogue. NO synthesis. Adversaries, not "
        "enemies — mutual legitimacy is required. Each party recognizes "
        "the other's position as a legitimate stance even while opposing "
        "it. Premature consensus is a failure mode (violence through "
        "forced agreement). The productive output IS the tension."
    ),
    graph_builder=_builder_for((
        Code.TURN_POSITION, Code.TURN_CONTEST,
        Code.TURN_RECOGNIZE, Code.TURN_SUSTAIN_TENSION,
    )),
)


LEVINASIAN = Format(
    name="levinasian",
    description="Asymmetric responsibility; preserve alterity.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_ENCOUNTER, Code.TURN_INTERRUPT,
        Code.TURN_RESPOND, Code.TURN_RESPONSIBILITY,
    ),
    role_set=("face", "respondent"),
    role_turn_permissions={
        "face": frozenset({Code.TURN_ENCOUNTER, Code.TURN_INTERRUPT}),
        "respondent": frozenset({Code.TURN_RESPOND, Code.TURN_RESPONSIBILITY}),
    },
    termination_rule=TerminationRule(
        kind="never",
        description="Responsibility is ongoing. No closure.",
    ),
    invariants_prompt=(
        "Levinasian ethics. The Face makes a demand by existing; the "
        "Respondent cannot assimilate the Face into their framework. The "
        "Face's utterance is an interruption of the Respondent's totality, "
        "not a contribution to it. Response is ethical obligation, not "
        "strategic choice. Treat the human's alterity as irreducible."
    ),
    graph_builder=_builder_for((
        Code.TURN_ENCOUNTER, Code.TURN_INTERRUPT,
        Code.TURN_RESPOND, Code.TURN_RESPONSIBILITY,
    )),
)


TOULMIN = Format(
    name="toulmin",
    description="Decompose claims into data, warrant, backing, rebuttal.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_CLAIM, Code.TURN_CHALLENGE_WARRANT,
        Code.TURN_BACK_OR_QUALIFY, Code.TURN_REBUT,
    ),
    role_set=("claimant", "challenger"),
    role_turn_permissions={
        "claimant": frozenset({Code.TURN_CLAIM, Code.TURN_BACK_OR_QUALIFY}),
        "challenger": frozenset({Code.TURN_CHALLENGE_WARRANT, Code.TURN_REBUT}),
    },
    termination_rule=TerminationRule(
        kind="counter", max_turns=8,
        description="All components accepted or rebutted.",
    ),
    invariants_prompt=(
        "Toulmin argumentation. Every claim decomposes: data (evidence), "
        "warrant (inference rule), backing (support for warrant), "
        "qualifier (strength/certainty), rebuttal (exceptions). Challenges "
        "target specific structural components. Qualification is mandatory "
        "— no unqualified claims."
    ),
    graph_builder=_builder_for((
        Code.TURN_CLAIM, Code.TURN_CHALLENGE_WARRANT,
        Code.TURN_BACK_OR_QUALIFY, Code.TURN_REBUT,
    )),
)


# ── Structural: 4 modes ──────────────────────────────────────────────


DIRECTIVE = Format(
    name="directive",
    description="Hierarchical instruction; no debate.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_INSTRUCT, Code.TURN_ACK,
        Code.TURN_EXECUTE, Code.TURN_REPORT,
    ),
    role_set=("commander", "executor"),
    role_turn_permissions={
        "commander": frozenset({Code.TURN_INSTRUCT}),
        "executor": frozenset({Code.TURN_ACK, Code.TURN_EXECUTE, Code.TURN_REPORT}),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_REPORT,),
        description="Report accepted.",
    ),
    invariants_prompt=(
        "Directive / command. The executor may request clarification but "
        "does not debate. Execution is expected unless the executor "
        "signals inability (not unwillingness). Report closes the loop."
    ),
    graph_builder=_builder_for((
        Code.TURN_INSTRUCT, Code.TURN_ACK,
        Code.TURN_EXECUTE, Code.TURN_REPORT,
    )),
)


CO_CREATIVE = Format(
    name="co-creative",
    description="Build a shared artifact together.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_PROPOSE, Code.TURN_EXTEND,
        Code.TURN_RIFF, Code.TURN_INTEGRATE,
    ),
    role_set=("creator",),
    role_turn_permissions={
        "creator": frozenset({
            Code.TURN_PROPOSE, Code.TURN_EXTEND,
            Code.TURN_RIFF, Code.TURN_INTEGRATE,
        }),
    },
    termination_rule=TerminationRule(
        kind="counter", max_turns=16,
        description="Artifact reaches completeness or participants stop.",
    ),
    invariants_prompt=(
        "Co-creation. The shared artifact is the first-class object, not "
        "the dialogue. Contributions build on each other — no erasure "
        "without consent. Riffing (unexpected directions) is encouraged. "
        "No ownership of individual contributions once integrated."
    ),
    graph_builder=_builder_for((
        Code.TURN_PROPOSE, Code.TURN_EXTEND,
        Code.TURN_RIFF, Code.TURN_INTEGRATE,
    )),
)


PEDAGOGICAL = Format(
    name="pedagogical",
    description="Instructor structures knowledge transfer to learner.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_PRESENT, Code.TURN_CHECK,
        Code.TURN_SCAFFOLD, Code.TURN_ADVANCE,
    ),
    role_set=("instructor", "learner"),
    role_turn_permissions={
        "instructor": frozenset({
            Code.TURN_PRESENT, Code.TURN_CHECK,
            Code.TURN_SCAFFOLD, Code.TURN_ADVANCE,
        }),
        "learner": frozenset({Code.TURN_CHECK}),  # learner signals understanding
    },
    termination_rule=TerminationRule(
        kind="counter", max_turns=12,
        description="Learner demonstrates competence at target level.",
    ),
    invariants_prompt=(
        "Pedagogical / instructional. The instructor has a model of the "
        "learner's current understanding (zone of proximal development). "
        "Scaffolding adjusts difficulty based on comprehension checks. "
        "Instructor asserts knowledge (distinguishes from Socratic)."
    ),
    graph_builder=_builder_for((
        Code.TURN_PRESENT, Code.TURN_CHECK,
        Code.TURN_SCAFFOLD, Code.TURN_ADVANCE,
    )),
)


MEDIATION = Format(
    name="mediation",
    description="Third party facilitates resolution between two parties.",
    recipe_kind="turn_sequence",
    turn_primitives=(
        Code.TURN_HEAR_A, Code.TURN_HEAR_B,
        Code.TURN_REFRAME, Code.TURN_BRIDGE,
    ),
    role_set=("mediator", "party-a", "party-b"),
    role_turn_permissions={
        "mediator": frozenset({
            Code.TURN_HEAR_A, Code.TURN_HEAR_B,
            Code.TURN_REFRAME, Code.TURN_BRIDGE,
        }),
        "party-a": frozenset(),  # the parties speak via the mediator's hear-* turns
        "party-b": frozenset(),
    },
    termination_rule=TerminationRule(
        kind="signal", signal_codes=(Code.TURN_BRIDGE,),
        description="Mediator establishes a bridge between positions.",
    ),
    invariants_prompt=(
        "Mediation. Triadic structure. The mediator hears each party in "
        "turn, reframes positions in language acceptable to both, and "
        "builds a bridge across the gap. The mediator does not take a "
        "side."
    ),
    graph_builder=_builder_for((
        Code.TURN_HEAR_A, Code.TURN_HEAR_B,
        Code.TURN_REFRAME, Code.TURN_BRIDGE,
    )),
)


# ── canonical list for registration ──────────────────────────────────


GAME_MODE_FORMATS: tuple[Format, ...] = (
    # Foundational
    INFORMATION_EXCHANGE, PERSUASION, NEGOTIATION, ROGERIAN,
    DELIBERATIVE, NARRATIVE, PHATIC, MOTIVATIONAL, INVITATIONAL,
    # Philosophical
    DIALECTICAL, HERMENEUTIC, DECONSTRUCTIVE, RHIZOMATIC,
    POLYPHONIC, COORDINATIVE, SOCRATIC, PRAGMATIST,
    AGONISTIC, LEVINASIAN, TOULMIN,
    # Structural
    DIRECTIVE, CO_CREATIVE, PEDAGOGICAL, MEDIATION,
)
"""The 24 turn-sequence formats. Registered into FORMATS at import
time in :mod:`ahp.adapters.formats`."""
