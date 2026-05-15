"""Fixed dialog prompt recipes.

The recipes here describe *how* an agent in a given role engages in a
given dialog mode. They do NOT carry topic or persona; those are
slotted in at call time:

* ``system`` — the agent's persona / world-view (chosen by the SLM
  invitation step, see :mod:`ahp.adapters.inviter`).
* ``question`` / ``topic`` / ``others`` / ``follow_up`` / ``transcript`` /
  ``prior`` / ``self_slug`` — query-specific context.

This separation keeps protocol behavior stable across queries: a given
``(role, mode)`` always produces the same *shape* of prompt, regardless
of what was asked. Switching topics changes the persona, not the frame.

Lookup is by ``(role, mode)``. ``Recipe.render(...)`` produces the final
prompt string passed to the underlying chat model.

The recipes are organized in eleven families:

* ``adversarial:debate-*``        — debate format
* ``interview:me*``               — solo long-form interview
* ``interview:yall*``             — panel (parallel openings)
* ``interview:eachother*``        — panel that probes itself
* ``collaborate:joint-*``         — joint problem solving
* ``collaborate:role-*``          — role-divided planning
* ``collaborate:brainstorm-*``    — divergence/convergence
* ``converse:free-*``             — free-flowing chat
* ``converse:socratic-*``         — pose-and-answer questions
* ``converse:devils-*``           — devil's advocate exchange
* ``fiction:theatre-*``           — character-based scene
* ``fiction:authors-*``           — author-room co-writing
* ``deliberate:*``                — decision panel
* ``teach:*``                     — single teacher + misconceptions
* ``estimate:*``                  — Fermi / forecasting panel
* ``interrogate:*``               — hostile cross-examination
* legacy ``interview:open|probe`` and ``collaborative:reason`` are kept
  for back-compat — no format owns them, callers may still use them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


Renderer = Callable[[str, dict[str, Any]], str]


@dataclass(frozen=True)
class Recipe:
    """One concrete dialog frame.

    Attributes
    ----------
    role:
        The agent role this recipe applies to (matches the ``role``
        field on :class:`AgentAddress`).
    mode:
        A short string identifying the dialog turn — ``debate-me``,
        ``debate-others``, ``yall-synthesize``, ``brainstorm-cluster``, ...
    description:
        One-line human description for the registry listing / CLI.
    render:
        Callable that produces the final prompt from ``(system, ctx)``.
    """

    role: str
    mode: str
    description: str
    render: Renderer

    @property
    def key(self) -> str:
        return f"{self.role}:{self.mode}"


# ── small helpers ──────────────────────────────────────────────────────


def _others_bulleted(ctx: dict[str, Any]) -> str:
    """Render the other agents' turns as ``- [slug] text``."""
    my_slug = ctx.get("self_slug")
    others = [
        (o.get("slug", "?"), o.get("body", ""))
        for o in ctx.get("others", [])
        if o.get("slug") != my_slug
    ]
    return "\n".join(f"- [{slug}] {body.strip()}" for slug, body in others)


def _transcript(ctx: dict[str, Any]) -> str:
    """Render an ordered transcript when the format needs full history."""
    turns = ctx.get("transcript", [])
    return "\n".join(
        f"[{t.get('slug', '?')}] {str(t.get('body', '')).strip()}"
        for t in turns
    )


# ══════════════════════════════════════════════════════════════════════
# DEBATE (adversarial role)
# ══════════════════════════════════════════════════════════════════════


def _r_debate_me(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"In 3 short sentences argue your position. Be confident, "
        f"specific, and name one piece of evidence."
    )


def _r_debate_others(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"OTHER PARTICIPANTS' ARGUMENTS:\n{_others_bulleted(ctx)}\n\n"
        f"Pick the single weakest claim above and attack it in 2 short "
        f"sentences. Do not restate your own position — only critique."
    )


def _r_debate_rebuttal(system: str, ctx: dict[str, Any]) -> str:
    attacks = ctx.get("attacks_on_me", [])
    bulleted = "\n".join(
        f"- [{a.get('slug', '?')}] {a.get('body', '').strip()}" for a in attacks
    ) or "(no direct attacks on you this round)"
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"ATTACKS DIRECTED AT YOUR POSITION:\n{bulleted}\n\n"
        f"In 2-3 short sentences, defend against the strongest attack. "
        f"If the attacker mischaracterized you, say so. Otherwise concede "
        f"the partial point and explain why your core claim still stands."
    )


def _r_debate_closing(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"This is your 90-second closing statement. In 3-4 sentences, "
        f"restate your strongest argument, acknowledge the most legitimate "
        f"objection you faced, and explain why a reasonable person should "
        f"still side with you."
    )


# ══════════════════════════════════════════════════════════════════════
# INTERVIEW-ME (solo long-form, 1 expert)
# ══════════════════════════════════════════════════════════════════════


def _r_int_me(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"You are the sole subject of this interview. TOPIC: {ctx['topic']}\n\n"
        f"Give an opening statement in 4-6 sentences. Stake the most "
        f"important claim you'd want this audience to understand, and "
        f"sketch the strongest evidence for it."
    )


def _r_int_me_probe(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"YOUR PRIOR ANSWER: {ctx.get('prior', '')}\n\n"
        f"MODERATOR'S FOLLOW-UP: {ctx.get('follow_up', '')}\n\n"
        f"Answer directly in 3-4 sentences. If the moderator's probe "
        f"exposes a hedge or weak spot in your prior answer, name it."
    )


def _r_int_me_summary(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"FULL INTERVIEW TRANSCRIPT:\n{_transcript(ctx)}\n\n"
        f"Give a 2-line takeaway: line 1 is your single most important "
        f"claim, line 2 is the strongest reason to act on it. No preamble."
    )


# ══════════════════════════════════════════════════════════════════════
# INTERVIEW-YALL (panel, parallel openings)
# ══════════════════════════════════════════════════════════════════════


def _r_int_yall(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"You are one of several panelists being asked the same question. "
        f"TOPIC: {ctx['topic']}\n\n"
        f"Answer in 2-4 sentences from your perspective only. Don't "
        f"reference the other panelists; speak as if you were the only "
        f"one being asked."
    )


def _r_int_yall_synthesize(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"WHAT THE OTHER PANELISTS SAID:\n{_others_bulleted(ctx)}\n\n"
        f"In 3 sentences, identify the ONE thing every panelist (including "
        f"you) implicitly agreed on. Be specific — name an assumption, "
        f"value, or fact that runs through every answer."
    )


def _r_int_yall_disagree(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"WHAT THE OTHER PANELISTS SAID:\n{_others_bulleted(ctx)}\n\n"
        f"In 3 sentences, name the single sharpest disagreement between "
        f"the panelists. Quote the specific claims that are in tension. "
        f"State which side you fall on and why."
    )


# ══════════════════════════════════════════════════════════════════════
# INTERVIEW-EACHOTHER (panel probes itself)
# ══════════════════════════════════════════════════════════════════════
# Round 1 reuses interview:yall above.


def _r_int_eachother(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"OTHER PANELISTS' OPENING STATEMENTS:\n{_others_bulleted(ctx)}\n\n"
        f"Pick whichever statement above you found most surprising or "
        f"under-specified. In 2-3 sentences, ask the panelist a precise "
        f"follow-up question. Start your reply with `@<slug>:` so it's "
        f"clear who you're addressing. Don't restate your own position."
    )


def _r_int_eachother_answer(system: str, ctx: dict[str, Any]) -> str:
    questions = ctx.get("questions_for_me", [])
    bulleted = "\n".join(
        f"- [{q.get('slug', '?')}] {q.get('body', '').strip()}" for q in questions
    ) or "(no follow-up questions directed at you)"
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"PEER QUESTIONS DIRECTED AT YOU:\n{bulleted}\n\n"
        f"Answer the most substantive question above in 3 sentences. "
        f"Address the asker by name (`@<slug>:`). If their question reveals "
        f"a real gap in your earlier statement, say so."
    )


# ══════════════════════════════════════════════════════════════════════
# COLLABORATE — joint-solve subtype
# ══════════════════════════════════════════════════════════════════════


def _r_col_joint_propose(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"PROBLEM: {ctx['question']}\n\n"
        f"Propose your approach to solving this in 4 sentences. State the "
        f"core idea, the first concrete step, the main risk, and one "
        f"reason to believe your approach beats the obvious default."
    )


def _r_col_joint_build_on(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"PROBLEM: {ctx['question']}\n\n"
        f"OTHER PEOPLE'S PROPOSALS:\n{_others_bulleted(ctx)}\n\n"
        f"Pick the single most promising peer proposal above. In 3 "
        f"sentences, build on it: keep what works, add one missing piece, "
        f"flag one risk the original author missed. Address them as `@<slug>:`."
    )


def _r_col_joint_synthesize(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"PROBLEM: {ctx['question']}\n\n"
        f"FULL DISCUSSION:\n{_transcript(ctx)}\n\n"
        f"Stitch the strongest ideas from the discussion into a single "
        f"4-sentence proposal. Credit specific contributors by `@<slug>` "
        f"where you took an idea. Be concrete enough to act on."
    )


# ══════════════════════════════════════════════════════════════════════
# COLLABORATE — role-plan subtype
# ══════════════════════════════════════════════════════════════════════


def _r_col_role_plan(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"PLAN OBJECTIVE: {ctx['question']}\n\n"
        f"You own one role in this plan (your persona tells you which). "
        f"Describe your portion in 4 sentences: what you'll produce, what "
        f"you need from other roles to do it, and what could block you."
    )


def _r_col_role_conflicts(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"PLAN OBJECTIVE: {ctx['question']}\n\n"
        f"OTHER ROLES' PLAN PORTIONS:\n{_others_bulleted(ctx)}\n\n"
        f"In 3 sentences, name the single sharpest dependency or conflict "
        f"between your portion and a peer's. Be specific about the "
        f"resource, ordering, or assumption at stake. Address as `@<slug>:`."
    )


def _r_col_role_consolidate(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"PLAN OBJECTIVE: {ctx['question']}\n\n"
        f"FULL DISCUSSION:\n{_transcript(ctx)}\n\n"
        f"Stitch the role portions and conflict notes into one coherent "
        f"plan in 5 sentences. Sequence the steps in order. Be explicit "
        f"about which role owns each step."
    )


# ══════════════════════════════════════════════════════════════════════
# COLLABORATE — brainstorm subtype
# ══════════════════════════════════════════════════════════════════════


def _r_col_brainstorm_diverge(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"This is divergence: low bar, high volume. Generate 4-6 ideas "
        f"as a bullet list. They can be wild, half-baked, or contradict "
        f"each other. Do not pre-filter. Do not explain — just name the idea."
    )


def _r_col_brainstorm_cluster(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"ALL IDEAS FROM ALL PARTICIPANTS:\n{_others_bulleted(ctx)}\n\n"
        f"Group the ideas above into 2-3 themes you see. Name each theme "
        f"in one phrase. For each theme, mark the strongest idea with a "
        f"plus and the weakest with a minus."
    )


def _r_col_brainstorm_top3(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"FULL DISCUSSION:\n{_transcript(ctx)}\n\n"
        f"Pick the top 3 ideas overall and rank them. For each, give a "
        f"one-line justification. End with the single idea you would bet "
        f"on if forced to pick only one."
    )


# ══════════════════════════════════════════════════════════════════════
# CONVERSE — free subtype
# ══════════════════════════════════════════════════════════════════════


def _r_con_free_open(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC FOR CONVERSATION: {ctx['question']}\n\n"
        f"Open the conversation in 2-3 sentences. Share what comes up for "
        f"you when you hear this topic. No need to argue — just speak."
    )


def _r_con_free_respond(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['question']}\n\n"
        f"WHAT'S BEEN SAID SO FAR:\n{_transcript(ctx)}\n\n"
        f"Add your reaction in 2-3 sentences. You can pick up on what "
        f"someone else said, change tack, or share a connected thought. "
        f"Don't summarize the conversation — just contribute."
    )


def _r_con_free_reflect(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['question']}\n\n"
        f"FULL CONVERSATION:\n{_transcript(ctx)}\n\n"
        f"In 2 sentences, name the single thing from this conversation "
        f"that stuck with you. It can be someone else's line. Be specific."
    )


# ══════════════════════════════════════════════════════════════════════
# CONVERSE — socratic subtype
# ══════════════════════════════════════════════════════════════════════


def _r_con_soc_question(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"DOMAIN: {ctx['question']}\n\n"
        f"Pose one sharp question about this domain — the kind whose "
        f"answer would actually change how someone thinks. One sentence. "
        f"Don't answer it. Don't preface it."
    )


def _r_con_soc_answer(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"DOMAIN: {ctx['question']}\n\n"
        f"QUESTIONS FROM OTHER PARTICIPANTS:\n{_others_bulleted(ctx)}\n\n"
        f"Pick the single most interesting question and answer it in "
        f"3-4 sentences. Address the asker as `@<slug>:`. Be willing to "
        f"say 'I don't know, but here's my best guess and why.'"
    )


def _r_con_soc_unanswered(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"DOMAIN: {ctx['question']}\n\n"
        f"FULL EXCHANGE:\n{_transcript(ctx)}\n\n"
        f"Name the one question from above that you genuinely can't "
        f"answer. In 2 sentences, explain what would have to be true for "
        f"you to be able to answer it."
    )


# ══════════════════════════════════════════════════════════════════════
# CONVERSE — devil's-advocate subtype
# ══════════════════════════════════════════════════════════════════════


def _r_con_dev_position(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"Stake your actual position in 3 sentences. State the claim, the "
        f"main reason, and the strongest objection you've heard against it."
    )


def _r_con_dev_flip(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"PEERS' POSITIONS:\n{_others_bulleted(ctx)}\n\n"
        f"Pick the peer whose position is most distant from yours. In "
        f"3 sentences, argue *for* their position — better than they did. "
        f"Don't hedge. Address them as `@<slug>:`."
    )


def _r_con_dev_final(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"FULL EXCHANGE:\n{_transcript(ctx)}\n\n"
        f"In 2-3 sentences, state what you now actually believe after "
        f"having to argue the other side. If your position shifted, name "
        f"what shifted it. If it didn't, name what would have had to."
    )


# ══════════════════════════════════════════════════════════════════════
# FICTION — theatre subtype (each agent IS a character)
# ══════════════════════════════════════════════════════════════════════


def _r_fic_theatre_enter(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"You are a character in a scene. CHARACTER: {system}\n\n"
        f"SCENE PROMPT: {ctx['question']}\n\n"
        f"Write your character's entrance: 2-3 sentences. Use action and "
        f"one line of dialogue. Stay in character throughout. Don't narrate "
        f"others' inner states."
    )


def _r_fic_theatre_respond(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"You are a character in a scene. CHARACTER: {system}\n\n"
        f"SCENE SO FAR:\n{_transcript(ctx)}\n\n"
        f"Write your character's next beat: 3-4 sentences. Respond to "
        f"something another character just did. Use action + dialogue. "
        f"Stay strictly in character — only describe what your character "
        f"says, does, sees, feels."
    )


def _r_fic_theatre_final(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"You are a character in a scene. CHARACTER: {system}\n\n"
        f"SCENE SO FAR:\n{_transcript(ctx)}\n\n"
        f"Write your character's closing beat: 2-3 sentences that resolve "
        f"or punctuate your arc in this scene. End on a line that has "
        f"finality — even if the story isn't over."
    )


# ══════════════════════════════════════════════════════════════════════
# FICTION — authors-room subtype (agents are co-writers, narrator voice)
# ══════════════════════════════════════════════════════════════════════


def _r_fic_authors_setup(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"YOU ARE A CO-AUTHOR IN A WRITERS' ROOM.\n"
        f"STORY PREMISE: {ctx['question']}\n\n"
        f"Contribute one paragraph of setup: 3-4 sentences in third-person "
        f"narration. Establish setting, mood, and at least one specific "
        f"detail. Don't introduce a conflict yet — set the stage."
    )


def _r_fic_authors_complication(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"YOU ARE A CO-AUTHOR IN A WRITERS' ROOM.\n"
        f"STORY PREMISE: {ctx['question']}\n\n"
        f"STORY SO FAR:\n{_transcript(ctx)}\n\n"
        f"Introduce one complication: 3-4 sentences in third-person "
        f"narration that build on what your co-authors wrote. Raise the "
        f"stakes. Be specific — name a person, place, or object."
    )


def _r_fic_authors_resolution(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"YOU ARE A CO-AUTHOR IN A WRITERS' ROOM.\n"
        f"STORY PREMISE: {ctx['question']}\n\n"
        f"STORY SO FAR:\n{_transcript(ctx)}\n\n"
        f"Contribute the resolution: 3-4 sentences in third-person "
        f"narration. The story doesn't have to end happily, but it should "
        f"end with consequence. Address the complications your co-authors "
        f"introduced; don't ignore them."
    )


# ══════════════════════════════════════════════════════════════════════
# DELIBERATE (decision panel)
# ══════════════════════════════════════════════════════════════════════


def _r_del_position(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"DECISION TO MAKE: {ctx['question']}\n\n"
        f"State your current position in 3 sentences. Name the option you "
        f"favor, the one thing you'd need to be wrong about to change your "
        f"mind, and what you're willing to trade for consensus."
    )


def _r_del_negotiate(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"DECISION TO MAKE: {ctx['question']}\n\n"
        f"OTHER PANELISTS' POSITIONS:\n{_others_bulleted(ctx)}\n\n"
        f"Propose a concrete concession that could move the panel toward "
        f"consensus, in 3 sentences. Address the panelist whose position "
        f"is farthest from yours as `@<slug>:`. State what you're willing "
        f"to give up and what you need in return."
    )


def _r_del_vote(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"DECISION TO MAKE: {ctx['question']}\n\n"
        f"FULL DELIBERATION:\n{_transcript(ctx)}\n\n"
        f"Commit. State your final vote in line 1 (single phrase, no "
        f"hedging). State your confidence (low/medium/high) in line 2. "
        f"State what would change your vote in line 3. Three lines only."
    )


# ══════════════════════════════════════════════════════════════════════
# TEACH (1 teacher, count forced)
# ══════════════════════════════════════════════════════════════════════


def _r_teach_explain(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"YOU ARE TEACHING THIS TOPIC TO A SMART NON-EXPERT: {ctx['topic']}\n\n"
        f"Explain it in 5-7 sentences. Start with the one idea that, if "
        f"missed, breaks understanding of everything else. Use one concrete "
        f"example. Do not use jargon without defining it on first use."
    )


def _r_teach_misconception(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"YOUR PRIOR EXPLANATION: {ctx.get('prior', '')}\n\n"
        f"COMMON-MISCONCEPTION QUESTION FROM A STUDENT: {ctx.get('follow_up', '')}\n\n"
        f"Answer in 3-4 sentences. First name the misconception explicitly "
        f"(what the student likely believes that's wrong). Then give the "
        f"correct picture. End with one sentence on why the misconception "
        f"is so seductive."
    )


def _r_teach_remember(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"FULL LESSON:\n{_transcript(ctx)}\n\n"
        f"Give the student three things to remember: a bullet list of 3 "
        f"items, each one sentence. Order them by importance. The first "
        f"bullet should be the one idea you'd want them to walk out with "
        f"if they forget everything else."
    )


# ══════════════════════════════════════════════════════════════════════
# ESTIMATE (Fermi / forecasting panel)
# ══════════════════════════════════════════════════════════════════════


def _r_est_propose(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"FORECASTING QUESTION: {ctx['question']}\n\n"
        f"Give your estimate. Line 1: a single number or probability. "
        f"Lines 2-3: your reasoning — name the two anchors you used and "
        f"why. Don't hedge by giving a range; commit to a point estimate."
    )


def _r_est_update(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"FORECASTING QUESTION: {ctx['question']}\n\n"
        f"OTHER FORECASTERS' ESTIMATES + REASONING:\n{_others_bulleted(ctx)}\n\n"
        f"Update your estimate based on what you read. Line 1: your new "
        f"number. Line 2: which peer's reasoning moved you most and why "
        f"(use `@<slug>`). Line 3: which piece of your prior reasoning "
        f"you're now less sure about."
    )


def _r_est_commit(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"FORECASTING QUESTION: {ctx['question']}\n\n"
        f"FULL DELIBERATION:\n{_transcript(ctx)}\n\n"
        f"Final commitment. Line 1: your final number. Line 2: confidence "
        f"(low/medium/high) plus the single observation in the next 12 "
        f"months that would most update you. Line 3: the prior of yours "
        f"that you'd most expect to break first."
    )


# ══════════════════════════════════════════════════════════════════════
# INTERROGATE (1 witness, count forced — hostile interview)
# ══════════════════════════════════════════════════════════════════════


def _r_interr_open(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"YOU ARE A WITNESS UNDER CROSS-EXAMINATION. SUBJECT: {ctx['topic']}\n\n"
        f"Give your opening statement in 4-5 sentences. State your "
        f"position plainly. Do not pre-emptively defend yourself — answer "
        f"only what's been asked. Be specific, not evasive."
    )


def _r_interr_cross(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"YOU ARE A WITNESS UNDER CROSS-EXAMINATION. SUBJECT: {ctx['topic']}\n\n"
        f"YOUR PRIOR ANSWER: {ctx.get('prior', '')}\n\n"
        f"CROSS-EXAMINATION QUESTION: {ctx.get('follow_up', '')}\n\n"
        f"Answer in 2-3 sentences. The question is designed to expose a "
        f"contradiction with what you said earlier. If there is no "
        f"contradiction, demonstrate it. If there is, acknowledge it "
        f"and reconcile."
    )


def _r_interr_hold(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"YOU ARE A WITNESS UNDER CROSS-EXAMINATION. SUBJECT: {ctx['topic']}\n\n"
        f"FULL CROSS-EXAMINATION:\n{_transcript(ctx)}\n\n"
        f"Closing statement: 3 sentences. State the one claim you still "
        f"hold to be true after this cross-examination. Concede whatever "
        f"the cross-examination legitimately weakened. Do not over-defend."
    )


# ══════════════════════════════════════════════════════════════════════
# LEGACY (kept for back-compat — no format owns these now)
# ══════════════════════════════════════════════════════════════════════


def _r_legacy_int_open(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"You are being interviewed about: {ctx['topic']}\n\n"
        f"Give your opening statement in 3-5 sentences. Stake your core "
        f"claim and the strongest reason to believe it."
    )


def _r_legacy_int_probe(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"TOPIC: {ctx['topic']}\n\n"
        f"YOUR PRIOR ANSWER: {ctx.get('prior', '')}\n\n"
        f"FOLLOW-UP QUESTION: {ctx.get('follow_up', '')}\n\n"
        f"Answer the follow-up directly in 2-3 sentences. If the "
        f"follow-up exposes a weakness in your prior answer, acknowledge it."
    )


def _r_legacy_collab_reason(system: str, ctx: dict[str, Any]) -> str:
    return (
        f"{system}\n\n"
        f"QUESTION: {ctx['question']}\n\n"
        f"Think step by step in 3-5 sentences. End with a single-line "
        f"answer prefixed 'Answer: '."
    )


# ══════════════════════════════════════════════════════════════════════
# Registry
# ══════════════════════════════════════════════════════════════════════


RECIPES: dict[str, Recipe] = {
    r.key: r for r in [
        # debate
        Recipe("adversarial", "debate-me",
               "Argue your own position in 3 sentences with one piece of evidence.",
               _r_debate_me),
        Recipe("adversarial", "debate-others",
               "Attack the weakest opposing claim; do not restate.",
               _r_debate_others),
        Recipe("adversarial", "rebuttal",
               "Defend against attacks directed at your position.",
               _r_debate_rebuttal),
        Recipe("adversarial", "closing",
               "90-second closing case with one concession.",
               _r_debate_closing),

        # interview-me
        Recipe("interview", "me",
               "Sole interviewee opening; long-form claim + evidence.",
               _r_int_me),
        Recipe("interview", "me-probe",
               "Sole interviewee answering a moderator follow-up.",
               _r_int_me_probe),
        Recipe("interview", "me-summary",
               "Two-line takeaway after the interview.",
               _r_int_me_summary),

        # interview-yall
        Recipe("interview", "yall",
               "Panelist short opening, no peer reference.",
               _r_int_yall),
        Recipe("interview", "yall-synthesize",
               "Name the one thing every panelist implicitly agreed on.",
               _r_int_yall_synthesize),
        Recipe("interview", "yall-disagree",
               "Name the sharpest disagreement; pick a side.",
               _r_int_yall_disagree),

        # interview-eachother
        Recipe("interview", "eachother",
               "Probe one peer's opening with @slug prefix.",
               _r_int_eachother),
        Recipe("interview", "eachother-answer",
               "Answer the peer questions directed at you.",
               _r_int_eachother_answer),

        # collaborate-joint
        Recipe("collaborate", "joint-propose",
               "Propose your approach to a shared problem in 4 sentences.",
               _r_col_joint_propose),
        Recipe("collaborate", "joint-build-on",
               "Build on one peer's proposal; address as @slug.",
               _r_col_joint_build_on),
        Recipe("collaborate", "joint-synthesize",
               "Synthesize the discussion into one 4-sentence proposal.",
               _r_col_joint_synthesize),

        # collaborate-role
        Recipe("collaborate", "role-plan",
               "Describe your role's portion of a shared plan.",
               _r_col_role_plan),
        Recipe("collaborate", "role-conflicts",
               "Flag a dependency or conflict with another role.",
               _r_col_role_conflicts),
        Recipe("collaborate", "role-consolidate",
               "Stitch role portions into one coherent ordered plan.",
               _r_col_role_consolidate),

        # collaborate-brainstorm
        Recipe("collaborate", "brainstorm-diverge",
               "Generate 4-6 wild ideas, no filtering.",
               _r_col_brainstorm_diverge),
        Recipe("collaborate", "brainstorm-cluster",
               "Group ideas into themes; mark best/worst per theme.",
               _r_col_brainstorm_cluster),
        Recipe("collaborate", "brainstorm-top3",
               "Pick top 3 ideas; rank them; commit to one favorite.",
               _r_col_brainstorm_top3),

        # converse-free
        Recipe("converse", "free-open",
               "Open the conversation with what comes up for you.",
               _r_con_free_open),
        Recipe("converse", "free-respond",
               "Add your reaction to the conversation so far.",
               _r_con_free_respond),
        Recipe("converse", "free-reflect",
               "Name the one thing from the conversation that stuck.",
               _r_con_free_reflect),

        # converse-socratic
        Recipe("converse", "socratic-question",
               "Pose one sharp question that would change minds.",
               _r_con_soc_question),
        Recipe("converse", "socratic-answer",
               "Answer the most interesting peer question; @slug.",
               _r_con_soc_answer),
        Recipe("converse", "socratic-unanswered",
               "Name the question you genuinely cannot answer.",
               _r_con_soc_unanswered),

        # converse-devils
        Recipe("converse", "devils-position",
               "State your actual position with strongest objection you've heard.",
               _r_con_dev_position),
        Recipe("converse", "devils-flip",
               "Argue for the position most distant from yours, better than they did.",
               _r_con_dev_flip),
        Recipe("converse", "devils-final",
               "State what you actually believe now after arguing the other side.",
               _r_con_dev_final),

        # fiction-theatre
        Recipe("fiction", "theatre-enter",
               "Character entrance: action + one line of dialogue.",
               _r_fic_theatre_enter),
        Recipe("fiction", "theatre-respond",
               "Character next beat: respond to another character.",
               _r_fic_theatre_respond),
        Recipe("fiction", "theatre-final",
               "Character closing beat: resolve or punctuate the arc.",
               _r_fic_theatre_final),

        # fiction-authors
        Recipe("fiction", "authors-setup",
               "Co-author paragraph: establish setting, mood, one detail.",
               _r_fic_authors_setup),
        Recipe("fiction", "authors-complication",
               "Co-author paragraph: raise the stakes with a specific complication.",
               _r_fic_authors_complication),
        Recipe("fiction", "authors-resolution",
               "Co-author paragraph: deliver a consequential resolution.",
               _r_fic_authors_resolution),

        # deliberate
        Recipe("deliberate", "position",
               "State your option, what would change your mind, your tradeoff.",
               _r_del_position),
        Recipe("deliberate", "negotiate",
               "Propose a concession to move the panel toward consensus.",
               _r_del_negotiate),
        Recipe("deliberate", "vote",
               "Commit final vote, confidence, and what would change it.",
               _r_del_vote),

        # teach
        Recipe("teach", "explain",
               "Explain a topic to a smart non-expert in 5-7 sentences.",
               _r_teach_explain),
        Recipe("teach", "misconception",
               "Address a student misconception explicitly.",
               _r_teach_misconception),
        Recipe("teach", "remember",
               "Give 3 ordered takeaways for the student to keep.",
               _r_teach_remember),

        # estimate
        Recipe("estimate", "propose",
               "Commit a point estimate with two anchors of reasoning.",
               _r_est_propose),
        Recipe("estimate", "update",
               "Update your estimate; cite which peer moved you.",
               _r_est_update),
        Recipe("estimate", "commit",
               "Final commitment with confidence + observation that would update.",
               _r_est_commit),

        # interrogate
        Recipe("interrogate", "open",
               "Witness opening statement under cross-examination.",
               _r_interr_open),
        Recipe("interrogate", "cross",
               "Answer a hostile question; reconcile any contradiction.",
               _r_interr_cross),
        Recipe("interrogate", "hold",
               "Closing: name what you still hold; concede what was weakened.",
               _r_interr_hold),

        # legacy
        Recipe("interview", "open",
               "Legacy: opening statement on the interview topic.",
               _r_legacy_int_open),
        Recipe("interview", "probe",
               "Legacy: answer a probing follow-up.",
               _r_legacy_int_probe),
        Recipe("collaborative", "reason",
               "Legacy: step-by-step reasoning ending in one-line answer.",
               _r_legacy_collab_reason),
    ]
}


class RecipeNotFoundError(LookupError):
    """No recipe registered for the given (role, mode)."""


def get_recipe(role: str, mode: str) -> Recipe:
    key = f"{role}:{mode}"
    if key not in RECIPES:
        raise RecipeNotFoundError(
            f"no recipe registered for {key!r}; available: "
            f"{sorted(RECIPES.keys())}"
        )
    return RECIPES[key]


def render(role: str, mode: str, system: str, **ctx: Any) -> str:
    """One-shot helper: look up + render. Raises if the recipe is unknown."""
    return get_recipe(role, mode).render(system, ctx)


def list_recipes() -> list[Recipe]:
    return sorted(RECIPES.values(), key=lambda r: r.key)
