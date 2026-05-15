"""Fixed dialog prompt recipes.

The recipes here describe *how* an agent in a given role engages in a
given dialog mode — debate-me, debate-others, interview-open,
interview-probe, collaborative-reason. They do NOT carry topic or
persona; those are slotted in at call time:

* ``system`` — the agent's persona / world-view (chosen by the SLM
  invitation step, see :mod:`ahp.adapters.inviter`).
* ``question`` / ``others`` / ... — query-specific context.

This separation keeps protocol behavior stable across queries: a
``Code.ADVERSARIAL_DEBATE`` message with ``mode="debate-me"`` always
asks the agent for a 3-sentence position with one piece of evidence,
regardless of what was asked. Switching topics changes the persona,
not the frame.

Lookup is by ``(role, mode)``. ``Recipe.render(...)`` produces the
final prompt string passed to the underlying chat model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# A renderer takes a persona system prompt + a context dict and returns
# the final prompt string. Splitting it out keeps the recipe library
# free of Jinja-style string templating; recipes can be as smart as
# they need to be while still being plain Python.
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
        ``debate-others``, ``interview-open``, etc.
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


# ── adversarial ────────────────────────────────────────────────────────


def _render_debate_me(system: str, ctx: dict[str, Any]) -> str:
    question = ctx["question"]
    return (
        f"{system}\n\n"
        f"QUESTION: {question}\n\n"
        f"In 3 short sentences argue your position. Be confident, "
        f"specific, and name one piece of evidence."
    )


def _render_debate_others(system: str, ctx: dict[str, Any]) -> str:
    question = ctx["question"]
    my_slug = ctx.get("self_slug")
    others = [
        (o.get("slug", "?"), o.get("body", ""))
        for o in ctx.get("others", [])
        if o.get("slug") != my_slug
    ]
    bulleted = "\n".join(f"- [{slug}] {body.strip()}" for slug, body in others)
    return (
        f"{system}\n\n"
        f"QUESTION: {question}\n\n"
        f"OTHER PARTICIPANTS' ARGUMENTS:\n{bulleted}\n\n"
        f"Pick the single weakest claim above and attack it in 2 short "
        f"sentences. Do not restate your own position — only critique."
    )


# ── interview ──────────────────────────────────────────────────────────


def _render_interview_open(system: str, ctx: dict[str, Any]) -> str:
    topic = ctx["topic"]
    return (
        f"{system}\n\n"
        f"You are being interviewed about: {topic}\n\n"
        f"Give your opening statement in 3-5 sentences. Stake your core "
        f"claim and the strongest reason to believe it."
    )


def _render_interview_probe(system: str, ctx: dict[str, Any]) -> str:
    topic = ctx["topic"]
    prior = ctx.get("prior", "")
    follow_up = ctx.get("follow_up", "")
    return (
        f"{system}\n\n"
        f"TOPIC: {topic}\n\n"
        f"YOUR PRIOR ANSWER: {prior}\n\n"
        f"FOLLOW-UP QUESTION: {follow_up}\n\n"
        f"Answer the follow-up directly in 2-3 sentences. If the "
        f"follow-up exposes a weakness in your prior answer, acknowledge it."
    )


# ── collaborative ──────────────────────────────────────────────────────


def _render_collab_reason(system: str, ctx: dict[str, Any]) -> str:
    question = ctx["question"]
    return (
        f"{system}\n\n"
        f"QUESTION: {question}\n\n"
        f"Think step by step in 3-5 sentences. End with a single-line "
        f"answer prefixed 'Answer: '."
    )


RECIPES: dict[str, Recipe] = {
    r.key: r for r in [
        Recipe(
            role="adversarial", mode="debate-me",
            description="Argue your own position with one piece of evidence.",
            render=_render_debate_me,
        ),
        Recipe(
            role="adversarial", mode="debate-others",
            description="Attack the weakest opposing claim; do not restate.",
            render=_render_debate_others,
        ),
        Recipe(
            role="interview", mode="open",
            description="Opening statement on the interview topic.",
            render=_render_interview_open,
        ),
        Recipe(
            role="interview", mode="probe",
            description="Answer a probing follow-up; acknowledge weaknesses.",
            render=_render_interview_probe,
        ),
        Recipe(
            role="collaborative", mode="reason",
            description="Step-by-step reasoning ending in one-line answer.",
            render=_render_collab_reason,
        ),
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
