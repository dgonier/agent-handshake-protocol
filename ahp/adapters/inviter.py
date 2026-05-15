"""Inviter — turns a topic + domain into a slate of agent personas.

The :class:`Inviter` is the *invitation-time* SLM call. Given the
domain, subdomain, and the user's topic, it asks a small fast model
to enumerate ``count`` perspectives a real community in that field
would hold on that topic. The model returns JSON; the inviter parses
it into :class:`AgentInvitation` records.

What the SLM does NOT decide:

* Address shape — org, role, accept tier, lifecycle are set by the
  caller (the factory / demo).
* Dialog format — that's the recipe library
  (:mod:`ahp.adapters.prompts`).
* Number of agents — the caller passes ``count``.

The SLM only contributes *content*: a slug per perspective and a
system prompt that gives that perspective a coherent voice.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentInvitation:
    """One persona the SLM has picked for the slate.

    ``slug`` is a short, kebab-case identifier the factory turns into
    an :class:`AgentAddress` instance field. ``system`` is the persona
    prompt the chosen :class:`~ahp.adapters.prompts.Recipe` will slot
    into its ``system`` placeholder.
    """

    slug: str
    system: str


@runtime_checkable
class ChatModel(Protocol):
    """Minimal duck-typed interface the inviter calls into.

    LangChain ``BaseChatModel``-likes (including ``ChatBedrockConverse``)
    satisfy this via ``.invoke(prompt) -> response`` where ``response``
    has a ``.content`` attribute. The inviter accepts anything with
    ``.invoke`` so tests can hand it a fake.
    """

    def invoke(self, prompt: str) -> Any: ...


class Inviter:
    """Builds an agent slate by asking an SLM to enumerate perspectives.

    Parameters
    ----------
    model:
        Small language model used to generate the slate. Should be
        cheap and fast — Haiku-class.
    max_retries:
        How many times to retry if the model returns unparseable JSON.
        Each retry sends a tighter "JSON only, no prose" reminder.
    """

    def __init__(self, model: ChatModel, *, max_retries: int = 2) -> None:
        self._model = model
        self._max_retries = max_retries

    async def invite(
        self,
        *,
        domain: str,
        subdomain: str,
        topic: str,
        count: int,
        mode_hint: str | None = None,
    ) -> list[AgentInvitation]:
        """Ask the SLM for ``count`` perspectives on ``topic``.

        ``mode_hint`` is an optional plain-English nudge ("adversarial
        debate", "collaborative interview") that gives the model some
        sense of what these perspectives will be used for. It does not
        change the address shape or the dialog recipe.
        """
        if count <= 0:
            raise ValueError(f"count must be positive, got {count}")
        prompt = self._build_prompt(
            domain=domain, subdomain=subdomain, topic=topic,
            count=count, mode_hint=mode_hint,
        )
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                text = await self._call(prompt)
                slate = _parse_slate(text, expected=count)
                return slate
            except _ParseError as e:
                last_error = e
                prompt = self._tighten_prompt(prompt, error=str(e))
                log.warning(
                    "Inviter: SLM returned unparseable output on attempt %d/%d: %s",
                    attempt + 1, self._max_retries + 1, e,
                )
        assert last_error is not None
        raise RuntimeError(
            f"Inviter exhausted retries; last error: {last_error}"
        )

    async def _call(self, prompt: str) -> str:
        resp = await asyncio.to_thread(self._model.invoke, prompt)
        text = getattr(resp, "content", None)
        if text is None:
            text = str(resp)
        return text

    @staticmethod
    def _build_prompt(
        *,
        domain: str,
        subdomain: str,
        topic: str,
        count: int,
        mode_hint: str | None,
    ) -> str:
        usage = (
            f" They will participate in {mode_hint}." if mode_hint else ""
        )
        return (
            f"You are curating a panel of {count} experts from the field of "
            f"{domain} / {subdomain}.{usage}\n\n"
            f"TOPIC: {topic}\n\n"
            f"Return exactly {count} distinct, well-known perspectives a "
            f"real community of {subdomain} specialists holds on this topic. "
            f"Each perspective should be substantive and clearly differ from "
            f"the others.\n\n"
            f"Output STRICT JSON only (no prose, no markdown fences). Schema:\n"
            f'{{\n'
            f'  "agents": [\n'
            f'    {{"slug": "kebab-case-short-label",\n'
            f'     "system": "First-person system prompt that puts the model '
            f'in this perspective. 1-2 sentences. Start with \\"You hold ...\\" '
            f'or \\"You argue ...\\"."}}\n'
            f'  ]\n'
            f'}}\n\n'
            f"Constraints: slugs must be unique, lowercase, kebab-case, "
            f"<= 24 chars. system prompts must not include the topic verbatim; "
            f"they describe the *stance*, not the question."
        )

    @staticmethod
    def _tighten_prompt(prior: str, *, error: str) -> str:
        return (
            prior
            + "\n\nYour previous reply could not be parsed: "
            + error
            + "\nReturn STRICT JSON only. No prose. No code fences. No commentary."
        )


# ── parsing ────────────────────────────────────────────────────────────


class _ParseError(Exception):
    pass


_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,23}$")


def _parse_slate(text: str, *, expected: int) -> list[AgentInvitation]:
    raw = _extract_json(text)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise _ParseError(f"not valid JSON: {e.msg}") from e
    agents = data.get("agents") if isinstance(data, dict) else None
    if not isinstance(agents, list):
        raise _ParseError("missing or non-list 'agents' field")
    if len(agents) != expected:
        raise _ParseError(
            f"expected {expected} agents, got {len(agents)}"
        )
    seen: set[str] = set()
    out: list[AgentInvitation] = []
    for i, item in enumerate(agents):
        if not isinstance(item, dict):
            raise _ParseError(f"agent[{i}] is not an object")
        slug = item.get("slug")
        system = item.get("system")
        if not isinstance(slug, str) or not _SLUG_RE.match(slug):
            raise _ParseError(
                f"agent[{i}] slug {slug!r} must match {_SLUG_RE.pattern}"
            )
        if slug in seen:
            raise _ParseError(f"duplicate slug {slug!r}")
        seen.add(slug)
        if not isinstance(system, str) or not system.strip():
            raise _ParseError(f"agent[{i}] system must be a non-empty string")
        out.append(AgentInvitation(slug=slug, system=system.strip()))
    return out


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a model response.

    Handles three shapes:
    * raw JSON
    * fenced ```json ... ``` blocks
    * JSON embedded in prose (first ``{`` to matching ``}``)
    """
    text = text.strip()
    if text.startswith("{"):
        return text
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # Greedy first-{ to last-} fallback.
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last > first:
        return text[first : last + 1]
    raise _ParseError("no JSON object found in response")
