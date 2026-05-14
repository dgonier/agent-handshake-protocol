"""Capability resolution — address fields → agent configuration.

An :class:`AgentProfile` is the bundle of *what* an agent does: its
tools, skills, RAG sources, system prompt, and a hint about its
implementation kind (ReAct, deep, custom). The :class:`CapabilityRegistry`
maps address patterns to fragments of this profile and merges all
matching fragments into a single resolved profile for any given
address.

Design intent:

* ``domain`` and ``subdomain`` are the dominant axes for tool/skill/RAG
  selection (you register tools "for any finance agent").
* ``role`` partially determines the agent kind (e.g. interview/
  adversarial are typically ReAct loops; some planners are deep).
* Multiple fragments compose: lists concatenate, prompts join with a
  blank line, ``agent_kind`` follows the highest-priority provider.

The protocol layer is intentionally agnostic about what a Tool actually
*does* — we model it as a name + callable. Adapters (LangGraph, DSPy)
translate to their respective tool/skill primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Awaitable, Callable, Iterable, Literal, Mapping, Optional

from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern


AgentKind = Literal["react", "deep", "custom"]
"""How the agent is structured.

``react`` — a single ReAct loop (think → act → observe) over tools.
``deep`` — multi-step planner that may itself call other AHP agents.
``custom`` — caller supplies the implementation; profile is informational.
"""


@dataclass(frozen=True)
class Tool:
    """A named callable an agent can invoke.

    Adapters convert this into framework-specific tool types
    (LangChain ``BaseTool``, DSPy retriever, etc.). We model it
    minimally so the protocol layer stays neutral.
    """

    name: str
    description: str
    handler: Callable[..., Any]
    schema: dict | None = None
    """Optional JSON Schema for the handler's inputs."""


@dataclass(frozen=True)
class Skill:
    """A named bundle of tools + prompt fragment.

    Skills group related tools and provide a snippet the system prompt
    composer can use to describe the capability to the model.
    """

    name: str
    description: str
    tools: tuple[Tool, ...] = ()
    prompt_fragment: str = ""


@dataclass(frozen=True)
class RagSource:
    """A retrieval source for an agent.

    ``retrieve`` returns a list of relevant text snippets for a query.
    Used by adapters to attach a retrieval step into the agent's loop.
    """

    name: str
    retrieve: Callable[[str], Awaitable[list[str]]]
    description: str = ""


@dataclass
class AgentProfile:
    """Resolved configuration for an agent at a specific address.

    ``tools`` and ``skills`` come from the capability registry (inline)
    plus the tool registry (address-keyed). ``resources`` is a
    ``{name: instance}`` map of lazy-constructed shared objects pulled
    from the resource registry — agents grab a DB client / vector
    store / FS backend by name and use it inside their tool handlers.
    """

    address: AgentAddress
    tools: tuple[Tool, ...] = field(default_factory=tuple)
    skills: tuple[Skill, ...] = field(default_factory=tuple)
    rag_sources: tuple[RagSource, ...] = field(default_factory=tuple)
    prompt: str = ""
    agent_kind: AgentKind = "react"
    resources: dict[str, Any] = field(default_factory=dict)

    @property
    def all_tools(self) -> tuple[Tool, ...]:
        """Standalone tools + every tool nested inside a skill."""
        result: list[Tool] = list(self.tools)
        for skill in self.skills:
            result.extend(skill.tools)
        return tuple(result)


@dataclass
class CapabilityProvider:
    """A pattern-bound contribution to an :class:`AgentProfile`."""

    pattern: AddressPattern
    priority: int = 0
    tools: tuple[Tool, ...] = ()
    skills: tuple[Skill, ...] = ()
    rag_sources: tuple[RagSource, ...] = ()
    prompt: str = ""
    agent_kind: AgentKind | None = None


class CapabilityRegistry:
    """Pattern-keyed providers of profile fragments.

    Multiple matching providers are merged. Higher-priority providers
    are processed first; the first non-None ``agent_kind`` wins, while
    tools/skills/rag-sources concatenate (in priority order) and
    prompts join with a blank line.
    """

    def __init__(self) -> None:
        self._providers: list[CapabilityProvider] = []

    # ── registration ────────────────────────────────────────────────────

    def register(
        self,
        pattern: AddressPattern | str,
        *,
        priority: int = 0,
        tools: Iterable[Tool] = (),
        skills: Iterable[Skill] = (),
        rag_sources: Iterable[RagSource] = (),
        prompt: str = "",
        agent_kind: AgentKind | None = None,
    ) -> CapabilityProvider:
        if isinstance(pattern, str):
            pattern = AddressPattern.parse(pattern)
        provider = CapabilityProvider(
            pattern=pattern,
            priority=priority,
            tools=tuple(tools),
            skills=tuple(skills),
            rag_sources=tuple(rag_sources),
            prompt=prompt,
            agent_kind=agent_kind,
        )
        self._providers.append(provider)
        # Stable sort by descending priority.
        self._providers.sort(key=lambda p: -p.priority)
        return provider

    def providers(self) -> list[CapabilityProvider]:
        return list(self._providers)

    def clear(self) -> None:
        self._providers.clear()

    # ── resolution ──────────────────────────────────────────────────────

    def resolve(self, address: AgentAddress) -> AgentProfile:
        tools: list[Tool] = []
        skills: list[Skill] = []
        rag_sources: list[RagSource] = []
        prompt_parts: list[str] = []
        agent_kind: AgentKind | None = None

        for provider in self._providers:
            if not provider.pattern.matches(address):
                continue
            tools.extend(provider.tools)
            skills.extend(provider.skills)
            rag_sources.extend(provider.rag_sources)
            if provider.prompt:
                prompt_parts.append(provider.prompt)
            if agent_kind is None and provider.agent_kind is not None:
                agent_kind = provider.agent_kind

        return AgentProfile(
            address=address,
            tools=tuple(tools),
            skills=tuple(skills),
            rag_sources=tuple(rag_sources),
            prompt="\n\n".join(prompt_parts),
            agent_kind=agent_kind or "react",
        )

    def __len__(self) -> int:
        return len(self._providers)
