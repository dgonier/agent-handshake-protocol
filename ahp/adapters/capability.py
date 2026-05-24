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
from ahp.adapters.tool_address import ResourceAddress, ToolAddress


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
    """An executable workflow playbook with suggested-address bundles.

    A skill is more than a capability tag — it's a description of a
    *workflow* the agent can run, plus the addresses of the things
    that workflow expects to coordinate:

    * ``tools`` — concrete callables stitched into the workflow (the
      original "bundle of tools" use case).
    * ``prompt_fragment`` — system-prompt snippet describing the
      skill to the model.
    * ``when_to_use`` — natural-language guidance on when this skill
      applies (used by the model to decide whether to invoke it).
    * ``graph`` — a compiled LangGraph DAG (or any callable graph).
      ``Any`` rather than a tight type so this module stays
      framework-agnostic; production skills ship a
      :class:`langgraph.graph.CompiledStateGraph`. The graph IS the
      executable workflow; everything else is metadata or
      suggested-context for the agent that runs it.
    * ``suggested_tools`` — tool addresses the workflow is built
      around. Distinct from ``tools``: ``tools`` is the inline
      callables the skill bundles; ``suggested_tools`` is the broader
      address space the skill expects to operate over (and other
      orgs' tools by address too).
    * ``suggested_specialists`` — agent addresses (or patterns) the
      workflow may consult — specialists with persistent memory and
      LoRAs that contribute domain expertise.
    * ``suggested_loras`` — LoRA resource addresses an agent
      executing this skill should compose into its model.
    * ``suggested_information_sources`` — resource/agent addresses of
      databases, vector stores, document corpora, knowledge bases
      the workflow reads from or writes to. Same pattern as agent
      accept tiers — info sources are addressable, tier-declaring,
      and billable per query.

    The four ``suggested_*`` lists are advisory: a sophisticated
    agent could ignore them and pick differently, but they document
    the canonical execution recipe and let the broker pre-check that
    the addresses are reachable.
    """

    name: str
    description: str
    tools: tuple[Tool, ...] = ()
    prompt_fragment: str = ""

    # ── workflow + suggested addresses ──────────────────────────────
    when_to_use: str = ""
    graph: Any = None
    """A compiled workflow graph (typically LangGraph). The skill IS
    this graph; everything else is metadata describing how to run it."""

    suggested_tools: tuple[ToolAddress, ...] = ()
    suggested_specialists: tuple[AgentAddress | AddressPattern, ...] = ()
    suggested_loras: tuple[ResourceAddress, ...] = ()
    suggested_information_sources: tuple[
        ResourceAddress | AgentAddress | AddressPattern, ...
    ] = ()


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
