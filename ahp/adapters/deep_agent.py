"""DeepAgent — LLM-driven planner with subagents + virtual filesystem.

Wraps :func:`deepagents.create_deep_agent` so an :class:`AgentProfile`
+ a chat model produces a fully-orchestrated deep agent. The deep
agent gets:

* ``profile.tools``               → tools the planner can invoke
* ``profile.skills``              → ``SubAgent`` entries (each skill
  becomes a delegate the planner can hand work off to)
* ``profile.prompt``              → top-level system prompt

Subagents are not addressable from the AHP protocol — they live inside
the deep agent and are LLM-orchestrated. To call OTHER AHP agents from
inside a deep agent, expose those calls as ordinary tools via
``extra_tools`` (a closure over the engine + this agent's address).
See ``ahp/demo/finance_react.py`` for the canonical example.

The state shape returned by ``create_deep_agent`` is the same
``{"messages": [...]}`` LangChain agent state used by
:class:`ReactAgent`, so the input/output mappers are shared.
"""

from __future__ import annotations

from typing import Any, Iterable

from deepagents import SubAgent, create_deep_agent

from ahp.adapters.capability import AgentProfile, Skill, Tool
from ahp.adapters.langgraph_agent import LangGraphAgent
from ahp.adapters.react_agent import (
    _react_input_mapper,
    _react_output_mapper_for,
    _to_langchain_tool,
)
from ahp.adapters.resources import ResourceRegistry
from ahp.adapters.storage import build_fs_backend, fs_mount_description
from ahp.core.address import AgentAddress
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


def _skill_to_subagent(skill: Skill) -> SubAgent:
    """Map an AHP :class:`Skill` to a ``deepagents.SubAgent`` TypedDict."""
    subagent: SubAgent = {
        "name": skill.name,
        "description": skill.description,
        "system_prompt": skill.prompt_fragment or skill.description,
    }
    if skill.tools:
        subagent["tools"] = [_to_langchain_tool(t) for t in skill.tools]
    return subagent


class DeepAgent(LangGraphAgent):
    """LLM-driven deep agent (planner + subagents) backed by ``deepagents``.

    Usage::

        agent = DeepAgent.from_profile(
            address, engine, profile, model=bedrock_chat_model(),
            extra_tools=[ahp_callout_tool],  # tools that reach into AHP
        )
        await agent.register(); await agent.start()
    """

    @classmethod
    def from_profile(
        cls,
        address: AgentAddress,
        engine: ProtocolEngine,
        profile: AgentProfile,
        model: Any,
        *,
        metadata: AgentMeta | None = None,
        extra_tools: Iterable[Tool] | None = None,
        extra_subagents: Iterable[SubAgent] | None = None,
        fs_resources: ResourceRegistry | None = None,
        **kwargs: Any,
    ) -> "DeepAgent":
        """Build a DeepAgent from a profile.

        ``fs_resources`` (optional): a :class:`ResourceRegistry`. Any
        resource registered with ``kind="fs"`` whose ``allowed_for``
        matches ``address`` is mounted into the agent's virtual
        filesystem (default mount path ``/<name>/``). The system prompt
        is appended with a list of available mounts so the LLM knows
        where to read/write. Pass ``factory.resources`` for the common
        case.
        """
        tools_in: list[Tool] = list(profile.tools)
        if extra_tools:
            tools_in.extend(extra_tools)
        lc_tools = [_to_langchain_tool(t) for t in tools_in]

        subagents: list[SubAgent] = [_skill_to_subagent(s) for s in profile.skills]
        if extra_subagents:
            subagents.extend(extra_subagents)

        backend = None
        prompt = profile.prompt or ""
        if fs_resources is not None:
            backend = build_fs_backend(fs_resources, address)
            mount_desc = fs_mount_description(fs_resources, address)
            if mount_desc:
                prompt = (prompt + "\n\n" if prompt else "") + mount_desc

        graph = create_deep_agent(
            model=model,
            tools=lc_tools,
            system_prompt=prompt or None,
            subagents=subagents or None,
            backend=backend,
        )
        return cls(
            address=address,
            engine=engine,
            graph=graph,
            input_mapper=_react_input_mapper,
            output_mapper=_react_output_mapper_for(address),
            metadata=metadata,
            **kwargs,
        )
