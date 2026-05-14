"""AgentFactory — address → agent construction.

The factory holds ``(AddressPattern, builder)`` registrations.
``create(address)`` finds the first matching builder and invokes it.
``spawn(provisioning_spec)`` materializes a :class:`ProvisioningPattern`
and builds one agent per resulting address — consulting the registry for
any field whose syntax does not include a dash (the reuse-then-top-up
default).

The factory is optional. Agents can always be constructed directly.
It's useful when:

* Multiple implementations exist for the same address pattern (e.g.
  cheap vs. expensive variant of the same role).
* You want bulk provisioning (``4*.adversarial.finance.2*.s.session.*``).
* You want the system to reuse already-live agents instead of cloning
  them on every spawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ahp.adapters.base import AHPAgent
from ahp.adapters.capability import AgentProfile, CapabilityRegistry
from ahp.adapters.provisioning import (
    FieldNamer,
    ProvisioningPattern,
    default_namer,
)
from ahp.adapters.resources import ResourceRegistry
from ahp.adapters.tool_registry import ToolRegistry
from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern
from ahp.engine.router import ProtocolEngine


Builder = Callable[[AgentAddress, ProtocolEngine, AgentProfile], AHPAgent]
"""``builder(address, engine, profile) -> AHPAgent`` (sync construction).

The ``profile`` is the merged :class:`AgentProfile` produced by the
factory's :class:`CapabilityRegistry` for that address. Builders may
ignore it (e.g. simple test agents) or consume it to wire tools,
prompts, and agent-kind into the constructed instance.
"""


@dataclass(frozen=True)
class _Registration:
    pattern: AddressPattern
    builder: Builder
    priority: int


@dataclass
class SpawnResult:
    """Outcome of :meth:`AgentFactory.spawn`.

    ``new`` agents were constructed because their address wasn't already
    alive in the registry. ``reused`` are addresses that ARE alive — no
    builder ran, callers can address them as usual.
    """

    new: list[AHPAgent] = field(default_factory=list)
    reused: list[AgentAddress] = field(default_factory=list)

    @property
    def all_addresses(self) -> list[AgentAddress]:
        return [a.address for a in self.new] + list(self.reused)

    def __len__(self) -> int:
        return len(self.new) + len(self.reused)


class AgentFactory:
    """Pattern-keyed registry of agent builders.

    Builders receive an :class:`AgentProfile` derived from the optional
    :class:`CapabilityRegistry`. If no capability registry is supplied,
    every builder gets an empty profile keyed to the agent's address.
    """

    def __init__(
        self,
        engine: ProtocolEngine,
        capabilities: CapabilityRegistry | None = None,
        tools: ToolRegistry | None = None,
        resources: ResourceRegistry | None = None,
    ) -> None:
        self._engine = engine
        self._regs: list[_Registration] = []
        self._capabilities = capabilities or CapabilityRegistry()
        self._tools = tools or ToolRegistry()
        self._resources = resources or ResourceRegistry()

    @property
    def capabilities(self) -> CapabilityRegistry:
        return self._capabilities

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def resources(self) -> ResourceRegistry:
        return self._resources

    @property
    def engine(self) -> ProtocolEngine:
        return self._engine

    def profile_for(self, address: AgentAddress | str) -> AgentProfile:
        """Build the merged :class:`AgentProfile` for ``address``.

        Combines:

        * Capability-registry contributions (inline tools, skills,
          prompt fragments, agent_kind).
        * Tool-registry tools whose ``allowed_for`` pattern matches.
        * Resource-registry resources whose ``allowed_for`` pattern
          matches (lazily constructed on first access).
        """
        addr = (
            address if isinstance(address, AgentAddress)
            else AgentAddress.parse(address)
        )
        base = self._capabilities.resolve(addr)
        extra_tools = self._tools.for_address(addr)
        if extra_tools:
            base.tools = tuple(base.tools) + tuple(extra_tools)
        base.resources = self._resources.for_address(addr)
        return base

    # ── registration ────────────────────────────────────────────────────

    def register(
        self,
        pattern: AddressPattern | str,
        builder: Builder,
        *,
        priority: int = 0,
    ) -> None:
        """Bind ``builder`` to any address matching ``pattern``.

        Higher priority wins. Ties broken by registration order.
        """
        if isinstance(pattern, str):
            pattern = AddressPattern.parse(pattern)
        self._regs.append(_Registration(pattern, builder, priority))
        self._regs.sort(key=lambda r: -r.priority)

    def unregister_all(self) -> None:
        self._regs.clear()

    def registrations(self) -> list[tuple[AddressPattern, Builder, int]]:
        return [(r.pattern, r.builder, r.priority) for r in self._regs]

    # ── single-address lookup ───────────────────────────────────────────

    def can_create(self, address: AgentAddress | str) -> bool:
        addr = (
            address if isinstance(address, AgentAddress)
            else AgentAddress.parse(address)
        )
        return any(r.pattern.matches(addr) for r in self._regs)

    def create(self, address: AgentAddress | str) -> AHPAgent:
        """Build the agent for ``address`` using the first matching builder.

        The builder receives the merged :class:`AgentProfile` (capability
        registry + tool registry + resource registry) as its third
        argument.
        """
        addr = (
            address if isinstance(address, AgentAddress)
            else AgentAddress.parse(address)
        )
        profile = self.profile_for(addr)
        for reg in self._regs:
            if reg.pattern.matches(addr):
                return reg.builder(addr, self._engine, profile)
        raise LookupError(
            f"no registered builder matches {addr}; "
            f"registered patterns: {[str(r.pattern) for r in self._regs]}"
        )

    # ── bulk provisioning ──────────────────────────────────────────────

    async def spawn(
        self,
        spec: ProvisioningPattern | str,
        *,
        namer: FieldNamer = default_namer,
    ) -> SpawnResult:
        """Materialize a :class:`ProvisioningPattern` and build the *new* agents.

        Reuse-mode fields (no ``-``) consult the registry; matching alive
        agents are recorded in ``SpawnResult.reused`` and no builder runs
        for them. Fresh-mode fields (``-``) always spawn new.
        """
        pattern = (
            spec
            if isinstance(spec, ProvisioningPattern)
            else ProvisioningPattern.parse(spec)
        )
        addresses = await pattern.materialize_async(
            registry=self._engine.registry, namer=namer,
        )
        result = SpawnResult()
        for addr in addresses:
            if await self._engine.registry.is_alive(addr):
                result.reused.append(addr)
            else:
                result.new.append(self.create(addr))
        return result

    async def spawn_and_start(
        self,
        spec: ProvisioningPattern | str,
        *,
        namer: FieldNamer = default_namer,
    ) -> SpawnResult:
        """Spawn, then ``register()`` and ``start()`` each newly built agent."""
        result = await self.spawn(spec, namer=namer)
        for agent in result.new:
            await agent.register()
        for agent in result.new:
            await agent.start()
        return result

    def spawn_fresh(
        self,
        spec: ProvisioningPattern | str,
        *,
        namer: FieldNamer = default_namer,
    ) -> list[AHPAgent]:
        """Synchronous, registry-free spawn — equivalent to all fields using ``-``.

        Useful in unit tests where there's no registry, or when you
        explicitly don't want reuse semantics.
        """
        pattern = (
            spec
            if isinstance(spec, ProvisioningPattern)
            else ProvisioningPattern.parse(spec)
        )
        return [self.create(addr) for addr in pattern.materialize(namer=namer)]
