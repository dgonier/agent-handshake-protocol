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

import logging
from dataclasses import dataclass, field
from typing import Callable

from ahp.adapters.base import AHPAgent
from ahp.adapters.capability import AgentProfile, CapabilityRegistry
from ahp.adapters.groups import GroupRegistry
from ahp.adapters.inviter import AgentInvitation, ChatModel, Inviter
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
from ahp.engine.scope import ScopePolicy


log = logging.getLogger(__name__)


# Re-exported from ahp.adapters.errors so `ahp.adapters` users see these
# without an extra import path.
from ahp.adapters.errors import (    # noqa: E402,F401
    ResolutionConflictError,
    ResourceNameCollisionError,
    ToolNameCollisionError,
)


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
        groups: GroupRegistry | None = None,
        scope: ScopePolicy | None = None,
        slm: ChatModel | None = None,
    ) -> None:
        self._engine = engine
        self._regs: list[_Registration] = []
        # Optional SLM used by :meth:`invite` to generate per-query
        # persona slates. Builders never see this; they read the
        # resulting persona via :meth:`persona_for`.
        self._slm = slm
        self._inviter: Inviter | None = Inviter(slm) if slm is not None else None
        # Address → persona system prompt, populated by invite().
        self._personas: dict[str, str] = {}
        # Use explicit None checks rather than ``x or DefaultFactory()`` —
        # all the registries define ``__len__`` so an empty user-supplied
        # instance would be falsy and silently replaced with a fresh one.
        self._capabilities = (
            capabilities if capabilities is not None else CapabilityRegistry()
        )
        self._tools = tools if tools is not None else ToolRegistry()
        self._resources = (
            resources if resources is not None else ResourceRegistry()
        )
        self._groups = groups if groups is not None else GroupRegistry()
        self._scope = scope
        # Expose the group registry through the engine so adapters can
        # resolve a group name without needing a factory reference.
        # Warn loudly when overwriting non-default state — typical in
        # tests, dangerous in apps that wire two factories on one engine.
        prior_groups = getattr(engine, "groups", None)
        if prior_groups is not None and prior_groups is not self._groups:
            log.warning(
                "AgentFactory: engine.groups is being overwritten by this "
                "factory (was %r, now %r). Two factories sharing an engine "
                "will fight over group resolution.",
                prior_groups, self._groups,
            )
        engine.groups = self._groups
        # Same for the (optional) scope policy. None = open default.
        if scope is not None:
            prior_scope = getattr(engine, "scope", None)
            if prior_scope is not None and prior_scope is not scope:
                log.warning(
                    "AgentFactory: engine.scope is being overwritten by this "
                    "factory (was %r, now %r). Scope policies don't compose "
                    "across factories — pick one source of truth.",
                    prior_scope, scope,
                )
            engine.scope = scope

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
    def groups(self) -> GroupRegistry:
        return self._groups

    @property
    def scope(self) -> ScopePolicy | None:
        return self._scope

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

        Raises :class:`ToolNameCollisionError` when two tools at
        different :class:`ToolAddress`-es share an operation name
        (LangChain can't disambiguate) and
        :class:`ResourceNameCollisionError` when two resources share
        a name (the profile dict would silently overwrite).
        """
        addr = (
            address if isinstance(address, AgentAddress)
            else AgentAddress.parse(address)
        )
        base = self._capabilities.resolve(addr)

        # ── tool-registry contribution + collision detection ──────────
        registry_bindings = self._tools.bindings_for_address(addr)
        merged_tools = list(base.tools)
        # Track the source (ToolAddress or "inline capability") of each
        # short name so error messages are concrete.
        provenance: dict[str, str] = {
            t.name: "inline capability provider" for t in base.tools
        }
        for binding in registry_bindings:
            short = binding.tool.name
            if short in provenance and provenance[short] != str(binding.address):
                raise ToolNameCollisionError(
                    f"two tools claim the short name {short!r} for agent "
                    f"{addr}: {provenance[short]} and {binding.address}. "
                    f"Rename one or tighten its allowed_for so they don't "
                    f"both apply to this agent."
                )
            if short not in provenance:
                provenance[short] = str(binding.address)
                merged_tools.append(binding.tool)
        base.tools = tuple(merged_tools)

        # ── resource-registry contribution (collisions raised inside) ──
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

    # ── SLM-driven invitation ──────────────────────────────────────────

    @property
    def slm(self) -> ChatModel | None:
        return self._slm

    def set_slm(self, model: ChatModel) -> None:
        """Attach an SLM after construction. Rebuilds the inviter."""
        self._slm = model
        self._inviter = Inviter(model)

    def persona_for(self, address: AgentAddress) -> str | None:
        """Return the persona system prompt set by :meth:`invite`, if any."""
        return self._personas.get(str(address))

    def invitations(self) -> dict[str, str]:
        """All persona system prompts keyed by address (debug helper)."""
        return dict(self._personas)

    async def invite(
        self,
        *,
        org: str,
        role: str,
        domain: str,
        subdomain: str,
        topic: str,
        count: int,
        accept: str = "s",
        lifecycle: str = "session",
        mode_hint: str | None = None,
    ) -> SpawnResult:
        """Use the configured SLM to populate a slate of agents for ``topic``.

        Asks the SLM for ``count`` perspectives a community in
        ``domain/subdomain`` would hold on ``topic``. Materializes one
        agent per perspective at::

            {org}.{role}.{domain}.{subdomain}.{accept}.{lifecycle}.{slug}

        The per-agent persona system prompt is stored on the factory
        and retrievable via :meth:`persona_for`. Builders typically
        read it during construction (see the live demo).

        Returns the standard :class:`SpawnResult` so callers can
        ``register()`` / ``start()`` themselves, or use
        :meth:`invite_and_start` for the one-shot version.
        """
        if self._inviter is None:
            raise RuntimeError(
                "AgentFactory.invite requires an SLM; pass slm=... to "
                "AgentFactory(...) or call set_slm() first."
            )
        invitations = await self._inviter.invite(
            domain=domain, subdomain=subdomain, topic=topic,
            count=count, mode_hint=mode_hint,
        )
        result = SpawnResult()
        for inv in invitations:
            address = AgentAddress(
                org=org, role=role, domain=domain, subdomain=subdomain,
                accept=accept, lifecycle=lifecycle, instance=inv.slug,
            )
            self._personas[str(address)] = inv.system
            if await self._engine.registry.is_alive(address):
                result.reused.append(address)
            else:
                result.new.append(self.create(address))
        return result

    async def invite_and_start(
        self,
        *,
        org: str,
        role: str,
        domain: str,
        subdomain: str,
        topic: str,
        count: int,
        accept: str = "s",
        lifecycle: str = "session",
        mode_hint: str | None = None,
    ) -> SpawnResult:
        """:meth:`invite` + ``register()`` + ``start()`` for each new agent."""
        result = await self.invite(
            org=org, role=role, domain=domain, subdomain=subdomain,
            topic=topic, count=count, accept=accept, lifecycle=lifecycle,
            mode_hint=mode_hint,
        )
        for agent in result.new:
            await agent.register()
        for agent in result.new:
            await agent.start()
        return result
