"""Address-keyed skill catalog with a decorator API.

Mirrors :mod:`ahp.adapters.tool_registry` but for :class:`Skill`. A
skill is a workflow playbook with suggested addresses (tools,
specialists, LoRAs, info sources) — see :class:`ahp.adapters.Skill`
for the full data model.

Skills are addressable resources::

    {scope}.skill.{domain}.{subdomain}.{name}

The address pattern matches :class:`ResourceAddress` (kind="skill")
so discovery / access scope follows the same convention as every
other resource in the protocol.

Typical use::

    from ahp.adapters import DEFAULT_SKILL_REGISTRY, skill
    from ahp.adapters import Skill

    @skill("acme", "support", "refunds", name="refund-investigation")
    def make_refund_investigation():
        return Skill(
            name="refund-investigation",
            description="Investigate a refund request",
            when_to_use="Customer asks for a refund on an order",
            graph=_compile_my_langgraph(),  # CompiledStateGraph
            suggested_tools=(
                ToolAddress.parse("acme.api.*.orders.lookup_order"),
                ToolAddress.parse("acme.api.*.refunds.issue_refund"),
            ),
            suggested_specialists=(
                AgentAddress.parse(
                    "legalcorp.tos-reviewer.consumer.refunds.s.longterm.primary"
                ),
            ),
            suggested_information_sources=(
                ResourceAddress.parse("acme.data.policy.refunds"),
                ResourceAddress.parse("legalcorp.data.regulation.consumer-protection-us"),
            ),
        )

Default access scope (``allowed_for``) follows the same convention
as tools: a skill at ``{scope}.skill.{domain}.{subdomain}.{name}``
is visible by default to agents matching
``{scope}.*.{domain}.{subdomain}.*.*.*``. Override with
``allowed_for=`` when narrower targeting is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ahp.adapters.capability import Skill
from ahp.adapters.tool_address import ResourceAddress
from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern


SKILL_KIND: str = "skill"
"""``ResourceAddress.kind`` that marks a resource as a Skill."""


@dataclass(frozen=True)
class SkillBinding:
    """A registered skill: address + Skill object + access scope + tags."""

    address: ResourceAddress
    skill: Skill
    allowed_for: AddressPattern
    tags: frozenset[str]


class SkillRegistry:
    """Address-keyed skill catalog with pattern-based access scope.

    Same shape as :class:`ahp.adapters.tool_registry.ToolRegistry` —
    addresses are unique keys, ``allowed_for`` is an
    :class:`AddressPattern` deciding which agents see the skill, tags
    are free-form labels for filtering.
    """

    def __init__(self) -> None:
        self._bindings: dict[str, SkillBinding] = {}

    # ── registration ───────────────────────────────────────────────────

    def register(
        self,
        factory: Callable[[], Skill],
        scope: str,
        domain: str,
        subdomain: str,
        *,
        name: str | None = None,
        allowed_for: AddressPattern | str | None = None,
        tags: Iterable[str] = (),
    ) -> SkillBinding:
        """Register a skill factory at the given address.

        ``factory`` is a zero-arg callable returning a :class:`Skill`.
        Called once at registration time — the skill is materialized
        eagerly so failures surface immediately (matches the
        tool-registration shape).

        ``name`` defaults to the factory's ``__name__`` stripped of a
        ``make_`` prefix when present. ``allowed_for`` defaults to
        ``{scope}.*.{domain}.{subdomain}.*.*.*`` — the convention
        skills inherit from the tool / resource address layer.
        """
        skill_name = name or _strip_make_prefix(factory.__name__)
        address = ResourceAddress(
            scope=scope, kind=SKILL_KIND,
            domain=domain, subdomain=subdomain, name=skill_name,
        )
        key = str(address)
        if key in self._bindings:
            raise ValueError(f"skill already registered at {key!r}")

        if allowed_for is None:
            pattern = AddressPattern.parse(
                f"{scope}.*.{domain}.{subdomain}.*.*.*"
            )
        elif isinstance(allowed_for, str):
            pattern = AddressPattern.parse(allowed_for)
        else:
            pattern = allowed_for

        skill_obj = factory()
        if not isinstance(skill_obj, Skill):
            raise TypeError(
                f"skill factory for {key!r} returned {type(skill_obj).__name__}, "
                f"expected Skill"
            )

        binding = SkillBinding(
            address=address,
            skill=skill_obj,
            allowed_for=pattern,
            tags=frozenset(tags),
        )
        self._bindings[key] = binding
        return binding

    def skill(
        self,
        scope: str,
        domain: str,
        subdomain: str,
        *,
        name: str | None = None,
        allowed_for: AddressPattern | str | None = None,
        tags: Iterable[str] = (),
    ) -> Callable[[Callable[[], Skill]], Callable[[], Skill]]:
        """Decorator form of :meth:`register`. Returns the factory unchanged.

        Use on a zero-arg function that constructs and returns the
        :class:`Skill`::

            @skill("acme", "support", "refunds")
            def make_refund_investigation():
                return Skill(...)
        """

        def decorator(factory: Callable[[], Skill]) -> Callable[[], Skill]:
            self.register(
                factory, scope, domain, subdomain,
                name=name, allowed_for=allowed_for, tags=tags,
            )
            return factory

        return decorator

    def unregister(self, address: ResourceAddress | str) -> bool:
        return self._bindings.pop(str(address), None) is not None

    # ── lookup ─────────────────────────────────────────────────────────

    def get(self, address: ResourceAddress | str) -> Skill:
        binding = self._bindings.get(str(address))
        if binding is None:
            raise KeyError(address)
        return binding.skill

    def binding_at(self, address: ResourceAddress | str) -> SkillBinding:
        binding = self._bindings.get(str(address))
        if binding is None:
            raise KeyError(address)
        return binding

    def addresses(self) -> list[ResourceAddress]:
        return [b.address for b in self._bindings.values()]

    def __len__(self) -> int:
        return len(self._bindings)

    def __contains__(self, address: ResourceAddress | str) -> bool:
        return str(address) in self._bindings

    def bindings(self) -> Iterable[SkillBinding]:
        return self._bindings.values()

    def clear(self) -> None:
        self._bindings.clear()

    def bindings_for_address(
        self,
        agent_address: AgentAddress,
        *,
        tags: Iterable[str] | None = None,
    ) -> list[SkillBinding]:
        """Every binding visible to ``agent_address``, optionally
        filtered by tag (ANY-of)."""
        tag_set: set[str] | None = None
        if tags is not None:
            tag_set = set(tags)
        out: list[SkillBinding] = []
        for binding in self._bindings.values():
            if not binding.allowed_for.matches(agent_address):
                continue
            if tag_set is not None and not (tag_set & binding.tags):
                continue
            out.append(binding)
        return out

    def for_address(
        self,
        agent_address: AgentAddress,
        *,
        tags: Iterable[str] | None = None,
    ) -> list[Skill]:
        """Every :class:`Skill` visible to ``agent_address``."""
        return [b.skill for b in self.bindings_for_address(
            agent_address, tags=tags,
        )]


def _strip_make_prefix(name: str) -> str:
    """Drop a leading ``make_`` so ``make_refund_investigation`` becomes
    ``refund-investigation`` (also kebab-cases for the convention)."""
    base = name[len("make_"):] if name.startswith("make_") else name
    return base.replace("_", "-")


# ── module-level default ─────────────────────────────────────────────


DEFAULT_SKILL_REGISTRY: SkillRegistry = SkillRegistry()
"""Module-level default registry — the @skill decorator writes here."""


def skill(
    scope: str,
    domain: str,
    subdomain: str,
    *,
    name: str | None = None,
    allowed_for: AddressPattern | str | None = None,
    tags: Iterable[str] = (),
) -> Callable[[Callable[[], Skill]], Callable[[], Skill]]:
    """Module-level decorator that writes to :data:`DEFAULT_SKILL_REGISTRY`.

    Mirrors how :func:`ahp.adapters.tool` writes to
    :data:`DEFAULT_TOOL_REGISTRY` and :func:`ahp.adapters.resource`
    writes to :data:`DEFAULT_RESOURCE_REGISTRY`.
    """
    return DEFAULT_SKILL_REGISTRY.skill(
        scope, domain, subdomain,
        name=name, allowed_for=allowed_for, tags=tags,
    )
