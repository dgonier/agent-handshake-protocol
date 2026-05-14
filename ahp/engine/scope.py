"""Address-based access control with open-default semantics.

The protocol routes any source to any target by default — no central
authority gates registration or delivery. :class:`ScopePolicy` adds
opt-in restrictions on top of that:

* If **no rule** covers a target, the target stays open.
* Once **at least one rule** covers a target, only sources matching
  one of those rules' ``allow_sources`` patterns may reach it.
* Rules with the same target pattern UNION (any matching source
  passes), so you can loosen a restriction by adding more rules.

Use the address fields to scope progressively:

::

    # Loose: only tifin agents can reach tifin's DB plane
    scope.restrict(target="tifin.*.*.*.*.*.*",
                   allow_sources="tifin.*.*.*.*.*.*")

    # Tighter: only adversarial agents touch the adversarial DB
    scope.restrict(target="tifin.adversarial.*.*.*.*.*",
                   allow_sources="tifin.adversarial.*.*.*.*.*")

    # Tightest: only the finance team touches finance DB rows
    scope.restrict(target="tifin.adversarial.finance.*.*.*.*",
                   allow_sources="tifin.adversarial.finance.*.*.*.*")

Each rule can also constrain by interaction-code glob (``code=
"adversarial.*"``) so different verbs against the same target can have
different access policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.pattern import AddressPattern


@dataclass(frozen=True)
class ScopeRule:
    """One allow rule. Target + source patterns + optional code glob."""

    target: AddressPattern
    allow_sources: AddressPattern
    code: str = "*"
    description: str = ""


class ScopePolicy:
    """Open-default access control. Adding rules tightens. Empty = open."""

    def __init__(self) -> None:
        self._rules: list[ScopeRule] = []

    # ── registration ───────────────────────────────────────────────────

    def restrict(
        self,
        target: AddressPattern | str,
        allow_sources: AddressPattern | str,
        *,
        code: str = "*",
        description: str = "",
    ) -> ScopeRule:
        """Add an allow rule.

        After this call, any source NOT matching ``allow_sources`` (and
        no other rule's allow_sources for the same target) is blocked
        from reaching matching targets for matching codes.
        """
        if isinstance(target, str):
            target = AddressPattern.parse(target)
        if isinstance(allow_sources, str):
            allow_sources = AddressPattern.parse(allow_sources)
        rule = ScopeRule(
            target=target,
            allow_sources=allow_sources,
            code=code,
            description=description,
        )
        self._rules.append(rule)
        return rule

    def clear(self) -> None:
        self._rules.clear()

    def rules(self) -> Iterable[ScopeRule]:
        return tuple(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    # ── policy decision ────────────────────────────────────────────────

    def is_allowed(
        self,
        source: AgentAddress,
        target: AgentAddress,
        code: str,
    ) -> bool:
        """``True`` iff ``source`` may reach ``target`` for ``code``."""
        covering = [
            rule for rule in self._rules
            if rule.target.matches(target) and Code.matches(code, rule.code)
        ]
        if not covering:
            return True   # nothing covers this target — open
        return any(rule.allow_sources.matches(source) for rule in covering)

    def filter_targets(
        self,
        source: AgentAddress,
        targets: Iterable[AgentAddress],
        code: str,
    ) -> list[AgentAddress]:
        """Return the subset of ``targets`` reachable by ``source`` for ``code``."""
        return [t for t in targets if self.is_allowed(source, t, code)]
