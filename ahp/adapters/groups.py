"""Named pattern aliases — broadcast to a group by simple string.

A *group* is a human-friendly name for an :class:`AddressPattern`. The
registry lets callers write::

    groups.register("debaters", "*.adversarial.*.*.*.*.*")

and then broadcast or subscribe by name::

    replies = await alice.broadcast_to(
        "debaters", code=Code.ADVERSARIAL_DEBATE, body="argue Tesla",
    )

Resolution accepts either a registered group name or a raw 7-field
pattern, in that order. Strings that look like patterns (have at
least one ``.``) are parsed directly when no group of that name
exists, so callers can mix declarative aliases and ad-hoc patterns in
the same call site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ahp.core.pattern import AddressPattern


@dataclass(frozen=True)
class Group:
    name: str
    pattern: AddressPattern
    description: str = ""


class GroupRegistry:
    """Name → :class:`AddressPattern` lookup for broadcast targets."""

    def __init__(self) -> None:
        self._groups: dict[str, Group] = {}

    # ── registration ───────────────────────────────────────────────────

    def register(
        self,
        name: str,
        pattern: AddressPattern | str,
        *,
        description: str = "",
    ) -> Group:
        if not name:
            raise ValueError("group name must be non-empty")
        if "." in name:
            raise ValueError(
                f"group name {name!r} must not contain '.' — that's "
                f"reserved for pattern syntax"
            )
        if name in self._groups:
            raise ValueError(f"group already registered: {name!r}")
        if isinstance(pattern, str):
            pattern = AddressPattern.parse(pattern)
        group = Group(name=name, pattern=pattern, description=description)
        self._groups[name] = group
        return group

    def unregister(self, name: str) -> bool:
        return self._groups.pop(name, None) is not None

    def names(self) -> list[str]:
        return list(self._groups.keys())

    def __len__(self) -> int:
        return len(self._groups)

    def __contains__(self, name: str) -> bool:
        return name in self._groups

    def groups(self) -> Iterable[Group]:
        return self._groups.values()

    # ── resolution ─────────────────────────────────────────────────────

    def get(self, name: str) -> Group:
        if name not in self._groups:
            raise KeyError(name)
        return self._groups[name]

    def resolve(self, name_or_pattern: str | AddressPattern) -> AddressPattern:
        """Return the :class:`AddressPattern` for a group name, or parse as pattern.

        Order of resolution:

        1. Already an :class:`AddressPattern` → returned as-is.
        2. Registered group name → its pattern.
        3. Otherwise parse the string as a raw 7-field pattern.
        """
        if isinstance(name_or_pattern, AddressPattern):
            return name_or_pattern
        if name_or_pattern in self._groups:
            return self._groups[name_or_pattern].pattern
        return AddressPattern.parse(name_or_pattern)


DEFAULT_GROUP_REGISTRY = GroupRegistry()
"""Process-wide default. Module-level :func:`group` registers here."""


def group(
    name: str,
    pattern: AddressPattern | str,
    *,
    description: str = "",
) -> Group:
    """Module-level helper that registers into :data:`DEFAULT_GROUP_REGISTRY`."""
    return DEFAULT_GROUP_REGISTRY.register(name, pattern, description=description)
