"""Address-keyed tool catalog with a decorator API.

Every tool has a structured 5-field :class:`ToolAddress` that drives
discovery and access control. Tools are usually declared via the
:meth:`ToolRegistry.tool` decorator, which infers the ``operation``
field from the decorated function's name::

    from ahp.adapters import DEFAULT_TOOL_REGISTRY, tool

    @tool("tifin", "db", "adversarial", "crud")
    def update_record(table: str, row_id: str, fields: dict) -> dict:
        \"\"\"Update a row in the table.\"\"\"
        ...

    # → registered as ToolAddress("tifin", "db", "adversarial", "crud",
    #                              "update_record")
    # → default allowed_for: agents matching "tifin.adversarial.*.*.*.*.*"

Override the default access scope via ``allowed_for=``::

    @tool("tifin", "db", "*", "crud",
          allowed_for="*.adversarial.finance.*.*.*.*")
    def fetch(...): ...

The factory pulls every matching tool into the agent's profile at
build time.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ahp.adapters.capability import Tool
from ahp.adapters.tool_address import ToolAddress
from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern


@dataclass(frozen=True)
class ToolBinding:
    """A registered tool: address + Tool object + access scope + tags."""

    address: ToolAddress
    tool: Tool
    allowed_for: AddressPattern
    tags: frozenset[str]


class ToolRegistry:
    """Address-keyed tool catalog with pattern-based access scope."""

    def __init__(self) -> None:
        self._bindings: dict[str, ToolBinding] = {}

    # ── registration ───────────────────────────────────────────────────

    def register(
        self,
        func: Callable[..., Any],
        scope: str,
        kind: str,
        role: str,
        category: str,
        *,
        operation: str | None = None,
        description: str | None = None,
        allowed_for: AddressPattern | str | None = None,
        tags: Iterable[str] = (),
        schema: dict | None = None,
    ) -> ToolBinding:
        """Register a callable as a tool at the given address.

        ``operation`` defaults to ``func.__name__``. ``description``
        defaults to the function's docstring (first line). ``allowed_for``
        defaults to the convention derived from the tool address (see
        :meth:`ToolAddress.derived_allowed_for`).
        """
        op = operation or func.__name__
        address = ToolAddress(scope, kind, role, category, op)
        key = str(address)
        if key in self._bindings:
            raise ValueError(f"tool already registered at {key!r}")

        if allowed_for is None:
            pattern = address.derived_allowed_for()
        elif isinstance(allowed_for, str):
            pattern = AddressPattern.parse(allowed_for)
        else:
            pattern = allowed_for

        desc = description or (func.__doc__ or "").strip().splitlines()[0] if (description or func.__doc__) else op
        tool = Tool(
            name=op, description=desc, handler=func, schema=schema,
        )
        binding = ToolBinding(
            address=address,
            tool=tool,
            allowed_for=pattern,
            tags=frozenset(tags),
        )
        self._bindings[key] = binding
        return binding

    def tool(
        self,
        scope: str,
        kind: str,
        role: str,
        category: str,
        *,
        operation: str | None = None,
        description: str | None = None,
        allowed_for: AddressPattern | str | None = None,
        tags: Iterable[str] = (),
        schema: dict | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator form of :meth:`register`. Returns the function unchanged."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                func, scope, kind, role, category,
                operation=operation, description=description,
                allowed_for=allowed_for, tags=tags, schema=schema,
            )
            return func

        return decorator

    def unregister(self, address: ToolAddress | str) -> bool:
        key = str(address)
        return self._bindings.pop(key, None) is not None

    # ── lookup ─────────────────────────────────────────────────────────

    def get(self, address: ToolAddress | str) -> Tool:
        binding = self._bindings.get(str(address))
        if binding is None:
            raise KeyError(address)
        return binding.tool

    def binding_at(self, address: ToolAddress | str) -> ToolBinding:
        binding = self._bindings.get(str(address))
        if binding is None:
            raise KeyError(address)
        return binding

    def addresses(self) -> list[ToolAddress]:
        return [b.address for b in self._bindings.values()]

    def __len__(self) -> int:
        return len(self._bindings)

    def __contains__(self, address: ToolAddress | str) -> bool:
        return str(address) in self._bindings

    def bindings(self) -> Iterable[ToolBinding]:
        return self._bindings.values()

    def for_address(
        self,
        agent_address: AgentAddress,
        *,
        tags: Iterable[str] | None = None,
    ) -> list[Tool]:
        """Every tool visible to ``agent_address``, filtered by tags.

        Tags filter as ANY-of: a tool passes if any of its tags
        intersects the requested set. Pass ``None`` (default) to skip
        tag filtering.
        """
        tag_set: set[str] | None = None
        if tags is not None:
            tag_set = set(tags)
        out: list[Tool] = []
        for binding in self._bindings.values():
            if not binding.allowed_for.matches(agent_address):
                continue
            if tag_set is not None and not (tag_set & binding.tags):
                continue
            out.append(binding.tool)
        return out


# ── module-level default registry + decorator convenience ─────────────

DEFAULT_TOOL_REGISTRY = ToolRegistry()
"""Process-wide default. Convenient for the ``@tool`` decorator pattern."""


def tool(
    scope: str,
    kind: str,
    role: str,
    category: str,
    *,
    operation: str | None = None,
    description: str | None = None,
    allowed_for: AddressPattern | str | None = None,
    tags: Iterable[str] = (),
    schema: dict | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Module-level decorator that registers into :data:`DEFAULT_TOOL_REGISTRY`.

    Equivalent to ``DEFAULT_TOOL_REGISTRY.tool(...)``. Use a per-app
    :class:`ToolRegistry` instance if you want isolation.
    """
    return DEFAULT_TOOL_REGISTRY.tool(
        scope, kind, role, category,
        operation=operation, description=description,
        allowed_for=allowed_for, tags=tags, schema=schema,
    )
