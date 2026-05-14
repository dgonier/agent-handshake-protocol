"""Storage as an addressable resource — wire ``kind="fs"`` resources
into deepagents' :class:`FilesystemMiddleware` / ``CompositeBackend``
so the virtual filesystem lives in the same address space as
everything else.

A storage backend is just a Resource at a ``scope.fs.domain.subdomain.name``
address that returns an object implementing deepagents'
:class:`BackendProtocol` (or any duck-compatible type — at minimum
``read`` / ``write`` / ``ls``).

By default each backend mounts at ``/<name>/`` in the agent's virtual
FS. The :class:`DeepAgent` adapter consumes this via
``build_fs_backend(resources, agent_address)``, which returns a single
:class:`BackendProtocol` (one backend → returned directly, multiple →
wrapped in :class:`CompositeBackend`).

Example::

    from ahp.adapters import resource
    from deepagents.backends import StateBackend

    @resource("tifin", "fs", "finance", "documents", name="docs",
              description="finance team scratch + uploads")
    def make_docs_backend():
        return StateBackend()

    # Inside DeepAgent.from_profile(..., fs_resources=factory.resources):
    #   /docs/  is mounted as this backend
    #   the agent's system prompt is appended with the mount list
"""

from __future__ import annotations

from typing import Iterable

from ahp.adapters.resources import ResourceBinding, ResourceRegistry
from ahp.adapters.tool_address import ResourceAddress
from ahp.core.address import AgentAddress


FS_KIND: str = "fs"
"""``ResourceAddress.kind`` that marks a resource as a filesystem backend."""


def default_mount_path(address: ResourceAddress) -> str:
    """Default mount: ``/<name>/`` from the address's ``name`` field."""
    return f"/{address.name}/"


def _matching_fs_bindings(
    resources: ResourceRegistry,
    agent_address: AgentAddress,
) -> list[ResourceBinding]:
    out: list[ResourceBinding] = []
    for binding in resources.bindings():
        if binding.address.kind != FS_KIND:
            continue
        if not binding.allowed_for.matches(agent_address):
            continue
        out.append(binding)
    return out


def build_fs_backend(
    resources: ResourceRegistry,
    agent_address: AgentAddress,
    *,
    default=None,
    mount_path=default_mount_path,
):
    """Construct a deepagents :class:`BackendProtocol` for ``agent_address``.

    * No fs resources match → returns ``default`` if given, otherwise a
      fresh :class:`deepagents.backends.StateBackend`.
    * Exactly one fs resource matches and no explicit ``default`` →
      returns that backend directly (no composite wrapper).
    * Multiple match (or ``default`` is provided) → returns a
      :class:`deepagents.backends.CompositeBackend` routing each one to
      its mount path.

    Raises :class:`ValueError` if two resources resolve to the same
    mount path for this agent (a network-mapping conflict the user
    should fix at registration time).
    """
    from deepagents.backends import CompositeBackend, StateBackend

    bindings = _matching_fs_bindings(resources, agent_address)

    if not bindings:
        return default if default is not None else StateBackend()

    routes: dict[str, object] = {}
    for binding in bindings:
        path = mount_path(binding.address)
        if path in routes:
            raise ValueError(
                f"mount-path collision at {path!r} for agent {agent_address}: "
                f"resources at {binding.address} and another both want this "
                f"prefix. Override mount_path= per resource or rename one."
            )
        backend = resources.get(str(binding.address))
        routes[path] = backend

    if len(routes) == 1 and default is None:
        return next(iter(routes.values()))

    return CompositeBackend(
        default=default if default is not None else StateBackend(),
        routes=routes,
    )


def fs_mount_description(
    resources: ResourceRegistry,
    agent_address: AgentAddress,
    *,
    mount_path=default_mount_path,
    header: str = "Available filesystem mounts:",
) -> str:
    """A system-prompt fragment listing this agent's FS mounts.

    Empty string when nothing matches — easy to concatenate
    unconditionally with the rest of the prompt.
    """
    lines: list[str] = []
    for binding in _matching_fs_bindings(resources, agent_address):
        path = mount_path(binding.address)
        desc = binding.description or binding.address.name
        lines.append(f"- {path} — {desc}")
    if not lines:
        return ""
    return header + "\n" + "\n".join(lines)


def fs_resource_addresses(
    resources: ResourceRegistry,
    agent_address: AgentAddress,
) -> list[ResourceAddress]:
    """The :class:`ResourceAddress`-es of every fs backend visible to the agent."""
    return [b.address for b in _matching_fs_bindings(resources, agent_address)]
