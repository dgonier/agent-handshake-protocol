"""Address-keyed resource catalog with lazy init + decorator API.

A *resource* is a long-lived stateful object shared across agents — a
vector store, a Redis client, an API SDK, a filesystem backend. Each
resource has a structured :class:`ResourceAddress` driving discovery
and access control, and a factory (a class or callable) constructed
lazily on first access.

Two registration styles:

::

    from ahp.adapters import DEFAULT_RESOURCE_REGISTRY, resource

    # 1. Decorate a class — its no-arg constructor is the factory.
    @resource("tifin", "fs", "finance", "documents")
    class FinanceDocs:
        def __init__(self):
            self.root = "/data/finance"
        def read(self, path: str) -> bytes:
            ...
        def aclose(self):
            ...

    # 2. Decorate a factory function.
    @resource("tifin", "vector", "finance", "filings",
              name="sec-edgar", cleanup=lambda c: c.aclose())
    def make_sec_vector():
        return ChromaClient(...)

The factory passes the agent's :class:`AddressPattern`-matched
resources into its :class:`AgentProfile` so tools can pull them out by
name.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterable

from ahp.adapters.tool_address import ResourceAddress
from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern


log = logging.getLogger(__name__)

ResourceFactory = Callable[[], Any]
ResourceCleanup = Callable[[Any], Awaitable[None] | None]


@dataclass(frozen=True)
class ResourceBinding:
    """A registered resource: address + factory + cleanup + access scope."""

    address: ResourceAddress
    factory: ResourceFactory
    cleanup: ResourceCleanup | None
    allowed_for: AddressPattern
    description: str


class ResourceRegistry:
    """Lazy, address-keyed shared resources with shutdown teardown."""

    def __init__(self) -> None:
        self._bindings: dict[str, ResourceBinding] = {}
        self._instances: dict[str, Any] = {}
        self._construct_order: list[str] = []

    # ── registration ───────────────────────────────────────────────────

    def register(
        self,
        factory: ResourceFactory,
        scope: str,
        kind: str,
        domain: str,
        subdomain: str,
        *,
        name: str | None = None,
        description: str = "",
        allowed_for: AddressPattern | str | None = None,
        cleanup: ResourceCleanup | None = None,
    ) -> ResourceBinding:
        """Register a factory under the given resource address.

        ``name`` defaults to the factory's ``__name__`` (function name
        or class name). ``allowed_for`` defaults to the convention
        derived from the resource address.
        """
        resolved_name = name or getattr(factory, "__name__", None) or "anonymous"
        address = ResourceAddress(scope, kind, domain, subdomain, resolved_name)
        key = str(address)
        if key in self._bindings:
            raise ValueError(f"resource already registered at {key!r}")

        if allowed_for is None:
            pattern = address.derived_allowed_for()
        elif isinstance(allowed_for, str):
            pattern = AddressPattern.parse(allowed_for)
        else:
            pattern = allowed_for

        # If the factory looks like a class, infer cleanup from the
        # presence of an `aclose`/`close` method on instances.
        if cleanup is None and inspect.isclass(factory):
            cleanup = _auto_cleanup_for(factory)

        binding = ResourceBinding(
            address=address,
            factory=factory,
            cleanup=cleanup,
            allowed_for=pattern,
            description=description,
        )
        self._bindings[key] = binding
        return binding

    def resource(
        self,
        scope: str,
        kind: str,
        domain: str,
        subdomain: str,
        *,
        name: str | None = None,
        description: str = "",
        allowed_for: AddressPattern | str | None = None,
        cleanup: ResourceCleanup | None = None,
    ) -> Callable[[ResourceFactory], ResourceFactory]:
        """Decorator form of :meth:`register`. Returns the factory unchanged."""

        def decorator(factory: ResourceFactory) -> ResourceFactory:
            self.register(
                factory, scope, kind, domain, subdomain,
                name=name, description=description,
                allowed_for=allowed_for, cleanup=cleanup,
            )
            return factory

        return decorator

    def unregister(self, address: ResourceAddress | str) -> bool:
        key = str(address)
        if key in self._instances:
            # Leaving an instance dangling is worse than blocking — refuse.
            raise RuntimeError(
                f"cannot unregister live resource {key!r}; close_all first"
            )
        return self._bindings.pop(key, None) is not None

    # ── access ─────────────────────────────────────────────────────────

    def get(self, address: ResourceAddress | str) -> Any:
        key = str(address)
        if key not in self._instances:
            if key not in self._bindings:
                raise KeyError(key)
            self._instances[key] = self._bindings[key].factory()
            self._construct_order.append(key)
        return self._instances[key]

    def addresses(self) -> list[ResourceAddress]:
        return [b.address for b in self._bindings.values()]

    def __len__(self) -> int:
        return len(self._bindings)

    def __contains__(self, address: ResourceAddress | str) -> bool:
        return str(address) in self._bindings

    def bindings(self) -> Iterable[ResourceBinding]:
        return self._bindings.values()

    def for_address(self, agent_address: AgentAddress) -> dict[str, Any]:
        """Resources visible to ``agent_address`` as ``{name: instance}``.

        The dict key is the resource's ``name`` field (the final dot
        segment of the :class:`ResourceAddress`). Two resources at
        different full addresses that share the same ``name`` would
        otherwise silently clobber each other in the agent's profile;
        this method raises :class:`ResourceNameCollisionError`
        instead, so the failure surfaces at wiring time. Either
        rename one resource or tighten its ``allowed_for`` so they
        don't both apply to the same agent.
        """
        from ahp.adapters.errors import ResourceNameCollisionError

        out: dict[str, Any] = {}
        provenance: dict[str, ResourceAddress] = {}
        for binding in self._bindings.values():
            if not binding.allowed_for.matches(agent_address):
                continue
            name = binding.address.name
            if name in provenance and provenance[name] != binding.address:
                raise ResourceNameCollisionError(
                    f"two resources claim the short name {name!r} for agent "
                    f"{agent_address}: {provenance[name]} and "
                    f"{binding.address}. Rename one or tighten its "
                    f"allowed_for so they don't both apply to this agent."
                )
            provenance[name] = binding.address
            out[name] = self.get(str(binding.address))
        return out

    # ── lifecycle ──────────────────────────────────────────────────────

    async def close_all(self) -> None:
        """Tear down constructed resources in reverse-construction order."""
        first_error: BaseException | None = None
        for key in reversed(self._construct_order):
            binding = self._bindings.get(key)
            instance = self._instances.get(key)
            if binding is None or instance is None:
                continue
            try:
                if binding.cleanup is not None:
                    result = binding.cleanup(instance)
                    if inspect.isawaitable(result):
                        await result
            except Exception as exc:
                log.exception("cleanup failed for resource %s", key)
                if first_error is None:
                    first_error = exc
        self._instances.clear()
        self._construct_order.clear()
        if first_error is not None:
            raise first_error


def _auto_cleanup_for(cls: type) -> ResourceCleanup | None:
    """Best-effort discovery of an ``aclose`` or ``close`` method on a class."""
    if hasattr(cls, "aclose"):
        async def _cleanup(instance: Any) -> None:
            result = instance.aclose()
            if inspect.isawaitable(result):
                await result
        return _cleanup
    if hasattr(cls, "close"):
        def _cleanup_sync(instance: Any) -> None:
            instance.close()
        return _cleanup_sync
    return None


# ── module-level default registry + decorator convenience ─────────────

DEFAULT_RESOURCE_REGISTRY = ResourceRegistry()


def resource(
    scope: str,
    kind: str,
    domain: str,
    subdomain: str,
    *,
    name: str | None = None,
    description: str = "",
    allowed_for: AddressPattern | str | None = None,
    cleanup: ResourceCleanup | None = None,
) -> Callable[[ResourceFactory], ResourceFactory]:
    """Module-level decorator that registers into :data:`DEFAULT_RESOURCE_REGISTRY`."""
    return DEFAULT_RESOURCE_REGISTRY.resource(
        scope, kind, domain, subdomain,
        name=name, description=description,
        allowed_for=allowed_for, cleanup=cleanup,
    )
