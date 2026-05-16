"""ahp.broker — the routing + settlement broker.

This package is the *source of truth* for everything money- and
identity-related at the server level:

* :mod:`ahp.broker.server_registry` — server identities, capabilities,
  rate cards, compute bindings. Mirror of
  :class:`ahp.registry.AgentRegistry` but one level up.
* :mod:`ahp.broker.compute_registry` — compute provider directory and
  their advertised menu leaves.
* :mod:`ahp.broker.router` — the three-stage routing pipeline that
  picks one server + one compute leaf per dispatch.
* :mod:`ahp.broker.broker` — the high-level facade. Engines and the
  viewer talk to this; it composes the pieces.

All persistence is in the same Redis the rest of AHP uses, under the
``ahp:server:*``, ``ahp:compute_provider:*``, ``ahp:compute_menu:*``,
``ahp:wallet:*``, ``ahp:reputation:*`` keyspaces.
"""

from ahp.broker.broker import Broker
from ahp.broker.compute_registry import ComputeProviderRegistry
from ahp.broker.router import (
    NoCandidatesError,
    RoutingDecision,
    RoutingPreferences,
    Router,
)
from ahp.broker.server_registry import ServerMeta, ServerRegistry


__all__ = [
    "Broker",
    "ComputeProviderRegistry",
    "NoCandidatesError",
    "Router",
    "RoutingDecision",
    "RoutingPreferences",
    "ServerMeta",
    "ServerRegistry",
]
