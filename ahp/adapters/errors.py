"""Resolution-conflict errors raised at profile-build time.

The address layer is unambiguous by construction (every tool / resource
has a unique full address). But agent profiles expose tools to
LangChain by their short ``operation`` name and resources by their
short ``name`` field. Two bindings whose short names collide for the
same agent would silently clobber each other in the profile; these
errors surface the conflict at wiring time instead.
"""

from __future__ import annotations


class ResolutionConflictError(Exception):
    """Base class for address-mapping name-collision errors."""


class ToolNameCollisionError(ResolutionConflictError):
    """Two tools at different ToolAddresses share an operation name."""


class ResourceNameCollisionError(ResolutionConflictError):
    """Two resources at different ResourceAddresses share a name."""
