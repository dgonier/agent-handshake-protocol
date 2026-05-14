"""Engine-level exceptions."""

from __future__ import annotations


class ProtocolError(Exception):
    """Raised when a message can't be routed for protocol reasons.

    Distinct from transport errors (network failure, bad serialization)
    and from agent-level errors raised inside handlers.
    """


class IncompatibleTargetError(ProtocolError):
    """Target agent's accept set doesn't satisfy the code's tier requirements."""


class InvalidTargetTypeError(ProtocolError):
    """Verb expected an AgentAddress but got an AddressPattern (or vice versa)."""


class UnauthorizedError(ProtocolError):
    """Source isn't permitted to reach target by the active ScopePolicy.

    Raised on point-to-point verbs (SEND, SEND-GET). For broadcasts
    (CAST, CAST-GET) the engine silently drops disallowed targets
    from the resolved set — same pattern compatibility uses.
    """
