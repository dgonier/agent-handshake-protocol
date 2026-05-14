"""AHP — Agentic Handshake Protocol."""

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.compatibility import CompatibilityMatrix
from ahp.core.message import LIFECYCLE_TTL, Message, Verb
from ahp.core.pattern import AddressPattern

__all__ = [
    "AgentAddress",
    "AddressPattern",
    "Code",
    "Message",
    "Verb",
    "LIFECYCLE_TTL",
    "CompatibilityMatrix",
]

__version__ = "0.12.0"
