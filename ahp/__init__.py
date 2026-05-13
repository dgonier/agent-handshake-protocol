"""AHP — Agentic Handshake Protocol."""

from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern
from ahp.core.codes import Code
from ahp.core.message import Message, Verb, LIFECYCLE_TTL
from ahp.core.compatibility import CompatibilityMatrix

__all__ = [
    "AgentAddress",
    "AddressPattern",
    "Code",
    "Message",
    "Verb",
    "LIFECYCLE_TTL",
    "CompatibilityMatrix",
]

__version__ = "0.1.0"
