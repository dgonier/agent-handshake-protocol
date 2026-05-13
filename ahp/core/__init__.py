"""ahp.core — protocol primitives: addresses, patterns, codes, messages, compatibility."""

from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern, WILDCARD
from ahp.core.codes import Code
from ahp.core.message import Message, Verb, VALID_VERBS, LIFECYCLE_TTL
from ahp.core.compatibility import CompatibilityMatrix

__all__ = [
    "AgentAddress",
    "AddressPattern",
    "WILDCARD",
    "Code",
    "Message",
    "Verb",
    "VALID_VERBS",
    "LIFECYCLE_TTL",
    "CompatibilityMatrix",
]
