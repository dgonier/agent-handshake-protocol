"""ahp.adapters — agent base class, factory, and framework adapters.

Always-importable (no optional deps):

* :class:`AHPAgent` — abstract base
* :class:`AgentFactory` — pattern-keyed agent construction + bulk spawn
* :class:`HumanAgent`
* :class:`ProvisioningPattern` — bulk-spawn spec with N*/star-N syntax

Framework-specific adapters live in their own submodules and are
imported only on demand so the optional deps stay optional:

* ``from ahp.adapters.langgraph_agent import LangGraphAgent, DeepAgentDAG``
  (requires ``langgraph``)
* ``from ahp.adapters.dspy_agent import DSPyAgent`` (requires ``dspy-ai``)
"""

from ahp.adapters.base import AHPAgent
from ahp.adapters.capability import (
    AgentKind,
    AgentProfile,
    CapabilityProvider,
    CapabilityRegistry,
    RagSource,
    Skill,
    Tool,
)
from ahp.adapters.factory import AgentFactory, Builder, SpawnResult
from ahp.adapters.human import HumanAgent, ObservationLevel
from ahp.adapters.provisioning import (
    FieldNamer,
    ProvisioningField,
    ProvisioningPattern,
    default_namer,
)

__all__ = [
    "AHPAgent",
    "AgentFactory",
    "AgentKind",
    "AgentProfile",
    "Builder",
    "CapabilityProvider",
    "CapabilityRegistry",
    "FieldNamer",
    "HumanAgent",
    "ObservationLevel",
    "ProvisioningField",
    "ProvisioningPattern",
    "RagSource",
    "Skill",
    "SpawnResult",
    "Tool",
    "default_namer",
]
