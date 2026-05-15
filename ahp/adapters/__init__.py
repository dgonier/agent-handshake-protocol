"""ahp.adapters — agent base class, factory, registries, and adapters.

Always-importable (no optional deps):

* :class:`AHPAgent` — abstract base
* :class:`AgentFactory` — pattern-keyed agent construction + bulk spawn
* :class:`HumanAgent`
* :class:`ProvisioningPattern` — bulk-spawn spec with N*/star-N syntax
* :class:`ToolRegistry` / :class:`ToolAddress` / :func:`tool`
* :class:`ResourceRegistry` / :class:`ResourceAddress` / :func:`resource`
* :class:`CapabilityRegistry`

Framework-specific adapters live in their own submodules and are
imported only on demand so the optional deps stay opt-in:

* ``from ahp.adapters.langgraph_agent import LangGraphAgent, DeepAgentDAG``
* ``from ahp.adapters.react_agent import ReactAgent``
* ``from ahp.adapters.deep_agent import DeepAgent``
* ``from ahp.adapters.dspy_agent import DSPyAgent``
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
from ahp.adapters.factory import (
    AgentFactory,
    Builder,
    ResolutionConflictError,
    ResourceNameCollisionError,
    SpawnResult,
    ToolNameCollisionError,
)
from ahp.adapters.groups import (
    DEFAULT_GROUP_REGISTRY,
    Group,
    GroupRegistry,
    group,
)
from ahp.adapters.human import HumanAgent, ObservationLevel
from ahp.adapters.formats import (
    FORMATS,
    Format,
    FormatNotFoundError,
    get_format,
    list_formats,
)
from ahp.adapters.inviter import AgentInvitation, ChatModel, Inviter
from ahp.adapters.prompts import (
    RECIPES,
    Recipe,
    RecipeNotFoundError,
    get_recipe,
    list_recipes,
    render,
)
from ahp.adapters.provisioning import (
    FieldNamer,
    ProvisioningField,
    ProvisioningPattern,
    default_namer,
)
from ahp.adapters.resources import (
    DEFAULT_RESOURCE_REGISTRY,
    ResourceBinding,
    ResourceRegistry,
    resource,
)
from ahp.adapters.storage import (
    FS_KIND,
    build_fs_backend,
    default_mount_path,
    fs_mount_description,
    fs_resource_addresses,
)
from ahp.adapters.tool_address import ResourceAddress, ToolAddress
from ahp.adapters.tool_registry import (
    DEFAULT_TOOL_REGISTRY,
    ToolBinding,
    ToolRegistry,
    tool,
)

__all__ = [
    "AHPAgent",
    "AgentFactory",
    "AgentInvitation",
    "AgentKind",
    "AgentProfile",
    "Builder",
    "ChatModel",
    "Inviter",
    "RECIPES",
    "Recipe",
    "RecipeNotFoundError",
    "CapabilityProvider",
    "CapabilityRegistry",
    "DEFAULT_GROUP_REGISTRY",
    "DEFAULT_RESOURCE_REGISTRY",
    "DEFAULT_TOOL_REGISTRY",
    "FS_KIND",
    "FieldNamer",
    "Group",
    "GroupRegistry",
    "HumanAgent",
    "ObservationLevel",
    "ProvisioningField",
    "ProvisioningPattern",
    "RagSource",
    "ResolutionConflictError",
    "ResourceAddress",
    "ResourceBinding",
    "ResourceNameCollisionError",
    "ResourceRegistry",
    "Skill",
    "SpawnResult",
    "Tool",
    "ToolAddress",
    "ToolBinding",
    "ToolNameCollisionError",
    "ToolRegistry",
    "build_fs_backend",
    "default_mount_path",
    "default_namer",
    "fs_mount_description",
    "fs_resource_addresses",
    "get_recipe",
    "group",
    "list_recipes",
    "render",
    "resource",
    "tool",
]
