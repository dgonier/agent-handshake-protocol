"""ahp.tools — first-party tools that any agent can use.

Importing this package registers each tool in the
:data:`ahp.adapters.DEFAULT_TOOL_REGISTRY` via the ``@tool`` decorator.
Tools that need optional dependencies (HTTP clients, API keys) import
their modules lazily and degrade gracefully when those deps are missing.

To opt into the bundled tool set, just::

    import ahp.tools  # registers everything below

Currently bundled:

* :func:`ahp.tools.research.search_tavily` — web search via Tavily.
"""

from __future__ import annotations

# Side-effect imports: each module registers its tools when loaded.
from ahp.tools import research  # noqa: F401

__all__ = ["research"]
