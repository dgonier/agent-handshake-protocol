"""MCP (Model Context Protocol) passthrough.

Register an entire MCP server's tool surface under a scope/kind/role/
category prefix in one call. Each discovered MCP tool becomes an
addressable :class:`ToolBinding`; the MCP client itself is registered
as a :class:`Resource` so its connection is torn down with the rest of
the application.

Example::

    from langchain_mcp_adapters.client import MultiServerMCPClient

    factory = AgentFactory(engine)
    await register_mcp_server(
        factory.tools, factory.resources,
        scope="tifin", kind="api", role="*", category="mcp-github",
        connection={"command": "uvx",
                    "args": ["mcp-server-github"],
                    "transport": "stdio"},
    )

Every tool the GitHub MCP server exposes (`search_repos`,
`get_issue`, ...) is now reachable as e.g.
``tifin.api.*.mcp-github.search_repos`` and gets auto-bound to any
agent matching the convention pattern ``tifin.*.*.*.*.*.*``.

For tests that don't want to spawn a real MCP process, use
:func:`register_mcp_tools` directly with hand-rolled LangChain tools.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Iterable

from ahp.adapters.resources import ResourceRegistry
from ahp.adapters.tool_registry import ToolBinding, ToolRegistry
from ahp.core.pattern import AddressPattern


def _wrap_langchain_tool(lc_tool: Any) -> tuple[Callable[..., Awaitable[Any]], str | None, dict | None]:
    """Build an AHP-shaped handler around a LangChain BaseTool.

    Returns ``(handler, description, schema)``. The handler is async —
    it calls the tool's ``ainvoke`` so the agent's event loop is not
    blocked. ``schema`` is the JSON-schema dict for the tool's inputs
    when available.
    """

    async def handler(**kwargs: Any) -> Any:
        return await lc_tool.ainvoke(kwargs)

    handler.__name__ = lc_tool.name  # type: ignore[attr-defined]

    schema: dict | None = None
    args_schema = getattr(lc_tool, "args_schema", None)
    if args_schema is not None:
        # LangChain tools commonly use a pydantic BaseModel for args.
        # Try the pydantic v2 path first, then v1 fallback.
        try:
            schema = args_schema.model_json_schema()  # type: ignore[union-attr]
        except AttributeError:
            try:
                schema = args_schema.schema()  # type: ignore[union-attr]
            except Exception:
                schema = None

    description = getattr(lc_tool, "description", None) or None
    return handler, description, schema


def register_mcp_tools(
    tools_registry: ToolRegistry,
    scope: str,
    kind: str,
    role: str,
    category: str,
    *,
    langchain_tools: Iterable[Any],
    allowed_for: AddressPattern | str | None = None,
    tags: Iterable[str] = (),
) -> list[ToolBinding]:
    """Register a batch of LangChain (or LangChain-compatible) tools at one prefix.

    Each tool's ``operation`` field is its LangChain ``name`` —
    addresses come out as ``{scope}.{kind}.{role}.{category}.{tool.name}``.
    """
    bindings: list[ToolBinding] = []
    for lc_tool in langchain_tools:
        handler, description, schema = _wrap_langchain_tool(lc_tool)
        binding = tools_registry.register(
            handler,
            scope, kind, role, category,
            operation=lc_tool.name,
            description=description,
            allowed_for=allowed_for,
            tags=tags,
            schema=schema,
        )
        bindings.append(binding)
    return bindings


async def register_mcp_server(
    tools_registry: ToolRegistry,
    resources_registry: ResourceRegistry,
    scope: str,
    kind: str,
    role: str,
    category: str,
    *,
    connection: dict[str, Any],
    server_name: str | None = None,
    allowed_for: AddressPattern | str | None = None,
    tags: Iterable[str] = (),
) -> list[ToolBinding]:
    """Connect to an MCP server, discover its tools, register everything.

    The MCP client is also registered as a :class:`Resource` at
    ``{scope}.api.mcp.{category}.{server_name}`` so :meth:`ResourceRegistry.close_all`
    tears down the connection on shutdown.

    Requires ``langchain-mcp-adapters`` (install via the ``[mcp]`` extra).
    """
    from langchain_mcp_adapters.client import MultiServerMCPClient

    server_name = server_name or category

    # MultiServerMCPClient keys configurations by server name; we use the
    # one this call is registering.
    client = MultiServerMCPClient({server_name: connection})
    lc_tools = await client.get_tools(server_name=server_name)

    bindings = register_mcp_tools(
        tools_registry, scope, kind, role, category,
        langchain_tools=lc_tools, allowed_for=allowed_for, tags=tags,
    )

    # Stash the client as a resource so its session cleanup runs on
    # ResourceRegistry.close_all. Most MCP transports don't strictly
    # need an explicit close, but registering keeps lifecycle visible.
    async def _cleanup(c: MultiServerMCPClient) -> None:
        close = getattr(c, "aclose", None) or getattr(c, "close", None)
        if close is None:
            return
        result = close()
        import inspect
        if inspect.isawaitable(result):
            await result

    resources_registry.register(
        lambda c=client: c,
        scope, "api", "mcp", category,
        name=server_name,
        description=f"MCP server connection: {server_name}",
        cleanup=_cleanup,
        # Resource access scope: any agent that could use the tools also
        # legitimately could need the client handle (rare but possible).
        allowed_for=allowed_for,
    )

    return bindings
