"""Tests for MCP passthrough — bulk-registering an MCP server under a scope.

Uses a hand-rolled fake LangChain tool to avoid spawning a real MCP
process. The integration with a real ``MultiServerMCPClient`` is
exercised manually (e.g. against ``mcp-server-github``) — that test
would be too environment-dependent for CI.
"""

from __future__ import annotations

import pytest
from langchain_core.tools import StructuredTool

from ahp.adapters.mcp import register_mcp_tools
from ahp.adapters.tool_registry import ToolRegistry
from ahp.core.address import AgentAddress


def _agent(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


def _fake_lc_tool(name: str, description: str, fn):
    return StructuredTool.from_function(coroutine=fn, name=name, description=description)


async def test_register_mcp_tools_addresses_each_under_prefix():
    reg = ToolRegistry()

    async def search_repos(query: str) -> list:
        return [f"repo-for-{query}"]

    async def get_issue(repo: str, number: int) -> dict:
        return {"repo": repo, "number": number}

    fake_tools = [
        _fake_lc_tool("search_repos", "search github repos", search_repos),
        _fake_lc_tool("get_issue", "fetch a github issue", get_issue),
    ]

    bindings = register_mcp_tools(
        reg, "tifin", "api", "*", "mcp-github",
        langchain_tools=fake_tools,
    )

    addresses = sorted(str(b.address) for b in bindings)
    assert addresses == [
        "tifin.api.*.mcp-github.get_issue",
        "tifin.api.*.mcp-github.search_repos",
    ]
    # The address-derived allowed_for puts these in scope for any tifin agent.
    fin_agent = _agent("tifin.adversarial.finance.equities.s.session.frank")
    tools = reg.for_address(fin_agent)
    assert sorted(t.name for t in tools) == ["get_issue", "search_repos"]


async def test_mcp_tool_invocation_through_registry():
    """The wrapped handler delegates to the underlying LangChain tool."""
    reg = ToolRegistry()

    async def echo(text: str) -> str:
        return f"echoed:{text}"

    register_mcp_tools(
        reg, "tifin", "api", "*", "mcp-echo",
        langchain_tools=[_fake_lc_tool("echo", "echo back", echo)],
    )
    tool = reg.get("tifin.api.*.mcp-echo.echo")
    # Handler is async — await it directly inside this async test.
    result = await tool.handler(text="hi")
    assert result == "echoed:hi"


def test_mcp_tools_inherit_tags():
    reg = ToolRegistry()

    async def fn(x: int) -> int:
        return x

    register_mcp_tools(
        reg, "tifin", "api", "*", "mcp-github",
        langchain_tools=[_fake_lc_tool("noop", "no-op", fn)],
        tags=["external", "rate-limited"],
    )
    only_external = reg.for_address(
        _agent("tifin.adversarial.x.y.s.session.f"),
        tags=["external"],
    )
    assert [t.name for t in only_external] == ["noop"]


def test_mcp_tools_can_override_allowed_for():
    reg = ToolRegistry()

    async def fn(x: int) -> int:
        return x

    register_mcp_tools(
        reg, "tifin", "api", "*", "mcp-strict",
        langchain_tools=[_fake_lc_tool("strict_tool", "x", fn)],
        allowed_for="*.adversarial.finance.*.*.*.*",
    )
    fin_adv = _agent("public.adversarial.finance.equities.s.session.f")
    other = _agent("tifin.collaborative.science.x.s.session.f")
    assert [t.name for t in reg.for_address(fin_adv)] == ["strict_tool"]
    assert reg.for_address(other) == []


def test_mcp_schema_extracted_when_args_schema_present():
    reg = ToolRegistry()

    from pydantic import BaseModel

    class EchoArgs(BaseModel):
        text: str

    async def echo(text: str) -> str:
        return text

    lc_tool = StructuredTool.from_function(
        coroutine=echo, name="echo_typed", description="echo",
        args_schema=EchoArgs,
    )
    register_mcp_tools(
        reg, "tifin", "api", "*", "mcp-test",
        langchain_tools=[lc_tool],
    )
    tool = reg.get("tifin.api.*.mcp-test.echo_typed")
    assert tool.schema is not None
    # JSON Schema for the args.
    assert tool.schema.get("type") == "object"
    assert "text" in tool.schema.get("properties", {})
