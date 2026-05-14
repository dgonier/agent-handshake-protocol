"""Tests for the ``python -m ahp`` CLI.

Invokes the argparse main directly with captured stdout — no
subprocess, so failures show up in pytest with full tracebacks.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from ahp.adapters import (
    DEFAULT_GROUP_REGISTRY,
    DEFAULT_RESOURCE_REGISTRY,
    DEFAULT_TOOL_REGISTRY,
    tool,
)
from ahp.cli import main


@pytest.fixture(autouse=True)
def _clean_default_registries():
    """Each test gets fresh default registries (the CLI reads from them)."""
    DEFAULT_TOOL_REGISTRY._bindings.clear()
    DEFAULT_RESOURCE_REGISTRY._bindings.clear()
    DEFAULT_RESOURCE_REGISTRY._instances.clear()
    DEFAULT_RESOURCE_REGISTRY._construct_order.clear()
    DEFAULT_GROUP_REGISTRY._groups.clear()
    yield
    DEFAULT_TOOL_REGISTRY._bindings.clear()
    DEFAULT_RESOURCE_REGISTRY._bindings.clear()
    DEFAULT_RESOURCE_REGISTRY._instances.clear()
    DEFAULT_RESOURCE_REGISTRY._construct_order.clear()
    DEFAULT_GROUP_REGISTRY._groups.clear()


def _run(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    rc = main(list(argv), out=buf)
    return rc, buf.getvalue()


# ── list-tools ────────────────────────────────────────────────────────


def test_list_tools_empty():
    rc, out = _run("list-tools")
    assert rc == 0
    assert "no tools registered" in out


def test_list_tools_after_decorator_registration():
    @tool("tifin", "db", "*", "crud")
    def update_record(table: str, row_id: str, fields: dict) -> dict:
        """Update a row in the table."""
        return {"ok": True}

    rc, out = _run("list-tools")
    assert rc == 0
    assert "tifin.db.*.crud.update_record" in out
    assert "update_record" in out
    assert "Update a row" in out


def test_list_tools_filter_by_for_addr():
    @tool("tifin", "db", "adversarial", "crud")
    def adv_only(): pass

    @tool("public", "db", "*", "crud")
    def public_tool(): pass

    # tifin agent: sees only adv_only (tifin scope).
    rc, out = _run(
        "list-tools",
        "--for", "tifin.adversarial.finance.equities.s.session.f",
    )
    assert rc == 0
    assert "adv_only" in out
    assert "public_tool" not in out

    # Public agent: sees only public_tool.
    rc, out = _run(
        "list-tools",
        "--for", "public.collaborative.x.y.s.session.f",
    )
    assert "public_tool" in out
    assert "adv_only" not in out


def test_list_tools_filter_by_tag():
    @tool("tifin", "db", "*", "crud", tags=["read-only"])
    def fetch(): pass

    @tool("tifin", "db", "*", "crud", tags=["mutating"], operation="purge")
    def _purge(): pass

    rc, out = _run("list-tools", "--tag", "read-only")
    assert "fetch" in out
    assert "purge" not in out


# ── list-resources ────────────────────────────────────────────────────


def test_list_resources_empty():
    rc, out = _run("list-resources")
    assert rc == 0
    assert "no resources registered" in out


def test_list_resources_after_registration():
    from ahp.adapters import resource

    @resource("tifin", "fs", "finance", "documents", name="docs",
              description="finance scratch")
    def docs_factory():
        return {"id": "docs"}

    rc, out = _run("list-resources")
    assert rc == 0
    assert "tifin.fs.finance.documents.docs" in out
    assert "fs" in out
    assert "finance scratch" in out


# ── list-groups ───────────────────────────────────────────────────────


def test_list_groups_empty():
    rc, out = _run("list-groups")
    assert rc == 0
    assert "no groups registered" in out


def test_list_groups_after_registration():
    from ahp.adapters import group

    group("debaters", "*.adversarial.*.*.*.*.*",
          description="bull and bear pool")

    rc, out = _run("list-groups")
    assert rc == 0
    assert "debaters" in out
    assert "adversarial" in out
    assert "bull and bear pool" in out


# ── profile ───────────────────────────────────────────────────────────


def test_profile_shows_resolved_tools_and_resources():
    @tool("tifin", "db", "*", "crud")
    def lookup(): pass

    from ahp.adapters import resource

    @resource("tifin", "fs", "finance", "equities", name="store",
              allowed_for="*.*.*.*.*.*.*")
    def store_factory():
        return {"backend": "fake"}

    rc, out = _run(
        "profile",
        "tifin.adversarial.finance.equities.s.session.frank",
    )
    assert rc == 0
    assert "agent address:" in out
    assert "agent kind:" in out
    assert "lookup" in out
    assert "store" in out


def test_profile_unknown_module_errors():
    rc, _ = _run(
        "profile",
        "tifin.adversarial.finance.equities.s.session.f",
        "-m", "definitely_not_a_real_module_xyzzy",
    )
    assert rc != 0


# ── template / scaffold ───────────────────────────────────────────────


def test_template_tool_emits_runnable_code():
    rc, out = _run(
        "template", "tool",
        "--scope", "tifin", "--kind", "db", "--category", "crud",
        "--operation", "update_record",
        "--signature", "table: str, row_id: str, fields: dict",
    )
    assert rc == 0
    assert "@tool('tifin', 'db', '*', 'crud')" in out
    assert "def update_record(table: str, row_id: str, fields: dict):" in out


def test_template_resource_emits_factory():
    rc, out = _run(
        "template", "resource",
        "--scope", "tifin", "--kind", "fs",
        "--domain", "finance", "--subdomain", "documents",
        "--name", "docs",
    )
    assert rc == 0
    assert "@resource('tifin', 'fs', 'finance', 'documents', name='docs')" in out
    assert "def make_docs():" in out


def test_scaffold_writes_file(tmp_path: Path):
    target = tmp_path / "my_tools.py"
    rc, out = _run(
        "scaffold", "tool",
        "--scope", "acme", "--kind", "api", "--category", "search",
        "--operation", "find_repo",
        "--signature", "query: str",
        "-o", str(target),
    )
    assert rc == 0
    assert target.exists()
    body = target.read_text()
    assert "@tool('acme', 'api', '*', 'search')" in body
    assert "def find_repo(query: str):" in body


def test_scaffold_refuses_to_overwrite_without_force(tmp_path: Path):
    target = tmp_path / "exists.py"
    target.write_text("existing content")
    rc, _ = _run(
        "scaffold", "tool",
        "--scope", "x", "--kind", "y", "--category", "z",
        "--operation", "op", "-o", str(target),
    )
    assert rc != 0
    assert target.read_text() == "existing content"   # untouched


def test_scaffold_force_overwrites(tmp_path: Path):
    target = tmp_path / "ow.py"
    target.write_text("stale")
    rc, _ = _run(
        "scaffold", "tool",
        "--scope", "x", "--kind", "y", "--category", "z",
        "--operation", "op", "-o", str(target), "--force",
    )
    assert rc == 0
    assert "stale" not in target.read_text()


# ── list-agents (live Redis via fakeredis) ────────────────────────────
#
# list-agents is the only CLI command that does real async I/O. The
# sync entry point (cmd_list_agents) calls asyncio.run(), which can't
# run inside pytest-asyncio's already-running loop — so we invoke the
# async worker directly with an already-parsed args namespace.


async def _arun(*argv: str) -> tuple[int, str]:
    import ahp.cli
    parser = ahp.cli.build_parser()
    args = parser.parse_args(list(argv))
    buf = io.StringIO()
    rc = await ahp.cli._list_agents_async(args, buf)
    return rc, buf.getvalue()


async def test_list_agents_empty_registry(redis_client, monkeypatch):
    import ahp.cli
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, out = await _arun("list-agents", "--redis-url", "redis://test/0")
    assert rc == 0
    assert "registry is empty" in out


async def test_list_agents_lists_alive_agents(redis_client, monkeypatch):
    """A populated registry shows up in list-agents output."""
    import ahp.cli
    from ahp.core.address import AgentAddress
    from ahp.registry import AgentMeta, AgentRegistry

    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    registry = AgentRegistry(redis_client, heartbeat_ttl=60)
    await registry.register(
        AgentAddress.parse("tifin.adversarial.finance.equities.s.session.bull"),
        AgentMeta(capabilities=["debate", "valuation"], reputation=0.85,
                  description="bull-case agent"),
    )
    await registry.register(
        AgentAddress.parse("tifin.adversarial.finance.equities.s.session.bear"),
        AgentMeta(capabilities=["debate", "risk"], reputation=0.80,
                  description="bear-case agent"),
    )

    rc, out = await _arun("list-agents", "--redis-url", "redis://test/0")
    assert rc == 0
    assert "tifin.adversarial.finance.equities.s.session.bull" in out
    assert "tifin.adversarial.finance.equities.s.session.bear" in out
    assert "alive" in out
    assert "0.85" in out
    assert "bull-case agent" in out


async def test_list_agents_filters_by_pattern(redis_client, monkeypatch):
    import ahp.cli
    from ahp.core.address import AgentAddress
    from ahp.registry import AgentRegistry

    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    registry = AgentRegistry(redis_client, heartbeat_ttl=60)
    await registry.register(
        AgentAddress.parse("tifin.adversarial.finance.equities.s.session.bull"),
    )
    await registry.register(
        AgentAddress.parse("tifin.collaborative.finance.equities.s.session.alice"),
    )

    rc, out = await _arun(
        "list-agents",
        "--redis-url", "redis://test/0",
        "--pattern", "*.adversarial.*.*.*.*.*",
    )
    assert rc == 0
    assert "bull" in out
    assert "alice" not in out
    assert "no matching agents" not in out


async def test_list_agents_no_matches_under_pattern(redis_client, monkeypatch):
    import ahp.cli
    from ahp.core.address import AgentAddress
    from ahp.registry import AgentRegistry

    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    registry = AgentRegistry(redis_client, heartbeat_ttl=60)
    await registry.register(
        AgentAddress.parse("tifin.adversarial.finance.equities.s.session.bull"),
    )

    rc, out = await _arun(
        "list-agents",
        "--redis-url", "redis://test/0",
        "--pattern", "nobody.*.*.*.*.*.*",
    )
    assert rc == 0
    assert "no matching agents" in out


async def test_list_agents_all_includes_stale(redis_client, monkeypatch):
    """--all surfaces registry entries whose liveness has expired."""
    import ahp.cli
    from ahp.core.address import AgentAddress
    from ahp.registry import AgentRegistry
    from ahp.transport.keys import Keys

    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    registry = AgentRegistry(redis_client, heartbeat_ttl=60)
    addr = AgentAddress.parse(
        "tifin.adversarial.finance.equities.s.session.zombie",
    )
    await registry.register(addr)
    # Kill the liveness marker — the registry hash entry remains.
    await redis_client.delete(Keys.alive_key(addr))

    # Default (alive_only): nothing.
    rc, out = await _arun("list-agents", "--redis-url", "redis://test/0")
    assert rc == 0
    assert "zombie" not in out

    # --all: shows up with status=stale.
    rc, out = await _arun(
        "list-agents", "--redis-url", "redis://test/0", "--all",
    )
    assert rc == 0
    assert "zombie" in out
    assert "stale" in out
