"""CLI tests for `ahp new` scaffolders and `ahp register/start/stop/deregister agent`.

Two clusters:

* Filesystem scaffolders — drive ``ahp new {tool,integration,agent}``
  against ``tmp_path``, assert files land in the convention paths,
  parse as Python, and refuse to overwrite without ``--force``.
* Lifecycle commands — drive ``ahp register/start/stop/deregister
  agent`` against a fakeredis client (via ``monkeypatch`` on
  ``_connect_redis``, matching the existing ``list-agents`` test
  pattern). Verifies the durable-record vs heartbeat split.
"""

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path

import pytest

import ahp.cli
from ahp.core.address import AgentAddress
from ahp.registry import AgentMeta, AgentRegistry
from ahp.transport.keys import Keys


# ── filesystem scaffolders ────────────────────────────────────────────


def _run(*argv: str, out_buf: io.StringIO | None = None) -> tuple[int, str]:
    """Run a CLI invocation synchronously and capture stdout."""
    buf = out_buf or io.StringIO()
    rc = ahp.cli.main(list(argv), out=buf)
    return rc, buf.getvalue()


def test_new_tool_writes_into_tools_dir(tmp_path: Path):
    rc, out = _run(
        "new", "tool",
        "--name", "find_nearby_restaurants",
        "--path", str(tmp_path),
    )
    assert rc == 0
    target = tmp_path / "tools" / "find_nearby_restaurants.py"
    assert target.is_file()
    assert "wrote" in out and str(target) in out
    text = target.read_text()
    # The generated file mentions the tool name AND parses as Python.
    assert "find_nearby_restaurants" in text
    ast.parse(text)


def test_new_tool_kebab_to_snake(tmp_path: Path):
    """`--name find-nearby-restaurants` is normalized to snake_case."""
    rc, _ = _run(
        "new", "tool", "--name", "find-nearby-restaurants",
        "--path", str(tmp_path),
    )
    assert rc == 0
    assert (tmp_path / "tools" / "find_nearby_restaurants.py").is_file()


def test_new_tool_refuses_overwrite(tmp_path: Path):
    rc, _ = _run("new", "tool", "--name", "x", "--path", str(tmp_path))
    assert rc == 0
    # Second run without --force fails.
    buf = io.StringIO()
    rc2 = ahp.cli.main(
        ["new", "tool", "--name", "x", "--path", str(tmp_path)],
        out=buf,
    )
    assert rc2 == 2


def test_new_tool_force_overwrites(tmp_path: Path):
    rc, _ = _run("new", "tool", "--name", "x", "--path", str(tmp_path))
    assert rc == 0
    target = tmp_path / "tools" / "x.py"
    target.write_text("stale\n")
    rc2, _ = _run(
        "new", "tool", "--name", "x", "--path", str(tmp_path), "--force",
    )
    assert rc2 == 0
    assert "stale" not in target.read_text()


def test_new_integration_oauth_stub(tmp_path: Path):
    rc, _ = _run(
        "new", "integration",
        "--name", "google_maps", "--type", "oauth",
        "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "integrations" / "google_maps.py").read_text()
    assert "google_maps" in text
    # OAuth scaffold is intentionally a stub — no opinionated wiring.
    assert "TODO" in text
    ast.parse(text)


def test_new_integration_api_key(tmp_path: Path):
    rc, _ = _run(
        "new", "integration", "--name", "tavily", "--type", "api_key",
        "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "integrations" / "tavily.py").read_text()
    assert "TAVILY_API_KEY" in text  # convention-derived env key
    ast.parse(text)


def test_new_integration_webhook(tmp_path: Path):
    rc, _ = _run(
        "new", "integration", "--name", "stripe", "--type", "webhook",
        "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "integrations" / "stripe.py").read_text()
    assert "verify_signature" in text
    ast.parse(text)


def test_new_agent_simple(tmp_path: Path):
    rc, _ = _run(
        "new", "agent", "--name", "restaurant_finder", "--type", "simple",
        "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "agents" / "restaurant_finder.py").read_text()
    # Class name is CamelCase + "Agent" suffix.
    assert "class RestaurantFinderAgent" in text
    # Has the broker-side visibility comment block.
    assert "register agent" in text and "start agent" in text
    ast.parse(text)


def test_new_agent_react(tmp_path: Path):
    rc, _ = _run(
        "new", "agent", "--name", "scout", "--type", "react",
        "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "agents" / "scout.py").read_text()
    assert "ReactAgent" in text and "from_profile" in text
    ast.parse(text)


def test_new_agent_deepagent(tmp_path: Path):
    rc, _ = _run(
        "new", "agent", "--name", "planner", "--type", "deepagent",
        "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "agents" / "planner.py").read_text()
    assert "DeepAgent" in text
    ast.parse(text)


def test_new_tool_invalid_name(tmp_path: Path):
    """Names with leading digits / dots / colons are rejected."""
    rc, _ = _run(
        "new", "tool", "--name", "1bad", "--path", str(tmp_path),
        out_buf=io.StringIO(),
    )
    assert rc == 2


# ── lifecycle: register / start / stop / deregister ───────────────────


async def _arun_register(argv: list[str]) -> tuple[int, str]:
    """Run a single lifecycle subcommand by calling the async worker
    directly. Mirrors the existing ``_arun`` pattern used by
    list-agents tests (avoids asyncio.run inside the pytest loop)."""
    parser = ahp.cli.build_parser()
    args = parser.parse_args(argv)
    buf = io.StringIO()
    # Each subcommand has its own async worker; dispatch by entity+cmd.
    if argv[0] == "register":
        rc = await ahp.cli._register_agent_async(args, buf)
    elif argv[0] == "start":
        rc = await ahp.cli._start_agent_async(args, buf)
    elif argv[0] == "stop":
        rc = await ahp.cli._stop_agent_async(args, buf)
    elif argv[0] == "deregister":
        rc = await ahp.cli._deregister_agent_async(args, buf)
    else:
        raise AssertionError(f"unexpected command {argv[0]}")
    return rc, buf.getvalue()


_BASE_ARGS = [
    "--name", "restaurant_finder",
    "--scope", "tifin",
    "--role", "researcher",
    "--domain", "food",
    "--subdomain", "local",
    "--accept", "s",
    "--lifecycle", "session",
    "--redis-url", "redis://test/0",
]

_EXPECTED_ADDRESS = (
    "tifin.researcher.food.local.s.session.restaurant_finder"
)


async def test_register_writes_durable_record_but_not_alive(
    redis_client, monkeypatch,
):
    """`ahp register agent` writes AgentMeta and does NOT mark alive."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    rc, out = await _arun_register([
        "register", "agent",
        *_BASE_ARGS,
        "--description", "finds nearby restaurants",
        "--capability", "search",
        "--capability", "recommend",
    ])
    assert rc == 0
    assert _EXPECTED_ADDRESS in out

    # Durable record present.
    addr = AgentAddress.parse(_EXPECTED_ADDRESS)
    raw = await redis_client.hget(Keys.registry_hash(), str(addr))
    assert raw is not None
    meta = AgentMeta.from_json(raw)
    assert set(meta.capabilities) == {"search", "recommend"}
    assert meta.description == "finds nearby restaurants"

    # NOT visible on the menu.
    registry = AgentRegistry(redis_client)
    assert await registry.is_alive(addr) is False
    assert addr not in await registry.list_all(alive_only=True)


async def test_start_makes_registered_agent_visible(
    redis_client, monkeypatch,
):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    rc, _ = await _arun_register(["register", "agent", *_BASE_ARGS])
    assert rc == 0

    rc, out = await _arun_register(["start", "agent", *_BASE_ARGS])
    assert rc == 0
    assert "visible" in out

    addr = AgentAddress.parse(_EXPECTED_ADDRESS)
    registry = AgentRegistry(redis_client)
    assert await registry.is_alive(addr) is True


async def test_start_without_registration_errors(redis_client, monkeypatch):
    """Trying to start an agent that was never registered surfaces a
    clear error rather than silently inserting a half-state."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, _ = await _arun_register(["start", "agent", *_BASE_ARGS])
    assert rc == 2


async def test_stop_hides_but_keeps_record(redis_client, monkeypatch):
    """`stop` clears heartbeat but leaves the durable record so a
    subsequent `start` flips visibility back without re-registering."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    await _arun_register(["register", "agent", *_BASE_ARGS])
    await _arun_register(["start", "agent", *_BASE_ARGS])

    addr = AgentAddress.parse(_EXPECTED_ADDRESS)
    registry = AgentRegistry(redis_client)
    assert await registry.is_alive(addr) is True

    rc, out = await _arun_register(["stop", "agent", *_BASE_ARGS])
    assert rc == 0
    assert "hidden" in out

    # Liveness gone.
    assert await registry.is_alive(addr) is False
    # But record still present.
    assert (await registry.get(addr)) is not None

    # `start` flips it back without re-registering.
    rc, _ = await _arun_register(["start", "agent", *_BASE_ARGS])
    assert rc == 0
    assert await registry.is_alive(addr) is True


async def test_stop_with_no_record_errors(redis_client, monkeypatch):
    """`stop` against an address that was never registered errors out
    instead of pretending there was something to do."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, _ = await _arun_register(["stop", "agent", *_BASE_ARGS])
    assert rc == 2


async def test_deregister_drops_record_entirely(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)

    await _arun_register(["register", "agent", *_BASE_ARGS])
    await _arun_register(["start", "agent", *_BASE_ARGS])

    rc, out = await _arun_register(["deregister", "agent", *_BASE_ARGS])
    assert rc == 0
    assert "deregistered" in out

    addr = AgentAddress.parse(_EXPECTED_ADDRESS)
    registry = AgentRegistry(redis_client)
    assert await registry.is_alive(addr) is False
    assert (await registry.get(addr)) is None


async def test_invalid_address_fragment_errors_cleanly(
    redis_client, monkeypatch,
):
    """A bad scope (with a dot) should be rejected before Redis is touched."""
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, _ = await _arun_register([
        "register", "agent",
        "--name", "x",
        "--scope", "bad.scope",   # extra dot breaks the 7-field address
        "--role", "researcher",
        "--domain", "d", "--subdomain", "s",
        "--accept", "s", "--lifecycle", "session",
        "--redis-url", "redis://test/0",
    ])
    assert rc == 2
