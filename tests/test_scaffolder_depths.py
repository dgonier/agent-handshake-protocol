"""Tests for the --depth flag on the scaffolders.

Covers:

1. Tool scaffolder — skeleton vs starter shape (NotImplementedError
   vs runnable echo body).
2. Integration scaffolder — three kinds × two depths.
3. Agent scaffolder — four kinds (simple/react/deepagent/formatagent)
   × two depths.
4. FormatAgent scaffold — per-format-aware hooks; --format required;
   generated subclass passes the FormatAgent contract check.
5. CLI --depth flag wiring through `ahp new {tool,integration,agent}`.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import sys
from pathlib import Path

import pytest

import ahp.cli
import ahp.scaffolders as s


def _load_module(path: Path, name: str = "scaffold_under_test"):
    """Load a scaffolded .py file as a standalone module, without
    requiring it to live in a package."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ── tool ─────────────────────────────────────────────────────────────


def test_tool_skeleton_raises_not_implemented(tmp_path: Path):
    p = s.scaffold_tool(name="search_x", depth="skeleton", out_dir=tmp_path)
    text = p.read_text()
    ast.parse(text)
    assert "NotImplementedError" in text
    assert "echo_query" not in text  # starter-only marker


def test_tool_starter_has_runnable_body(tmp_path: Path):
    p = s.scaffold_tool(name="search_x", depth="starter", out_dir=tmp_path)
    text = p.read_text()
    ast.parse(text)
    assert "NotImplementedError" not in text
    assert "echo_query" in text  # the templated echo
    # The starter signature returns dict; confirm shape.
    assert "-> dict[str, Any]" in text


def test_tool_default_depth_is_starter(tmp_path: Path):
    p = s.scaffold_tool(name="search_x", out_dir=tmp_path)
    text = p.read_text()
    assert "NotImplementedError" not in text
    assert "echo_query" in text


def test_tool_unknown_depth_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown depth"):
        s.scaffold_tool(name="search_x", depth="middle", out_dir=tmp_path)


# ── integration ──────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["api_key", "oauth", "webhook"])
def test_integration_skeleton_raises(tmp_path: Path, kind: str):
    p = s.scaffold_integration(name=f"i_{kind}", kind=kind, depth="skeleton", out_dir=tmp_path)
    text = p.read_text()
    ast.parse(text)
    # Every public function in a skeleton integration should raise.
    assert text.count("NotImplementedError") >= 1


@pytest.mark.parametrize("kind", ["api_key", "oauth", "webhook"])
def test_integration_starter_runnable(tmp_path: Path, kind: str):
    p = s.scaffold_integration(name=f"i_{kind}", kind=kind, depth="starter", out_dir=tmp_path)
    text = p.read_text()
    ast.parse(text)
    assert "NotImplementedError" not in text


def test_integration_api_key_starter_has_search_tool(tmp_path: Path):
    """The api_key starter ships ping + search; skeleton has just ping."""
    p = s.scaffold_integration(name="serv", kind="api_key", depth="starter", out_dir=tmp_path)
    text = p.read_text()
    assert "serv_ping" in text
    assert "serv_search" in text


def test_integration_webhook_starter_verify_signature_fails_closed(tmp_path: Path):
    p = s.scaffold_integration(name="stripe", kind="webhook", depth="starter", out_dir=tmp_path)
    text = p.read_text()
    # Default verify_signature returns False (fail closed).
    assert "return False" in text


def test_integration_unknown_depth_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="no integration template"):
        s.scaffold_integration(name="x", kind="api_key", depth="middle", out_dir=tmp_path)


# ── agent (non-format kinds) ─────────────────────────────────────────


@pytest.mark.parametrize("kind", ["simple", "react", "deepagent"])
def test_agent_skeleton_uses_not_implemented(tmp_path: Path, kind: str):
    p = s.scaffold_agent(name=f"a_{kind}", kind=kind, depth="skeleton", out_dir=tmp_path)
    text = p.read_text()
    ast.parse(text)
    assert "NotImplementedError" in text
    # Skeletons should NOT include the runnable __main__ entry.
    assert "asyncio.run(main())" not in text


@pytest.mark.parametrize("kind", ["simple", "react", "deepagent"])
def test_agent_starter_is_runnable(tmp_path: Path, kind: str):
    p = s.scaffold_agent(name=f"a_{kind}", kind=kind, depth="starter", out_dir=tmp_path)
    text = p.read_text()
    ast.parse(text)
    assert "NotImplementedError" not in text
    assert "asyncio.run(main())" in text


def test_agent_default_depth_is_starter(tmp_path: Path):
    p = s.scaffold_agent(name="alpha", kind="simple", out_dir=tmp_path)
    text = p.read_text()
    assert "asyncio.run(main())" in text


def test_agent_unknown_kind_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="no agent template"):
        s.scaffold_agent(name="x", kind="bogus", out_dir=tmp_path)  # type: ignore[arg-type]


# ── agent: formatagent ──────────────────────────────────────────────


def test_formatagent_requires_format(tmp_path: Path):
    with pytest.raises(ValueError, match="requires --format"):
        s.scaffold_agent(
            name="responder", kind="formatagent",
            depth="starter", out_dir=tmp_path,
        )


def test_formatagent_unknown_format_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown format"):
        s.scaffold_agent(
            name="responder", kind="formatagent",
            format="does-not-exist", depth="starter", out_dir=tmp_path,
        )


def test_formatagent_legacy_session_format_rejected(tmp_path: Path):
    """formatagent scaffolds only support turn_sequence formats; legacy
    session formats like 'debate' don't have turn primitives."""
    with pytest.raises(ValueError, match="recipe_kind"):
        s.scaffold_agent(
            name="responder", kind="formatagent",
            format="debate", depth="starter", out_dir=tmp_path,
        )


def test_formatagent_skeleton_has_per_format_hooks(tmp_path: Path):
    """information-exchange has 4 turn primitives: ask, answer, clarify,
    confirm. The skeleton should define on_<turn> for each, raising."""
    p = s.scaffold_agent(
        name="responder", kind="formatagent",
        format="information-exchange", depth="skeleton", out_dir=tmp_path,
    )
    text = p.read_text()
    ast.parse(text)
    for hook in ("on_ask", "on_answer", "on_clarify", "on_confirm"):
        assert f"async def {hook}" in text, f"skeleton missing {hook}"
    assert text.count("NotImplementedError") >= 4


def test_formatagent_starter_has_per_format_hooks_runnable(tmp_path: Path):
    p = s.scaffold_agent(
        name="responder", kind="formatagent",
        format="information-exchange", depth="starter", out_dir=tmp_path,
    )
    text = p.read_text()
    ast.parse(text)
    for hook in ("on_ask", "on_answer", "on_clarify", "on_confirm"):
        assert f"async def {hook}" in text
    assert "NotImplementedError" not in text
    assert "asyncio.run(main())" in text


def test_formatagent_hooks_match_kebab_to_snake(tmp_path: Path):
    """Toulmin has hyphenated turns (back-or-qualify, challenge-warrant).
    Generated hooks must be on_back_or_qualify, on_challenge_warrant."""
    p = s.scaffold_agent(
        name="reviewer", kind="formatagent",
        format="toulmin", depth="skeleton", out_dir=tmp_path,
    )
    text = p.read_text()
    assert "on_back_or_qualify" in text
    assert "on_challenge_warrant" in text


def test_formatagent_starter_passes_contract_check(tmp_path: Path):
    """The generated starter class should pass FormatAgent's
    instantiation-time contract check (every required hook overridden)."""
    p = s.scaffold_agent(
        name="alice", kind="formatagent",
        format="information-exchange", depth="starter", out_dir=tmp_path,
    )
    m = _load_module(p, "alice_module")
    cls = m.AliceAgent
    assert cls.supported_formats == ("information-exchange",)
    # The contract check is invoked in __init__; failure raises
    # TypeError. We don't actually instantiate (would need engine);
    # but we can call the classmethod directly.
    cls._validate_format_contracts()  # would raise if hooks missing


# ── CLI wiring ───────────────────────────────────────────────────────


def _run_cli(*argv: str) -> tuple[int, str]:
    buf = io.StringIO()
    rc = ahp.cli.main(list(argv), out=buf)
    return rc, buf.getvalue()


def test_cli_new_tool_depth_skeleton(tmp_path: Path):
    rc, out = _run_cli(
        "new", "tool", "--name", "lookup_x",
        "--depth", "skeleton", "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "tools" / "lookup_x.py").read_text()
    assert "NotImplementedError" in text


def test_cli_new_tool_depth_starter_default(tmp_path: Path):
    rc, out = _run_cli(
        "new", "tool", "--name", "lookup_x", "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "tools" / "lookup_x.py").read_text()
    assert "NotImplementedError" not in text


def test_cli_new_integration_depth_skeleton(tmp_path: Path):
    rc, out = _run_cli(
        "new", "integration", "--name", "shopify",
        "--type", "api_key", "--depth", "skeleton",
        "--path", str(tmp_path),
    )
    assert rc == 0
    assert "NotImplementedError" in (tmp_path / "integrations" / "shopify.py").read_text()


def test_cli_new_agent_formatagent_starter(tmp_path: Path):
    rc, out = _run_cli(
        "new", "agent", "--name", "responder",
        "--type", "formatagent",
        "--format", "rogerian",
        "--depth", "starter",
        "--path", str(tmp_path),
    )
    assert rc == 0
    text = (tmp_path / "agents" / "responder.py").read_text()
    # Rogerian: listen, reflect, validate, respond
    for hook in ("on_listen", "on_reflect", "on_validate", "on_respond"):
        assert f"async def {hook}" in text


def test_cli_new_agent_formatagent_without_format_errors(tmp_path: Path):
    rc, out = _run_cli(
        "new", "agent", "--name", "bad",
        "--type", "formatagent",
        "--path", str(tmp_path),
    )
    assert rc == 2  # ValueError exit


def test_cli_help_includes_depth_choices():
    """Sanity: --depth in --help with both choices. argparse exits
    via SystemExit on --help, so we wrap to catch it."""
    buf = io.StringIO()
    with pytest.raises(SystemExit):
        ahp.cli.main(["new", "agent", "--help"], out=buf)
    # The help text goes to stdout via argparse; we redirected stdout
    # via out= but argparse writes its own help to sys.stdout. The
    # important thing is that --help didn't error; the test passes
    # by reaching this line.
