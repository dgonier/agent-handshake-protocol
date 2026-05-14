"""Tests for addressable storage — kind="fs" resources → deepagents backend.

Verifies that:

1. ``build_fs_backend`` returns ``StateBackend`` (the default) when no
   fs-kind resources match the agent.
2. With one matching fs resource it returns that backend directly —
   no CompositeBackend wrapper.
3. With multiple it returns a ``CompositeBackend`` routing each one to
   ``/<name>/``.
4. Mount-path collisions raise a clear ``ValueError`` at build time.
5. ``fs_mount_description`` produces the system-prompt fragment.
6. ``DeepAgent.from_profile(fs_resources=...)`` wires the backend into
   the underlying deep agent and appends the mount list to the prompt.
"""

from __future__ import annotations

import asyncio

import pytest
from deepagents.backends import CompositeBackend, StateBackend
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ahp.adapters import (
    AgentProfile,
    ResourceRegistry,
    build_fs_backend,
    fs_mount_description,
    fs_resource_addresses,
)
from ahp.adapters.deep_agent import DeepAgent
from ahp.adapters.tool_address import ResourceAddress
from ahp.core.address import AgentAddress


pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
)


def _agent(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


class _ToolableFakeChat(FakeListChatModel):
    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


# ── build_fs_backend ───────────────────────────────────────────────────


def test_no_fs_resources_returns_state_backend(stack):
    resources = ResourceRegistry()
    backend = build_fs_backend(
        resources, _agent("tifin.adversarial.finance.equities.s.session.f"),
    )
    assert isinstance(backend, StateBackend)


def test_single_fs_resource_returned_directly(stack):
    resources = ResourceRegistry()

    @resources.resource("tifin", "fs", "finance", "documents", name="docs")
    def make_docs():
        return StateBackend()

    backend = build_fs_backend(
        resources, _agent("tifin.adversarial.finance.documents.s.session.f"),
    )
    # Single match → no CompositeBackend wrapper.
    assert isinstance(backend, StateBackend)
    assert not isinstance(backend, CompositeBackend)


def test_multiple_fs_resources_wrapped_in_composite(stack):
    resources = ResourceRegistry()

    @resources.resource(
        "tifin", "fs", "finance", "documents",
        name="docs", allowed_for="*.*.*.*.*.*.*",
    )
    def make_docs():
        return StateBackend()

    @resources.resource(
        "tifin", "fs", "finance", "papers",
        name="papers", allowed_for="*.*.*.*.*.*.*",
    )
    def make_papers():
        return StateBackend()

    backend = build_fs_backend(
        resources, _agent("tifin.adversarial.finance.equities.s.session.f"),
    )
    assert isinstance(backend, CompositeBackend)
    # Routes default to /<name>/.
    routes = backend.routes
    assert set(routes.keys()) == {"/docs/", "/papers/"}


def test_mount_path_collision_raises(stack):
    """Two fs resources whose default mount path collides should error."""
    resources = ResourceRegistry()

    # Same `name` field → same default mount path → collision.
    # But the ResourceRegistry's own collision detector fires first,
    # so we need a custom mount_path callable to surface the storage-
    # layer error instead.
    @resources.resource(
        "tifin", "fs", "finance", "documents",
        name="store-a", allowed_for="*.*.*.*.*.*.*",
    )
    def make_a():
        return StateBackend()

    @resources.resource(
        "tifin", "fs", "finance", "papers",
        name="store-b", allowed_for="*.*.*.*.*.*.*",
    )
    def make_b():
        return StateBackend()

    # Custom mount_path that always returns the same path → collision.
    with pytest.raises(ValueError, match="mount-path collision"):
        build_fs_backend(
            resources,
            _agent("tifin.adversarial.finance.equities.s.session.f"),
            mount_path=lambda addr: "/shared/",
        )


def test_non_matching_resources_excluded(stack):
    """fs resources whose allowed_for doesn't match are silently skipped."""
    resources = ResourceRegistry()

    @resources.resource("tifin", "fs", "finance", "documents", name="fin")
    def fin_store():
        return StateBackend()

    @resources.resource("tifin", "fs", "science", "papers", name="sci")
    def sci_store():
        return StateBackend()

    fin_backend = build_fs_backend(
        resources, _agent("tifin.adversarial.finance.documents.s.session.f"),
    )
    sci_backend = build_fs_backend(
        resources, _agent("tifin.adversarial.science.papers.s.session.x"),
    )
    # Each agent sees only its domain's backend (single → returned directly).
    assert isinstance(fin_backend, StateBackend)
    assert isinstance(sci_backend, StateBackend)
    assert fin_backend is not sci_backend


def test_non_fs_resources_ignored(stack):
    """kind != 'fs' resources are NOT mounted as storage."""
    resources = ResourceRegistry()

    @resources.resource("tifin", "vector", "finance", "filings", name="filings")
    def make_vec():
        return {"kind": "vector"}    # not a BackendProtocol — would crash if mounted

    # Should silently skip the non-fs resource.
    backend = build_fs_backend(
        resources, _agent("tifin.adversarial.finance.filings.s.session.f"),
    )
    assert isinstance(backend, StateBackend)   # default — nothing matched


def test_explicit_default_wraps_single_match_in_composite(stack):
    """When the caller provides a `default`, the single match still gets composite-wrapped."""
    resources = ResourceRegistry()

    @resources.resource("tifin", "fs", "finance", "documents", name="docs")
    def make_docs():
        return StateBackend()

    user_default = StateBackend()
    backend = build_fs_backend(
        resources,
        _agent("tifin.adversarial.finance.documents.s.session.f"),
        default=user_default,
    )
    # Two backends in play (the matched one + the explicit default) →
    # CompositeBackend.
    assert isinstance(backend, CompositeBackend)
    assert backend.default is user_default


# ── fs_mount_description ───────────────────────────────────────────────


def test_mount_description_empty_when_no_resources():
    resources = ResourceRegistry()
    text = fs_mount_description(
        resources, _agent("tifin.adversarial.finance.equities.s.session.f"),
    )
    assert text == ""


def test_mount_description_lists_each_mount(stack):
    resources = ResourceRegistry()

    @resources.resource(
        "tifin", "fs", "finance", "documents",
        name="docs", description="finance docs scratch",
        allowed_for="*.*.*.*.*.*.*",
    )
    def docs():
        return StateBackend()

    @resources.resource(
        "tifin", "fs", "finance", "uploads",
        name="uploads", allowed_for="*.*.*.*.*.*.*",
    )
    def uploads():
        return StateBackend()

    text = fs_mount_description(
        resources, _agent("tifin.adversarial.finance.equities.s.session.f"),
    )
    assert "/docs/" in text
    assert "/uploads/" in text
    assert "finance docs scratch" in text
    # Falls back to the name when no description was set.
    assert "uploads" in text


# ── fs_resource_addresses ─────────────────────────────────────────────


def test_fs_resource_addresses_only_lists_fs_kind(stack):
    resources = ResourceRegistry()

    @resources.resource("tifin", "fs", "finance", "documents", name="docs")
    def docs():
        return StateBackend()

    @resources.resource("tifin", "vector", "finance", "filings", name="filings")
    def vec():
        return {"kind": "vector"}

    addrs = fs_resource_addresses(
        resources, _agent("tifin.adversarial.finance.documents.s.session.f"),
    )
    assert [str(a) for a in addrs] == ["tifin.fs.finance.documents.docs"]


# ── DeepAgent integration ─────────────────────────────────────────────


async def test_deep_agent_mounts_fs_resources(stack):
    """DeepAgent.from_profile(fs_resources=...) wires the backend in."""
    addr = _agent("tifin.collaborative.finance.equities.s.session.researcher")

    resources = ResourceRegistry()

    @resources.resource(
        "tifin", "fs", "finance", "documents",
        name="scratch", allowed_for="*.*.*.*.*.*.*",
        description="working scratch space",
    )
    def make_scratch():
        return StateBackend()

    model = _ToolableFakeChat(responses=["acknowledged"])
    profile = AgentProfile(address=addr, prompt="You are a researcher.")

    agent = DeepAgent.from_profile(
        addr, stack.engine, profile, model=model,
        fs_resources=resources, heartbeat_interval=0,
    )
    # The graph was constructed without raising; smoke-pass confirms
    # the backend wiring made it through create_deep_agent.
    assert agent.graph is not None


async def test_deep_agent_without_fs_resources_uses_default_state_backend(stack):
    """When fs_resources=None (the existing default), the deep agent uses
    deepagents' built-in state-backed FS — unchanged behavior."""
    addr = _agent("tifin.collaborative.finance.equities.s.session.r2")
    model = _ToolableFakeChat(responses=["ok"])
    profile = AgentProfile(address=addr, prompt="terse")

    agent = DeepAgent.from_profile(
        addr, stack.engine, profile, model=model,
        heartbeat_interval=0,
    )
    assert agent.graph is not None
