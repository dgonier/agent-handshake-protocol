"""Tests for the in-memory knowledge graph backend and resource wiring."""

from __future__ import annotations

import pytest

from ahp.adapters.knowledge_graph import (
    KG_KIND,
    InMemoryKnowledgeGraph,
    KGEdge,
    KGNode,
    KnowledgeGraphBackend,
    build_kg_backend,
    kg_mount_description,
    kg_resource_addresses,
    node_id_for_agent,
    node_id_for_judgement,
    node_id_for_rubric,
)
from ahp.adapters.resources import ResourceRegistry
from ahp.adapters.tool_address import ResourceAddress
from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


AGENT = _addr("acme.adversarial.finance.equities.s.session.alpha")
OTHER = _addr("acme.collaborative.finance.equities.s.session.beta")


# ── KGNode / KGEdge invariants ─────────────────────────────────────────


def test_kgnode_with_id_returns_new_instance():
    original = KGNode(id="", kind="Belief", label="x")
    updated = original.with_id("belief-123")
    assert original.id == ""
    assert updated.id == "belief-123"
    assert updated.kind == "Belief"


def test_kgnode_protocol_conformance():
    backend = InMemoryKnowledgeGraph()
    assert isinstance(backend, KnowledgeGraphBackend)


# ── in-memory CRUD ─────────────────────────────────────────────────────


def test_write_and_get_node_assigns_id_when_blank():
    kg = InMemoryKnowledgeGraph()
    stored = kg.write_node(KGNode(id="", kind="Belief", label="Tesla is overvalued"))
    assert stored.id != ""
    fetched = kg.get_node(stored.id)
    assert fetched is not None
    assert fetched.label == "Tesla is overvalued"


def test_write_edge_requires_known_endpoints():
    kg = InMemoryKnowledgeGraph()
    kg.write_node(KGNode(id="a", kind="Belief"))
    with pytest.raises(KeyError):
        kg.write_edge(KGEdge(source_id="a", target_id="b", kind="SUPPORTS"))


def test_list_nodes_filters_by_kind():
    kg = InMemoryKnowledgeGraph()
    kg.write_node(KGNode(id="b1", kind="Belief"))
    kg.write_node(KGNode(id="e1", kind="Evidence"))
    kg.write_node(KGNode(id="b2", kind="Belief"))
    beliefs = kg.list_nodes(kind="Belief")
    assert {n.id for n in beliefs} == {"b1", "b2"}


def test_list_edges_filters_by_source_target_kind():
    kg = InMemoryKnowledgeGraph()
    for nid in ["a", "b", "c"]:
        kg.write_node(KGNode(id=nid, kind="Belief"))
    kg.write_edge(KGEdge("a", "b", kind="SUPPORTS"))
    kg.write_edge(KGEdge("a", "c", kind="CONTRADICTS"))
    kg.write_edge(KGEdge("b", "c", kind="SUPPORTS"))
    assert len(kg.list_edges(source_id="a")) == 2
    assert len(kg.list_edges(kind="SUPPORTS")) == 2
    assert len(kg.list_edges(source_id="a", kind="CONTRADICTS")) == 1


def test_delete_node_removes_incident_edges():
    kg = InMemoryKnowledgeGraph()
    for nid in ["a", "b"]:
        kg.write_node(KGNode(id=nid, kind="Belief"))
    kg.write_edge(KGEdge("a", "b", kind="SUPPORTS"))
    assert kg.delete_node("a") is True
    assert kg.list_edges() == []
    assert kg.delete_node("a") is False  # second delete is a no-op


# ── vector similarity ──────────────────────────────────────────────────


def test_similarity_query_returns_top_k_by_cosine():
    kg = InMemoryKnowledgeGraph()
    kg.write_node(KGNode(
        id="match", kind="Belief", embedding=(1.0, 0.0, 0.0),
    ))
    kg.write_node(KGNode(
        id="orthogonal", kind="Belief", embedding=(0.0, 1.0, 0.0),
    ))
    kg.write_node(KGNode(
        id="opposite", kind="Belief", embedding=(-1.0, 0.0, 0.0),
    ))
    hits = kg.query_by_similarity((1.0, 0.0, 0.0), top_k=2)
    assert [h.node.id for h in hits] == ["match", "orthogonal"]
    assert hits[0].score == pytest.approx(1.0)


def test_similarity_skips_nodes_without_embedding():
    kg = InMemoryKnowledgeGraph()
    kg.write_node(KGNode(id="no-embed", kind="Belief"))
    kg.write_node(KGNode(id="embed", kind="Belief", embedding=(0.5, 0.5)))
    hits = kg.query_by_similarity((0.5, 0.5), top_k=5)
    assert [h.node.id for h in hits] == ["embed"]


def test_similarity_filters_by_kind():
    kg = InMemoryKnowledgeGraph()
    kg.write_node(KGNode(id="b", kind="Belief", embedding=(1.0, 0.0)))
    kg.write_node(KGNode(id="e", kind="Evidence", embedding=(1.0, 0.0)))
    hits = kg.query_by_similarity((1.0, 0.0), top_k=5, kind="Evidence")
    assert [h.node.id for h in hits] == ["e"]


# ── address-keyed resource integration ────────────────────────────────


def test_build_kg_backend_returns_inmemory_when_no_match():
    resources = ResourceRegistry()
    backend = build_kg_backend(resources, AGENT)
    assert isinstance(backend, InMemoryKnowledgeGraph)


def test_build_kg_backend_returns_registered_backend():
    resources = ResourceRegistry()
    resources.register(
        InMemoryKnowledgeGraph, "acme", KG_KIND, "finance", "equities",
        name="alpha-graph",
    )
    backend = build_kg_backend(resources, AGENT)
    # Singleton — repeated builds return the same instance.
    assert backend is build_kg_backend(resources, AGENT)


def test_build_kg_backend_skips_resource_outside_allowed_pattern():
    resources = ResourceRegistry()
    resources.register(
        InMemoryKnowledgeGraph, "other-org", KG_KIND, "finance", "equities",
        name="off-org",
    )
    backend = build_kg_backend(resources, AGENT)
    # Falls back to a fresh in-memory KG (a different instance every call).
    assert isinstance(backend, InMemoryKnowledgeGraph)


def test_build_kg_backend_raises_on_multiple_matches():
    resources = ResourceRegistry()
    resources.register(
        InMemoryKnowledgeGraph, "acme", KG_KIND, "finance", "equities",
        name="one",
        allowed_for=AddressPattern.parse("acme.*.*.*.*.*.*"),
    )
    resources.register(
        InMemoryKnowledgeGraph, "acme", KG_KIND, "finance", "equities",
        name="two",
        allowed_for=AddressPattern.parse("acme.*.*.*.*.*.*"),
    )
    with pytest.raises(ValueError, match="multiple kg resources"):
        build_kg_backend(resources, AGENT)


def test_kg_mount_description_lists_visible_graphs():
    resources = ResourceRegistry()
    resources.register(
        InMemoryKnowledgeGraph, "acme", KG_KIND, "finance", "equities",
        name="primary", description="the canonical equities graph",
    )
    desc = kg_mount_description(resources, AGENT)
    assert "primary" in desc
    assert "equities graph" in desc


def test_kg_mount_description_empty_when_no_match():
    resources = ResourceRegistry()
    assert kg_mount_description(resources, AGENT) == ""


def test_kg_resource_addresses_returns_visible_addresses():
    resources = ResourceRegistry()
    resources.register(
        InMemoryKnowledgeGraph, "acme", KG_KIND, "finance", "equities",
        name="primary",
    )
    resources.register(
        InMemoryKnowledgeGraph, "acme", "fs", "finance", "equities",
        name="docs",
    )
    addrs = kg_resource_addresses(resources, AGENT)
    assert len(addrs) == 1
    assert isinstance(addrs[0], ResourceAddress)
    assert addrs[0].name == "primary"


# ── deterministic id helpers ───────────────────────────────────────────


def test_node_id_for_agent_is_stable():
    assert node_id_for_agent(AGENT) == node_id_for_agent(AGENT)
    assert node_id_for_agent(AGENT) != node_id_for_agent(OTHER)


def test_node_id_for_judgement_separates_by_timestamp():
    a = node_id_for_judgement(AGENT, OTHER, ts=1.0)
    b = node_id_for_judgement(AGENT, OTHER, ts=2.0)
    assert a != b


def test_node_id_for_rubric_matches_name():
    assert "my-rubric" in node_id_for_rubric("my-rubric")
