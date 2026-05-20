"""Knowledge graph as an addressable resource — sketch.

A *knowledge graph* is a long-lived store of typed nodes and typed edges
that AHP agents read from and write to. It sits in the address space at
``scope.kg.domain.subdomain.name`` (resource kind ``"kg"``), the same
way filesystem backends sit at ``scope.fs.*``.

The protocol layer ships only the shape:

* :class:`KGNode` — id, kind, label, props, optional embedding.
* :class:`KGEdge` — source/target ids, kind, props.
* :class:`KnowledgeGraphBackend` — Protocol with the minimal CRUD +
  query surface.
* :class:`InMemoryKnowledgeGraph` — zero-dep reference implementation
  used by tests and small demos.
* :func:`build_kg_backend`, :func:`kg_mount_description`,
  :func:`kg_resource_addresses` — helpers that mirror
  ``ahp.adapters.storage`` for the ``fs`` kind, so the same address-keyed
  resolution pattern works.

A real backend (Neo4j, KuzuDB, in-process embedded) plugs in by
implementing :class:`KnowledgeGraphBackend` and registering as a
:func:`~ahp.adapters.resources.resource` of kind ``"kg"``. See
``ahp.adapters.neo4j_kg`` for the Neo4j adapter (opt-in extra).

This module is intentionally a *sketch*: enough shape to let the Teacher
agent and example wiring work end-to-end. The next phase fleshes out
versioning, provenance, conflict resolution, and federation across
multiple KG resources.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Protocol, runtime_checkable

from ahp.adapters.resources import ResourceBinding, ResourceRegistry
from ahp.adapters.tool_address import ResourceAddress
from ahp.core.address import AgentAddress


KG_KIND: str = "kg"
"""``ResourceAddress.kind`` that marks a resource as a knowledge graph backend."""


# ── data shapes ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class KGNode:
    """A typed node in the knowledge graph.

    * ``id`` — caller-stable id (deterministic preferred). Auto-assigned
      via :meth:`KnowledgeGraphBackend.write_node` if blank.
    * ``kind`` — coarse label (``"Agent"``, ``"Belief"``, ``"Evidence"``,
      ``"Judgement"``, ``"Rubric"`` …). Maps onto a Neo4j label.
    * ``label`` — human-readable display string.
    * ``props`` — free-form attributes. Backend-side serialization decides
      what's queryable.
    * ``embedding`` — optional vector for similarity search. The Neo4j
      adapter wires this into a native vector index; the in-memory
      backend exposes a cosine-similarity ``query_by_similarity``.
    """

    id: str
    kind: str
    label: str = ""
    props: Mapping[str, Any] = field(default_factory=dict)
    embedding: tuple[float, ...] | None = None

    def with_id(self, new_id: str) -> "KGNode":
        return KGNode(
            id=new_id, kind=self.kind, label=self.label,
            props=dict(self.props), embedding=self.embedding,
        )


@dataclass(frozen=True)
class KGEdge:
    """A typed directed edge between two :class:`KGNode`-s."""

    source_id: str
    target_id: str
    kind: str
    props: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KGSimilarityHit:
    """A single result from a similarity query."""

    node: KGNode
    score: float


# ── backend protocol ───────────────────────────────────────────────────


@runtime_checkable
class KnowledgeGraphBackend(Protocol):
    """The minimal surface every KG backend must implement.

    Methods are intentionally sync-friendly; an async backend can return
    awaitables and the Teacher agent awaits them. Tests use the sync
    in-memory backend.
    """

    def write_node(self, node: KGNode) -> KGNode: ...
    def write_edge(self, edge: KGEdge) -> KGEdge: ...
    def get_node(self, node_id: str) -> KGNode | None: ...
    def list_nodes(self, *, kind: str | None = None) -> list[KGNode]: ...
    def list_edges(
        self, *, source_id: str | None = None, target_id: str | None = None,
        kind: str | None = None,
    ) -> list[KGEdge]: ...
    def delete_node(self, node_id: str) -> bool: ...
    def close(self) -> None: ...


# ── in-memory reference implementation ────────────────────────────────


class InMemoryKnowledgeGraph:
    """Zero-dep KG backend.

    Holds nodes and edges in dicts. Useful for tests, single-process
    demos, and as the default when no resource of kind ``kg`` is wired.

    The :meth:`query_by_similarity` method does brute-force cosine over
    all nodes that have an ``embedding``. Fine for hundreds of nodes;
    swap in a Neo4j-backed adapter (or any vector store) for production.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, KGNode] = {}
        self._edges: list[KGEdge] = []

    # ── writes ────────────────────────────────────────────────────────

    def write_node(self, node: KGNode) -> KGNode:
        stored = node if node.id else node.with_id(_new_id("node"))
        self._nodes[stored.id] = stored
        return stored

    def write_edge(self, edge: KGEdge) -> KGEdge:
        if edge.source_id not in self._nodes:
            raise KeyError(f"unknown source_id {edge.source_id!r}")
        if edge.target_id not in self._nodes:
            raise KeyError(f"unknown target_id {edge.target_id!r}")
        self._edges.append(edge)
        return edge

    # ── reads ─────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> KGNode | None:
        return self._nodes.get(node_id)

    def list_nodes(self, *, kind: str | None = None) -> list[KGNode]:
        if kind is None:
            return list(self._nodes.values())
        return [n for n in self._nodes.values() if n.kind == kind]

    def list_edges(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: str | None = None,
    ) -> list[KGEdge]:
        out: list[KGEdge] = []
        for edge in self._edges:
            if source_id is not None and edge.source_id != source_id:
                continue
            if target_id is not None and edge.target_id != target_id:
                continue
            if kind is not None and edge.kind != kind:
                continue
            out.append(edge)
        return out

    # ── delete / lifecycle ────────────────────────────────────────────

    def delete_node(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        self._edges = [
            e for e in self._edges
            if e.source_id != node_id and e.target_id != node_id
        ]
        return True

    def close(self) -> None:
        self._nodes.clear()
        self._edges.clear()

    # ── vector search ─────────────────────────────────────────────────

    def query_by_similarity(
        self, embedding: Iterable[float], *, top_k: int = 5,
        kind: str | None = None,
    ) -> list[KGSimilarityHit]:
        target = tuple(embedding)
        hits: list[KGSimilarityHit] = []
        for node in self._nodes.values():
            if node.embedding is None:
                continue
            if kind is not None and node.kind != kind:
                continue
            score = _cosine(target, node.embedding)
            hits.append(KGSimilarityHit(node=node, score=score))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ── address-keyed resource integration ────────────────────────────────


def _matching_kg_bindings(
    resources: ResourceRegistry,
    agent_address: AgentAddress,
) -> list[ResourceBinding]:
    out: list[ResourceBinding] = []
    for binding in resources.bindings():
        if binding.address.kind != KG_KIND:
            continue
        if not binding.allowed_for.matches(agent_address):
            continue
        out.append(binding)
    return out


def build_kg_backend(
    resources: ResourceRegistry,
    agent_address: AgentAddress,
    *,
    default: KnowledgeGraphBackend | None = None,
) -> KnowledgeGraphBackend:
    """Return the KG backend an agent at ``agent_address`` should use.

    * No matching ``kg`` resource → returns ``default`` if given,
      otherwise a fresh :class:`InMemoryKnowledgeGraph`.
    * Exactly one match → that backend.
    * Multiple matches → raises :class:`ValueError`. Future phases can
      add a routing layer (sharding nodes across multiple graphs by
      domain) but for the sketch we surface the ambiguity at wiring
      time.
    """
    bindings = _matching_kg_bindings(resources, agent_address)
    if not bindings:
        return default if default is not None else InMemoryKnowledgeGraph()
    if len(bindings) == 1:
        return resources.get(str(bindings[0].address))
    addresses = ", ".join(str(b.address) for b in bindings)
    raise ValueError(
        f"agent {agent_address} matches multiple kg resources ({addresses}); "
        f"tighten allowed_for on one of them or wire a routing layer."
    )


def kg_mount_description(
    resources: ResourceRegistry,
    agent_address: AgentAddress,
    *,
    header: str = "Available knowledge graphs:",
) -> str:
    """System-prompt fragment listing this agent's KG mounts. Empty when none."""
    lines: list[str] = []
    for binding in _matching_kg_bindings(resources, agent_address):
        desc = binding.description or binding.address.name
        lines.append(f"- {binding.address} — {desc}")
    if not lines:
        return ""
    return header + "\n" + "\n".join(lines)


def kg_resource_addresses(
    resources: ResourceRegistry,
    agent_address: AgentAddress,
) -> list[ResourceAddress]:
    """The :class:`ResourceAddress`-es of every KG backend visible to the agent."""
    return [b.address for b in _matching_kg_bindings(resources, agent_address)]


# ── deterministic node ids ────────────────────────────────────────────


def node_id_for_agent(agent_address: AgentAddress) -> str:
    """Stable KG node id for an :class:`AgentAddress`.

    Useful when multiple teachers write judgements about the same agent
    and we want them all hanging off one ``Agent`` node.
    """
    return f"agent::{agent_address}"


def node_id_for_rubric(name: str) -> str:
    return f"rubric::{name}"


def node_id_for_judgement(
    judge: AgentAddress, subject: AgentAddress, ts: float | None = None,
) -> str:
    when = ts if ts is not None else time.time()
    return f"judgement::{judge}::{subject}::{when:.6f}"
