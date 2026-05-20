"""Neo4j-backed :class:`KnowledgeGraphBackend`.

Opt-in adapter — requires the ``neo4j`` Python driver (install via the
``[kg]`` extra). The in-memory backend in :mod:`ahp.adapters.knowledge_graph`
is sufficient for tests and small demos; switch to this adapter when you
want persistence, multi-process sharing, or native vector indexing.

Schema (mirrors :class:`~ahp.adapters.knowledge_graph.KGNode` /
:class:`~ahp.adapters.knowledge_graph.KGEdge`):

::

    (:KGNode {id, kind, label, props..., embedding?})
    (:KGNode)-[:KG_EDGE {kind, props...}]->(:KGNode)

Nodes carry their ``kind`` as both a label (``:KGNode:Belief``) and a
property, so a Cypher ``MATCH (n:Belief)`` works as well as a property
filter. Edges keep ``kind`` as a property because Neo4j relationship
types must be known at write time and we want callers to pass any string.

Vector index
------------

The thing we got wrong last time was bolting vector search on as an
afterthought. Here it's first-class:

* :meth:`ensure_vector_index` creates a native Neo4j vector index on
  ``KGNode.embedding`` at the configured dimensions and similarity
  function. Idempotent — safe to call on every connection.
* :meth:`query_by_similarity` uses ``db.index.vector.queryNodes`` so
  the search is index-backed, not a Cypher-side cosine over every row.
* The index name, dimensions, and similarity function are constructor
  args, *not* hardcoded — different KG resources can run at different
  embedding dims (e.g. 384 for bge-small, 1536 for OpenAI ada-002).

Connection
----------

Standard ``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD`` env vars
(matches the AuraDB / community defaults). The driver is created lazily
on first use so ``Neo4jKnowledgeGraph(...)`` is cheap to instantiate at
factory-decoration time.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

from ahp.adapters.knowledge_graph import (
    KGEdge,
    KGNode,
    KGSimilarityHit,
    KnowledgeGraphBackend,
)


log = logging.getLogger(__name__)


_DEFAULT_INDEX_NAME = "kg_node_embedding"
_DEFAULT_DIMENSIONS = 1536
_DEFAULT_SIMILARITY = "cosine"  # one of: cosine, euclidean


class Neo4jKnowledgeGraph(KnowledgeGraphBackend):
    """Neo4j-backed implementation of :class:`KnowledgeGraphBackend`.

    Construct with explicit credentials or rely on the standard
    ``NEO4J_URI`` / ``NEO4J_USERNAME`` / ``NEO4J_PASSWORD`` env vars.

    The vector index is *not* created on construction — call
    :meth:`ensure_vector_index` once at startup (or pass
    ``auto_create_vector_index=True``).
    """

    def __init__(
        self,
        uri: str | None = None,
        username: str | None = None,
        password: str | None = None,
        *,
        database: str = "neo4j",
        vector_index_name: str = _DEFAULT_INDEX_NAME,
        vector_dimensions: int = _DEFAULT_DIMENSIONS,
        vector_similarity: str = _DEFAULT_SIMILARITY,
        auto_create_vector_index: bool = False,
    ) -> None:
        self._uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        self._username = username or os.environ.get("NEO4J_USERNAME", "neo4j")
        self._password = password or os.environ.get("NEO4J_PASSWORD", "neo4j")
        self._database = database
        self._vector_index_name = vector_index_name
        self._vector_dimensions = vector_dimensions
        self._vector_similarity = vector_similarity
        self._driver: Any | None = None
        if auto_create_vector_index:
            self.ensure_vector_index()

    # ── connection ────────────────────────────────────────────────────

    def _connect(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase  # local import; optional dep

            self._driver = GraphDatabase.driver(
                self._uri, auth=(self._username, self._password),
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _session(self):
        return self._connect().session(database=self._database)

    # ── index setup ───────────────────────────────────────────────────

    def ensure_vector_index(self) -> None:
        """Idempotently create the vector index on ``KGNode.embedding``.

        Neo4j 5.13+ syntax. The index name, dimensions, and similarity
        function are configurable on the backend; this method just
        materializes them. Safe to call repeatedly — Cypher's
        ``IF NOT EXISTS`` makes it a no-op when the index is already
        present at the same shape.
        """
        cypher = (
            "CREATE VECTOR INDEX $name IF NOT EXISTS "
            "FOR (n:KGNode) ON (n.embedding) "
            "OPTIONS {indexConfig: {"
            "  `vector.dimensions`: $dims, "
            "  `vector.similarity_function`: $sim"
            "}}"
        )
        with self._session() as s:
            s.run(
                cypher,
                name=self._vector_index_name,
                dims=self._vector_dimensions,
                sim=self._vector_similarity,
            )

    # ── writes ────────────────────────────────────────────────────────

    def write_node(self, node: KGNode) -> KGNode:
        stored = node if node.id else node.with_id(_new_id("node"))
        params = {
            "id": stored.id,
            "kind": stored.kind,
            "label": stored.label,
            "props": dict(stored.props),
            "embedding": list(stored.embedding) if stored.embedding else None,
        }
        cypher = (
            "MERGE (n:KGNode {id: $id}) "
            "SET n.kind = $kind, n.label = $label, n += $props "
            "WITH n "
            "CALL apoc.create.addLabels(n, [$kind]) YIELD node "
            "RETURN node"
        )
        cypher_no_apoc = (
            "MERGE (n:KGNode {id: $id}) "
            "SET n.kind = $kind, n.label = $label, n += $props "
            "RETURN n"
        )
        embedding_cypher = (
            "MATCH (n:KGNode {id: $id}) "
            "CALL db.create.setNodeVectorProperty(n, 'embedding', $embedding) "
            "RETURN n"
        )
        with self._session() as s:
            # Try APOC for dynamic label; fall back gracefully if not installed.
            try:
                s.run(cypher, **params)
            except Exception:
                s.run(cypher_no_apoc, **params)
            if params["embedding"] is not None:
                s.run(
                    embedding_cypher,
                    id=params["id"], embedding=params["embedding"],
                )
        return stored

    def write_edge(self, edge: KGEdge) -> KGEdge:
        cypher = (
            "MATCH (a:KGNode {id: $source_id}), (b:KGNode {id: $target_id}) "
            "MERGE (a)-[r:KG_EDGE {kind: $kind}]->(b) "
            "SET r += $props "
            "RETURN r"
        )
        with self._session() as s:
            result = s.run(
                cypher,
                source_id=edge.source_id,
                target_id=edge.target_id,
                kind=edge.kind,
                props=dict(edge.props),
            )
            if result.peek() is None:
                raise KeyError(
                    f"edge endpoints not found: {edge.source_id!r} -> "
                    f"{edge.target_id!r}"
                )
        return edge

    # ── reads ─────────────────────────────────────────────────────────

    def get_node(self, node_id: str) -> KGNode | None:
        with self._session() as s:
            record = s.run(
                "MATCH (n:KGNode {id: $id}) RETURN n", id=node_id,
            ).single()
        if record is None:
            return None
        return _node_from_record(record["n"])

    def list_nodes(self, *, kind: str | None = None) -> list[KGNode]:
        if kind is None:
            cypher = "MATCH (n:KGNode) RETURN n"
            params: dict[str, Any] = {}
        else:
            cypher = "MATCH (n:KGNode {kind: $kind}) RETURN n"
            params = {"kind": kind}
        with self._session() as s:
            return [_node_from_record(r["n"]) for r in s.run(cypher, **params)]

    def list_edges(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        kind: str | None = None,
    ) -> list[KGEdge]:
        conditions: list[str] = []
        params: dict[str, Any] = {}
        if source_id is not None:
            conditions.append("a.id = $source_id")
            params["source_id"] = source_id
        if target_id is not None:
            conditions.append("b.id = $target_id")
            params["target_id"] = target_id
        if kind is not None:
            conditions.append("r.kind = $kind")
            params["kind"] = kind
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        cypher = (
            f"MATCH (a:KGNode)-[r:KG_EDGE]->(b:KGNode) {where} "
            "RETURN a.id AS source_id, b.id AS target_id, r"
        )
        with self._session() as s:
            return [
                KGEdge(
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    kind=r["r"].get("kind", ""),
                    props={
                        k: v for k, v in dict(r["r"]).items() if k != "kind"
                    },
                )
                for r in s.run(cypher, **params)
            ]

    def delete_node(self, node_id: str) -> bool:
        with self._session() as s:
            result = s.run(
                "MATCH (n:KGNode {id: $id}) "
                "DETACH DELETE n RETURN count(n) AS deleted",
                id=node_id,
            ).single()
        return bool(result and result["deleted"])

    # ── vector search ─────────────────────────────────────────────────

    def query_by_similarity(
        self,
        embedding: Iterable[float],
        *,
        top_k: int = 5,
        kind: str | None = None,
    ) -> list[KGSimilarityHit]:
        """Index-backed similarity over :class:`KGNode` embeddings."""
        vector = list(embedding)
        if len(vector) != self._vector_dimensions:
            raise ValueError(
                f"embedding has {len(vector)} dims; index expects "
                f"{self._vector_dimensions}. Construct the backend with the "
                f"right vector_dimensions or pad/truncate the query."
            )
        cypher = (
            "CALL db.index.vector.queryNodes($name, $top_k, $vector) "
            "YIELD node, score "
            + ("WHERE node.kind = $kind " if kind is not None else "")
            + "RETURN node, score"
        )
        params: dict[str, Any] = {
            "name": self._vector_index_name,
            "top_k": top_k,
            "vector": vector,
        }
        if kind is not None:
            params["kind"] = kind
        with self._session() as s:
            return [
                KGSimilarityHit(
                    node=_node_from_record(r["node"]),
                    score=float(r["score"]),
                )
                for r in s.run(cypher, **params)
            ]


def _node_from_record(node_record: Any) -> KGNode:
    data = dict(node_record)
    kind = data.pop("kind", "")
    node_id = data.pop("id", "")
    label = data.pop("label", "")
    embedding_raw = data.pop("embedding", None)
    embedding = tuple(embedding_raw) if embedding_raw else None
    return KGNode(
        id=node_id, kind=kind, label=label, props=data, embedding=embedding,
    )


def _new_id(prefix: str) -> str:
    import uuid as _uuid
    return f"{prefix}-{_uuid.uuid4().hex[:12]}"
