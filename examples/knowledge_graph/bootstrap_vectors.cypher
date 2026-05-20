// Bootstrap script for the AHP knowledge graph schema + vector index.
//
// Idempotent — safe to run on every deploy. The IF NOT EXISTS clauses
// turn re-runs into no-ops if the constraints / indexes are already
// at the requested shape.
//
// Run::
//
//     docker exec -i ahp-neo4j cypher-shell -u neo4j -p ahp-local-dev \
//         < bootstrap_vectors.cypher
//
// To change embedding dimensions, edit the OPTIONS block below. 1536 is
// OpenAI ada-002 / text-embedding-3-small (truncated); 768 is bge-base;
// 384 is bge-small; 4096 is Mistral large. The dimension MUST match
// whatever embedder the writer agent is using — Neo4j rejects vectors
// that don't fit the index at write time.

// ── 1. Uniqueness on KGNode.id ─────────────────────────────────────────
// Lets MERGE (n:KGNode {id: $id}) act as a fast upsert. Required for
// the adapter to be safe under concurrent writers.

CREATE CONSTRAINT kg_node_id_unique IF NOT EXISTS
FOR (n:KGNode) REQUIRE n.id IS UNIQUE;


// ── 2. Vector index on KGNode.embedding ───────────────────────────────
// Native Neo4j vector index. The mistake from the previous deployment
// was using a plain SET to write embeddings, which can leave the index
// stale. The adapter uses db.create.setNodeVectorProperty(...) which
// is the index-safe path.
//
// IMPORTANT: change `vector.dimensions` here AND in your Python code
// (Neo4jKnowledgeGraph(vector_dimensions=...)) at the same time. They
// must agree or queries will throw.

CREATE VECTOR INDEX kg_node_embedding IF NOT EXISTS
FOR (n:KGNode) ON (n.embedding)
OPTIONS {indexConfig: {
  `vector.dimensions`: 1536,
  `vector.similarity_function`: 'cosine'
}};


// ── 3. Helpful secondary indexes ──────────────────────────────────────
// Used for the agent-as-judge query patterns: "find all Judgements
// about agent X" and "all nodes of kind Y".

CREATE INDEX kg_node_kind_idx IF NOT EXISTS
FOR (n:KGNode) ON (n.kind);

CREATE INDEX kg_edge_kind_idx IF NOT EXISTS
FOR ()-[r:KG_EDGE]-() ON (r.kind);


// ── 4. Sanity check ───────────────────────────────────────────────────
// Run a no-op query so the script exits non-zero if the connection
// itself was broken. SHOW INDEXES returns rows that the operator can
// eyeball to confirm everything got created.

SHOW INDEXES WHERE name STARTS WITH 'kg_';
