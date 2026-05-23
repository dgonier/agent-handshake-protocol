# Knowledge graph backend for AHP

This example wires a Neo4j-backed `KnowledgeGraphBackend` into AHP and
demonstrates the `TeacherAgent` writing judgements into it. It exists
as boilerplate so the next deployment doesn't have to redo the
infra-side work — particularly the vector index, which is the piece
that was easy to get wrong in the previous round.

## What's here

```
examples/knowledge_graph/
├── README.md                       you are here
├── docker-compose.neo4j.yml        local Neo4j 5 with APOC + GDS, ports 7474/7687
├── bootstrap_vectors.cypher        idempotent vector-index setup
├── teacher_demo.py                 end-to-end: spin up an in-process AHP stack,
│                                   register a Neo4j KG resource, run a Teacher
└── terraform/
    ├── main.tf                     AWS EC2 + Docker Compose deployment
    ├── variables.tf
    ├── outputs.tf
    └── user_data.sh                cloud-init that brings Neo4j up
```

## Quickstart (local)

```bash
# 1. Bring up Neo4j locally.
cd examples/knowledge_graph
docker compose -f docker-compose.neo4j.yml up -d

# 2. Apply the vector-index bootstrap. Cypher is idempotent — safe to
# rerun on every deploy. Edit the EMBEDDING_DIMS to match your model
# (1536 = OpenAI ada-002, 768 = bge-base, 384 = bge-small, 4096 = Mistral).
docker exec -i ahp-neo4j cypher-shell -u neo4j -p ahp-local-dev \
    < bootstrap_vectors.cypher

# 3. Drive it from Python.
pip install -e ".[kg]"
NEO4J_URI=bolt://localhost:7687 \
NEO4J_USERNAME=neo4j \
NEO4J_PASSWORD=ahp-local-dev \
python teacher_demo.py
```

The Neo4j browser is at http://localhost:7474. Login is
`neo4j / ahp-local-dev`. The default schema after bootstrap:

* `:KGNode` is the generic label; `:KGNode:Belief`, `:KGNode:Judgement`,
  `:KGNode:Agent` etc. are added on write so Cypher
  `MATCH (j:Judgement)` works the way you'd expect.
* `[:KG_EDGE {kind, ...}]` carries the edge type as a property because
  Neo4j relationship types must be known at write time.
* `kg_node_embedding` is a native vector index on `KGNode.embedding`.
  Default dimensions: 1536. Override via the env var
  `AHP_KG_VECTOR_DIMS` or pass the dimension to `Neo4jKnowledgeGraph(...)`
  at registration.

## The vector-index gotcha

When the vector property is set with a plain Cypher `SET n.embedding = $v`,
Neo4j stores the list but doesn't always wire it into the index — version
drift between 5.13, 5.15, and 5.18 changed the behavior. The adapter
uses `db.create.setNodeVectorProperty(n, 'embedding', $v)`, which is
the supported path across all 5.x and is what `bootstrap_vectors.cypher`
expects. **Don't switch back to plain SET** — that's the mistake from
last time.

## Terraform

Single-node Neo4j on a `t3.medium` EC2 instance with EBS storage. Not
HA — for production you'd swap in AuraDB or a 3-node cluster.

```bash
cd examples/knowledge_graph/terraform
terraform init
terraform apply \
    -var="key_name=$YOUR_AWS_KEY" \
    -var="allowed_cidr=$YOUR_IP/32" \
    -var="neo4j_password=$(openssl rand -base64 24)"
```

Outputs the bolt URL + browser URL. The user_data script bootstraps
Docker, pulls `neo4j:5.20-enterprise`, and runs the same
`bootstrap_vectors.cypher` after startup.

## Wiring into your AHP setup

```python
from ahp.adapters import resource
from ahp.adapters.neo4j_kg import Neo4jKnowledgeGraph

@resource("acme", "kg", "finance", "equities",
          name="primary",
          description="canonical Tesla/SPY belief graph",
          cleanup=lambda g: g.close())
def make_primary_kg():
    return Neo4jKnowledgeGraph(
        vector_dimensions=1536,
        auto_create_vector_index=True,
    )
```

Any agent whose address matches `acme.*.finance.equities.*.*.*` will
get this backend resolved by `build_kg_backend(...)`. The
`TeacherAgent.from_profile(..., resources=registry)` path picks it up
automatically.
