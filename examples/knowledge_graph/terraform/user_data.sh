#!/bin/bash
# Cloud-init bootstrap for the AHP Neo4j box.
#
# Installs Docker, pulls neo4j:5.20 with APOC + GDS plugins, and runs
# the same vector-index setup as the local docker-compose path. Logs
# go to /var/log/cloud-init-output.log on the instance.

set -euxo pipefail

# Templated by Terraform:
NEO4J_PASSWORD='${neo4j_password}'
VECTOR_DIMENSIONS='${vector_dimensions}'

# ── packages ──────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y ca-certificates curl gnupg lsb-release

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# shellcheck disable=SC2155,SC1091
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
    > /etc/apt/sources.list.d/docker.list

apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# ── Neo4j data dirs ───────────────────────────────────────────────────
mkdir -p /opt/ahp-neo4j/{data,logs,plugins,import}
chown -R 7474:7474 /opt/ahp-neo4j

# ── compose file ──────────────────────────────────────────────────────
cat > /opt/ahp-neo4j/docker-compose.yml <<EOF
services:
  neo4j:
    image: neo4j:5.20-enterprise
    container_name: ahp-neo4j
    restart: unless-stopped
    ports:
      - "0.0.0.0:7474:7474"
      - "0.0.0.0:7687:7687"
    environment:
      NEO4J_AUTH: "neo4j/$${NEO4J_PASSWORD}"
      NEO4J_ACCEPT_LICENSE_AGREEMENT: "yes"
      NEO4J_PLUGINS: '["apoc", "graph-data-science"]'
      NEO4J_dbms_security_procedures_unrestricted: "apoc.*,gds.*"
      NEO4J_dbms_security_procedures_allowlist: "apoc.*,gds.*,db.*"
      NEO4J_server_memory_heap_initial__size: "1G"
      NEO4J_server_memory_heap_max__size: "3G"
      NEO4J_server_memory_pagecache_size: "1G"
    volumes:
      - /opt/ahp-neo4j/data:/data
      - /opt/ahp-neo4j/logs:/logs
      - /opt/ahp-neo4j/plugins:/plugins
      - /opt/ahp-neo4j/import:/var/lib/neo4j/import
EOF

# Pass the password into the compose env without baking it into the file.
export NEO4J_PASSWORD
cd /opt/ahp-neo4j
docker compose up -d

# ── wait for Neo4j to come up ─────────────────────────────────────────
# 5.20 with plugins typically takes 60-90s to be query-ready.
for i in $(seq 1 30); do
    if docker exec ahp-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
        'RETURN 1' >/dev/null 2>&1; then
        echo "neo4j ready after $i attempts"
        break
    fi
    sleep 5
done

# ── vector + constraint bootstrap ─────────────────────────────────────
docker exec -i ahp-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" <<EOF
CREATE CONSTRAINT kg_node_id_unique IF NOT EXISTS
FOR (n:KGNode) REQUIRE n.id IS UNIQUE;

CREATE VECTOR INDEX kg_node_embedding IF NOT EXISTS
FOR (n:KGNode) ON (n.embedding)
OPTIONS {indexConfig: {
  \`vector.dimensions\`: $VECTOR_DIMENSIONS,
  \`vector.similarity_function\`: 'cosine'
}};

CREATE INDEX kg_node_kind_idx IF NOT EXISTS
FOR (n:KGNode) ON (n.kind);

CREATE INDEX kg_edge_kind_idx IF NOT EXISTS
FOR ()-[r:KG_EDGE]-() ON (r.kind);
EOF

echo "ahp-neo4j bootstrap complete"
