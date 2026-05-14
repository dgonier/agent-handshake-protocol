#!/usr/bin/env bash
# Start two AHP nodes pointing at the same Redis.
#
# Usage:
#   ./start.sh                         # uses AHP_REDIS_URL or redis://localhost:6379/0
#   AHP_REDIS_URL=redis://... ./start.sh
#
# Spawns:
#   - Node A (adversarial: Bull + Bear) on :8001
#   - Node B (researcher + HTTP) on :8002
# Both share the same Redis. Curl Node B's /query and watch the call
# fan out to Bull and Bear hosted on Node A.

set -eu

export AHP_REDIS_URL="${AHP_REDIS_URL:-redis://localhost:6379/0}"
echo "Using AHP_REDIS_URL=$AHP_REDIS_URL"

# Sanity-check Redis is reachable before booting either node.
python -c "
import os
import redis
url = os.environ['AHP_REDIS_URL']
client = redis.from_url(url, socket_connect_timeout=2)
client.ping()
print(f'Redis OK at {url}')
" || {
  echo "ERROR: Redis not reachable at $AHP_REDIS_URL"
  echo "Start one with: docker run --rm -p 6379:6379 redis:7-alpine"
  exit 1
}

# Start Node A in the background, Node B in the foreground.
trap 'kill 0' EXIT
uvicorn node_a:app --port 8001 --reload &
sleep 1
uvicorn node_b:app --port 8002 --reload
