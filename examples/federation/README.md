# AHP Federation — two FastAPI processes, one network

Concrete proof that AHP addresses are **universal strings**: two
independent Python processes share one Redis, and an agent on Node B
calls agents on Node A by URI alone — no service discovery, no import,
no inter-process plumbing.

## Topology

```
                     ┌────────────────────────┐
                     │   Redis (the network)  │
                     └────────────────────────┘
                           ▲              ▲
              port 6379    │              │   port 6379
                           │              │
        ┌──────────────────┘              └────────────────────┐
        │                                                       │
┌────────────────────┐                          ┌────────────────────┐
│ Node A  :8001      │                          │ Node B  :8002      │
│ (FastAPI + AHP)    │                          │ (FastAPI + AHP)    │
│                    │                          │                    │
│ Bull               │                          │ Researcher         │
│  tifin.adversarial │                          │  tifin.collaborative│
│  .finance.equities │                          │  .finance.equities │
│  .s.session.bull   │                          │  .s.session.researcher│
│                    │                          │                    │
│ Bear               │                          │ (talks to A by URI)│
│  ...session.bear   │                          │                    │
└────────────────────┘                          └────────────────────┘
```

Node B's researcher broadcasts `CAST-GET` to
`*.adversarial.finance.*.s.*.*`. The registry on the shared Redis
resolves that to Bull and Bear (hosted on Node A). Replies flow back
to B over the same Redis pub/sub. Node B's source code never imports
or references Node A.

## Run it

### 1. Start a Redis (any will do)

```bash
docker run --rm -p 6379:6379 redis:7-alpine
```

### 2. Install deps

```bash
pip install -e ../../             # the AHP library
pip install -r requirements.txt   # FastAPI + uvicorn + redis client
```

### 3. Start both nodes

```bash
./start.sh
# or manually:
#   uvicorn node_a:app --port 8001 &
#   uvicorn node_b:app --port 8002
```

### 4. Ask Node B a question

```bash
curl -X POST http://localhost:8002/query \
  -H "Content-Type: application/json" \
  -d '{
    "target": "tifin.collaborative.finance.equities.s.session.researcher",
    "body": "Tesla",
    "code": "human.query",
    "thread": "thread::devin",
    "timeout": 30
  }'
```

You'll see something like:

```json
{
  "target": "tifin.collaborative.finance.equities.s.session.researcher",
  "code": "human.query",
  "thread": "thread::devin",
  "body": "=== Analysis for Tesla ===\n\nBull view:\n  BULL view on Tesla: durable moat...\n\nBear view:\n  BEAR view on Tesla: regulatory headwinds...\n"
}
```

The researcher (on Node B) called `*.adversarial.finance.*.s.*.*`, the
registry resolved that to Bull and Bear (on Node A), and the replies
came back through Redis.

### 5. Watch live cross-node traffic

Open a tap on either node:

```bash
websocat 'ws://localhost:8001/observe?pattern=*.*.*.*.*.*.*&code=*'
# or :8002 — both see everything because the tap is the same Redis channel
```

### 6. Inspect the registry from either side

```bash
# Both nodes see all four agents — they're sharing the registry hash.
curl http://localhost:8001/agents
curl http://localhost:8002/agents
```

## What this proves

| Claim | How it's demonstrated |
|------|------|
| Addresses are universal | Node B references Node A's agents by URI string, never by Python object |
| The "network" is just the Redis substrate | Both nodes only know `AHP_REDIS_URL` to find each other |
| Discovery is shared, not federated | `*.adversarial.finance.*` resolves cross-process in one HGETALL |
| Tap is global | `/observe` on Node A or Node B sees the same traffic |
| The library is the SDK, not a service | No "AHP server" — two equal peers on the bus |

## Adding more nodes

Add `node_c.py` (a data agent, a vector-search service, another
researcher variant — anything). It just needs:

```python
from shared import build_stack
client, bus, registry, cache, engine, factory = build_stack()
my_agent = MyAgent(AgentAddress.parse("..."), engine, heartbeat_interval=0)
app = build_app(factory, agents=[my_agent], ...)
```

Same `AHP_REDIS_URL`. New port. That's it.

## What's deliberately NOT here

* **Auth.** Anyone with Redis access can register at any address. For
  multi-tenant deployments add a control plane that gates `register()`
  (and validates the `org` field of incoming addresses against signed
  tokens).
* **Replication / HA.** A single Redis is a single point of failure;
  use Redis Sentinel or Cluster in production.
* **Persistent threads beyond Redis stream limits.** The bus's xadd
  history grows unbounded by default; configure `MAXLEN` policies on
  thread streams when you ship.
