# AHP — Agentic Handshake Protocol

A typed, addressable, format-aware messaging protocol for heterogeneous
agents. Agents are identified by structured URIs, exchange messages
tagged with hierarchical interaction codes, and negotiate payload format
through a single shared compatibility matrix.

This repository implements **Phases 1 + 2**: core primitives plus the
Redis-backed transport, cache, and registry. The engine and framework
adapters land in later phases.

## Status

| Phase | Module | State |
|-------|--------|-------|
| 1 | `ahp.core` (addresses, patterns, codes, messages, compatibility) | implemented |
| 2 | `ahp.transport` (RedisBus, ProtocolCache), `ahp.registry` | implemented |
| 3 | `ahp.engine` | not started |
| 4 | `ahp.adapters` (LangGraph, DSPy, deep agent, human) | not started |
| 5 | `ahp.demo` | not started |

## Install

```bash
pip install -e ".[test]"          # for development
pip install -e ".[redis]"         # core + transport/registry
```

`ahp.core` has zero runtime dependencies. Importing `ahp.transport` or
`ahp.registry` requires `redis>=5.0` (install via the `redis` extra).
The `test` extras also pull in `pytest`, `pytest-asyncio`, `fakeredis`,
and `hypothesis`.

## Address Format

```
{org}.{role}.{domain}.{subdomain}.{accept}.{lifecycle}.{instance}?{params}
```

Example: `tifin.adversarial.finance.projections.j.longterm.frank?stock=Tesla`

| Field | Meaning |
|-------|---------|
| `org` | Namespace (`tifin`, `public`, `user-devin`) |
| `role` | `adversarial`, `collaborative`, `interview`, `human`, … |
| `domain` | Top-level subject (`finance`, `science`, `general`) |
| `subdomain` | Specialization (`projections`, `biology`) |
| `accept` | Payload tier set in canonical order (`s`, `j`, `b`, `e` → string, JSON, bytes, embeddings); combinations like `sj`, `sjbe` |
| `lifecycle` | `longterm` / `session` / `ephemeral` / `stale-ok` (drives cache TTL) |
| `instance` | Agent identity within the same role/domain |
| `params` | URL-encoded query string |

```python
from ahp.core import AgentAddress

addr = AgentAddress.parse("tifin.adversarial.finance.projections.j.longterm.frank?stock=Tesla")
addr.accepts("j")               # True
addr.cache_key("interview.text")  # deterministic SHA-256
```

## Patterns

Wildcard patterns route broadcast messages. Accept fields use **subset
semantics** — a pattern accept of `sj` requires the target to accept at
least both `s` and `j`.

```python
from ahp.core import AddressPattern

pat = AddressPattern.parse("*.adversarial.science.*.s.*.*")
pat.matches(addr)
```

## Interaction Codes

Hierarchical, dot-delimited code constants live on `Code`:

```python
from ahp.core import Code

Code.INTERVIEW_TEXT          # "interview.text"
Code.ADVERSARIAL_DEBATE      # "adversarial.debate"
Code.family(Code.HUMAN_HALT) # "human"
Code.matches("interview.text", "interview.*")  # True
```

## Messages

```python
from ahp.core import AgentAddress, AddressPattern, Code, Message

src = AgentAddress.parse("demo.collaborative.finance.equities.s.session.alice")
pat = AddressPattern.parse("*.adversarial.finance.*.s.*.*")

msg = Message(
    source=src,
    target=pat,
    verb="CAST-GET",
    code=Code.ADVERSARIAL_DEBATE,
    thread="thread::tesla-12m",
    body="Make the case against Tesla.",
)
msg.is_broadcast       # True
msg.expects_response   # True
msg.ttl                # 3600 (derived from "session" lifecycle)
restored = Message.from_dict(msg.to_dict())
```

Valid verbs: `SEND`, `SEND-GET`, `CAST`, `CAST-GET`, `CAST-SUB`,
`INVALIDATE`.

## Compatibility Matrix

```python
from ahp.core import CompatibilityMatrix, Code, AgentAddress

m = CompatibilityMatrix()
m.required_tiers(Code.INTERVIEW_EMBEDDINGS)   # {"b", "e"}
m.can_route(src, AgentAddress.parse("o.r.d.sd.j.session.i"), Code.INTERVIEW_SCHEMA)  # True
```

A target satisfies a code if its `accept` set intersects the code's
required tier set (any-of semantics).

## Transport — `RedisBus`

`RedisBus` carries messages over Redis pub/sub (delivery) and Redis
streams (durable thread history). It does not resolve address patterns —
callers pass pre-resolved target lists, and the engine handles registry
lookups in Phase 3.

```python
from ahp.transport import RedisBus
import redis.asyncio as aioredis

client = aioredis.from_url("redis://localhost", decode_responses=True)
bus = RedisBus(client)

# point-to-point with reply collection
reply = await bus.send_get(request_msg, timeout=5.0)

# broadcast fan-out with bounded collection
replies = await bus.cast_get(
    request_msg, targets=[bob, carol], timeout=5.0, max_responses=2,
)

# durable thread history
history = await bus.get_thread("thread::tesla-12m")
```

Verb semantics: `SEND` / `SEND-GET` require an `AgentAddress` target;
`CAST*` accept patterns at the envelope layer but the bus's `cast()` /
`cast_get()` take a pre-resolved target list. Bodies must be
JSON-serializable (strings, dicts, lists, numbers); base64-encode bytes
upstream.

## Cache — `ProtocolCache`

Read-through cache keyed by SHA-256 of `(target_uri, code)`. TTL is
derived from the target's lifecycle field: `longterm`=24h, `session`=1h,
`stale-ok`=7d, `ephemeral` skips caching entirely. `invalidate()`
supports pattern + param filters by scanning the namespace.

```python
from ahp.transport import ProtocolCache

cache = ProtocolCache(client)
hit = await cache.get(request)
if hit is None:
    response = await bus.send_get(request, timeout=5.0)
    await cache.put(request, response)

# bust everything for a specific stock
await cache.invalidate(
    AddressPattern.parse("*.adversarial.finance.*.*.*.*"),
    params={"stock": "Tesla"},
)
```

## Registry — `AgentRegistry`

Redis-backed agent directory with TTL liveness markers. `register()`
stores `AgentMeta` and marks the agent live for `heartbeat_ttl` seconds
(default 30); subsequent `heartbeat()` calls refresh the marker.
`resolve()` returns alive agents matching a pattern.

```python
from ahp.registry import AgentRegistry, AgentMeta

registry = AgentRegistry(client, heartbeat_ttl=30)
await registry.register(
    AgentAddress.parse("demo.adversarial.finance.equities.s.session.frank"),
    AgentMeta(capabilities=["debate", "valuation"], reputation=0.9),
)

# pattern resolution (alive_only=True by default)
candidates = await registry.resolve(
    AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
)

# rich discovery with capability + reputation filters
experts = await registry.discover(
    role="adversarial", domain="finance",
    capability="valuation", min_reputation=0.7,
)
```

## Tests

```bash
pytest
```

142 tests, all passing. Includes async tests over `fakeredis` for the
bus (point-to-point delivery, send/get reply collection, broadcast
fan-out, bounded `cast_get`, background consumers, dict body
round-trips, bytes rejection), the cache (TTL derivation, key
stability under param reordering, pattern + param invalidation), and
the registry (pattern resolution, liveness expiry, discovery filters).

## Layout

```
ahp/
├── core/
│   ├── address.py        AgentAddress
│   ├── pattern.py        AddressPattern
│   ├── codes.py          Code constants + family helpers
│   ├── message.py        Message envelope + verbs + TTL table
│   └── compatibility.py  CompatibilityMatrix
├── transport/
│   ├── keys.py           Redis key/channel name conventions
│   ├── redis_bus.py      RedisBus + Subscription
│   └── cache.py          ProtocolCache + CachedEntry
└── registry/
    └── registry.py       AgentRegistry + AgentMeta
tests/
    test_address.py  test_pattern.py  test_codes.py  test_message.py
    test_compatibility.py  test_keys.py
    test_redis_bus.py  test_cache.py  test_registry.py
```
