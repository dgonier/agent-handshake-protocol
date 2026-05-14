# AHP — Agentic Handshake Protocol

A typed, addressable, format-aware messaging protocol for heterogeneous
agents. Agents are identified by structured URIs, exchange messages
tagged with hierarchical interaction codes, and negotiate payload format
through a single shared compatibility matrix.

All five phases of the plan are implemented: core primitives, Redis
transport, cache, registry, the protocol engine + thread manager, the
framework adapter layer (LangGraph, DSPy, deep agent, human) with
factory, provisioning patterns, and a capability registry, plus a
runnable end-to-end demo.

## Status

| Phase | Module | State |
|-------|--------|-------|
| 1 | `ahp.core` (addresses, patterns, codes, messages, compatibility) | implemented |
| 2 | `ahp.transport` (RedisBus, ProtocolCache), `ahp.registry` | implemented |
| 3 | `ahp.engine` (ProtocolEngine, ThreadManager) | implemented |
| 4 | `ahp.adapters` (AHPAgent, LangGraph, DSPy, deep agent, human, factory, provisioning, capabilities) | implemented |
| 5 | `ahp.demo.finance_analysis` (runnable end-to-end demo) | implemented |

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

## Engine — `ProtocolEngine`

The engine is the outbound gate: agents construct a `Message` and hand
it to `engine.handle()`, which validates the envelope, checks the
cache for `SEND-GET`, resolves patterns via the registry, filters by
the compatibility matrix, dispatches via the bus, and caches the
response on the way out.

```python
from ahp.engine import ProtocolEngine

engine = ProtocolEngine(bus, registry, cache, matrix=None, threads=None)

# point-to-point
delivered = await engine.handle(send_msg)             # int
reply     = await engine.handle(send_get_msg)         # Message | None

# broadcast
delivered = await engine.handle(cast_msg)             # int
replies   = await engine.handle(
    cast_get_msg, timeout=5.0, max_responses=3,
)                                                     # list[Message]

# cache busting
n = await engine.handle(invalidate_msg)               # int
```

Return shapes by verb: `SEND` → delivery count; `SEND-GET` → response
or `None` (cache hits short-circuit the bus); `CAST` → fan-out count;
`CAST-GET` → list of responses; `CAST-SUB` → a `Subscription` over
matching tap traffic; `INVALIDATE` → entries cleared.

The cache key combines the target address, the code, and a digest of
the request body — so two queries with different bodies against the
same `(target, code)` don't collide on a cache slot.

`CAST-SUB` opens a long-lived `Subscription` on the bus's tap channel
(every published message is mirrored to it). The engine builds a
predicate from the verb's pattern target + code glob:

```python
sub = await engine.handle(Message(
    source=alice,
    target=AddressPattern.parse("*.adversarial.*.*.*.*.*"),
    verb="CAST-SUB",
    code="adversarial.*",   # glob — matches every adversarial.* code
    thread="thread::observer",
    body=None,
))
async for msg in sub.messages():
    print(msg.source, msg.code, msg.body)
```

Concrete-address targets are supported too — useful when you want to
audit every message addressed to a specific agent.

Errors raised: `IncompatibleTargetError` when the target's accept set
doesn't satisfy the code's tier requirements; `InvalidTargetTypeError`
when a verb's target shape is wrong (pattern where address is
required, or vice versa).

## Thread manager — `ThreadManager`

Thread metadata (topic, initiator, status), participation set, and
tier-filtered history reads layered on top of the bus's stream:

```python
tid = await engine.spawn_thread("Tesla outlook", initiator=alice)
await engine.join_thread(tid, bob)

# Human observer view: drop anything the code can't render as a string.
history = await engine.threads.get_history(tid, tier_filter="s")

# Slice the stream by Redis stream IDs.
recent = await engine.threads.get_history(tid, min_id="-", max_id="+", count=20)
```

## Adapters — `ahp.adapters`

`AHPAgent` is the abstract base every framework adapter inherits from.
Subclasses override `handle_message()`; the base handles registration,
inbox consumption, heartbeats, auto-reply, and error wrapping.

Available adapters:

| Adapter | Wraps | Optional dep |
|---------|-------|--------------|
| `HumanAgent` | callbacks (`on_message`, `input_provider`) with L0–L3 observation levels | — |
| `LangGraphAgent` | a compiled `StateGraph` | `langgraph` |
| `DeepAgentDAG` | a graph whose nodes recurse via `config["configurable"]["ahp_engine"]` | `langgraph` |
| `DSPyAgent` | a `dspy.Module` (run in a worker thread) | `dspy-ai` |

```python
from langgraph.graph import END, START, StateGraph
from ahp.adapters.langgraph_agent import LangGraphAgent

# build a graph; here a trivial uppercase node
g = StateGraph(dict)
g.add_node("up", lambda s: {"output": s["input"].upper()})
g.add_edge(START, "up"); g.add_edge("up", END)
agent = LangGraphAgent(addr, engine, g.compile())
await agent.register(); await agent.start()
```

## Provisioning patterns — `ProvisioningPattern`

Bulk-spawn spec with per-field counts. The two count syntaxes have
distinct semantics — *prefix* is "ceiling-with-cycling", *suffix* is
Cartesian:

| Syntax | Meaning |
|--------|---------|
| `N*` | up to N for this field, cycle modulo N. Multiple prefix-N fields share one outer loop → total = `max(N_i)`. |
| `*N` | Cartesian multiplier. Stacks with prefix-N: total = `max × prod(suffix)`. |
| `N-*` | same as `N*` but **fresh-only** — ignore the registry, always spawn N new. |
| `*-N` | same as `*N` but fresh-only. |
| no dash | reuse-then-top-up — pull existing alive agents matching the spec's fixed skeleton, top up with fresh names. |

```text
4*.adversarial.finance.2*.s.session.*     → 4 agents (subdomain cycles)
*4.adversarial.finance.*2.s.session.*     → 8 agents (Cartesian)
*4.adversarial.finance.2*.s.session.*     → 8 agents (4 orgs × 2 iters)
4-*.adversarial.finance.2-*.s.session.*   → 4 fresh agents, no reuse
```

## Factory — `AgentFactory`

Pattern-keyed registry of builders that turn addresses into agents,
optionally informed by an `AgentProfile` from the
`CapabilityRegistry`.

```python
from ahp.adapters import (
    AgentFactory, CapabilityRegistry, Tool, Skill, RagSource,
)

caps = CapabilityRegistry()
caps.register("*.*.finance.*.*.*.*",
              tools=[Tool("get_quote", "fetch stock quote", get_quote_fn)])
caps.register("*.adversarial.*.*.*.*.*",
              prompt="Argue the bear case.", agent_kind="react", priority=5)

factory = AgentFactory(engine, capabilities=caps)
factory.register(
    "*.adversarial.*.*.*.*.*",
    lambda address, engine, profile: MyReactAgent(
        address, engine,
        tools=profile.all_tools, prompt=profile.prompt,
    ),
)

# spawn 4 fresh adversarial finance analysts, cycling 2 subdomains
result = await factory.spawn_and_start(
    "4-*.adversarial.finance.2-*.s.session.*",
)
print(len(result.new), len(result.reused))   # → 4, 0
```

## Capabilities — `CapabilityRegistry`

Address fields drive agent configuration: `domain`/`subdomain` selects
tools/skills/RAG, `role` partially determines agent kind. Capability
providers are pattern-keyed fragments; the registry merges every
matching fragment for an address into a single `AgentProfile` that the
factory passes to builders.

```python
@dataclass
class AgentProfile:
    address: AgentAddress
    tools: tuple[Tool, ...]
    skills: tuple[Skill, ...]
    rag_sources: tuple[RagSource, ...]
    prompt: str
    agent_kind: Literal["react", "deep", "custom"]
```

Composition rules: lists concatenate (priority-first, registration
order on ties), `prompt` joins with blank lines, `agent_kind` follows
the highest-priority specifier (default `"react"`).

## LLM-backed agents — `ReactAgent` + Bedrock

`ReactAgent` (in `ahp.adapters.react_agent`) wraps
`langgraph.prebuilt.create_react_agent` so an `AgentProfile` + a
LangChain chat model produces a fully-wired AHP agent. Profile tools
are translated to LangChain `StructuredTool`s; `profile.prompt` becomes
the system prompt; inbox messages enter the graph as a `HumanMessage`
and the last `AIMessage` is sent back as the reply.

```python
from ahp.adapters.capability import AgentProfile, Tool
from ahp.adapters.react_agent import ReactAgent
from ahp.llm import bedrock_chat_model

model = bedrock_chat_model()  # reads BEDROCK_MODEL_ID + AWS_REGION from env
profile = AgentProfile(address=addr, prompt="You are the bear case.")
agent = ReactAgent.from_profile(addr, engine, profile, model=model)
await agent.register(); await agent.start()
```

`ahp.llm.bedrock` builds `ChatBedrockConverse` (cached per model id +
region) and exposes `has_aws_credentials()` for graceful skipping in
tests. Credentials themselves come from the standard boto3 chain (the
AWS CLI, env vars, IAM role) — this package never touches keys.

Copy `.env.example` to `.env` to override the region or model id:

```env
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
# AWS_PROFILE=default
```

Install extras: `pip install -e ".[aws]"` (pulls `langchain-aws`,
`boto3`, `python-dotenv`).

## Demo — `ahp.demo.finance_analysis`

End-to-end working pipeline against `fakeredis` (no real Redis or LLM
needed):

```bash
python -m ahp.demo.finance_analysis
```

What it does:

1. Builds bus + registry + cache + engine.
2. Wires a `CapabilityRegistry` (per-role prompts) into an `AgentFactory`.
3. Constructs five agents via the factory:
   * **Bull** — `LangGraphAgent` over a tiny `StateGraph`.
   * **Bear** — `DSPyAgent` over a stubbed `dspy.Module`.
   * **Data** — plain `AHPAgent` returning canned fundamentals.
   * **Researcher** — `DeepAgentDAG` whose node calls back into the
     engine, doing a `SEND-GET` to data and a `CAST-GET` to the
     adversarial pattern.
   * **Human** — `HumanAgent` at observation level L2.
4. Drives a `HUMAN_QUERY` `SEND-GET` from the human to the researcher.
   The deep agent fans out concurrently and composes a single brief.
5. Sends the same query a second time → response cache short-circuits;
   typical speedup ~700×.

Sample output:

```
=== Devin asks the researcher (cold) ===
=== Analysis for Tesla ===

Fundamentals:
  Revenue $96B, EPS $4.30, P/E 70, EV/EBITDA 45, FCF $7.5B. ...

Bull view:
  Bull case for Tesla: durable competitive moat, expanding TAM, ...

Bear view:
  Bear case for Tesla: regulatory headwinds, margin compression, ...

--- timing: cold=120.6ms, warm=0.2ms (speedup ~735x) ---
```

The demo is also runnable as a library function (`from ahp.demo.finance_analysis
import run`) and is exercised by `tests/test_demo.py`.

### LLM-backed variant — `ahp.demo.finance_react`

Same pipeline, but Bull and Bear are `ReactAgent` instances driven by
Bedrock chat models. Requires AWS credentials reachable through the
boto3 chain. Run with:

```bash
python -m ahp.demo.finance_react
```

For tests, pass `model=` to `run()` with a fake chat model that
implements `bind_tools`; `tests/test_demo_react.py` does exactly this
so the LLM demo's wiring is exercised in CI without hitting AWS. The
live Bedrock path is gated behind `AHP_RUN_BEDROCK=1` so it stays
opt-in.

## Tests

```bash
pytest
```

273 tests passing + 1 cleanly skipped (live Bedrock smoke). Phase 4 / 5
/ 6 coverage includes:

* `test_agent_base.py` — register/deregister/start/stop, auto-reply,
  handler exception → `error.internal` wrapping, send/broadcast helpers.
* `test_provisioning.py` — prefix vs suffix semantics, dash variants,
  field constraints, custom namers, user's company×subdomain example.
* `test_factory.py` — pattern + priority dispatch, reuse-then-top-up,
  dash skips registry, dead agents not reused.
* `test_capability.py` — fragment merging, priority ordering, prompt
  composition, profile passed through factory + spawn.
* `test_human_agent.py` — L0–L3 observation levels, truncation,
  input-provider reply flow.
* `test_langgraph_agent.py` — graph round-trips, custom mappers,
  `DeepAgentDAG` recursion through the engine.
* `test_dspy_agent.py` — module round-trips, custom field names.
* `test_demo.py` — full Phase-5 pipeline: deep researcher composes
  data + bull + bear into a single brief, second query hits cache,
  unknown ticker degrades gracefully, registry holds all five URIs.
* `test_react_agent.py` — `ReactAgent.from_profile` round-trips through
  a fake chat model (no AWS), profile tools accepted by LangChain,
  `extra_tools` injection, list-shaped content coercion.
* `test_demo_react.py` — LLM demo wired against a fake chat model
  (cold + cache hit), `run()` without AWS credentials raises a clear
  error, opt-in live-Bedrock smoke test.

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
├── registry/
│   └── registry.py       AgentRegistry + AgentMeta
├── engine/
│   ├── router.py         ProtocolEngine (verb dispatcher)
│   ├── thread_manager.py ThreadManager + Thread
│   └── errors.py         ProtocolError / IncompatibleTargetError / ...
├── adapters/
│   ├── base.py             AHPAgent
│   ├── factory.py          AgentFactory + SpawnResult
│   ├── provisioning.py     ProvisioningPattern + N* / *N / dash variants
│   ├── capability.py       Tool / Skill / RagSource / AgentProfile / CapabilityRegistry
│   ├── human.py            HumanAgent
│   ├── langgraph_agent.py  LangGraphAgent + DeepAgentDAG  (needs langgraph)
│   └── dspy_agent.py       DSPyAgent  (needs dspy-ai)
├── adapters/
│   └── react_agent.py    ReactAgent (wraps create_react_agent)
├── llm/
│   └── bedrock.py        ChatBedrockConverse helper + has_aws_credentials()
└── demo/
    ├── finance_analysis.py end-to-end pipeline (deterministic stubs)
    └── finance_react.py    same pipeline, Bedrock-driven Bull + Bear
tests/
    test_address.py  test_pattern.py  test_codes.py  test_message.py
    test_compatibility.py  test_keys.py
    test_redis_bus.py  test_cache.py  test_registry.py
    test_engine.py  test_thread_manager.py
    test_agent_base.py  test_factory.py  test_provisioning.py
    test_capability.py  test_human_agent.py
    test_langgraph_agent.py  test_dspy_agent.py
    test_demo.py
```
