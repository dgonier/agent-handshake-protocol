# AHP — Agentic Handshake Protocol

> A typed protocol for talking to N AI agents at once — the same way every time.

If you've built more than one multi-agent system you've probably written
this code three times:

* **Addressing.** Every project invents its own way to name agents and
  then can't broadcast generically. `bull_agent`, `bear_agent`,
  `data_agent` — fine for two agents, miserable for fifty.
* **Format negotiation.** Some agents speak strings, some JSON, some
  bytes, some embeddings. The first integration is a switch statement;
  the tenth is an unmaintainable rats' nest of glue.
* **Orchestration.** Every debate, interview, panel, or critique loop
  is a one-off script. Adding a new dialog shape means rewriting the
  controller from scratch.

**AHP solves these three problems together with a small Redis-backed
library.** Agents have structured URI addresses. Messages carry a
hierarchical interaction code and a payload-tier set. Dialog patterns
— debate, interview, deliberation, brainstorm — are *recipes* defined
once and reused across topics. An SLM-driven `AgentFactory` picks the
panel from a domain + topic; a `ProtocolEngine` routes everything
through a single matrix.

`ahp.core` has zero runtime dependencies. Everything else (transport,
framework adapters, web search, audit) is opt-in.

## In 30 lines

```python
import asyncio
from ahp.adapters import AgentFactory, CapabilityRegistry
from ahp.adapters.formats import get_format
from ahp.core import AgentAddress, AddressPattern, Message
from ahp.engine.router import ProtocolEngine
from ahp.llm.bedrock import bedrock_chat_model
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus
import fakeredis.aioredis

async def main():
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    bus, reg, cache = RedisBus(redis), AgentRegistry(redis), ProtocolCache(redis)
    engine = ProtocolEngine(bus, reg, cache)

    slm = bedrock_chat_model()  # picks Haiku 4.5 by default
    factory = AgentFactory(engine, capabilities=CapabilityRegistry(), slm=slm)

    # The SLM picks 4 cosmology perspectives appropriate to the topic.
    spawn = await factory.invite_and_start(
        org="public", role="adversarial",
        domain="science", subdomain="astrophysics",
        topic="What caused the Big Bang?", count=4,
    )

    # Broadcast: every agent argues its position in 3 sentences.
    pattern = AddressPattern.parse("*.adversarial.science.astrophysics.*.*.*")
    msg = Message(source=AgentAddress.parse("you.adversarial.science.astrophysics.s.session.you"),
                  target=pattern, code="adversarial.debate", verb="CAST-GET",
                  body={"recipe": "adversarial:debate-me", "question": "What caused the Big Bang?"})
    replies = await engine.handle(msg, max_responses=4)
    for r in replies:
        print(r.source.instance, "→", r.body["text"][:120])

asyncio.run(main())
```

That snippet covers: addressing, an SLM-picked panel, a pattern-routed
broadcast, a reusable recipe, and a typed response. Same code can run
`interview-yall`, `collaborate-brainstorm`, `fiction-theatre`, or any
of the 16 built-in formats by swapping one string.

## What you get

* **A 7-field URI address space** for agents. Wildcards + payload-tier
  subset semantics mean one pattern can fan out to a hundred targets,
  filtered by what they actually understand.
* **51 dialog recipes across 11 interaction roles** (debate, interview
  variants, collaboration, conversation, fiction, deliberation,
  teaching, estimation, interrogation). Each recipe is a 5-line render
  function — extend with your own as easily as you write a Python
  function.
* **16 prebuilt session formats** that compose recipes into
  opening/middle/closing rounds. `debate`, `interview-me`,
  `interview-yall`, `collaborate-brainstorm`, `fiction-theatre`,
  `teach`, `estimate`, … swap them by name.
* **`AgentFactory` with an SLM**: hand it a domain + topic and it
  populates a panel with appropriate personas. No hard-coded prompts
  to maintain across topics.
* **A global tool registry** that any agent can use. `search_tavily`
  ships in `ahp.tools.research`; the LLM (LangChain `create_agent`)
  decides when to call it.
* **Audit + observability**: every protocol op emits a typed
  `AuditEvent`. Sinks ship for in-memory, stdlib logging, and
  CloudWatch Logs.
* **Adapters for LangGraph, DSPy, deepagents** (≥0.6), and a
  human-in-the-loop primitive. MCP servers can be mounted under any
  scope.
* **A FastAPI + Docker Compose viewer** under `examples/viewer/` —
  mobile-first browser UI that runs the live stack against your AWS
  Bedrock credentials.

## Show me

The viewer is one command:

```bash
docker compose up --build         # http://localhost:9876
# optional public URL via Cloudflare quick-tunnel:
docker compose --profile tunnel up
```

Pick a topic, a format, a domain/subdomain, hit Run. The page shows
the SLM-picked panel, every round, and which tools each agent called.
Full instructions in [`examples/viewer/README.md`](examples/viewer/README.md).

## Concepts in 5 minutes

**Address.** Every agent is a 7-field URI:

```
{org}.{role}.{domain}.{subdomain}.{accept}.{lifecycle}.{instance}?{params}
```

Example: `tifin.adversarial.finance.equities.j.session.bear-42` —
TIFIN's adversarial finance/equities agent that accepts JSON, lives
for one session, named `bear-42`.

**Pattern.** A wildcard URI that matches a set of agents. Routing,
broadcast, and discovery all use the same pattern syntax:

```
*.adversarial.science.*.s.*.*
```

**Code.** Hierarchical dotted message type, with families:
`adversarial.debate`, `interview.text`, `human.query`,
`collaborative.reason`. Codes carry payload-tier requirements (does
this code need a string? JSON? embeddings?) which the
compatibility matrix enforces.

**Recipe.** A `(role, mode) → render(system, ctx) → str` function in
`ahp.adapters.prompts`. Format-agnostic; the *system* slot is the
agent's persona, the *ctx* carries query-specific context. 51 ship
in the box; you can register more.

**Format.** A named bundle that composes recipes into an
opening/middle/closing turn pattern: `debate` uses
`adversarial:debate-me` then `adversarial:debate-others` then
`adversarial:closing`. 16 ship; one entry in
[`ahp/adapters/formats.py`](ahp/adapters/formats.py) per format.

**Tool.** An async function decorated with
`@tool(scope, kind, role, category, operation=...)` registered to a
shared `ToolRegistry`. Global tools at `scope="*"` are visible to
every agent. Agents bind the visible tools to their chat model via
LangChain's `create_agent`.

## Status

This is a pre-1.0 protocol. Primitives (addresses, codes, the
compatibility matrix, the engine verbs) are stable enough that
breaking them would hurt. Periphery (recipes, formats, adapters,
tools, the viewer) is wide open and will keep growing.

* **Tests**: 478 passing on the suite, 1 live-Bedrock smoke gated
  behind `AHP_RUN_BEDROCK=1`.
* **Verified end-to-end**: the docker-compose viewer drives real
  Bedrock + Redis + CloudWatch end-to-end.
* **Production-readiness**: not yet. No auth on the viewer; no rate
  limiting on the engine; no replay protection on the bus. Issue list
  is open.

## Not designed for

So you can self-select before sinking time in:

* **Streaming.** AHP messages are request/response or fan-out/fan-in.
  There's no per-token streaming primitive. If you need that, this
  isn't the right layer.
* **Multi-user IAM.** `AuthPolicy` decides who can register at which
  address; that's it. Bring your own user system.
* **Cross-network federation by default.** Federation works (see the
  example) but only over a *shared* Redis. There's no DHT, no
  cross-cluster discovery.
* **Replacing your agent framework.** AHP is the *messaging layer*
  between agents; it adapts to LangGraph / DSPy / deepagents, it does
  not replace them.

## Phase index

This branch has shipped 19 phases. Each phase commit explains *why*
not just *what*. Skim `git log --oneline` for the narrative arc.

| Phase | Module | Notes |
|-------|--------|-------|
| 1 | `ahp.core` | addresses, patterns, codes, messages, compatibility |
| 2 | `ahp.transport` + `ahp.registry` | RedisBus, ProtocolCache, registry |
| 3 | `ahp.engine` | ProtocolEngine, ThreadManager |
| 4 | `ahp.adapters` | base, LangGraph, DSPy, deep agent, human, factory, provisioning, capabilities, tool/resource registries, MCP passthrough |
| 5 | `ahp.demo.*` | stubbed + Bedrock end-to-end demos |
| 14–17 | addressable storage, auth, CLI, federation |
| 18 | `ahp.audit` | typed events, CloudWatch sink |
| 19 | recipes (51), formats (16), tools, docker viewer | this phase |

## Contributing

This is open source. We're early enough that meaningful contributions
don't require committee approval.

The protocol primitives are settled; the periphery is wide open.
Good first contributions:

* **A new recipe.** Add a `Recipe(role, mode, description, render)`
  entry to [`ahp/adapters/prompts.py`](ahp/adapters/prompts.py). Each
  is ~5 lines. We'd love more interview-style recipes for specific
  domains.
* **A new format.** Compose existing recipes into a turn pattern in
  [`ahp/adapters/formats.py`](ahp/adapters/formats.py). One dataclass
  entry, no code changes needed.
* **A new tool.** Decorate a function with `@tool(scope="*", ...)`.
  See [`ahp/tools/research.py`](ahp/tools/research.py) for the
  shape. Tools we'd love: web fetch + reader, ArXiv search, code
  execution sandbox, vector-DB query.
* **A new adapter.** Wrap your favorite framework
  ([CrewAI](https://github.com/joaomdmoura/crewai),
  [PydanticAI](https://github.com/pydantic/pydantic-ai),
  [smolagents](https://github.com/huggingface/smolagents),
  AutoGen) as an `AHPAgent` subclass so its agents speak the
  protocol.
* **A new audit sink.** Loki, OpenSearch, Honeycomb, S3 — all small
  classes that implement the `AuditSink` protocol.
* **Documentation and examples.** Real-world session transcripts,
  blog posts about specific recipes, a Jupyter notebook walkthrough.

How to ship one:

1. Fork, branch, write your change.
2. Run `pytest` from the repo root — needs to stay green.
3. Open a PR. We'll review and merge fast for small additions to the
   periphery; protocol-touching changes need a short discussion first.

For ideas, bug reports, or "what about X?" discussions, open an issue
or start a GitHub discussion. No formal contributor agreement yet.

---

## Reference

Everything below is API reference for the existing primitives. If
you've read this far and want depth, keep going. If you just want to
ship something, the [examples](examples/) directory and the snippet
at the top are the best starts.

### Status (legacy phases)

| Phase | Module | State |
|-------|--------|-------|
| 1 | `ahp.core` (addresses, patterns, codes, messages, compatibility) | implemented |
| 2 | `ahp.transport` (RedisBus, ProtocolCache), `ahp.registry` | implemented |
| 3 | `ahp.engine` (ProtocolEngine, ThreadManager) | implemented |
| 4 | `ahp.adapters` (AHPAgent, LangGraph, DSPy, deep agent, human, factory, provisioning, capabilities, tool/resource registries, MCP passthrough) | implemented |
| 5 | `ahp.demo.finance_analysis` (stubbed) + `ahp.demo.finance_react` (Bedrock) + `ahp.demo.serve` (FastAPI) | implemented |

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

## Addressable tools — `ToolRegistry`

Tools are first-class citizens with their own 5-field address:

```
{scope}.{kind}.{role}.{category}.{operation}
```

Register declaratively with the `@tool` decorator — the operation
defaults to the function's name, and access scope is derived from the
address. **Use `*` liberally**: the role field describes which agent
*kinds* may use the tool, not which kinds the tool *belongs to*. Most
DB / FS / API tools are usable by any role in scope, so `role="*"`
is the right default:

```python
from ahp.adapters import tool

@tool("tifin", "db", "*", "crud")            # any tifin role
def update_record(table: str, row_id: str, fields: dict) -> dict:
    """Update a row in the table."""
    return run_sql(...)

# → registered at ToolAddress("tifin", "db", "*", "crud",
#                              "update_record")
# → default allowed_for: agents matching "tifin.*.*.*.*.*.*"
#   (the convention projects tool.scope/role onto agent org/role; `*`
#   on role means "any role in the tifin org")
```

When to be concrete vs `*`:

* **`scope`** — concrete when the tool is org-private; `"*"` for
  platform-wide utilities.
* **`kind`** — concrete (`db`, `fs`, `api`, `compute`). This labels
  the tool's *type*, not its access.
* **`role`** — `*` by default. Only constrain when the tool is
  semantically tied to one role (e.g. a `redteam` tool that should
  only appear in adversarial agents' profiles).
* **`category`** — concrete (`crud`, `search`, `read`, `write`).
  Like `kind`, this is descriptive metadata, not access control.
* **`operation`** — derived from `func.__name__`.

Override the convention with explicit `allowed_for=` or tag tools for
selective inclusion (`tags=["read-only", "slow"]`, then
`registry.for_address(addr, tags=["read-only"])`).

## Addressable resources — `ResourceRegistry`

Long-lived shared objects (vector stores, DB clients, API SDKs, FS
backends) have parallel `{scope}.{kind}.{domain}.{subdomain}.{name}`
addresses. Lazy-instantiated on first access, torn down via
`close_all()` during shutdown. Default access scope is by
`org/domain/subdomain` (shared across roles):

```python
from ahp.adapters import resource

@resource("tifin", "fs", "finance", "documents")
class FinanceDocs:
    def __init__(self):
        self.root = "/data/finance"
    def aclose(self):                       # auto-detected for cleanup
        ...

@resource("tifin", "vector", "finance", "filings",
          name="sec-edgar", cleanup=lambda c: c.aclose())
def make_sec_vector():
    return ChromaClient(...)
```

Agents matching the resource's `allowed_for` pattern get a
`profile.resources["sec-edgar"]` map handed to their builder — tools
inside the agent grab the client by name.

## Addressable storage — `kind="fs"` resources

Filesystem-backed resources are just `Resource`s with `kind="fs"`.
They register at `{scope}.fs.{domain}.{subdomain}.{name}` like
anything else, and the `DeepAgent` adapter automatically mounts every
matching one into its virtual filesystem:

```python
from ahp.adapters import resource
from ahp.adapters.deep_agent import DeepAgent
from deepagents.backends import StateBackend

@resource("tifin", "fs", "finance", "documents",
          name="scratch", description="finance team scratch + uploads")
def make_scratch_fs():
    return StateBackend()   # or any BackendProtocol implementation

@resource("tifin", "fs", "finance", "filings",
          name="filings", description="SEC EDGAR cache")
def make_filings_fs():
    return _load_chroma_backed_fs(...)

# In your factory wiring:
researcher = DeepAgent.from_profile(
    address, engine, profile, model=bedrock_chat_model(),
    fs_resources=factory.resources,   # ← that's the whole integration
)
```

What the integration does:

1. **Filters by address.** Only fs-kind resources whose `allowed_for`
   pattern matches the agent end up in its virtual FS — same
   convention as tools (default: `{scope}/{*}/{domain}/{subdomain}`).
2. **Mounts each at `/<name>/` by default.** `scratch` mounts at
   `/scratch/`, `filings` at `/filings/`. Override via the
   `mount_path=` callable to `build_fs_backend` if you want a
   different scheme.
3. **Wraps in `CompositeBackend` when there's more than one.** A
   single matching backend is returned directly; multiple are
   composed with the default backend handling the unmatched-prefix
   case.
4. **Appends a system-prompt fragment listing every mount** so the
   LLM knows where to read/write. Format:
   ```
   Available filesystem mounts:
   - /scratch/ — finance team scratch + uploads
   - /filings/ — SEC EDGAR cache
   ```
5. **Mount-path collisions raise** `ValueError` at build time — same
   philosophy as the other resolution-conflict errors.

Tools that the LLM sees come from deepagents' `FilesystemMiddleware`:
`ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`. Your
backend implements the deepagents `BackendProtocol` (`StateBackend`
ships with the library; `LocalShellBackend`, `StoreBackend`,
`ContextHubBackend` are also available).

## Addressable knowledge graphs — `kind="kg"` resources

Knowledge-graph backends live alongside filesystems in the resource
address space, at `{scope}.kg.{domain}.{subdomain}.{name}`. The
shape (`KGNode`, `KGEdge`, `KnowledgeGraphBackend`), an in-memory
reference implementation, and a Neo4j-backed adapter (vector index
included) live in `ahp.adapters.knowledge_graph` and
`ahp.adapters.neo4j_kg`. Wire one in the same way as any other
resource:

```python
from ahp.adapters import resource
from ahp.adapters.neo4j_kg import Neo4jKnowledgeGraph

@resource("acme", "kg", "finance", "equities", name="primary",
          cleanup=lambda g: g.close())
def make_primary_kg():
    return Neo4jKnowledgeGraph(vector_dimensions=1536,
                               auto_create_vector_index=True)
```

`build_kg_backend(resources, agent_address)` resolves the right
backend for an agent the same way `build_fs_backend` does — single
match wins, multiple matches raise. See
`examples/knowledge_graph/` for docker-compose + Terraform
boilerplate (the vector-index Cypher is the piece that's easy to
get wrong).

## TeacherAgent — agent-as-judge into the KG

`TeacherAgent` is the writer-side companion to the KG: an
`AHPAgent` that scores other agents against a `Rubric` and persists
the resulting `Judgement` into a KG backend. Three inbound codes:

* `teacher.judge` — score one body, write a `Judgement` node linked
  to subject + rubric, return the verdict.
* `teacher.survey` — broadcast a prompt, judge each reply, persist
  per-respondent judgements.
* `teacher.observe` — write a free-form `Observation` linked to the
  reporting agent.

`judge_fn` is any callable (sync or async) returning a `Judgement`
or a `{criterion_name: score}` map — drop in a LangChain chain, a
deterministic scorer, or a HEXIS-side judge. See
`examples/knowledge_graph/teacher_demo.py` for an end-to-end run
against a real Neo4j.

## MCP passthrough

Register an entire MCP server's tool surface under one scope:

```python
from ahp.adapters.mcp import register_mcp_server

await register_mcp_server(
    factory.tools, factory.resources,
    scope="tifin", kind="api", role="*", category="mcp-github",
    connection={"command": "uvx", "args": ["mcp-server-github"],
                "transport": "stdio"},
)
```

Every tool the MCP server exposes (e.g. `search_repos`, `get_issue`)
is now addressable as `tifin.api.*.mcp-github.<tool_name>` and gets
auto-bound to agents whose address matches the derived pattern. The
MCP client itself is registered as a `Resource` so its connection is
closed on shutdown. Optional dep: `pip install -e ".[mcp]"`.

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

## Model / compute sources

Three natural sources, each at a different layer:

| Source | What it gives you | When to use |
|---|---|---|
| **Bedrock** (`ahp.llm.bedrock`) | Hosted Claude / Llama / Titan on AWS | Default "just give me a smart model" — pay per token, no compute concerns |
| **OpenRouter** (`ahp.llm.openrouter`) | Unified OpenAI-compatible API to 100+ models | A/B test recipes across providers without rewiring; cheap/fast OSS models alongside frontier ones |
| **Modal** | Serverless GPUs by the second | Where you actually *run* AHP nodes that need GPUs — HF + peft for LoRA composition, vLLM serving, custom inference. Two patterns: Modal-as-endpoint (use `openrouter_chat_model` pointed at your Modal URL) or Modal-as-AHP-node (a Modal app boots a factory and joins the Redis network). |

```python
# Bedrock
from ahp.llm import bedrock_chat_model
agent = ReactAgent.from_profile(addr, engine, profile, model=bedrock_chat_model())

# OpenRouter
from ahp.llm import openrouter_chat_model
agent = ReactAgent.from_profile(
    addr, engine, profile,
    model=openrouter_chat_model("anthropic/claude-3.5-sonnet"),
)

# Modal-hosted endpoint (your own OpenAI-compatible server)
agent = ReactAgent.from_profile(
    addr, engine, profile,
    model=openrouter_chat_model(
        model="my-llama3-with-loras",
        base_url=f"https://you--llama-app.modal.run/v1",
    ),
)
```

Extras: `pip install -e ".[aws]"` for Bedrock, `pip install -e ".[openrouter]"`
for OpenRouter. Modal is BYO — see [examples (planned)](#) for runnable
patterns once Modal credentials are in your environment.

## LLM recipes — base model + LoRAs as addressable resources

The natural extension of "everything is addressable": **base models
and LoRA adapters register as `Resource`s with `kind="model"` and
`kind="lora"`**. The agent's recipe (which base + which adapters)
becomes a pure consequence of its address.

```python
from ahp.adapters import resource
from ahp.llm import LoRAHandle, ModelHandle

@resource("tifin", "model", "*", "*", name="llama3-8b")
def make_base():
    return ModelHandle(
        name="llama3-8b",
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    )

@resource(
    "tifin", "lora", "finance", "*", name="bearish-v2",
    allowed_for="tifin.adversarial.finance.*.*.*.*",  # role gating
)
def make_bearish():
    return LoRAHandle(
        name="bearish-v2",
        repo_id="tifin/finance-bearish-v2",
        weight=1.0,
    )
```

An agent at `tifin.adversarial.finance.equities.s.session.bull` picks
up both: the base (matches `domain="*"`/`subdomain="*"`) plus
`bearish-v2` (matches the explicit `allowed_for`). A collaborative
finance agent gets the base but not `bearish-v2`. Same registry,
different recipe, by address alone.

Consumers introspect via:

```python
from ahp.llm import find_model, find_loras, recipe_summary

profile = factory.profile_for(addr)
base   = find_model(profile)              # ModelHandle | None
loras  = find_loras(profile)              # list[LoRAHandle] sorted by name
prompt_fragment = recipe_summary(profile) # for the agent's system prompt
```

The handles are pure metadata — they don't load weights. A future
`HuggingFaceAgent` (planned) will consume them via `peft` /
`transformers` on the Modal side, while the address-level composition
stays usable on machines without GPUs.

**Wart worth knowing:** `ResourceAddress` has no `role` field, so
role-discriminated LoRAs need an explicit `allowed_for=` instead of
inferring "for adversarial agents" from the address alone. The
convention's default projection only covers scope / domain /
subdomain.

## LLM-backed agents — `ReactAgent`, `DeepAgent`, Bedrock

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

### Deep agents — `DeepAgent`

`DeepAgent` (in `ahp.adapters.deep_agent`) wraps
`deepagents.create_deep_agent`, which adds a planner, subagents, and a
virtual filesystem on top of the ReAct loop. From an `AgentProfile`:

* `profile.tools` → planner tools
* `profile.skills` → `SubAgent` entries the planner can delegate to
  (each `Skill.name`/`description`/`prompt_fragment`/`tools` is mapped
  to the matching `SubAgent` field)
* `profile.prompt` → top-level system prompt
* `extra_tools=` → AHP-aware closures that let the planner reach back
  into the protocol (see the LLM demo for `lookup_fundamentals` and
  `hold_debate` examples)

```python
from ahp.adapters.deep_agent import DeepAgent

researcher = DeepAgent.from_profile(
    address, engine, profile, model=bedrock_chat_model(),
    extra_tools=[lookup_fundamentals_tool, hold_debate_tool],
)
```

The translator (`_to_langchain_tool`) detects coroutine handlers and
wires them as the LangChain tool's async path — so AHP-aware tools can
`await engine.handle(...)` inside an already-running event loop without
the usual `asyncio.run` re-entrancy crash.

Install extras: `pip install -e ".[deepagents]"`.

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

Same pipeline as the stub demo, but the agents are LLM-driven:

* **Bull** and **Bear** are `ReactAgent` instances with role-specific
  prompts layered on top of the capability registry.
* **Researcher** is a `DeepAgent` (`deepagents.create_deep_agent`) with
  two AHP-aware tools — `lookup_fundamentals` calls the Data agent over
  the protocol, `hold_debate` fans the question out to every
  `*.adversarial.finance.*` agent via `CAST-GET`. The planner decides
  when to call each one.

Requires AWS credentials reachable through the boto3 chain. Run with:

```bash
python -m ahp.demo.finance_react
```

For tests, pass `model=` to `run()` with a fake chat model that
implements `bind_tools`; `tests/test_demo_react.py` does exactly this
so the LLM demo's wiring is exercised in CI without hitting AWS. The
live Bedrock path is gated behind `AHP_RUN_BEDROCK=1` so it stays
opt-in.

## Broadcast by name — `GroupRegistry`

Name a pattern, broadcast by string:

```python
from ahp.adapters import AgentFactory, GroupRegistry

groups = GroupRegistry()
groups.register("debaters",     "*.adversarial.*.*.*.*.*")
groups.register("research-team", "*.collaborative.finance.*.s.*.*")

factory = AgentFactory(engine, groups=groups)  # wires engine.groups

# Now any agent can fan out by a single string:
replies = await alice.broadcast_to(
    "debaters",                            # ← simple group name
    code=Code.ADVERSARIAL_DEBATE,
    body="argue Tesla",
)
```

Resolution order on `broadcast_to(name_or_pattern, ...)`:
1. Already an `AddressPattern` → used as-is.
2. Registered group name → its pattern.
3. Otherwise the string is parsed as a 7-field pattern (so ad-hoc
   patterns and named groups can mix at the same call site).

## Hosting it behind HTTP

FastAPI is intentionally NOT a dependency of `ahp`. A complete FastAPI
consumer that turns the library into a runnable service lives in
[`examples/fastapi_serve/`](examples/fastapi_serve/) — copy the
directory into your own project as a starting point.

The example wires `AgentFactory` + agents inside a FastAPI lifespan
and exposes the protocol over HTTP/WebSocket:

| Verb | Path | Purpose |
|------|------|---------|
| POST | `/query`           | HUMAN_QUERY → target (SEND-GET) |
| POST | `/send`            | arbitrary AHP message |
| GET  | `/agents`          | list registered agents |
| GET  | `/threads/{id}`    | read thread history |
| GET  | `/tools`           | list tool addresses |
| GET  | `/resources`       | list resource addresses |
| WS   | `/observe`         | live CAST-SUB stream over the bus tap |

```bash
cd examples/fastapi_serve
pip install -r requirements.txt
uvicorn app:app --reload          # stub variant
AHP_DEMO_VARIANT=react uvicorn app:app    # Bedrock-driven
```

## Access control — `ScopePolicy`

**Default is open** — meaning *no extra restrictions beyond what the
protocol already enforces*. The normal layers always run:

* The compatibility matrix gates messages whose code requires a tier
  the target doesn't accept.
* Liveness markers gate routing to expired agents.
* Address-pattern matching gates broadcasts.

Adding a `ScopePolicy` layers *additional* address-pattern allow rules
on top, restricting who can reach whom. The protocol stays open
unless you add the policy and the rules. Progressive tightening:

```python
from ahp.adapters import AgentFactory
from ahp.engine import ScopePolicy

scope = ScopePolicy()

# Step 1 — only tifin agents can reach tifin's address space:
scope.restrict(
    target="tifin.*.*.*.*.*.*",
    allow_sources="tifin.*.*.*.*.*.*",
)

# Step 2 — only finance agents touch the finance subdomain:
scope.restrict(
    target="tifin.*.finance.*.*.*.*",
    allow_sources="tifin.*.finance.*.*.*.*",
)

# Step 3 — only adversarial agents can mutate the DB plane:
scope.restrict(
    target="tifin.db.adversarial.*.*.*.*",
    allow_sources="tifin.adversarial.*.*.*.*.*",
    code="collaborative.delegate",        # optional code glob filter
)

factory = AgentFactory(engine, scope=scope)   # wires engine.scope
```

Semantics:

* **A target is "covered"** when at least one rule's `target` pattern
  matches it. Uncovered targets remain open.
* **Covered targets allow** any source matching any of the rules'
  `allow_sources` patterns for that target (union).
* **Tighter rules don't displace looser ones** — adding a narrower
  rule with a narrower source pattern doesn't shrink access granted
  by an existing broader rule. You shrink access by removing rules,
  not by adding them.
* **Point-to-point verbs** (`SEND`, `SEND-GET`) raise
  `UnauthorizedError` on denial.
* **Broadcast verbs** (`CAST`, `CAST-GET`) silently drop disallowed
  targets — same UX as the compatibility matrix.
* **`INVALIDATE` is not gated** by scope; cache control is a separate
  plane.

Scope is configured per-engine (each FastAPI process can carry its
own policy), but because the addresses are universal, every node on
the network should typically share the same policy or coordinate via
a central source of truth.

## Network-mapping resolution conflicts

The address-mapping layer is unambiguous *by address* — every tool /
resource has a unique full address. But agent profiles surface tools
to LangChain by their short `operation` name and resources by their
short `name` field. Two bindings with different addresses that share
a short name applied to the same agent would silently clobber each
other in the profile. To prevent that, the factory raises at
profile-build time:

* `ToolNameCollisionError` — two tools at different `ToolAddress`-es
  share an `operation` name for one agent. The error message names
  both addresses so you can decide which to rename or which to
  narrow with `allowed_for=`.
* `ResourceNameCollisionError` — two resources at different
  `ResourceAddress`-es share a `name` field for one agent. Same fix
  shape: rename or tighten `allowed_for`.

Both inherit from `ResolutionConflictError` for catch-all handlers.

Additionally, if you attach two `AgentFactory` instances to the same
engine and they carry different `groups` or `scope` registries, the
second factory warns via `logging` before overwriting. Re-attaching
the same registry (idempotent) does not warn.

## CLI — inspect and scaffold

```bash
python -m ahp <command> [options]
```

Subcommands:

| Command | What it does |
|--------|---|
| `list-tools` | list registered tools (filter by `--for ADDR` and `--tag T`) |
| `list-resources` | list registered resources (filter by `--for ADDR`) |
| `list-groups` | list named address-pattern groups |
| `list-agents` | query a live Redis registry; alive by default, `--all` to include stale entries, `--pattern PAT` to filter |
| `profile ADDR` | show the resolved AgentProfile for an address — tools, skills, resources, prompt |
| `template tool` / `template resource` | print a starter module to stdout |
| `scaffold tool` / `scaffold resource` | write the starter module to a file (`-o PATH`, `--force` to overwrite) |

Tools and resources only appear after their modules are imported, so
pass `-m DOTTED.PATH` one or more times to import user modules first:

```bash
python -m ahp list-tools -m my_project.tools -m my_project.db
python -m ahp profile tifin.adversarial.finance.equities.s.session.bull \
  -m my_project.tools -m my_project.db
```

Scaffolding a new DB tool:

```bash
python -m ahp scaffold tool \
  --scope tifin --kind db --category crud \
  --operation update_record \
  --signature 'table: str, row_id: str, fields: dict' \
  -o my_project/db_tools.py
```

Produces:

```python
"""Tools registered at scope=tifin kind=db role=* category=crud."""

from ahp.adapters import tool


@tool('tifin', 'db', '*', 'crud')
def update_record(table: str, row_id: str, fields: dict):
    """One-line description."""
    # TODO: implement
    raise NotImplementedError
```

Note the generated tool uses `role='*'` by default — see the
"Addressable tools" section above for when to narrow it.

Querying a live AHP network:

```bash
# What's actually registered on this Redis right now?
AHP_REDIS_URL=redis://localhost:6379/0 python -m ahp list-agents

# Only the adversarial finance pool:
python -m ahp list-agents --pattern '*.adversarial.finance.*.s.*.*'

# Include stale entries (registry hash entry exists but liveness expired):
python -m ahp list-agents --all
```

`list-agents` is the only CLI command that does live I/O — the others
inspect in-process registries. Output columns: address, status
(`alive`/`stale`), capabilities, reputation, description.

## Auth — who may register at which addresses

Same "default open, opt in to tighten" pattern as `ScopePolicy`, but
for *registration* rather than *routing*. By default any process with
Redis access can `register()` at any address. To restrict, give the
registry a `Principal` (the identity of the process) and an
`AddressClaimPolicy`:

```python
from ahp.registry import (
    AddressClaimPolicy, AgentRegistry, Principal,
    UnauthorizedRegistrationError,
)

# Node A's identity. Claims are AddressPatterns; only addresses
# matching one of these patterns may be registered by this node.
node_a = Principal.with_claims(
    "node-a",
    "tifin.adversarial.finance.*.*.*.*",
    "tifin.collaborative.finance.*.*.*.*",
)

registry = AgentRegistry(
    redis_client,
    principal=node_a,
    policy=AddressClaimPolicy(),
)

await registry.register(AgentAddress.parse(
    "tifin.adversarial.finance.equities.s.session.bull",
))   # OK

await registry.register(AgentAddress.parse(
    "public.adversarial.finance.equities.s.session.bull",
))   # raises UnauthorizedRegistrationError — node-a's claims don't
     # cover the public org
```

The same gate fires on `deregister()` and `heartbeat()` — owners can
take down what they put up, but a foreign node can't kick someone
else's agent off the network.

Available policies:

| Policy | Default? | Behavior |
|--------|----------|----------|
| `OpenAuthPolicy` | yes | accepts everything, including anonymous (`Principal=None`) |
| `AddressClaimPolicy` | — | a principal can mutate iff one of their claims matches the address. Anonymous principals are denied. |
| `DenyAllPolicy` | — | refuses every mutation. Useful for read-only mirrors. |

Important non-feature: **reads are not gated.** `resolve()`,
`discover()`, `get()`, `is_alive()` stay open even under
`AddressClaimPolicy`. A federation node needs to see who's on the
network to send messages; gating read paths would break that. If you
want read-side restrictions too, wrap the registry yourself or layer
a `ScopePolicy` on the engine.

In production this module is *agnostic* about how the principal's
claims got there — wrap it with whatever verification you prefer
(signed JWT, OAuth tokens, mTLS client cert subject). The protocol
just consumes the parsed claim list.

## Federation — multiple processes, one network

AHP addresses are universal strings. Any process that connects to the
same Redis is a node on the same network: it sees the same registry,
the same tap, the same cache. There's no "AHP service" daemon —
peers federate by sharing the substrate.

[`examples/federation/`](examples/federation/) runs two FastAPI
processes against one Redis to prove this end-to-end:

```
Node A :8001          Redis           Node B :8002
─────────────         ──────          ──────────────
Bull  (...bull) ◀──── HSET ────▶  Researcher (...researcher)
Bear  (...bear) ◀──── PUBSUB ──▶  (calls A's agents by URI alone)
                ◀──── XADD ───▶
```

Node B's researcher broadcasts `CAST-GET` to
`*.adversarial.finance.*.s.*.*`. The registry resolves that to Bull
and Bear (hosted on Node A); replies flow back over Redis pub/sub.
Node B's source code never imports or references Node A.

```bash
docker run --rm -p 6379:6379 redis:7-alpine                    # the substrate
cd examples/federation && ./start.sh                            # both nodes
curl -X POST http://localhost:8002/query \
  -d '{"target":"tifin.collaborative.finance.equities.s.session.researcher",
       "body":"Tesla","thread":"t::devin","timeout":30}'
```

Library-level proof is in `tests/test_federation.py` (5 tests over a
shared fakeredis): registry sharing, cross-node SEND-GET, cross-node
CAST-GET that reaches both A and B, group-name resolution that routes
over the wire, and cache hits served from a node whose agent is now
dead.

## Tests

```bash
pytest
```

412 tests passing + 1 cleanly skipped (live Bedrock smoke). Coverage
across the library includes:

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
│   ├── registry.py       AgentRegistry + AgentMeta
│   └── auth.py           Principal + AuthPolicy + claim policies
├── engine/
│   ├── router.py         ProtocolEngine (verb dispatcher)
│   ├── thread_manager.py ThreadManager + Thread
│   ├── scope.py          ScopePolicy + ScopeRule (open-default access control)
│   └── errors.py         ProtocolError / IncompatibleTargetError / UnauthorizedError / ...
├── adapters/
│   ├── base.py             AHPAgent
│   ├── factory.py          AgentFactory + SpawnResult
│   ├── provisioning.py     ProvisioningPattern + N* / *N / dash variants
│   ├── capability.py       Tool / Skill / RagSource / AgentProfile / CapabilityRegistry
│   ├── human.py            HumanAgent
│   ├── langgraph_agent.py  LangGraphAgent + DeepAgentDAG  (needs langgraph)
│   └── dspy_agent.py       DSPyAgent  (needs dspy-ai)
├── adapters/
│   ├── tool_address.py   ToolAddress + ResourceAddress
│   ├── tool_registry.py  ToolRegistry + @tool decorator
│   ├── resources.py      ResourceRegistry + @resource decorator
│   ├── groups.py         GroupRegistry (named pattern aliases)
│   ├── errors.py         ResolutionConflictError + Tool/Resource collision types
│   ├── storage.py        build_fs_backend + fs_mount_description (kind=fs → deepagents)
│   ├── mcp.py            register_mcp_server + register_mcp_tools
│   ├── react_agent.py    ReactAgent (wraps create_react_agent)
│   └── deep_agent.py     DeepAgent (wraps deepagents.create_deep_agent)
├── llm/
│   └── bedrock.py        ChatBedrockConverse helper + has_aws_credentials()
├── demo/
│   ├── finance_analysis.py end-to-end pipeline (deterministic stubs)
│   └── finance_react.py    same pipeline, Bedrock-driven Bull + Bear
├── cli.py                  argparse CLI: list-tools / profile / scaffold / ...
└── __main__.py             `python -m ahp <subcommand>` entry point
examples/
├── fastapi_serve/         FastAPI consumer of the library (NOT in `ahp/`)
│   ├── server.py          generic build_app(factory, agents=...)
│   ├── app.py             wires the finance demo behind HTTP
│   ├── requirements.txt
│   └── README.md
└── federation/            two FastAPI processes, one shared Redis
    ├── shared.py          build_stack(redis_url) helper
    ├── node_a.py          Bull + Bear at universal addresses
    ├── node_b.py          Researcher; reaches A by URI alone
    ├── start.sh           launch both nodes
    └── README.md
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
