# CLAUDE.md — handoff for Claude Code

Orientation doc for whoever picks up this branch next (human or a
fresh Claude Code session). Skim the **Quick orient** section first;
the rest is reference.

---

## Quick orient

* **Project:** AHP — Agentic Handshake Protocol. A messaging fabric
  for AI agents: addressing, discovery, routing, format negotiation,
  caching, observability. Library/SDK, not a service. Redis-backed.
* **Branch:** `claude/ahp-core-primitives-7looL`. All work has been
  on this branch; no PRs created (per user instruction — do NOT open
  one unless asked).
* **Test gate:** `pytest` from repo root. Currently **412 passing +
  1 cleanly skipped** (the skipped one is a live-Bedrock smoke that
  needs `AWS_PROFILE` + `AHP_RUN_BEDROCK=1`).
* **CLI:** `python -m ahp <command>` — see [README "CLI" section](README.md).
* **Top-level entrypoints to read first:**
  * [README.md](README.md) — user-facing.
  * [ahp/core/address.py](ahp/core/address.py) — the 7-field agent address.
  * [ahp/engine/router.py](ahp/engine/router.py) — the verb dispatcher,
    where all the policy layers compose.
  * [ahp/adapters/factory.py](ahp/adapters/factory.py) — how a profile
    gets assembled from capability/tool/resource/group registries.
  * [examples/federation/](examples/federation/) — concrete proof that
    addresses are universal across processes sharing one Redis.

The branch has shipped 17 phases. Look at `git log --oneline` for the
narrative arc — each commit body explains *why* the change was made,
not just what.

---

## Architecture map

```
ahp/
├── core/                  protocol primitives (zero runtime deps)
│   ├── address.py          AgentAddress — 7-field, ?params, hashable
│   ├── pattern.py          AddressPattern — wildcards + accept-subset
│   ├── codes.py            Code constants + family glob matching
│   ├── message.py          Message envelope, verbs, lifecycle→TTL,
│                           cache_key() (target+code+body digest)
│   └── compatibility.py    code → tier requirements
├── transport/             Redis-backed wire layer
│   ├── keys.py             "ahp:" namespace conventions
│   ├── redis_bus.py        RedisBus + Subscription + tap channel
│   └── cache.py            ProtocolCache (lifecycle TTL, params-aware)
├── registry/              agent directory + auth
│   ├── registry.py         AgentRegistry + AgentMeta
│   └── auth.py             Principal + AuthPolicy (Open/Claim/DenyAll)
├── engine/                protocol routing
│   ├── router.py           ProtocolEngine — verb dispatcher
│   ├── thread_manager.py   ThreadManager + Thread
│   ├── scope.py            ScopePolicy — open-default routing rules
│   └── errors.py           ProtocolError / UnauthorizedError / ...
├── adapters/              agent base + adapters + registries
│   ├── base.py             AHPAgent (subclass-and-override)
│   ├── factory.py          AgentFactory + SpawnResult +
│                           ResolutionConflictError types
│   ├── provisioning.py     ProvisioningPattern (N*/star-N/dash)
│   ├── capability.py       Tool/Skill/RagSource/AgentProfile/
│                           CapabilityRegistry
│   ├── tool_address.py     ToolAddress + ResourceAddress
│   ├── tool_registry.py    @tool decorator + ToolRegistry
│   ├── resources.py        @resource decorator + ResourceRegistry
│   ├── groups.py           GroupRegistry (named pattern aliases)
│   ├── storage.py          kind="fs" resources → deepagents FS
│   ├── mcp.py              MCP passthrough (whole servers under a scope)
│   ├── errors.py           ResolutionConflictError + Tool/Resource
│                           collision types
│   ├── human.py            HumanAgent
│   ├── langgraph_agent.py  LangGraphAgent + DeepAgentDAG (legacy DAG)
│   ├── react_agent.py      ReactAgent (wraps create_react_agent)
│   ├── deep_agent.py       DeepAgent (wraps deepagents.create_deep_agent)
│   ├── dspy_agent.py       DSPyAgent
│   ├── knowledge_graph.py  KGNode/KGEdge + InMemoryKnowledgeGraph +
│                           build_kg_backend (kind="kg" resources)
│   ├── neo4j_kg.py         Neo4j-backed KnowledgeGraphBackend with
│                           native vector index (opt-in [kg] extra)
│   └── teacher_agent.py    TeacherAgent — agent-as-judge that writes
│                           Judgement nodes into a KG backend
├── llm/
│   ├── bedrock.py          ChatBedrockConverse helper + creds check
│   ├── openrouter.py       ChatOpenAI helper for OpenRouter / Modal endpoints
│   └── recipe.py           ModelHandle / LoRAHandle + recipe finders
├── demo/
│   ├── finance_analysis.py deterministic stub pipeline
│   └── finance_react.py    Bedrock-driven LLM pipeline
├── cli.py                  argparse CLI
└── __main__.py             `python -m ahp` entry

examples/
├── fastapi_serve/          generic FastAPI face over the protocol
├── federation/             two FastAPI processes, one Redis
└── knowledge_graph/        Neo4j boilerplate + teacher demo
    ├── docker-compose.neo4j.yml   local Neo4j 5 with APOC + GDS
    ├── bootstrap_vectors.cypher   idempotent vector-index setup
    ├── teacher_demo.py            end-to-end TeacherAgent run
    └── terraform/                 single-node EC2 deployment

tests/                      pytest suite (412 passing)
```

---

## Design decisions that should NOT be relitigated

When in doubt, preserve these. They came out of explicit discussion
and the codebase is shaped around them.

1. **Default open beyond normal filtering.** The protocol always
   enforces the compatibility matrix, liveness, and address-pattern
   matching. `ScopePolicy` (routing) and `AuthPolicy` (registration)
   are *optional* tighteners. With nothing wired, anyone can route to
   anyone and anyone can register at any address. This is intentional
   — single-tenant deployments stay ergonomic.

2. **Addresses are universal strings.** 7 fields for agents
   (`org.role.domain.subdomain.accept.lifecycle.instance?params`),
   5 fields for tools (`scope.kind.role.category.operation`),
   5 fields for resources (`scope.kind.domain.subdomain.name`).
   Universal = same string identifies the same thing across every
   process that shares the Redis substrate.

3. **`*` is the right default for many tool fields, especially
   `role`.** The role field describes which agent kinds *may use*
   the tool, not which kinds it belongs to. Most DB/FS/API tools
   are usable by any role in scope. The decorator allows positional
   args; teach `*` liberally in docs.

4. **Tuple-level swap-in for provisioning reuse.** When `N*` (no
   dash) provisions agents, registry-existing agents are reused
   WHOLE rather than field-by-field. Per-field independent reuse
   broke tuples (you'd combine org from agent A with instance from
   agent B). Tuple-level swap is preserved.

5. **FastAPI is not in the package.** It lives in
   `examples/fastapi_serve/`. The library stays framework-agnostic;
   consumers wire their own HTTP layer.

6. **Convention-derived access scope, not explicit allowed_for.**
   Tool address `scope.kind.role.category.operation` derives
   `allowed_for = scope.role.*.*.*.*.*` (the convention). Override
   only when the convention is wrong — don't make users write
   `allowed_for=` for every registration.

7. **Cache key includes a body digest.** Previously two distinct
   queries to the same (target, code) would collide on one cache
   slot. Fixed via `Message.cache_key()` which composes
   `sha256(target_uri, code, body_digest)`.

8. **MCP servers are namespaced under your address scheme, not
   bolted on.** `register_mcp_server(..., scope, kind, role,
   category)` puts every MCP tool at
   `{scope}.{kind}.{role}.{category}.{mcp_tool_name}`. Same factory
   + agent address pattern that binds native tools binds MCP tools.

9. **Reads are NOT gated by `AuthPolicy`.** `resolve()`,
   `discover()`, `is_alive()`, the tap channel — all open even when
   `AddressClaimPolicy` is active. Federation requires every node to
   see who's on the network. Read-side gating would break that.

10. **`INVALIDATE` bypasses `ScopePolicy`.** Cache control is a
    separate plane. Don't entangle it with routing policy.

11. **Group names live in a separate `GroupRegistry`, not on the
    pattern.** `broadcast_to("debaters", ...)` resolves the string
    name to an `AddressPattern`. Names are local to a process; the
    pattern they resolve to is universal.

12. **Three natural model/compute sources: Bedrock, OpenRouter,
    Modal.** Bedrock and OpenRouter are hosted endpoints (consumed
    via `ahp.llm.{bedrock,openrouter}`). Modal is where AHP nodes
    *run* when they need GPUs — either as OpenAI-compatible endpoints
    (use `openrouter_chat_model(base_url=...)`) or as full AHP nodes
    that join the Redis network. Don't add other providers ad-hoc;
    if a fourth source is needed, factor it through the same shape.

13. **LoRAs and base models are addressable resources, not bolted-on
    config.** `kind="model"` / `kind="lora"` register via the standard
    `@resource(...)` decorator and return `ModelHandle` / `LoRAHandle`
    instances (pure metadata, no weights loaded). Agent recipes
    compose by address: a base at `{scope}.model.*.*.{name}` is
    visible to every agent in scope; LoRAs at
    `{scope}.lora.{domain}.{subdomain}.{name}` apply via the standard
    convention. Caveat: `ResourceAddress` has no `role` field, so
    role-gated LoRAs need explicit `allowed_for=`. Consumers
    introspect via `find_model` / `find_loras` / `recipe_summary` in
    `ahp.llm.recipe`.

---

## Gotchas (real footguns we've already hit)

* **Empty registries are falsy.** `ToolRegistry`, `ResourceRegistry`,
  `GroupRegistry`, `CapabilityRegistry`, `ScopePolicy` all define
  `__len__`. Writing `tools or ToolRegistry()` silently replaces an
  empty user registry with a fresh one. Use `tools if tools is not
  None else ToolRegistry()`. Fixed in `factory.py`; watch for this
  if you add new registry types.

* **`asyncio.run()` inside pytest-asyncio's loop fails.** The CLI's
  sync entry point uses `asyncio.run()` which works for shell
  invocation but not from `async def test_…` functions. Tests for
  `list-agents` call `_list_agents_async` directly via an `_arun()`
  helper with pre-parsed args.

* **FakeRedis pubsub is per-client.** Two `FakeRedis` instances on a
  shared `FakeServer` don't share pubsub channels. For federation
  tests, share a single client instance between two `RedisBus`
  instances. Real Redis doesn't have this issue.

* **`langgraph.prebuilt.create_react_agent` is deprecated** in
  LangGraph V1 in favor of `langchain.agents.create_agent`. We
  haven't migrated; pytest filters out the warning. If you migrate,
  add `langchain` to deps.

* **LangChain tools need `bind_tools` on the chat model.** Plain
  `FakeListChatModel` doesn't implement it. Tests use a
  `_ToolableFakeChat` subclass with `def bind_tools(self, tools,
  **kw): return self`. Real Bedrock models work fine.

* **`_to_langchain_tool` detects coroutine handlers.** AHP-aware
  tools that `await engine.handle(...)` are async — the translator
  wires them as `coroutine=` on `StructuredTool` so LangChain
  doesn't try to `asyncio.run` them inside a running loop.

* **Tool / resource name collisions are surfaced at
  `factory.profile_for(addr)` time.** Two tools at different
  `ToolAddress`-es with the same `operation` name applied to one
  agent raise `ToolNameCollisionError`. Same for resources. Don't
  catch these silently — they indicate a real wiring bug.

* **Multiple `AgentFactory` instances on one engine.** The factory
  sets `engine.scope` and `engine.groups`. A second factory
  clobbers; a warning logs. Tests using two factories should expect
  the warning.

* **`StateBackend` is the default deepagents FS.** It requires being
  inside a LangGraph state. For raw tests it instantiates fine but
  read/write requires the graph to be running.

---

## Open threads / what could come next

Picked from the post-Phase summaries; user has been steering the
priorities. Don't start any of these unprompted unless the user
asks.

1. **Audit / observability layer.** Typed `AuditEvent`
   (timestamp/principal/op/target/success/error), `AuditSink` protocol,
   concrete sinks (`InMemoryAuditSink`, `LoggingAuditSink`,
   `RedisStreamAuditSink`, `MultiSink`). Emit on
   register/deregister/heartbeat. Was deferred pending a real
   cross-process target.

2. **Live Bedrock smoke run.** `tests/test_demo_react.py::test_react_demo_against_real_bedrock`
   is the test. Needs `AWS_PROFILE` (or `AWS_ACCESS_KEY_ID` env vars)
   + `AHP_RUN_BEDROCK=1`. Once verified, capture the output.

3. **More CLI subcommands.** `tap --pattern PAT --code CODE` to
   live-stream the `ahp:tap` channel from the shell; `send` to fire
   a one-off message; `describe-agent ADDR` for more detail than
   `list-agents` provides.

4. **`langgraph.prebuilt.create_react_agent` → `langchain.agents.create_agent`
   migration.** Adds `langchain` as a hard dep for the react adapter.
   Clears the deprecation warning.

5. **Reputation/trust scoring on `AgentMeta.reputation`.** Currently
   the field exists but nothing updates it. Plug into win-rate from
   adversarial debates or similar.

6. **Gateway agents.** Protocol translators between accept tiers
   (e.g. `e→s` summarizers letting human observers see embedding-tier
   traffic). The plan called these out as architecturally important
   but they haven't been built.

---

## Conventions

### Commit messages

Use `Phase N: <one-line summary>` for the title when shipping a
coherent unit of work. Body explains *why* not just *what* — and
records the test count delta. Always end with the magic trailer:

```
https://claude.ai/code/session_017BgNP4Ef7a3m1ebo1gRFny
```

(Your session ID — Claude Code will fill it in automatically when
you use the `git commit` HEREDOC pattern.)

### Versions

`__version__` lives in `ahp/__init__.py` AND `version = ` in
`pyproject.toml`. Keep them in sync. Phase commits bump the minor
(`0.16.0` → `0.17.0`). The repo is pre-1.0 so we don't promise
stability.

### Dependencies

`ahp.core` stays **zero runtime deps**. Everything else is an
optional extra in `pyproject.toml`:

* `[redis]` — `redis>=5.0` (transport/registry/engine)
* `[langgraph]` — for `LangGraphAgent` / `DeepAgentDAG` / `ReactAgent`
* `[deepagents]` — for `DeepAgent` and storage
* `[dspy]` — for `DSPyAgent`
* `[aws]` — for the Bedrock helper
* `[mcp]` — for MCP passthrough
* `[test]` — installs everything above plus `pytest`, `fakeredis`, etc.

Adding new framework deps: add a new extra, don't put it in
`dependencies`.

### Testing

* `pytest` from repo root runs everything.
* The conftest fixture `stack` provides a wired bus + registry +
  cache + engine + threads + factory on one fakeredis client.
* `redis_client` is a bare fakeredis fixture for tests that only
  need the client.
* Module-level decorator tests need to clear the
  `DEFAULT_TOOL_REGISTRY` / `DEFAULT_RESOURCE_REGISTRY` /
  `DEFAULT_GROUP_REGISTRY` in an autouse fixture — see
  `tests/test_cli.py` for the pattern.
* When adding LLM tests, use `_ToolableFakeChat` (a `FakeListChatModel`
  subclass with a no-op `bind_tools`).
* Async tests work via `pytest-asyncio` auto mode (configured in
  `pyproject.toml`).

### Style

* No emojis in code or commits unless the user asks for them.
* No comments restating what the code says — only when the *why* is
  non-obvious (a constraint, a subtle invariant, a deliberate
  workaround).
* No defensive validation at internal boundaries — trust internal
  callers. Validate at user/external boundaries (parsers,
  decorators, public APIs).
* Default to writing no docstrings on private helpers. Public APIs
  get docstrings that explain semantics, not just signatures.

---

## Useful one-liners

```bash
# Verify the suite still passes
pytest -q -p no:warnings

# Just one test file
pytest tests/test_engine.py -v

# Run the stub demo end-to-end
python -m ahp.demo.finance_analysis

# Run the LLM demo (needs AWS creds)
AHP_RUN_BEDROCK=1 python -m ahp.demo.finance_react

# Try the CLI against a real Redis
docker run --rm -d -p 6379:6379 redis:7-alpine
AHP_REDIS_URL=redis://localhost:6379/0 python -m ahp list-agents

# Two-process federation
cd examples/federation && ./start.sh
curl -X POST http://localhost:8002/query \
  -d '{"target":"tifin.collaborative.finance.equities.s.session.researcher",
       "body":"Tesla","thread":"t::test","timeout":30}'

# Check what's on the latest pushed commit
git log --oneline -1
```

---

## Git remote situation

The remote `origin` points at a local proxy
(`http://127.0.0.1:NNNN/git/dgonier/cc-agent-proxy-experiment`) that
forwards to the real GitHub repo at
`https://github.com/dgonier/cc-agent-proxy-experiment`. The proxy
URL is sandbox-internal — only reachable from inside this Claude
Code environment. For a fresh checkout, clone from the GitHub URL
directly.

To pull on a developer machine:

```bash
git clone https://github.com/dgonier/cc-agent-proxy-experiment
cd cc-agent-proxy-experiment
git checkout claude/ahp-core-primitives-7looL
pip install -e ".[test]"
pytest
```
