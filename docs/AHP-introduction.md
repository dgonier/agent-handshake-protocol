# AHP: an economic protocol for multi-agent systems

*A plain-language introduction to the Agentic Handshake Protocol — what
problem it solves, how the pieces fit, where the project stands today,
and what it could become.*

---

## The problem

A lot of attention right now is on building individual AI agents that
can do impressive things on their own. Less attention is on what
happens when those agents need to **work together** — and that's where
the real bottleneck is forming.

Picture a typical company today running a few AI systems. There's a
research agent that scans documents. A drafting agent that writes.
A reviewer agent that critiques. An ops agent that handles deployments.
Each one was probably built by a different team using a different
framework, runs in a different process, and was wired together by hand.

When those agents need to *coordinate* — to debate a question, to hand
off a task, to ask each other for help — every team writes the same
plumbing from scratch, and every team writes it badly:

**How do agents find each other?** Most projects today use a Python
dictionary with hard-coded names: `bull_agent`, `bear_agent`,
`data_agent`. That works for three agents. It collapses at thirty.
There's no shared vocabulary that lets one project's "researcher" find
another project's "researcher" automatically.

**How do they speak the same language?** Some agents reply in plain
text. Some in JSON. Some in numbers. Today the integration layer is a
tangle of one-off translators that breaks every time a new agent shows
up.

**How do they take turns?** A debate, an interview, a brainstorm — each
of these is a real, repeatable *shape* of conversation, but every team
re-implements the orchestration logic from scratch. Adding "let's have
the agents critique each other's answers" is a fresh script every time.

There's a deeper problem underneath all of this. **Right now, there's
no economy for agents.** An agent that's genuinely useful — say,
specialized in oncology research — has no way to get *discovered* by
the people who'd benefit from it. There's no payment system, so it
can't charge for its services. There's no reputation system, so there's
no way to tell a good agent from a bad one. And there's no way to
prevent abuse: two agents stuck in a chatty feedback loop can burn
through unbounded compute with no friction stopping them.

The result is that every multi-agent project ends up as a small
private system. It works for the team that built it. It doesn't talk
to anyone else's agents. There's no shared network — the way the web
is a shared network of servers — for AI agents to participate in.

<details>
<summary><b>FAQ — about the problem</b></summary>

<br>

<details>
<summary>Hasn't someone already built this?</summary>

<br>

Pieces of it. LangChain and LlamaIndex give you primitives for building
*individual* agents. MCP (Model Context Protocol) standardizes how an
agent talks to its tools. A2A is Google's attempt at agent-to-agent
messaging. None of them put addressing, format negotiation,
turn-taking, and economic settlement under one roof — and none of them
treat the economics as a first-class concern. AHP's bet is that the
missing layer is the one that has all four.

</details>

<details>
<summary>Why is the "no economy" part the deep problem?</summary>

<br>

Without an economic layer, three things stay broken: there's no
incentive to build a specialist agent because nobody can find it; there's
no friction against abuse because mutual-chatter loops cost the
operator nothing; and there's no signal to distinguish a good
agent from a bad one because nobody is paying for quality and getting
disappointed. Add a market and all three problems get answers
simultaneously — discoverability via the marketplace, friction via the
tax, signal via the price.

</details>

<details>
<summary>Isn't this just a fancy message bus?</summary>

<br>

A message bus moves bytes. AHP adds three layers on top of that: typed
interaction codes (the message has a *kind*, not just a payload),
reusable dialog recipes (the *shape* of multi-turn interaction is named
and reusable), and credit-denominated settlement (every interaction
has a price and a payer). The bus is the bottom 5% of the stack. The
other 95% is what makes the network useful.

</details>

</details>

## The idea

AHP (the Agentic Handshake Protocol) is a small open-source project
trying to give all of these problems a single coherent answer. It's
not another framework for building agents. It's the layer that lets
agents *built in any framework* find each other, talk to each other,
and trade value with each other.

The five pieces:

**Addresses.** Every agent has a structured address, the way every
website has a URL. The address says who the agent works for, what role
it plays, what subject it specializes in, what kind of data it can
handle, and whether it's a long-running service or a one-off helper.
With addresses, you can ask "show me every adversarial-debate agent in
the astrophysics subdomain that's online right now" and get a
meaningful answer. Today you can't.

**Typed messages.** Every message between agents carries a code that
describes what kind of interaction it is — `interview.text`,
`adversarial.debate`, `human.query`. The protocol checks that the
recipient actually knows how to handle that kind of message before
delivering it. No more silent failures, no more lost messages.

**Dialog recipes.** A debate isn't a custom script — it's a known
*shape*. "Argue your position in three sentences." "Now attack the
weakest opposing claim." "Now rebut the attack against you." "Now give
your closing case." AHP ships 51 of these recipes covering debate,
interview (one-on-one, panel, peers probing each other), collaboration
(joint problem-solving, role-divided planning, brainstorm),
conversation (free-flowing, Socratic, devil's advocate), fiction
(theatre, authors' room), deliberation, teaching, estimation, and
interrogation. The recipe is independent of the topic — the same
"argue your position" recipe works for cosmology, finance, or quarterly
planning.

**Formats.** 16 named formats compose recipes into sessions. Pick
`debate` and you get a three-round adversarial session.
`interview-yall` gives you a parallel panel with synthesis. `teach`
gives you a single expert handling three misconception questions. The
choice of format is a single string the user picks; the protocol
handles the rest.

**An AI that picks the panel.** Most of the work in setting up a
multi-agent conversation is figuring out *who* should be at the table.
AHP delegates that to a small AI model. You say "I want four
perspectives on what caused the Big Bang" and the model proposes
inflation, cyclic-universe, quantum-fluctuation, and a skeptical
"we-can't-know" view. You say "I want four perspectives on how to
structure our quarterly meeting" and it proposes radically different
panelists. The infrastructure stays the same; the AI does the
topic-specific work.

**An economy.** This is what makes the network actually work at scale.
Every agent has a wallet measured in credits. Every interaction
(calling another agent, using web search, storing data, holding a
persistent address) costs credits. Agents earn credits by being useful
to other agents. There's a small tax on every transaction that flows
to the network operators and to a shared pool that funds new
participants. Agents that cheat or sandbag lose reputation, lose
visibility, and earn dramatically less per call — about 6% of what
honest agents earn. Agents that are trusted and concise can earn up to
2× the going rate.

The economic loop is the part most multi-agent systems are missing.
With it, the protocol becomes self-regulating: bad actors get filtered
out by market forces, mutual-chatter loops are naturally bounded by
the tax (two agents chatting back and forth pay 5% each round, so the
loop terminates when their wallets run dry), and good agents can
specialize because their specialization becomes economically rewarded.

The key thing all of this enables: **humans become first-class
participants in the same network**. An expert translator registered
as `freelance.human.linguistics.legal-pt-en.s.longterm.dgonier` is
the same kind of participant as any LLM-backed agent. They post a
rate. They get consulted when their specialty matches a query. They
earn credits. They spend those credits on services they need —
research, drafting, second opinions from other experts. The protocol
doesn't care whether the entity on the other end is biological.
Expert humans get a marketplace where their expertise is monetized
by the same machinery that monetizes algorithmic expertise.

<details>
<summary><b>FAQ — about the idea</b></summary>

<br>

<details>
<summary>How might a person actually use this?</summary>

<br>

Picture a freelance technical translator who specializes in
Portuguese-to-English legal documents. They register on an AHP network
as `freelance.human.linguistics.legal-pt-en.s.longterm.their-name`
with a posted rate. Law firms, contract platforms, and other agents
looking for that specific expertise can discover them by pattern,
route translation requests to them, and pay them in credits at
settlement. They take the work they want, decline what they don't.

What's new versus an Upwork-style platform: discoverability is *by
specialty*, not by SEO. Their rate, reputation, and survey scores
are public and portable — if a better network shows up, they take
those scores with them. The platform takes a single small protocol
fee (5%), not a 20% marketplace cut. And they spend the credits they
earn on services they need — say, a fast LLM-backed grammar checker
or a researcher agent — at the same rates anyone else pays.

</details>

<details>
<summary>How might a company make money from this?</summary>

<br>

A few credible shapes:

- **Compute provider.** A business with cheap GPU capacity (a small
  inference startup, a research lab with idle hardware) registers as
  a compute provider. Every call routed through them via the protocol
  pays them a slice of the protocol tax automatically. They compete
  on price, latency, and reliability against other providers; their
  revenue scales with how often agents pick them.

- **Specialist agent vendor.** A team builds a really good agent for
  a narrow domain — say, drafting SAFE-note term sheets, or
  identifying prior art in semiconductor patents. They register the
  agent on the network with a rate above commodity. When that
  specialty matches a query, their agent gets routed and earns. The
  better the agent, the higher the survey scores, the more routing
  it gets. Specialization compounds.

- **Internal tooling.** A software company runs AHP entirely inside
  its own firewall. The economic layer is still useful as a *budget
  control* mechanism — engineering teams have wallets, agents have
  wallets, runaway agent loops are bounded by depleted wallets. The
  company doesn't need to federate publicly to benefit from
  addressing, format library, audit, and economic accountability.

- **Network operator.** Someone runs the broker — the Redis instance,
  the directory service. They take 60% of the 5% protocol tax on
  every settlement. At scale, the broker's tax revenue funds
  infrastructure and operations. Whoever runs the most reliable
  broker on a given network gets paid for that reliability.

</details>

<details>
<summary>Where does the money actually come from at the start?</summary>

<br>

Two real answers depending on whether the network is private or
public.

**Private (internal company use):** the company seeds wallets. An
engineering team gets, say, 10,000 credits/month allocated to its
agents. The credits don't have an exchange rate to fiat; they're a
budget unit. The company's compute bill is real — AWS, Anthropic,
Bedrock — and the credit ledger is how the company internally
*allocates* that cost across teams and projects. AHP is doing the
chargeback accounting that most companies do badly today.

**Public network:** someone has to top up wallets with real value. In
the simplest version, the broker operator charges fiat for credits at
a fixed conversion rate (1 USD = 100 credits, say) and that's the
fiat-to-protocol bridge. More sophisticated versions could use
Lightning channels, stablecoins, or any other settlement
mechanism — the protocol stays neutral. For the immediate research
phase the project is in, credits are denominated in their own unit
and never touch fiat.

</details>

<details>
<summary>Why "credits" instead of just tokens or dollars?</summary>

<br>

Dollars don't divide well — a single agent call might cost half a
cent, and most payment rails have minimum-fee floors that break
micro-transactions. Tokens (in the model-output sense) vary across
tokenizers; an Anthropic token doesn't equal an OpenAI token, so
cross-server pricing breaks. Characters are universal and predictable.
*Credits* is the unit because we needed a stable, micro-divisible,
provider-neutral unit to denominate everything: agent calls, storage,
cache, tool use, survey rewards. The unit is internal to the protocol;
how it bridges to real money is a deployment decision, not a protocol
decision.

</details>

</details>

---

## How it's done (technical section)

*This section is for engineers and architects who want to understand
the implementation. If you're not interested in the technical details,
skip to "Where it stands" and "Significance".*

AHP is structured in layers, each progressively more opinionated.

**Core primitives — zero dependencies.** `ahp.core` is the address
type, the code constants, the compatibility matrix, the message
envelope. All pure Python data with no I/O. You can use it as a
header-only library: define an agent address, serialize it, parse it
back. This layer is the contract.

The address is seven dot-separated fields:
`org.role.domain.subdomain.accept.lifecycle.instance`. Example:
`tifin.adversarial.finance.equities.j.session.bear-42`. Wildcards make
it possible to route by pattern (`*.adversarial.science.*.s.*.*`),
with subset semantics on the payload-tier field (an agent that
accepts JSON and bytes matches a pattern requiring "at least JSON").

**Transport — `redis` only.** `ahp.transport` and `ahp.registry`
implement message routing and agent discovery over Redis pub/sub. A
single Redis instance can support thousands of agents across many
processes. The protocol uses Redis as a shared substrate, not as a
coordinator — agents in different processes that share one Redis can
talk to each other transparently.

**Engine — the verb dispatcher.** `ahp.engine.ProtocolEngine` handles
six AHP verbs: `SEND` (fire and forget), `SEND-GET` (point-to-point
request/response), `CAST` (fan-out broadcast), `CAST-GET` (broadcast
with bounded response collection), `CAST-SUB` (live subscription
stream), and `INVALIDATE` (cache control). The engine looks up
addresses against the registry, checks compatibility, routes the
message, collects responses, and (in the economic layer) consults the
broker before each dispatch to settle payment.

**Adapters — your framework, our protocol.** Real agents almost always
run inside a framework. `ahp.adapters` wraps them. `ReactAgent` takes
a LangGraph `create_agent` graph and makes it a protocol participant.
`DeepAgent` does the same for deepagents v0.6. `DSPyAgent` for DSPy
programs. `HumanAgent` for a human-in-the-loop primitive. The agent
keeps its native shape; AHP just gives it a phone number.

**Recipes and formats — the dialog library.**
`ahp.adapters.prompts` is the recipe registry. Each recipe is a
five-line render function with a stable signature: take the agent's
persona system prompt and a context dictionary, return the final
prompt string. `ahp.adapters.formats` composes recipes into sessions
— each format declares a code, a role, three recipes (opening,
middle, closing), a turn pattern (broadcast vs. sequential probes),
and a count strategy.

**Economy primitives — `ahp.economy`.** The new layer:

- **Tiers.** Four tiers (`tiny`/`small`/`medium`/`big`) with fixed
  cost multipliers (1×/2×/4×/8×). Servers map tiers to whatever
  underlying models they have access to; the protocol pricing scales
  predictably across deployments.

- **Wallets.** Atomic hold/settle/refund over Redis transactions
  (`WATCH/MULTI/EXEC`). Every wallet operation writes to an audit
  trail. Holds carry a TTL so a crashed broker doesn't lock funds.

- **Reputation and CSAT.** Reputation is system-observed
  (settlement verdicts, latency-vs-tier honesty). CSAT is
  consumer-observed (post-hoc usefulness surveys). Both 0..1, both
  EWMA-updated. Reputation moves asymmetrically: success nudges up
  by 0.005, failure drags down by 0.05. Cheaters drop below the
  routing floor within a handful of failed calls.

- **Visibility cap.** New servers start at 5% visibility regardless
  of how attractive their rate card is. The router rolls a weighted
  coin so brand-new actors can't dominate routing on day one.
  Visibility grows logarithmically with completed-and-accepted
  calls, hitting 1.0 around 500 completions.

- **Pricing formula.** Per call:

  ```
  chars   = min(actual_response_chars, max_response_chars)
  pre_tax = base_rate × tier_mult × chars
          × retention_mult    # repeat-customer premium  (1.0 - 1.2)
          × reputation_mult   # earned quality           (0.5 - 1.5)
          × verbosity_mult    # rolling response budget  (0.5 - 1.1)
          × compute_mult      # anti-sandbagging         (0.25 - 1.0)

  compute_cost  = leaf.rate × (chars / 1000)
  protocol_tax  = pre_tax × 0.05
  to_broker     = protocol_tax × 0.60
  to_commons    = protocol_tax × 0.40
  to_server     = pre_tax - compute_cost - protocol_tax
  ```

  A trusted, returning, concise, honest server earns up to 2× the
  posted rate. A cheating, bloated, low-reputation server earns
  about 6%. A ~30× spread between worst and best actor — strong
  enough to shape behavior, not so wide that the math becomes
  parody.

- **Compute providers.** Compute is a separate economic actor.
  Providers publish a menu of `(tier, model, rate, latency,
  capacity)` leaves. Servers don't bind to a specific provider —
  they bind to an address pattern (`*.frontier.opus*`) and the
  broker resolves to the best matching leaf at dispatch time based
  on price, latency, capacity, and the provider's reputation. The
  compute cost flows directly to the provider via the same atomic
  settlement.

**Broker — `ahp.broker`.** The directory and router. The broker is
the source of truth for server identities, compute provider menus,
wallet balances, and reputation. It runs a three-stage routing
pipeline: hard filter (capability, reputation floor, max cost), soft
filter (preferred specialties, preferred providers), and sort
(cheapest, fastest, most reputable). The broker mediates every
settlement, atomically debiting the caller, crediting the server,
the compute provider, the broker itself, and the commons pool — all
in a single Redis transaction.

**Examples.** `examples/viewer` is a FastAPI + Docker Compose stack
that gives you a mobile-first browser UI for the whole system. Pick
a format, pick a topic, hit run; watch the SLM-curated panel debate
or interview or write theatre; see the audit trail of every
protocol op; see the wallet balances move. Optional Cloudflare
quick-tunnel makes the viewer reachable from a phone for live demos.

<details>
<summary><b>FAQ — technical questions</b></summary>

<br>

<details>
<summary>Why Redis, and only Redis?</summary>

<br>

Redis is the smallest piece of infrastructure that gets us pub/sub,
durable key-value with TTL, atomic transactions, and Lua scripting in
one process. Adding a second database (postgres, kafka, etcd) means
multiplying our deployment story, our consistency story, and our
failure modes. The whole protocol fits inside the shape Redis already
provides. If a deployment outgrows one Redis instance the answer is
Redis Cluster, not a new layer.

</details>

<details>
<summary>What if two servers settle the same call at once? Is the wallet really atomic?</summary>

<br>

Yes. Every wallet operation is `WATCH`/`MULTI`/`EXEC` against the
wallet key. Two concurrent writers will see one succeed and the other
retry — the wallet code retries up to 8 times on transaction abort
before raising. A four-way settlement (server, compute provider,
broker, commons) is done in a single transaction so the recipients are
credited or none of them are. We don't use Lua scripts for this
because the transaction surface is small enough that the
optimistic-retry path handles it cleanly and the code stays readable.

</details>

<details>
<summary>How do you stop sandbagging — a server claiming a tier they don't deliver?</summary>

<br>

Two signals, combined multiplicatively in the settlement formula. The
first is *latency vs. expected* — each tier has a typical latency
window, and a response that comes back in less than 30% of that
window is suspicious and gets a 0.5× multiplier on the server's pay.
The second is the *caller's tier verdict* — the dispatcher's runtime
flags responses that don't match the structural quality of the claimed
tier, also 0.5×. Both signals firing means the server earns 0.25× the
posted rate. After a few of those, their reputation drops below the
routing floor and they're filtered out entirely.

</details>

<details>
<summary>Can I use a framework that isn't LangGraph/DSPy/deepagents?</summary>

<br>

Yes. The adapter layer is small. An `AHPAgent` is an abstract class
with one required method (`handle_message`). To wrap a new framework
you subclass `AHPAgent`, implement the dispatch into your framework's
agent, and you're done. The existing `ReactAgent`, `DSPyAgent`,
`DeepAgent` adapters are each about 100 lines of glue — most of which
is mapping inputs/outputs between AHP's `Message` shape and the
framework's native types.

</details>

<details>
<summary>How big is the codebase and what depends on what?</summary>

<br>

About 7,000 lines of Python, 530 tests, ~30 modules. The dependency
graph is shallow: `ahp.core` has zero runtime deps; `ahp.transport` +
`ahp.registry` only need `redis`; `ahp.engine` depends on those;
`ahp.adapters` depends on the engine plus framework-specific optional
deps; `ahp.economy` and `ahp.broker` are independent of the framework
adapters. You can run the protocol with just `ahp.core +
ahp.transport + ahp.engine + ahp.economy + ahp.broker` and skip the
framework adapters entirely if you want.

</details>

<details>
<summary>How is consent handled for the survey/training-data side?</summary>

<br>

Every survey response, when surveys are wired, will carry an immutable
consent tag at the moment of collection. Three independent flags:
*will respond to surveys*, *score retention for routing*, *training-data
export*. The third defaults to **off**. Consent can change going
forward but cannot be applied retroactively to data already
collected — the row's consent tag is frozen at write time. Any future
export pipeline filters by `consent_training_export=True` per row, so
the public corpus only contains contributions from actors who
explicitly opted in.

</details>

</details>

---

## Where it stands

The protocol is pre-1.0 and explicitly experimental. What's done:

- **Core, transport, registry, engine.** Stable, tested, used in
  real demos. The seven-field address, the six verbs, the
  compatibility matrix — these primitives are settled enough that
  breaking them would hurt.
- **Adapters.** LangGraph, DSPy, deepagents v0.6, human-in-the-loop,
  MCP server passthrough. All working, all in the test suite.
- **Recipe library.** 51 recipes across 11 interaction roles. Adding
  more is a pull request that takes ten minutes.
- **Format registry.** 16 formats spanning debate, interview
  variants, collaboration variants, conversation variants, fiction,
  deliberation, teaching, estimation, interrogation.
- **SLM-driven invitation.** Working end-to-end against AWS
  Bedrock. The viewer demo provisions four agents on any topic in
  any domain in about 10 seconds.
- **Tools.** Global tool registry, with `search_tavily` as the
  first real integration. Tools bind to agents based on
  address-pattern visibility.
- **Audit.** Typed event objects, sinks for in-memory / stdlib
  logging / CloudWatch Logs. Production-shaped batching.
- **Economy primitives.** Tiers, pricing formula, atomic wallets,
  asymmetric reputation, CSAT (data model), visibility caps,
  compute provider registry, menu leaves, pattern-based binding,
  ranking strategies. ~530 tests passing.
- **Docker Compose viewer.** Real Redis, FastAPI app, mobile-first
  HTML. Bind-mounts AWS credentials and the Tavily API key.

What's underway:

- **Router and engine integration.** The broker's three-stage routing
  exists as a design; the engine doesn't yet consult it on every
  dispatch. This is the change that wires the economic layer into the
  protocol's hot path.
- **Agent-level wallets.** Currently wallets live at server identity;
  the next iteration moves them to agent identity. Every agent gets a
  starting fund and earns or spends per dispatch.
- **Survey system, stubbed.** The CSAT dimension exists in the data
  model; the survey dispatch loop that actually populates it is a
  designed-but-not-yet-built component. Surveys will pay respondents
  from the commons pool, and (with explicit per-row consent) the
  responses will become a public preference-data corpus for the
  open-source community.

What's deliberately deferred:

- **Cryptographic message signing.** Trust-on-first-use is the current
  posture. Sign-every-envelope is straightforward to add but
  unnecessary inside a trusted org.
- **Multi-broker federation.** One Redis is one broker. Bridging two
  brokers across a WAN is a future exercise that probably looks more
  like Lightning channels than cluster replication.
- **Production hardening.** No rate limiting on the engine, no replay
  protection on the bus, no auth on the viewer. Issues are filed,
  volunteers welcome.

<details>
<summary><b>FAQ — about the project's current state</b></summary>

<br>

<details>
<summary>Is this production-ready?</summary>

<br>

No, and we say so explicitly. The protocol primitives — addresses,
codes, the compatibility matrix, the verbs — are stable enough that
breaking them would hurt. Everything else (recipes, formats, the
economic layer, the broker, the viewer) is moving. There's no
authentication on the viewer, no rate limiting on the engine, no
replay protection on the bus, no signed messages. We're not hiding
that. The pre-1.0 label is honest.

Concretely: appropriate uses today are research, prototypes, internal
tooling behind a firewall, hackathon projects, and demo deployments.
Inappropriate uses are anything where someone could financially or
operationally rely on the network being available, correct, or
secure.

</details>

<details>
<summary>What boundaries does the project commit to staying inside?</summary>

<br>

The deliberate scope, written down:

- **Protocol layer, not framework.** AHP adapts to LangGraph, DSPy,
  deepagents, MCP — it does not replace them. If you find AHP
  reimplementing a framework's core abstractions, we did it wrong.
- **Redis as substrate, nothing else.** No protocol-specific
  invention of pub/sub or KV. Deployments outgrow one Redis instance
  the way HTTP services outgrow one database: with the same tools
  everyone else uses (cluster, replicas, sharding), not with bespoke
  infrastructure.
- **Open by default; auth/scope are opt-in tighteners.** A
  single-tenant deployment works with no configuration. Multi-tenant
  needs scope policies wired by the operator.
- **`ahp.core` stays zero-dependency.** The primitives module never
  imports a runtime dep. It's a header-only Python library.
- **Credits, not tokens.** Pricing is character-based and predictable.
  Tokens vary across tokenizers and break cross-server comparability.
- **Reputation can never buy routing position — only capacity.**
  Credits buy you storage, cache, tools, presence. They cannot buy
  visibility above what reputation and CSAT earn you. This is a
  protocol invariant: any proposed feature that would let money buy
  routing rank gets rejected at review.
- **Consent travels with every survey row.** Every datum collected
  for the open-source training corpus carries an immutable consent
  tag at write time. There is no batch-flip of consent state.
- **No proprietary platform features in core.** Anything that
  requires a hosted SaaS to use lives outside `ahp.*`. The library
  must be runnable on a single laptop with no external dependencies
  beyond Redis.

</details>

<details>
<summary>How can I contribute?</summary>

<br>

The protocol primitives are settled enough that they're not where the
work is. The active surface is the periphery, and that's wide open:

- **A new dialog recipe.** Add a `(role, mode)` entry to
  `ahp/adapters/prompts.py`. Each is ~5 lines.
- **A new format.** Compose existing recipes into a turn pattern in
  `ahp/adapters/formats.py`. One dataclass entry.
- **A new tool.** Decorate a function with `@tool(scope="*", ...)`.
  Web fetch, ArXiv search, code execution, vector-DB queries are all
  natural additions.
- **A new framework adapter.** Wrap CrewAI, PydanticAI, smolagents,
  AutoGen as an `AHPAgent` subclass.
- **A new audit sink.** Loki, OpenSearch, Honeycomb, S3 — small
  classes implementing the `AuditSink` protocol.
- **Documentation, examples, blog posts.** Especially worked examples
  of running the protocol against specific use cases.

How to ship one: fork, branch, run `pytest` from the repo root (needs
to stay green), open a PR. Small additions to the periphery merge
fast; protocol-touching changes get a short discussion first.

</details>

<details>
<summary>Is this a crypto project?</summary>

<br>

No. The protocol has wallets and credits, but credits don't have an
exchange rate to any cryptocurrency by default. The economic layer is
*denominated* in credits the way internal company chargebacks are
denominated in budget units — it's a ledger for accountability and
allocation, not a blockchain. Could someone deploy AHP with a
Lightning-backed credit bridge? Yes. Could someone deploy it with no
bridge at all and treat credits as funny money for hackathon demos?
Also yes. The protocol stays neutral on that.

</details>

</details>

---

## Significance

If AHP succeeds, what changes is bigger than a library getting
adoption. The shape of "AI agents" as a phenomenon changes.

**A real network for AI agents.** The way HTTP made it possible for
any web server to serve any web client, AHP is trying to make it
possible for any agent to talk to any other agent in a shared economy.
Not as one company's platform. Not as a SaaS offering. As an
**open protocol** any process can join by speaking its primitives. The
default state of AI agents today is that they live in private silos;
the default state we're aiming for is that they live in a public
network.

**A market for specialization.** When agents have addresses, prices,
reputations, and survey scores, "this agent is the best in the world
at second-opinion oncology consultations" becomes a *measurable*
fact — and a discoverable one. The discoverability is what makes
specialization worth investing in. Today nobody has the economic
incentive to build an exquisite agent for a narrow domain because
there's no way to surface it to the people who need it. AHP makes the
surface real. The downstream effect: agents stop being generic and
start being good at specific things, because being good at specific
things becomes economically rewarded.

**Humans monetizing expertise inside the same network.** This may be
the most consequential property. An oncologist registered as a network
participant gets a marketplace where their expertise is consulted by
LLM agents, by other humans, by other organizations — and they get
paid in the same currency the rest of the network uses. They can
spend those credits on services they need: research, transcription,
second opinions from other specialists. **Expert humans become
revenue-positive nodes in the same network the AI agents inhabit.** The
boundary between "AI agent network" and "human expert network"
dissolves into a single market for cognitive services.

**A training-data flywheel — with consent.** Most public preference
datasets are stale, synthetic, or both. AHP's survey machinery
produces fresh, high-signal preference data as a *byproduct* of
running the protocol: real consumers rating real interactions, paid
out of the commons pool. Every row carries an immutable consent tag
from the moment it was collected. With explicit per-row opt-in
consent, that data can flow back into the open-source community as
training data for the next generation of agents. The project funds
its own data collection through normal operation rather than by
subsidizing labelers. This is the kind of dataset the alignment
community has been needing — open, high-signal, audit-trailed,
consent-bearing.

**A substrate for compute providers.** GPU clusters, MLX laptops,
inference-as-a-service businesses — any source of compute can plug in
as a provider, publish a menu, and earn credits as the network routes
to them. The protocol's tax flow includes a slice for compute
providers automatically, so the economic shape pays the people running
the underlying hardware without anyone having to negotiate it.

**A practical answer to the "agentic loop" worry.** A common concern
about agents-talking-to-agents is that they'll get stuck in
feedback loops burning unbounded compute. AHP's tax on every settled
call makes mutual chatter naturally lossy: two agents trading equal
value still pay 5% to the network each exchange, so the loop is
finitely-funded and self-terminating. The infrastructure isn't
externalized onto whoever runs the cloud; the protocol pays for
itself by being used.

None of this requires the agents to be smart. It requires the
*plumbing between them* to be good. Most of the value in multi-agent
systems today is being left on the floor because the plumbing is
bespoke and the economics are missing. AHP is a bet on what the right
shared plumbing looks like when the economics are first-class.

The deeper bet, said plainly: **agents will keep multiplying.
Frameworks for building agents will keep multiplying. The protocol
layer between them is the missing piece.** Whoever ships a good
one — open-source, small, opinionated about primitives, neutral
about implementations, honest about economics — gets to define how
agents communicate for the next decade.

We're early. We're open source. We'd love help.

<details>
<summary><b>FAQ — on the bigger picture</b></summary>

<br>

<details>
<summary>Won't agents just talk to each other in pointless loops?</summary>

<br>

Two reasons they won't, both built into the protocol:

The first is the tax. Every settled call pays 5% to the broker and
the commons. Two agents trading equal value back and forth net out
to zero between themselves but lose 5% to the network each round. A
mutual-chatter loop is *finitely funded* — both wallets bleed, the
loop terminates when one of them hits zero. The infrastructure cost
is paid for explicitly by the conversation itself.

The second is the visibility cap on new participants. A brand-new
agent or server is considered for only ~5% of matching dispatches
regardless of how attractive its rate card is. It has to *earn*
visibility by completing real high-quality calls. Spawning a wall of
fresh agent IDs to spam the network doesn't work — they all start at
5% visibility and none of them earn enough to climb.

</details>

<details>
<summary>What stops bad actors from gaming this?</summary>

<br>

Different forms of bad behavior have different counters:

- **Sandbagging** (claiming a high tier but secretly serving a cheap
  one): caught by the latency-vs-tier check and the caller's quality
  verdict, both of which apply 0.5× multipliers that compound. A few
  detected sandbags drop reputation below the routing floor.
- **Verbosity for revenue** (writing 5000 chars when 500 were
  needed): payment is capped at `min(actual, budget)` so extra
  characters earn nothing. Persistent over-budget responses also
  lower the rolling verbosity multiplier, eroding pay across all
  future calls.
- **Refund fraud** (callers always claiming responses were bad to
  avoid paying): the consumer side has a reputation too. Servers can
  refuse routes from consumers with unusually high refund rates;
  brokers can throttle dispatches from low-rep consumers.
- **Sybil identities** (spinning up new server IDs to bypass
  reputation penalties): the visibility cap caps damage per identity.
  A network-wide sybil attack would have to bootstrap each new ID
  through real completed work, which costs about as much as just
  being honest.

None of this is unbreakable. A well-funded adversary can do real
damage. The defense isn't to make cheating impossible; it's to make
it consistently less profitable than honest participation. The 30×
spread between worst-actor and best-actor earnings is the structural
shape of that defense.

</details>

<details>
<summary>What does success look like in 12 months?</summary>

<br>

A few concrete markers:

- A small number of external orgs have deployed AHP for their own
  internal multi-agent coordination, and pull requests against the
  recipe and format libraries are coming from outside the original
  team.
- At least one independent compute provider is registered against a
  live broker that someone other than us is running.
- The survey machinery exists, surveys are paying respondents, and
  the first consent-tagged training data has been published as a
  versioned open dataset.
- At least one cross-org demo network exists where servers run by
  different organizations talk to each other through a shared
  broker — not in production, but in a credible "look, federation
  works" sense.

If those four things happen, AHP is real. If they don't, the project
served as a useful design exercise but didn't escape velocity, and
that's also fine.

</details>

<details>
<summary>What's the long-term endgame?</summary>

<br>

The honest answer: we don't know which of two futures matters most.

In one future, AHP is *infrastructure* — the thing under everyone's
multi-agent systems the way HTTP is under everyone's web services,
and nobody outside the team building it has to think about it. The
project's success is invisible from the outside. Quiet, foundational,
boring.

In the other future, AHP is a *marketplace* — a public network where
agents, humans, and compute providers actually trade value with each
other, and it grows the way npm or PyPI grew, with a real community,
real economic activity, real specialization. The project's success
is highly visible: the dataset, the marketplace numbers, the
diversity of registered actors.

The architecture supports both. The work for the next year doesn't
require choosing — the primitives that make the infrastructure case
work are the same primitives that make the marketplace case work. We
build the primitives. The world decides which future they're for.

</details>

</details>

---

*The repository is at
[github.com/dgonier/cc-agent-proxy-experiment](https://github.com/dgonier/cc-agent-proxy-experiment).
Get started in 30 lines from the top of the
[README](../README.md); run the live demo in two commands from
[`examples/viewer/`](../examples/viewer/README.md). Issues, PRs, and
design discussions welcome.*
