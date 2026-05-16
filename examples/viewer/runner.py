"""Generic session runner — dispatches on :class:`Format`.

A session has three rounds: opening, middle, closing. Each round is
either:

* **broadcast** — moderator pattern-CASTs to every agent.
* **sequential_probes** — moderator sends N sequential SEND-GETs to
  the single agent, generating a fresh follow-up each turn.
* **skip** — the round is omitted.

The format controls which kind each round uses and which recipe key
to render. The runner doesn't know anything about adversarial vs.
interview vs. fiction — it just orchestrates rounds against the
:class:`SessionAgent`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any


log = logging.getLogger("ahp.viewer.runner")

from ahp.adapters import (
    AgentFactory,
    AgentProfile,
    CapabilityRegistry,
    Format,
    get_format,
    render,
)
from ahp.adapters.base import AHPAgent
from ahp.adapters.tool_registry import DEFAULT_TOOL_REGISTRY, ToolRegistry
from ahp.audit import InMemoryAuditSink, MultiSink
from ahp.audit.cloudwatch import CloudWatchLogsSink
from ahp.audit.event import AuditEvent
from ahp.core import AgentAddress, AddressPattern, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.engine.router import ProtocolEngine
from ahp.llm.bedrock import bedrock_chat_model
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


# ── helpers ──────────────────────────────────────────────────────────


def _model_short(model_id: str) -> str:
    """Slugify a Bedrock model id for use in a MenuLeaf address.

    Bedrock inference profile ids contain colons and version suffixes
    that aren't valid in our menu-leaf address validator. Trim to the
    last segment and drop anything past a colon.
    """
    tail = model_id.split(".")[-1].split(":")[0]
    # MenuLeaf model validator: lowercase, kebab/dot/underscore.
    # Strip anything else.
    cleaned = "".join(
        c if (c.isalnum() or c in "-_.") else "-"
        for c in tail.lower()
    )
    return cleaned or "model"


# ── result types ──────────────────────────────────────────────────────


@dataclass
class AgentTurn:
    """One agent's contribution to a single round."""

    slug: str
    address: str
    text: str
    round_name: str = ""        # "round1", "round2", "closing", "probe-1", ...
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DebateResult:
    """Full transcript of one session.

    Named for back-compat with the original debate runner. ``round1``,
    ``round2``, ``closing`` mirror the format's three turns. For
    sequential-probe formats, ``round2`` holds the moderator's
    probes interleaved with the interviewee's answers (one entry per
    answer; the moderator's question is in ``extra_text`` on the turn).
    """

    topic: str
    org: str
    domain: str
    subdomain: str
    count: int
    model_id: str
    started_at: float
    format: str = "debate"
    finished_at: float = 0.0
    personas: dict[str, str] = field(default_factory=dict)
    round1: list[AgentTurn] = field(default_factory=list)
    round2: list[AgentTurn] = field(default_factory=list)
    closing: list[AgentTurn] = field(default_factory=list)
    audit_events: list[dict[str, Any]] = field(default_factory=list)
    cloudwatch_stream: str | None = None
    elapsed_round1: float = 0.0
    elapsed_round2: float = 0.0
    elapsed_closing: float = 0.0
    # Economy snapshot — captured at end-of-run.
    wallets: dict[str, float] = field(default_factory=dict)
    server_meta: dict[str, Any] = field(default_factory=dict)
    compute_menu: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── agent class ───────────────────────────────────────────────────────


class SessionAgent(AHPAgent):
    """One general-purpose adapter that handles every format.

    The agent reads ``message.body["recipe"]`` (a recipe key like
    ``"adversarial:debate-me"``) and dispatches to the right
    :mod:`ahp.adapters.prompts` render call. The body also carries
    whatever context that recipe needs (``question``, ``topic``,
    ``others``, ``transcript``, ``follow_up``, ``prior``, ...).

    Tools visible to the agent's address (looked up at construction
    time from :data:`DEFAULT_TOOL_REGISTRY`) are bound to the chat
    model via LangChain's ``create_agent``. The agent's reply records
    which tools were called as a parallel ``tool_calls`` list on the
    response body so the UI can render them.
    """

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        profile: AgentProfile,
        *,
        model: Any,
        persona_system: str,
        tools: list[Any] | None = None,
    ) -> None:
        super().__init__(address=address, engine=engine)
        self._profile = profile
        self._model = model
        self._persona = persona_system
        self._tools = tools or []

    async def handle_message(self, message: Message) -> Message | None:
        body = message.body if isinstance(message.body, dict) else {}
        recipe_key = body.get("recipe")
        if not recipe_key or ":" not in recipe_key:
            return None
        role, mode = recipe_key.split(":", 1)
        ctx = {k: v for k, v in body.items() if k != "recipe"}
        ctx.setdefault("self_slug", self.address.instance)
        try:
            prompt = render(role, mode, system=self._persona, **ctx)
        except KeyError:
            return None
        text, tool_calls = await self._invoke(prompt)
        return Message(
            source=self.address,
            target=message.source,
            code=message.code,
            verb="SEND",
            body={
                "slug": self.address.instance,
                "recipe": recipe_key,
                "text": text,
                "tool_calls": tool_calls,
            },
            thread=message.thread,
        )

    async def _invoke(self, prompt: str) -> tuple[str, list[dict[str, Any]]]:
        """Run the chat model with tool-calling enabled.

        Returns ``(final_text, tool_call_records)``. When there are no
        tools bound, falls back to a plain ``model.invoke(prompt)`` to
        avoid the ``create_agent`` overhead.
        """
        if not self._tools:
            resp = await asyncio.to_thread(self._model.invoke, prompt)
            return _extract_text(resp), []

        # Build a per-call agent. The graph is cheap to construct and
        # we want the cleanest possible state per turn.
        lc_tools = [_to_langchain_tool(t) for t in self._tools]
        try:
            from langchain.agents import create_agent  # v1
            graph = create_agent(model=self._model, tools=lc_tools)
        except ImportError:
            from langgraph.prebuilt import create_react_agent  # legacy
            graph = create_react_agent(self._model, lc_tools)

        from langchain_core.messages import HumanMessage
        state = await graph.ainvoke({"messages": [HumanMessage(content=prompt)]})
        messages = state.get("messages", [])
        # Collect tool_calls across the whole trace; the LLM may have
        # called multiple tools in series.
        tool_calls: list[dict[str, Any]] = []
        for m in messages:
            for call in getattr(m, "tool_calls", []) or []:
                tool_calls.append({
                    "name": call.get("name") if isinstance(call, dict) else getattr(call, "name", "?"),
                    "args": (
                        call.get("args") if isinstance(call, dict)
                        else getattr(call, "args", {})
                    ),
                })
        final = messages[-1] if messages else None
        return _extract_text(final), tool_calls


def _extract_text(resp: Any) -> str:
    if resp is None:
        return ""
    content = getattr(resp, "content", None)
    if content is None:
        return str(resp).strip()
    if isinstance(content, str):
        return content.strip()
    # Bedrock Converse sometimes returns a list of content blocks.
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text") or block.get("content")
            if text:
                parts.append(str(text))
        else:
            parts.append(str(block))
    return "\n".join(parts).strip()


def _to_langchain_tool(t: Any):
    """Defer to the ahp.adapters.react_agent converter (handles async)."""
    from ahp.adapters.react_agent import _to_langchain_tool as conv
    return conv(t)


def _builder_for(model: Any, factory: AgentFactory, tool_registry: ToolRegistry):
    def build(addr: AgentAddress, engine: ProtocolEngine, profile: AgentProfile):
        persona = factory.persona_for(addr)
        if persona is None:
            raise RuntimeError(f"no persona registered for {addr}")
        # Every tool whose allowed_for pattern matches my address.
        tools = tool_registry.for_address(addr)
        return SessionAgent(
            address=addr, engine=engine, profile=profile,
            model=model, persona_system=persona,
            tools=tools,
        )
    return build


# ── moderator-side probe generation ───────────────────────────────────


_PROBE_PROMPT_TEMPLATE = (
    "You are a sharp moderator probing an interviewee.\n\n"
    "TOPIC: {topic}\n\n"
    "INTERVIEWEE'S MOST RECENT ANSWER: {prior}\n\n"
    "Ask one short follow-up question (under 30 words). Push on the "
    "weakest point or biggest hidden assumption. Plain question only — "
    "no preamble, no quoting."
)


async def _generate_probe(model: Any, *, topic: str, prior: str) -> str:
    prompt = _PROBE_PROMPT_TEMPLATE.format(topic=topic, prior=prior)
    resp = await asyncio.to_thread(model.invoke, prompt)
    text = resp.content if hasattr(resp, "content") else str(resp)
    return text.strip()


# ── public entry point ────────────────────────────────────────────────


async def run_debate(
    *,
    topic: str,
    format: str = "debate",
    org: str = "tifin",
    domain: str = "science",
    subdomain: str = "astrophysics",
    count: int = 4,
    model_id: str | None = None,
    region: str | None = None,
    redis_url: str | None = None,
    cloudwatch_group: str | None = "/ahp/astrophysics-demo",
) -> DebateResult:
    """Run one full session in the chosen format.

    Function name kept as ``run_debate`` for back-compat with callers;
    it now drives any registered :class:`Format`.
    """
    fmt = get_format(format)

    # Honor the format's count strategy.
    effective_count = 1 if fmt.count_strategy == "force_one" else count

    region = region or os.environ.get("AWS_REGION", "us-east-1")
    model_id = model_id or os.environ.get(
        "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )

    if redis_url is None:
        import fakeredis.aioredis
        redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    else:
        import redis.asyncio as aioredis  # type: ignore[import-not-found]
        redis = aioredis.from_url(redis_url, decode_responses=True)

    result = DebateResult(
        topic=topic, format=fmt.name, org=org,
        domain=domain, subdomain=subdomain,
        count=effective_count, model_id=model_id,
        started_at=time.time(),
    )

    mem_sink = InMemoryAuditSink(capacity=1024)
    sinks: list[Any] = [mem_sink]
    cw_sink: CloudWatchLogsSink | None = None
    if cloudwatch_group:
        try:
            cw_sink = CloudWatchLogsSink(
                log_group=cloudwatch_group,
                log_stream=f"run-{int(time.time())}",
                region=region,
                batch_size=8,
                flush_interval=2.0,
            )
            sinks.append(cw_sink)
            result.cloudwatch_stream = cw_sink.log_stream
        except Exception:
            cw_sink = None
    audit = MultiSink(sinks)

    bus = RedisBus(redis)
    registry = AgentRegistry(redis, heartbeat_ttl=120, audit=audit)
    cache = ProtocolCache(redis)

    # Stand up a broker and register a self-hosted server for the
    # current run. The server's id matches the request's ``org`` so
    # the engine's broker path can find it when settling.
    from ahp.broker import Broker, ServerMeta
    from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
    from ahp.economy.reputation import ReputationEntry, VISIBILITY_FULL_AT
    broker = Broker(redis)
    server_meta = ServerMeta(
        server_id=org,
        org=org,
        operator="viewer-runner",
        base_rate=0.0002,
        compute_binding=f"{org}.small.{_model_short(model_id)}",
        supported_tiers=["small", "medium"],
    )
    await broker.register_server(server_meta)
    await broker.register_compute_provider(ComputeProvider(provider_id=org))
    await broker.register_leaf(MenuLeaf(
        provider_id=org, tier="small", model=_model_short(model_id),
        rate_per_1k_chars=0.0,  # self-hosted: compute slice returns to the server
        latency_p95_ms=600.0, capacity=1.0,
    ))
    # Give the demo server an established reputation so the visibility
    # coin-flip doesn't filter it out on the first call.
    await broker.set_reputation(ReputationEntry(
        owner=org, reputation=0.9, completed_accepted=VISIBILITY_FULL_AT,
    ))
    # Seed the broker + commons wallets so the tax has somewhere to flow.
    await broker.wallet("__broker__").topup(0.0, reason="init")
    await broker.wallet("__commons__").topup(50.0, reason="init commons pool")

    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(),
        default_timeout=90.0, audit=audit, broker=broker,
    )

    slm_model = bedrock_chat_model(
        model_id=model_id, region=region, temperature=0.5, max_tokens=600,
    )
    debate_model = bedrock_chat_model(
        model_id=model_id, region=region, temperature=0.7, max_tokens=320,
    )

    # Side-effect import: registers search_tavily and any other
    # first-party tools in DEFAULT_TOOL_REGISTRY.
    import ahp.tools  # noqa: F401

    factory = AgentFactory(
        engine, capabilities=CapabilityRegistry(),
        tools=DEFAULT_TOOL_REGISTRY,
        slm=slm_model,
        host_server_id=org,
        fund_agents=True,
    )
    # Top up the host server's wallet so it can fund the agents it
    # provisions. Without this, the banker would fail on the first
    # session-agent it tries to seed.
    from ahp.economy.agent_banker import STARTING_FUND_BY_LIFECYCLE
    await broker.wallet(org).topup(
        STARTING_FUND_BY_LIFECYCLE["session"] * (effective_count + 1),
        reason="seed host server for provisioning",
    )
    factory.register(
        AddressPattern.parse(f"*.{fmt.role}.{domain}.{subdomain}.*.*.*"),
        _builder_for(debate_model, factory, DEFAULT_TOOL_REGISTRY),
    )

    spawn = await factory.invite_and_start(
        org=org, role=fmt.role,
        domain=domain, subdomain=subdomain,
        topic=topic, count=effective_count,
        mode_hint=fmt.mode_hint or None,
    )
    # Let consumer loops finish subscribing.
    await asyncio.sleep(0.2)

    for agent in spawn.new:
        result.personas[str(agent.address)] = (
            factory.persona_for(agent.address) or ""
        )

    human = AgentAddress.parse(
        f"you.{fmt.role}.{domain}.{subdomain}.s.session.moderator"
    )
    await registry.register(human)
    # The moderator pays for every CAST-GET round; seed its wallet
    # generously so the demo never goes broke. Tax flows out of these
    # credits to broker + commons.
    await broker.wallet(str(human)).topup(
        100.0, reason="seed moderator wallet for demo",
    )

    pattern = AddressPattern.parse(
        f"*.{fmt.role}.{domain}.{subdomain}.*.*.*"
    )

    # Transcript so far, used by closing recipes that consume it.
    transcript: list[dict[str, str]] = []

    def _common_body(recipe_key: str) -> dict[str, Any]:
        """Slots both `topic` and `question` so recipes can use either."""
        return {
            "recipe": recipe_key,
            "topic": topic,
            "question": topic,
        }

    # ── Round 1 ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    round1_reply_objs = await _run_round(
        engine=engine, pattern=pattern, human=human,
        code=fmt.code, recipe=fmt.round1_recipe,
        body=_common_body(fmt.round1_recipe),
        kind="broadcast",
        max_responses=effective_count,
        thread=f"thread::{int(time.time())}::1",
        moderator_model=debate_model, topic=topic,
        probe_count=fmt.probe_count,
        prior_transcript=transcript,
        single_target=_single_target(spawn.new),
    )
    result.elapsed_round1 = time.perf_counter() - t0
    result.round1 = [_turn(r, round_name="round1") for r in round1_reply_objs]
    transcript.extend(_to_transcript_entries(result.round1))

    # ── Round 2 ─────────────────────────────────────────────────────
    if fmt.round2_recipe is not None and fmt.round2_kind != "skip":
        t0 = time.perf_counter()
        body = _common_body(fmt.round2_recipe)
        if fmt.round2_kind == "broadcast":
            body["others"] = [
                {"slug": t.slug, "body": t.text} for t in result.round1
            ]
            body["transcript"] = list(transcript)
            # Special-case: rebuttal needs attacks_on_me keyed per agent;
            # for v1 we hand everyone the full others set and let the
            # recipe note that none may be directed at them.
            body["attacks_on_me"] = body["others"]
            body["questions_for_me"] = body["others"]
        reply_objs = await _run_round(
            engine=engine, pattern=pattern, human=human,
            code=fmt.code, recipe=fmt.round2_recipe,
            body=body,
            kind=fmt.round2_kind,
            max_responses=effective_count,
            thread=f"thread::{int(time.time())}::2",
            moderator_model=debate_model, topic=topic,
            probe_count=fmt.probe_count,
            prior_transcript=transcript,
            single_target=_single_target(spawn.new),
        )
        result.elapsed_round2 = time.perf_counter() - t0
        result.round2 = [_turn(r, round_name="round2") for r in reply_objs]
        transcript.extend(_to_transcript_entries(result.round2))

    # ── Closing ─────────────────────────────────────────────────────
    if fmt.closing_recipe is not None and fmt.closing_kind != "skip":
        t0 = time.perf_counter()
        body = _common_body(fmt.closing_recipe)
        body["transcript"] = list(transcript)
        body["others"] = [
            {"slug": t.slug, "body": t.text} for t in result.round1
        ]
        reply_objs = await _run_round(
            engine=engine, pattern=pattern, human=human,
            code=fmt.code, recipe=fmt.closing_recipe,
            body=body,
            kind=fmt.closing_kind,
            max_responses=effective_count,
            thread=f"thread::{int(time.time())}::close",
            moderator_model=debate_model, topic=topic,
            probe_count=1,  # closing is one turn even in sequential mode
            prior_transcript=transcript,
            single_target=_single_target(spawn.new),
        )
        result.elapsed_closing = time.perf_counter() - t0
        result.closing = [_turn(r, round_name="closing") for r in reply_objs]

    # Capture audit transcript.
    result.audit_events = [_event_to_dict(e) for e in mem_sink.events]
    result.finished_at = time.time()

    # Capture an economy snapshot for the viewer to render.
    try:
        wallet_owners = [
            str(human), org, "__broker__", "__commons__",
        ] + [str(a.address) for a in spawn.new]
        for owner in wallet_owners:
            state = await broker.wallet(owner).get_state()
            result.wallets[owner] = round(state.balance, 6)
        result.server_meta = {
            "server_id": server_meta.server_id,
            "org": server_meta.org,
            "base_rate": server_meta.base_rate,
            "compute_binding": server_meta.compute_binding,
            "supported_tiers": list(server_meta.supported_tiers),
            "specialties": list(server_meta.specialties),
            "integrations": list(server_meta.integrations),
        }
        leaves = await broker.compute.list_leaves(only_alive_providers=False)
        result.compute_menu = [{
            "address": l.address,
            "provider": l.provider_id,
            "tier": l.tier,
            "model": l.model,
            "rate_per_1k_chars": l.rate_per_1k_chars,
            "latency_p95_ms": l.latency_p95_ms,
            "capacity": l.capacity,
            "healthy": l.healthy,
        } for l in leaves]
    except Exception:
        log.exception("economy snapshot failed; transcripts unaffected")

    # Teardown.
    for agent in spawn.new:
        await agent.stop()
        try:
            await agent.deregister()
        except Exception:
            pass
    try:
        await registry.deregister(human)
    except Exception:
        pass
    if cw_sink is not None:
        try:
            await cw_sink.flush()
        except Exception:
            pass
    await bus.close()
    try:
        await redis.aclose()
    except AttributeError:
        await redis.close()

    return result


# ── round dispatchers ─────────────────────────────────────────────────


async def _run_round(
    *,
    engine: ProtocolEngine,
    pattern: AddressPattern,
    human: AgentAddress,
    code: str,
    recipe: str,
    body: dict[str, Any],
    kind: str,
    max_responses: int,
    thread: str,
    moderator_model: Any,
    topic: str,
    probe_count: int,
    prior_transcript: list[dict[str, str]],
    single_target: AgentAddress | None,
) -> list[Message]:
    """Dispatch one round in the appropriate kind."""
    if kind == "broadcast":
        return await engine.handle(
            Message(
                source=human, target=pattern, code=code,
                verb="CAST-GET", body=body, thread=thread,
            ),
            timeout=90.0, max_responses=max_responses,
        )
    if kind == "sequential_probes":
        if single_target is None:
            return []
        replies: list[Message] = []
        prior_text = (
            prior_transcript[-1]["body"]
            if prior_transcript else "(opening statement)"
        )
        for i in range(probe_count):
            follow_up = await _generate_probe(
                moderator_model, topic=topic, prior=prior_text,
            )
            probe_body = dict(body)
            probe_body["prior"] = prior_text
            probe_body["follow_up"] = follow_up
            probe_body["moderator_question"] = follow_up
            reply = await engine.handle(
                Message(
                    source=human, target=single_target, code=code,
                    verb="SEND-GET", body=probe_body,
                    thread=f"{thread}::probe-{i}",
                ),
                timeout=60.0,
            )
            if reply is not None:
                replies.append(reply)
                if isinstance(reply.body, dict):
                    prior_text = str(reply.body.get("text", ""))
                else:
                    prior_text = str(reply.body)
        return replies
    return []


# ── helpers ───────────────────────────────────────────────────────────


def _single_target(new_agents: list[AHPAgent]) -> AgentAddress | None:
    return new_agents[0].address if new_agents else None


def _turn(m: Message, *, round_name: str = "") -> AgentTurn:
    body = m.body if isinstance(m.body, dict) else {}
    return AgentTurn(
        slug=body.get("slug", "?"),
        address=str(m.source),
        text=str(body.get("text", "")).strip(),
        round_name=round_name,
        tool_calls=list(body.get("tool_calls") or []),
    )


def _to_transcript_entries(turns: list[AgentTurn]) -> list[dict[str, str]]:
    return [{"slug": t.slug, "body": t.text} for t in turns]


def _event_to_dict(e: AuditEvent) -> dict[str, Any]:
    return {
        "op": e.op,
        "timestamp": e.timestamp,
        "principal": e.principal,
        "source": e.source,
        "target": e.target,
        "code": e.code,
        "verb": e.verb,
        "success": e.success,
        "error": e.error,
        "extra": dict(e.extra),
    }
