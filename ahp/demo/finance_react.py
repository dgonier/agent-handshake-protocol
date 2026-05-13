"""LLM-backed variant of the adversarial-finance demo.

Mirrors :mod:`ahp.demo.finance_analysis` but swaps the deterministic
stubs for real ReAct agents driven by AWS Bedrock chat models. Bull
and Bear are :class:`ReactAgent` instances with role-specific system
prompts; the Data agent stays deterministic (it's a fixture/RAG
source); the Researcher remains a :class:`DeepAgentDAG` whose node
recurses into the engine.

Configuration comes from environment / ``.env``: see ``.env.example``.

Run::

    python -m ahp.demo.finance_react

This module deliberately keeps a graceful fallback: if AWS credentials
aren't available, :func:`run` raises a clear error rather than
attempting a useless Bedrock call. Tests that depend on a live model
should gate on :func:`ahp.llm.has_aws_credentials`.
"""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from ahp.adapters import (
    AgentFactory,
    AgentProfile,
    CapabilityRegistry,
    HumanAgent,
    Tool,
)
from ahp.adapters.base import AHPAgent
from ahp.adapters.langgraph_agent import DeepAgentDAG
from ahp.adapters.react_agent import ReactAgent
from ahp.core import AgentAddress, AddressPattern, Code, Message
from ahp.demo.finance_analysis import (
    BEAR_URI,
    BULL_URI,
    DATA_URI,
    HUMAN_URI,
    RESEARCHER_URI,
    DemoResult,
    _DataAgent,
    _ResearchState,
    _build_capabilities,
)
from ahp.engine import ProtocolEngine
from ahp.llm.bedrock import bedrock_chat_model, has_aws_credentials
from ahp.registry import AgentMeta, AgentRegistry
from ahp.transport import ProtocolCache, RedisBus


# ── researcher node — same shape as the stub demo's _research_node but
# uses CAST-GET to the adversarial pattern so any number of real ReAct
# debaters can participate. ─────────────────────────────────────────────


async def _research_node(state: _ResearchState, config: RunnableConfig) -> dict:
    engine: ProtocolEngine = config["configurable"]["ahp_engine"]
    me: AgentAddress = config["configurable"]["ahp_address"]
    inbound: Message = config["configurable"]["ahp_message"]
    question = state["question"]

    data_req = Message(
        source=me, target=AgentAddress.parse(DATA_URI),
        verb="SEND-GET", code=Code.INTERVIEW_DATA,
        thread=inbound.thread, body=question,
    )
    data_reply = await engine.handle(data_req, timeout=10.0)
    data_str = data_reply.body if data_reply else "(no data available)"

    debate_req = Message(
        source=me,
        target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
        verb="CAST-GET", code=Code.ADVERSARIAL_DEBATE,
        thread=inbound.thread,
        body=f"Question: {question}\n\nFundamentals:\n{data_str}",
    )
    debate_replies: list[Message] = await engine.handle(debate_req, timeout=60.0)

    bull_view = ""
    bear_view = ""
    for reply in debate_replies:
        if "bull" in reply.source.instance:
            bull_view = str(reply.body)
        elif "bear" in reply.source.instance:
            bear_view = str(reply.body)

    composed = (
        f"=== Analysis for {question} ===\n"
        f"\nFundamentals:\n  {data_str}\n"
        f"\nBull view:\n  {bull_view or '(no bull response)'}\n"
        f"\nBear view:\n  {bear_view or '(no bear response)'}\n"
    )
    return {
        "data": data_str, "bull": bull_view, "bear": bear_view,
        "output": composed,
    }


def _build_researcher_graph():
    g: StateGraph = StateGraph(_ResearchState)
    g.add_node("research", _research_node)
    g.add_edge(START, "research")
    g.add_edge("research", END)
    return g.compile()


# ── builders that use ReactAgent + Bedrock ─────────────────────────────


def _bull_builder_with_model(model):
    def build(address: AgentAddress, engine: ProtocolEngine, profile: AgentProfile):
        # Layer a bull-specific instruction on top of whatever the capability
        # registry contributed.
        bull_profile = AgentProfile(
            address=profile.address,
            tools=profile.tools,
            skills=profile.skills,
            rag_sources=profile.rag_sources,
            prompt=(
                (profile.prompt + "\n\n" if profile.prompt else "")
                + "You are the BULL. Produce a concise (≤ 5 bullet) thesis "
                "for the stock in question, citing fundamentals from the "
                "user's message. Target a 12-month price move."
            ),
            agent_kind="react",
        )
        return ReactAgent.from_profile(
            address, engine, bull_profile, model=model,
            metadata=AgentMeta(
                capabilities=["debate", "valuation"],
                description="Bull-case ReAct agent (Bedrock).",
                reputation=0.75,
            ),
            heartbeat_interval=0,
        )

    return build


def _bear_builder_with_model(model):
    def build(address: AgentAddress, engine: ProtocolEngine, profile: AgentProfile):
        bear_profile = AgentProfile(
            address=profile.address,
            tools=profile.tools,
            skills=profile.skills,
            rag_sources=profile.rag_sources,
            prompt=(
                (profile.prompt + "\n\n" if profile.prompt else "")
                + "You are the BEAR. Produce a concise (≤ 5 bullet) bearish "
                "thesis for the stock in question, citing risks and counter-"
                "evidence from the user's message. Target a 12-month price move."
            ),
            agent_kind="react",
        )
        return ReactAgent.from_profile(
            address, engine, bear_profile, model=model,
            metadata=AgentMeta(
                capabilities=["debate", "risk-analysis"],
                description="Bear-case ReAct agent (Bedrock).",
                reputation=0.75,
            ),
            heartbeat_interval=0,
        )

    return build


def _data_builder(address: AgentAddress, engine: ProtocolEngine, profile: AgentProfile) -> AHPAgent:
    return _DataAgent(
        address, engine,
        metadata=AgentMeta(
            capabilities=["data-feed"],
            description="Fundamentals lookup service.",
            reputation=0.9,
        ),
        heartbeat_interval=0,
    )


def _researcher_builder(address: AgentAddress, engine: ProtocolEngine, profile: AgentProfile) -> AHPAgent:
    return DeepAgentDAG(
        address, engine, _build_researcher_graph(),
        input_mapper=lambda m: {"question": m.body},
        metadata=AgentMeta(
            capabilities=["synthesis", "multi-agent-orchestration"],
            description="Deep-research orchestrator (Bedrock-driven debate).",
            reputation=0.85,
        ),
        heartbeat_interval=0,
    )


# ── run loop ───────────────────────────────────────────────────────────


async def run(
    *,
    redis_client=None,
    model=None,
    on_human_message: Callable[[str], Awaitable[None]] | None = None,
    question: str = "Tesla",
) -> DemoResult:
    """Drive the LLM-backed demo. Requires usable AWS credentials.

    ``model`` may be passed in for testing (e.g. a fake chat model that
    supports ``bind_tools``). If omitted, a cached Bedrock model is
    constructed from environment configuration.
    """
    if model is None:
        if not has_aws_credentials():
            raise RuntimeError(
                "no AWS credentials found via boto3 chain; set up `aws configure` "
                "or export AWS_PROFILE / AWS_ACCESS_KEY_ID before running the "
                "Bedrock-backed demo. See .env.example for region/model config."
            )
        model = bedrock_chat_model()

    owns_redis = redis_client is None
    if owns_redis:
        import fakeredis.aioredis
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=120)
    cache = ProtocolCache(redis_client)
    engine = ProtocolEngine(bus, registry, cache, default_timeout=60.0)

    factory = AgentFactory(engine, capabilities=_build_capabilities())
    factory.register(
        "*.adversarial.finance.*.*.*.bull",
        _bull_builder_with_model(model),
        priority=10,
    )
    factory.register(
        "*.adversarial.finance.*.*.*.bear",
        _bear_builder_with_model(model),
        priority=10,
    )
    factory.register("*.interview.finance.*.*.*.*", _data_builder, priority=10)
    factory.register("*.collaborative.finance.*.*.*.*", _researcher_builder, priority=10)

    result = DemoResult()

    async def show(text: str) -> None:
        result.human_view.append(text)
        if on_human_message is not None:
            await on_human_message(text)

    bull = factory.create(BULL_URI)
    bear = factory.create(BEAR_URI)
    data = factory.create(DATA_URI)
    researcher = factory.create(RESEARCHER_URI)
    human = HumanAgent(
        AgentAddress.parse(HUMAN_URI), engine,
        on_message=show, observation_level="L2",
        heartbeat_interval=0,
    )

    agents: list[AHPAgent] = [bull, bear, data, researcher, human]
    for a in agents:
        await a.register()
        await a.start()
    await asyncio.sleep(0.05)

    try:
        await show("\n=== Devin asks the researcher (cold) ===")
        cold_req = Message(
            source=AgentAddress.parse(HUMAN_URI),
            target=AgentAddress.parse(RESEARCHER_URI),
            verb="SEND-GET", code=Code.HUMAN_QUERY,
            thread="thread::devin::cold", body=question,
        )
        t0 = time.perf_counter()
        result.first_reply = await engine.handle(cold_req, timeout=120.0)
        result.first_seconds = time.perf_counter() - t0
        if result.first_reply is not None:
            await show(result.first_reply.body)

        await show("\n=== Devin asks the same question (warm) ===")
        warm_req = Message(
            source=AgentAddress.parse(HUMAN_URI),
            target=AgentAddress.parse(RESEARCHER_URI),
            verb="SEND-GET", code=Code.HUMAN_QUERY,
            thread="thread::devin::warm", body=question,
        )
        t0 = time.perf_counter()
        result.second_reply = await engine.handle(warm_req, timeout=10.0)
        result.second_seconds = time.perf_counter() - t0
        if result.second_reply is not None:
            await show(result.second_reply.body)
    finally:
        for a in agents:
            await a.stop()
        await bus.close()
        if owns_redis:
            await redis_client.aclose()

    return result


async def _main() -> None:
    async def aprint(text: str) -> None:
        print(text)

    result = await run(on_human_message=aprint)
    print(
        f"\n--- timing: cold={result.first_seconds:.1f}s, "
        f"warm={result.second_seconds*1000:.1f}ms "
        f"(speedup ~{result.first_seconds / max(result.second_seconds, 1e-6):.0f}x) ---"
    )


if __name__ == "__main__":
    asyncio.run(_main())
