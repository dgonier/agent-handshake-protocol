"""End-to-end AHP demo: adversarial finance analysis.

Pipeline:

* Human sends ``human.query`` to the researcher (a :class:`DeepAgentDAG`).
* Researcher recurses through the engine:
   1. ``interview.data`` SEND-GET to the data agent (returns fixtures).
   2. ``adversarial.debate`` CAST-GET to ``*.adversarial.finance.*`` →
      Bull (LangGraph) and Bear (DSPy) reply concurrently.
* Researcher composes the responses into a single brief.
* The reply flows back to the human at observation level L2.
* A second identical query short-circuits at the response cache.

The demo is fully self-contained: it uses ``fakeredis`` and stubbed
LangGraph / DSPy nodes (no LLM required). Run it with::

    python -m ahp.demo.finance_analysis

or invoke :func:`run` programmatically from a test.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, TypedDict

import dspy
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
from ahp.adapters.dspy_agent import DSPyAgent
from ahp.adapters.langgraph_agent import DeepAgentDAG, LangGraphAgent
from ahp.core import AgentAddress, AddressPattern, Code, Message
from ahp.engine import ProtocolEngine
from ahp.registry import AgentMeta, AgentRegistry
from ahp.transport import ProtocolCache, RedisBus


# ── addresses ───────────────────────────────────────────────────────────

HUMAN_URI      = "demo.human.finance.equities.s.session.devin"
RESEARCHER_URI = "demo.collaborative.finance.equities.s.session.researcher"
BULL_URI       = "demo.adversarial.finance.equities.s.session.bull"
BEAR_URI       = "demo.adversarial.finance.equities.s.session.bear"
DATA_URI       = "demo.interview.finance.equities.sj.longterm.data"


# Fictitious fixture-data, keyed by ticker. Deterministic for tests.
_FIXTURES: dict[str, str] = {
    "Tesla": (
        "Revenue $96B, EPS $4.30, P/E 70, EV/EBITDA 45, FCF $7.5B. "
        "Q4 deliveries up 12% YoY; margins compressed 380bps."
    ),
    "Apple": (
        "Revenue $383B, EPS $6.13, P/E 32, EV/EBITDA 24, FCF $99B. "
        "Services growth 11%; iPhone units flat YoY."
    ),
}


# ── bull: LangGraph state-machine ───────────────────────────────────────


class _BullState(TypedDict, total=False):
    input: str
    output: str


def _bull_node(state: _BullState) -> dict:
    ticker = state["input"]
    return {
        "output": (
            f"Bull case for {ticker}: durable competitive moat, expanding "
            f"TAM, superior execution. Target +35% over 12 months."
        )
    }


def _build_bull_graph():
    g: StateGraph = StateGraph(_BullState)
    g.add_node("bull", _bull_node)
    g.add_edge(START, "bull")
    g.add_edge("bull", END)
    return g.compile()


# ── bear: DSPy module (stub forward, no LM) ─────────────────────────────


class _BearModule(dspy.Module):
    def forward(self, text: str):  # type: ignore[override]
        return dspy.Prediction(
            answer=(
                f"Bear case for {text}: regulatory headwinds, margin "
                f"compression, valuation extended. Target -20% over 12 months."
            )
        )


# ── data agent: plain AHPAgent serving fixtures ─────────────────────────


class _DataAgent(AHPAgent):
    """Returns canned fundamentals for a ticker."""

    async def handle_message(self, message: Message) -> Message | None:
        if not message.expects_response:
            return None
        ticker = message.body if isinstance(message.body, str) else ""
        body = _FIXTURES.get(ticker, f"no fixtures for {ticker!r}")
        return Message(
            source=self.address,
            target=message.source,
            verb="SEND",
            code=Code.INTERVIEW_DATA,
            thread=message.thread,
            body=body,
        )


# ── researcher: DeepAgentDAG whose node recurses into the engine ────────


class _ResearchState(TypedDict, total=False):
    question: str
    data: str
    bull: str
    bear: str
    output: str


async def _research_node(state: _ResearchState, config: RunnableConfig) -> dict:
    """Fetch data, then debate bull/bear, then compose a single brief."""
    engine: ProtocolEngine = config["configurable"]["ahp_engine"]
    me: AgentAddress = config["configurable"]["ahp_address"]
    inbound: Message = config["configurable"]["ahp_message"]
    question = state["question"]

    # 1) Ask the data agent for fundamentals (interview.data, JSON-required).
    data_req = Message(
        source=me,
        target=AgentAddress.parse(DATA_URI),
        verb="SEND-GET",
        code=Code.INTERVIEW_DATA,
        thread=inbound.thread,
        body=question,
    )
    data_reply = await engine.handle(data_req, timeout=3.0)
    data_str = data_reply.body if data_reply else "(no data available)"

    # 2) Broadcast an adversarial debate request to bull + bear in parallel.
    debate_req = Message(
        source=me,
        target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
        verb="CAST-GET",
        code=Code.ADVERSARIAL_DEBATE,
        thread=inbound.thread,
        body=question,
    )
    debate_replies: list[Message] = await engine.handle(debate_req, timeout=3.0)

    # Sort by source instance for deterministic output.
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
        "data": data_str,
        "bull": bull_view,
        "bear": bear_view,
        "output": composed,
    }


def _build_researcher_graph():
    g: StateGraph = StateGraph(_ResearchState)
    g.add_node("research", _research_node)
    g.add_edge(START, "research")
    g.add_edge("research", END)
    return g.compile()


# ── builders for the AgentFactory ───────────────────────────────────────


def _bull_builder(address: AgentAddress, engine: ProtocolEngine, profile: AgentProfile) -> AHPAgent:
    return LangGraphAgent(
        address, engine, _build_bull_graph(),
        input_mapper=lambda m: {"input": m.body},
        metadata=AgentMeta(
            capabilities=["debate", "valuation"],
            description="Bull case generator for equities.",
            reputation=0.75,
        ),
        heartbeat_interval=0,
    )


def _bear_builder(address: AgentAddress, engine: ProtocolEngine, profile: AgentProfile) -> AHPAgent:
    return DSPyAgent(
        address, engine, _BearModule(),
        input_field="text", output_field="answer",
        metadata=AgentMeta(
            capabilities=["debate", "risk-analysis"],
            description="Bear case generator for equities.",
            reputation=0.75,
        ),
        heartbeat_interval=0,
    )


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
            description="Deep-research orchestrator.",
            reputation=0.85,
        ),
        heartbeat_interval=0,
    )


# ── capability registry: prompts that *would* steer real LLMs ──────────


def _build_capabilities() -> CapabilityRegistry:
    caps = CapabilityRegistry()
    # Finance-wide tools could go here (omitted — stub agents don't call them).
    caps.register(
        "*.*.finance.*.*.*.*",
        prompt="You operate over equities. Cite numbers when you have them.",
    )
    caps.register(
        "*.adversarial.*.*.*.*.*",
        prompt="Argue your assigned case forcefully. Anticipate counterarguments.",
        priority=5,
    )
    caps.register(
        "*.interview.*.*.*.*.*",
        prompt="Return raw facts. Avoid opinion.",
        priority=5,
    )
    caps.register(
        "*.collaborative.*.*.*.*.*",
        prompt="Synthesize. Be balanced and concise.",
        agent_kind="deep",
        priority=5,
    )
    return caps


# ── run loop ────────────────────────────────────────────────────────────


@dataclass
class DemoResult:
    """Captured artifacts from a demo run, for tests / inspection."""

    human_view: list[str] = field(default_factory=list)
    first_reply: Message | None = None
    second_reply: Message | None = None
    first_seconds: float = 0.0
    second_seconds: float = 0.0


async def run(
    *,
    redis_client=None,
    on_human_message: Callable[[str], Awaitable[None]] | None = None,
    question: str = "Tesla",
) -> DemoResult:
    """Drive the demo end-to-end. Returns a :class:`DemoResult` for inspection.

    Pass ``redis_client`` to share an existing Redis connection; if omitted,
    a fresh ``fakeredis`` instance is created and torn down automatically.
    """
    owns_redis = redis_client is None
    if owns_redis:
        import fakeredis.aioredis  # local import — only when used
        redis_client = fakeredis.aioredis.FakeRedis(decode_responses=True)

    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=60)
    cache = ProtocolCache(redis_client)
    engine = ProtocolEngine(bus, registry, cache, default_timeout=5.0)

    # Wire the factory + capability registry.
    factory = AgentFactory(engine, capabilities=_build_capabilities())
    factory.register("*.adversarial.finance.*.*.*.bull", _bull_builder, priority=10)
    factory.register("*.adversarial.finance.*.*.*.bear", _bear_builder, priority=10)
    factory.register("*.interview.finance.*.*.*.*", _data_builder, priority=10)
    factory.register("*.collaborative.finance.*.*.*.*", _researcher_builder, priority=10)

    result = DemoResult()

    async def show(text: str) -> None:
        result.human_view.append(text)
        if on_human_message is not None:
            await on_human_message(text)

    # Build agents via factory.create(). Spawn would also work if we wanted
    # bulk provisioning, but the demo's roster is fixed.
    bull = factory.create(BULL_URI)
    bear = factory.create(BEAR_URI)
    data = factory.create(DATA_URI)
    researcher = factory.create(RESEARCHER_URI)
    human = HumanAgent(
        AgentAddress.parse(HUMAN_URI),
        engine,
        on_message=show,
        observation_level="L2",
        heartbeat_interval=0,
    )

    agents: list[AHPAgent] = [bull, bear, data, researcher, human]
    for agent in agents:
        await agent.register()
        await agent.start()
    # Give pub/sub subscriptions time to land before publishing.
    await asyncio.sleep(0.05)

    try:
        # ── first query: cold; full pipeline runs ───────────────────────
        await show("\n=== Devin asks the researcher (cold) ===")
        first_req = Message(
            source=AgentAddress.parse(HUMAN_URI),
            target=AgentAddress.parse(RESEARCHER_URI),
            verb="SEND-GET",
            code=Code.HUMAN_QUERY,
            thread="thread::devin::cold",
            body=question,
        )
        t0 = time.perf_counter()
        result.first_reply = await engine.handle(first_req, timeout=5.0)
        result.first_seconds = time.perf_counter() - t0
        if result.first_reply is not None:
            await show(result.first_reply.body)

        # ── second query: hot; cache should short-circuit ───────────────
        await show("\n=== Devin asks the same question (warm) ===")
        second_req = Message(
            source=AgentAddress.parse(HUMAN_URI),
            target=AgentAddress.parse(RESEARCHER_URI),
            verb="SEND-GET",
            code=Code.HUMAN_QUERY,
            thread="thread::devin::warm",
            body=question,
        )
        t0 = time.perf_counter()
        result.second_reply = await engine.handle(second_req, timeout=1.0)
        result.second_seconds = time.perf_counter() - t0
        if result.second_reply is not None:
            await show(result.second_reply.body)
    finally:
        for agent in agents:
            await agent.stop()
        await bus.close()
        if owns_redis:
            await redis_client.aclose()

    return result


def _print(text: str) -> None:
    print(text)


async def _main() -> None:
    async def aprint(text: str) -> None:
        _print(text)

    result = await run(on_human_message=aprint)
    print(
        f"\n--- timing: cold={result.first_seconds*1000:.1f}ms, "
        f"warm={result.second_seconds*1000:.1f}ms "
        f"(speedup ~{result.first_seconds / max(result.second_seconds, 1e-6):.1f}x) ---"
    )


if __name__ == "__main__":
    asyncio.run(_main())
