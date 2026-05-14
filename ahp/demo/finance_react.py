"""LLM-backed adversarial-finance demo.

Mirrors :mod:`ahp.demo.finance_analysis` but swaps the deterministic
stubs for real LLM-driven agents:

* **Researcher** is a :class:`DeepAgent` (``deepagents.create_deep_agent``)
  whose tool set includes AHP-aware closures so the planner can invoke
  Bull, Bear, and the Data agent over the protocol.
* **Bull** / **Bear** are :class:`ReactAgent` instances with role-specific
  system prompts layered on top of the capability registry.
* **Data** stays deterministic — it's a fixture/RAG source, not a model.

Configuration comes from environment / ``.env``: see ``.env.example``.

Run::

    python -m ahp.demo.finance_react

If AWS credentials aren't reachable, :func:`run` raises a clear error
rather than attempting a useless Bedrock call. Tests pass a fake chat
model via ``model=`` to exercise the wiring without real credentials.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from ahp.adapters import (
    AgentFactory,
    AgentProfile,
    CapabilityRegistry,
    HumanAgent,
    Tool,
)
from ahp.adapters.base import AHPAgent
from ahp.adapters.deep_agent import DeepAgent
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
    _build_capabilities,
)
from ahp.engine import ProtocolEngine
from ahp.llm.bedrock import bedrock_chat_model, has_aws_credentials
from ahp.registry import AgentMeta, AgentRegistry
from ahp.transport import ProtocolCache, RedisBus


# ── AHP-aware tools for the deep researcher ────────────────────────────
#
# These factories return AHP :class:`Tool` instances whose handlers
# close over the engine + the researcher's own address, turning protocol
# calls into ordinary LLM-callable functions.


def _make_lookup_fundamentals_tool(
    engine: ProtocolEngine,
    self_address: AgentAddress,
    thread: str,
) -> Tool:
    async def lookup_fundamentals(ticker: str) -> str:
        """Look up canned fundamentals for a stock ticker (e.g. 'Tesla', 'Apple')."""
        msg = Message(
            source=self_address,
            target=AgentAddress.parse(DATA_URI),
            verb="SEND-GET",
            code=Code.INTERVIEW_DATA,
            thread=thread,
            body=ticker,
        )
        reply = await engine.handle(msg, timeout=20.0)
        return reply.body if reply else f"(no data for {ticker})"

    return Tool(
        name="lookup_fundamentals",
        description="Look up canned fundamentals for a stock ticker.",
        handler=lookup_fundamentals,
    )


def _make_debate_tool(
    engine: ProtocolEngine,
    self_address: AgentAddress,
    thread: str,
) -> Tool:
    async def hold_debate(question: str) -> str:
        """Fan a question out to every adversarial finance agent and collect their cases.

        Returns the concatenated bull/bear views, labeled by side."""
        msg = Message(
            source=self_address,
            target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
            verb="CAST-GET",
            code=Code.ADVERSARIAL_DEBATE,
            thread=thread,
            body=question,
        )
        replies: list[Message] = await engine.handle(msg, timeout=60.0)
        if not replies:
            return "(no debaters responded)"
        lines = []
        for r in replies:
            label = "BULL" if "bull" in r.source.instance else (
                "BEAR" if "bear" in r.source.instance else r.source.instance
            )
            lines.append(f"--- {label} ({r.source.instance}) ---\n{r.body}")
        return "\n\n".join(lines)

    return Tool(
        name="hold_debate",
        description=(
            "Hold an adversarial debate by fanning the question out to every "
            "*.adversarial.finance.* agent and collecting their cases."
        ),
        handler=hold_debate,
    )


# ── builders ───────────────────────────────────────────────────────────


def _bull_builder_with_model(model):
    def build(address: AgentAddress, engine: ProtocolEngine, profile: AgentProfile):
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


def _data_builder(address, engine, profile):
    return _DataAgent(
        address, engine,
        metadata=AgentMeta(
            capabilities=["data-feed"],
            description="Fundamentals lookup service.",
            reputation=0.9,
        ),
        heartbeat_interval=0,
    )


def _researcher_builder_with_model(model, *, thread: str):
    """Researcher is a deepagents.create_deep_agent with AHP-aware tools."""

    def build(address: AgentAddress, engine: ProtocolEngine, profile: AgentProfile):
        # Layer a researcher-specific instruction on top of the profile prompt
        # the capability registry already contributed.
        researcher_profile = AgentProfile(
            address=profile.address,
            tools=profile.tools,
            skills=profile.skills,
            rag_sources=profile.rag_sources,
            prompt=(
                (profile.prompt + "\n\n" if profile.prompt else "")
                + "You are an equities research coordinator. For each user "
                "question: (1) call `lookup_fundamentals` to get the data, "
                "(2) call `hold_debate` to gather bull and bear views, "
                "(3) compose a single concise brief that includes the "
                "fundamentals, bull view, and bear view."
            ),
            agent_kind="deep",
        )
        return DeepAgent.from_profile(
            address, engine, researcher_profile, model=model,
            extra_tools=[
                _make_lookup_fundamentals_tool(engine, address, thread),
                _make_debate_tool(engine, address, thread),
            ],
            metadata=AgentMeta(
                capabilities=["synthesis", "multi-agent-orchestration"],
                description="Deep-research orchestrator (deepagents + Bedrock).",
                reputation=0.85,
            ),
            heartbeat_interval=0,
        )

    return build


# ── run loop ───────────────────────────────────────────────────────────


async def run(
    *,
    redis_client=None,
    model=None,
    on_human_message: Callable[[str], Awaitable[None]] | None = None,
    question: str = "Tesla",
    thread: str = "thread::devin::main",
) -> DemoResult:
    """Drive the LLM-backed demo. Requires usable AWS credentials.

    ``model`` may be passed in for testing (a fake chat model that
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
    engine = ProtocolEngine(bus, registry, cache, default_timeout=120.0)

    factory = AgentFactory(engine, capabilities=_build_capabilities())
    factory.register(
        "*.adversarial.finance.*.*.*.bull",
        _bull_builder_with_model(model), priority=10,
    )
    factory.register(
        "*.adversarial.finance.*.*.*.bear",
        _bear_builder_with_model(model), priority=10,
    )
    factory.register("*.interview.finance.*.*.*.*", _data_builder, priority=10)
    factory.register(
        "*.collaborative.finance.*.*.*.*",
        _researcher_builder_with_model(model, thread=thread),
        priority=10,
    )

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
            thread=thread, body=question,
        )
        t0 = time.perf_counter()
        result.first_reply = await engine.handle(cold_req, timeout=180.0)
        result.first_seconds = time.perf_counter() - t0
        if result.first_reply is not None:
            await show(result.first_reply.body)

        await show("\n=== Devin asks the same question (warm) ===")
        warm_req = Message(
            source=AgentAddress.parse(HUMAN_URI),
            target=AgentAddress.parse(RESEARCHER_URI),
            verb="SEND-GET", code=Code.HUMAN_QUERY,
            thread=thread, body=question,
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
