"""Runnable FastAPI process: the AHP finance demo behind HTTP.

This is a *consumer* of the ``ahp`` package — FastAPI itself is NOT a
dependency of the library. Treat this directory as a starting point
to copy into your own service.

Run from this directory::

    pip install -r requirements.txt
    uvicorn app:app --reload

Endpoints (see ``server.py``):

* ``POST /query``         — ``{ "target": "...researcher", "body": "Tesla" }``
* ``POST /send``          — arbitrary AHP message
* ``GET  /agents``        — list registered agents
* ``GET  /threads/{id}``  — thread history
* ``GET  /tools``         — registered tool addresses
* ``GET  /resources``     — registered resource addresses
* ``WS   /observe``       — live tap stream, e.g.
  ``ws://host/observe?pattern=*.adversarial.*.*.*.*.*&code=adversarial.*``

By default this boots the stubbed (no-LLM) finance stack — same
agents as ``ahp.demo.finance_analysis``. To use the Bedrock-backed
variant, set ``AHP_DEMO_VARIANT=react`` before starting uvicorn.
"""

from __future__ import annotations

import os

import fakeredis.aioredis

from ahp.adapters import AgentFactory, HumanAgent
from ahp.core import AgentAddress
from ahp.demo.finance_analysis import (
    BEAR_URI,
    BULL_URI,
    DATA_URI,
    HUMAN_URI,
    RESEARCHER_URI,
    _bear_builder,
    _build_capabilities,
    _bull_builder,
    _data_builder,
    _researcher_builder,
)
from ahp.engine import ProtocolEngine
from ahp.registry import AgentRegistry
from ahp.transport import ProtocolCache, RedisBus
from server import build_app    # local module — sits next to this file


def _build_stub_app():
    """Stub variant — deterministic, no LLM, no AWS credentials needed."""
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    bus = RedisBus(redis)
    registry = AgentRegistry(redis, heartbeat_ttl=120)
    cache = ProtocolCache(redis)
    engine = ProtocolEngine(bus, registry, cache, default_timeout=10.0)

    factory = AgentFactory(engine, capabilities=_build_capabilities())
    factory.register("*.adversarial.finance.*.*.*.bull", _bull_builder, priority=10)
    factory.register("*.adversarial.finance.*.*.*.bear", _bear_builder, priority=10)
    factory.register("*.interview.finance.*.*.*.*", _data_builder, priority=10)
    factory.register(
        "*.collaborative.finance.*.*.*.*", _researcher_builder, priority=10,
    )

    bull = factory.create(BULL_URI)
    bear = factory.create(BEAR_URI)
    data = factory.create(DATA_URI)
    researcher = factory.create(RESEARCHER_URI)
    human = HumanAgent(
        AgentAddress.parse(HUMAN_URI), engine,
        on_message=None, observation_level="L2",
        heartbeat_interval=0,
    )

    return build_app(
        factory,
        agents=[bull, bear, data, researcher, human],
        title="AHP Finance Demo (stub)",
        description=(
            "Deterministic adversarial-finance pipeline behind HTTP. "
            "Set AHP_DEMO_VARIANT=react to use the Bedrock-driven agents."
        ),
        version="0.1.0",
    )


def _build_react_app():
    """LLM-driven variant — requires AWS credentials via the boto3 chain."""
    from ahp.demo.finance_react import (
        _bear_builder_with_model,
        _bull_builder_with_model,
        _data_builder as _react_data_builder,
        _researcher_builder_with_model,
    )
    from ahp.llm.bedrock import bedrock_chat_model

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    bus = RedisBus(redis)
    registry = AgentRegistry(redis, heartbeat_ttl=300)
    cache = ProtocolCache(redis)
    engine = ProtocolEngine(bus, registry, cache, default_timeout=120.0)

    model = bedrock_chat_model()

    factory = AgentFactory(engine, capabilities=_build_capabilities())
    thread = "thread::http::main"
    factory.register(
        "*.adversarial.finance.*.*.*.bull",
        _bull_builder_with_model(model), priority=10,
    )
    factory.register(
        "*.adversarial.finance.*.*.*.bear",
        _bear_builder_with_model(model), priority=10,
    )
    factory.register("*.interview.finance.*.*.*.*", _react_data_builder, priority=10)
    factory.register(
        "*.collaborative.finance.*.*.*.*",
        _researcher_builder_with_model(model, thread=thread),
        priority=10,
    )

    bull = factory.create(BULL_URI)
    bear = factory.create(BEAR_URI)
    data = factory.create(DATA_URI)
    researcher = factory.create(RESEARCHER_URI)
    human = HumanAgent(
        AgentAddress.parse(HUMAN_URI), engine,
        on_message=None, observation_level="L2",
        heartbeat_interval=0,
    )

    return build_app(
        factory,
        agents=[bull, bear, data, researcher, human],
        title="AHP Finance Demo (Bedrock)",
        description=(
            "LLM-driven adversarial-finance pipeline behind HTTP. "
            "Researcher is a DeepAgent with AHP-aware tools that call "
            "Bull and Bear via the protocol."
        ),
        version="0.1.0",
    )


variant = os.environ.get("AHP_DEMO_VARIANT", "stub").lower()
app = _build_react_app() if variant == "react" else _build_stub_app()
