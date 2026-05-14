"""Node A — hosts the adversarial agents (Bull + Bear).

This process registers two AHP agents at canonical, universal
addresses. Anyone else on the same Redis (Node B, a CLI, another
service) can reach them by address — no service discovery beyond
"point at the same Redis URL".

Run::

    AHP_REDIS_URL=redis://localhost:6379/0 \\
        uvicorn node_a:app --port 8001 --reload
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

# Re-use the generic build_app from the sibling fastapi_serve example.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fastapi_serve"))
from server import build_app    # noqa: E402

from ahp.adapters.base import AHPAgent
from ahp.core import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.registry import Principal
from shared import BEAR_URI, BULL_URI, build_stack    # noqa: E402


# Node A's identity on the network. The auth policy only lets us
# register agents whose address matches one of these claim patterns.
# In a real deployment these claims would arrive as a signed token.
NODE_A_PRINCIPAL = Principal.with_claims(
    "node-a",
    "tifin.adversarial.finance.*.*.*.*",   # bull + bear live here
)


class _BullAgent(AHPAgent):
    """Stub bull — a real deployment would use ReactAgent + Bedrock."""

    async def handle_message(self, message: Message) -> Message | None:
        if not message.expects_response:
            return None
        return Message(
            source=self.address, target=message.source, verb="SEND",
            code=message.code, thread=message.thread,
            body=(
                f"BULL view on {message.body}: durable moat, expanding TAM, "
                f"superior execution. Target +35% over 12 months."
            ),
        )


class _BearAgent(AHPAgent):
    async def handle_message(self, message: Message) -> Message | None:
        if not message.expects_response:
            return None
        return Message(
            source=self.address, target=message.source, verb="SEND",
            code=message.code, thread=message.thread,
            body=(
                f"BEAR view on {message.body}: regulatory headwinds, margin "
                f"compression, valuation extended. Target -20% over 12 months."
            ),
        )


client, bus, registry, cache, engine, factory = build_stack(
    principal=NODE_A_PRINCIPAL,
)
bull = _BullAgent(AgentAddress.parse(BULL_URI), engine, heartbeat_interval=0)
bear = _BearAgent(AgentAddress.parse(BEAR_URI), engine, heartbeat_interval=0)


# The build_app helper provides /query, /agents, /threads, /tools,
# /resources, /observe. We add a tiny /health for orchestration.
app = build_app(
    factory,
    agents=[bull, bear],
    title="AHP Federation — Node A (adversarial)",
    description="Hosts Bull and Bear at universal addresses.",
)


@app.get("/health")
async def health() -> dict:
    n_alive = await registry.count(alive_only=True)
    return {
        "node": "A",
        "redis_url": str(client.connection_pool.connection_kwargs.get("host", "?")),
        "alive_agents": n_alive,
    }
