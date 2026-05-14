"""Node B — hosts the Researcher + the human-facing HTTP entry point.

This process never directly imports Node A. It addresses A's agents by
URI alone:

    target=AddressPattern.parse("*.adversarial.finance.*.s.*.*")

which the registry on the shared Redis resolves to ``BULL_URI`` and
``BEAR_URI`` (hosted on Node A). The message hops over Redis pub/sub
and Bull/Bear's replies come back the same way.

Run (alongside Node A)::

    AHP_REDIS_URL=redis://localhost:6379/0 \\
        uvicorn node_b:app --port 8002 --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fastapi_serve"))
from server import build_app    # noqa: E402

from ahp.adapters.base import AHPAgent
from ahp.core import AgentAddress, AddressPattern
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.registry import Principal
from shared import (    # noqa: E402
    ALICE_URI, BEAR_URI, BULL_URI, HUMAN_URI, RESEARCHER_URI,
    build_stack,
)


# Node B's identity. Claims cover collaborative + human addresses
# but NOT the adversarial ones — so if node_b tried to register
# bull or bear (which legitimately belong to node_a), the registry
# would raise UnauthorizedRegistrationError.
NODE_B_PRINCIPAL = Principal.with_claims(
    "node-b",
    "tifin.collaborative.finance.*.*.*.*",   # researcher
    "public.human.*.*.*.*.*",                # HTTP-origin humans
)


class _Researcher(AHPAgent):
    """Coordinator — calls Bull and Bear (on Node A) over the protocol."""

    async def handle_message(self, message: Message) -> Message | None:
        if not message.expects_response:
            return None
        question = str(message.body)

        # Broadcast to whichever adversarial agents are alive. They live
        # on Node A; this code doesn't know that.
        debate_req = Message(
            source=self.address,
            target=AddressPattern.parse("*.adversarial.finance.*.s.*.*"),
            verb="CAST-GET", code=Code.ADVERSARIAL_DEBATE,
            thread=message.thread, body=question,
        )
        debate_replies = await self.engine.handle(debate_req, timeout=30.0)

        bull_view = next(
            (r.body for r in debate_replies if "bull" in r.source.instance),
            "(no bull response)",
        )
        bear_view = next(
            (r.body for r in debate_replies if "bear" in r.source.instance),
            "(no bear response)",
        )

        body = (
            f"=== Analysis for {question} ===\n\n"
            f"Bull view:\n  {bull_view}\n\n"
            f"Bear view:\n  {bear_view}\n"
        )
        return Message(
            source=self.address, target=message.source, verb="SEND",
            code=message.code, thread=message.thread, body=body,
        )


client, bus, registry, cache, engine, factory = build_stack(
    principal=NODE_B_PRINCIPAL,
)
researcher = _Researcher(
    AgentAddress.parse(RESEARCHER_URI), engine, heartbeat_interval=0,
)


app = build_app(
    factory,
    agents=[researcher],
    title="AHP Federation — Node B (researcher)",
    description=(
        "Researcher that fans out to *.adversarial.finance.* via the shared "
        "Redis. The targets live on Node A; this process never imports them."
    ),
)


@app.get("/health")
async def health() -> dict:
    pat = AddressPattern.parse("*.adversarial.finance.*.s.*.*")
    visible = await registry.resolve(pat, alive_only=True)
    return {
        "node": "B",
        "adversarial_agents_visible": [str(a) for a in visible],
    }
