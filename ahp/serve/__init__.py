"""ahp.serve — host the protocol behind an HTTP / WebSocket app.

Provides :func:`build_app`, a generic FastAPI factory that takes a
fully-wired :class:`AgentFactory` and returns an application exposing:

* ``POST /query``           — send a HUMAN_QUERY SEND-GET and return the reply
* ``POST /send``            — send an arbitrary AHP message and return the reply
* ``GET /agents``           — list registered agents
* ``GET /threads/{id}``     — read thread history (optional tier filter)
* ``GET /tools``            — list registered tools (by ToolAddress)
* ``GET /resources``        — list registered resources (by ResourceAddress)
* ``WS  /observe``          — CAST-SUB stream over the bus's tap channel

The app's lifespan registers + starts the supplied agents on boot and
stops + deregisters them on shutdown — so a FastAPI process IS the AHP
runtime, and the factory inside it can self-call the protocol from
HTTP handlers.
"""

from ahp.serve.fastapi_app import build_app

__all__ = ["build_app"]
