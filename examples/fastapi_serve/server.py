"""FastAPI front-end for an :class:`AgentFactory`.

A FastAPI process IS the AHP runtime: at startup it registers/starts
every agent the caller supplied, and HTTP handlers route into the
protocol via the factory's engine. The factory is "self-callable" in
the sense that any HTTP handler can send AHP messages and observe the
same tap that intra-process agents see.

The exposed surface is small but covers the common cases:

============================  ============================================
``POST /query``               human query → researcher (SEND-GET)
``POST /send``                arbitrary message → engine.handle()
``GET  /agents``              list alive registry entries
``GET  /threads/{id}``        read a thread's history (tier_filter=?)
``GET  /tools``               list registered tool addresses
``GET  /resources``           list registered resource addresses
``WS   /observe``             CAST-SUB stream of filtered tap traffic
============================  ============================================
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from ahp.adapters.base import AHPAgent
from ahp.adapters.factory import AgentFactory
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern


log = logging.getLogger(__name__)


# ── request / response models ──────────────────────────────────────────


class QueryRequest(BaseModel):
    target: str = Field(..., description="Concrete agent address URI to query.")
    body: Any = Field(..., description="Question body (string or JSON).")
    code: str = Field(Code.HUMAN_QUERY, description="AHP interaction code.")
    thread: str = Field(
        "thread::http::query",
        description="Thread to use for this exchange.",
    )
    timeout: float = Field(60.0, gt=0, description="Seconds to wait for a reply.")


class QueryResponse(BaseModel):
    target: str
    code: str
    thread: str
    body: Any
    cached: bool = False


class SendRequest(BaseModel):
    source: str = Field(..., description="Sender agent address URI.")
    target: str = Field(..., description="Target address URI or pattern.")
    target_kind: str = Field(
        "address",
        description="'address' for point-to-point, 'pattern' for broadcast.",
    )
    verb: str
    code: str
    thread: str = "thread::http::send"
    body: Any = None
    timeout: float = Field(60.0, gt=0)
    max_responses: int | None = None


class AgentSummary(BaseModel):
    address: str
    alive: bool
    capabilities: list[str]
    reputation: float
    description: str | None


class ToolSummary(BaseModel):
    address: str
    name: str
    description: str
    tags: list[str]


class ResourceSummary(BaseModel):
    address: str
    name: str
    description: str


# ── app factory ────────────────────────────────────────────────────────


def build_app(
    factory: AgentFactory,
    *,
    agents: list[AHPAgent] | None = None,
    title: str = "AHP Service",
    description: str | None = None,
    version: str = "0.1.0",
) -> FastAPI:
    """Construct a FastAPI app wired to the supplied :class:`AgentFactory`.

    ``agents`` are managed by the app's lifespan: registered + started
    on boot, stopped + deregistered on shutdown. Pass ``None`` if you
    manage agent lifecycles externally.
    """
    agents_list: list[AHPAgent] = list(agents or [])

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        for agent in agents_list:
            await agent.register()
            await agent.start()
        # Small grace period so pub/sub subscriptions land before requests.
        await asyncio.sleep(0.05)
        try:
            yield
        finally:
            for agent in agents_list:
                with contextlib.suppress(Exception):
                    await agent.stop()
            for agent in agents_list:
                with contextlib.suppress(Exception):
                    await agent.deregister()
            with contextlib.suppress(Exception):
                await factory.resources.close_all()
            with contextlib.suppress(Exception):
                await factory.engine.bus.close()

    app = FastAPI(
        title=title,
        description=description or "Agentic Handshake Protocol HTTP face.",
        version=version,
        lifespan=lifespan,
    )

    engine = factory.engine

    # ── POST /query ────────────────────────────────────────────────────

    @app.post("/query", response_model=QueryResponse)
    async def post_query(req: QueryRequest) -> QueryResponse:
        try:
            target = AgentAddress.parse(req.target)
        except ValueError as exc:
            raise HTTPException(400, f"invalid target: {exc}")
        msg = Message(
            source=AgentAddress.parse(
                # HTTP-origin queries get a synthetic public-human address.
                # Replace ``http-client`` with a per-session identity for
                # real auth setups.
                "public.human.general.http.s.session.http-client",
            ),
            target=target,
            verb="SEND-GET",
            code=req.code,
            thread=req.thread,
            body=req.body,
        )
        reply = await engine.handle(msg, timeout=req.timeout)
        if reply is None:
            raise HTTPException(504, "timed out waiting for reply")
        return QueryResponse(
            target=str(target),
            code=reply.code,
            thread=reply.thread,
            body=reply.body,
            cached=False,  # engine doesn't surface cache-hit info today
        )

    # ── POST /send ─────────────────────────────────────────────────────

    @app.post("/send")
    async def post_send(req: SendRequest) -> Any:
        try:
            source = AgentAddress.parse(req.source)
            if req.target_kind == "pattern":
                target: Any = AddressPattern.parse(req.target)
            else:
                target = AgentAddress.parse(req.target)
        except ValueError as exc:
            raise HTTPException(400, f"invalid address: {exc}")

        msg = Message(
            source=source, target=target, verb=req.verb,
            code=req.code, thread=req.thread, body=req.body,
        )
        kwargs: dict[str, Any] = {}
        if req.verb in {"SEND-GET", "CAST-GET"}:
            kwargs["timeout"] = req.timeout
        if req.verb == "CAST-GET":
            kwargs["max_responses"] = req.max_responses
        result = await engine.handle(msg, **kwargs)

        # Serialize whichever shape came back.
        if isinstance(result, Message):
            return result.to_dict()
        if isinstance(result, list):
            return [m.to_dict() for m in result]
        return {"result": result}

    # ── GET /agents ────────────────────────────────────────────────────

    @app.get("/agents", response_model=list[AgentSummary])
    async def get_agents(alive_only: bool = Query(False)) -> list[AgentSummary]:
        registry = engine.registry
        pairs = await registry.discover(alive_only=alive_only)
        out: list[AgentSummary] = []
        for addr, meta in pairs:
            out.append(AgentSummary(
                address=str(addr),
                alive=await registry.is_alive(addr),
                capabilities=list(meta.capabilities),
                reputation=meta.reputation,
                description=meta.description,
            ))
        return out

    # ── GET /threads/{thread_id} ───────────────────────────────────────

    @app.get("/threads/{thread_id}")
    async def get_thread(
        thread_id: str,
        tier_filter: str | None = Query(None, description="Optional accept-tier set (e.g. 's' or 'sj')."),
        count: int | None = Query(None, ge=1, le=1000),
    ) -> list[dict]:
        history = await engine.threads.get_history(
            thread_id, tier_filter=tier_filter, count=count,
        )
        return [m.to_dict() for m in history]

    # ── GET /tools ─────────────────────────────────────────────────────

    @app.get("/tools", response_model=list[ToolSummary])
    async def get_tools() -> list[ToolSummary]:
        return [
            ToolSummary(
                address=str(b.address),
                name=b.tool.name,
                description=b.tool.description,
                tags=sorted(b.tags),
            )
            for b in factory.tools.bindings()
        ]

    # ── GET /resources ─────────────────────────────────────────────────

    @app.get("/resources", response_model=list[ResourceSummary])
    async def get_resources() -> list[ResourceSummary]:
        return [
            ResourceSummary(
                address=str(b.address),
                name=b.address.name,
                description=b.description,
            )
            for b in factory.resources.bindings()
        ]

    # ── WS /observe ────────────────────────────────────────────────────

    @app.websocket("/observe")
    async def observe(
        ws: WebSocket,
        pattern: str = Query(
            "*.*.*.*.*.*.*",
            description="AddressPattern to match against message targets/sources.",
        ),
        code: str = Query("*", description="Code glob (e.g. 'interview.*')."),
    ) -> None:
        try:
            target_pattern = AddressPattern.parse(pattern)
        except ValueError:
            await ws.close(code=1003, reason="invalid pattern")
            return

        await ws.accept()

        observe_msg = Message(
            source=AgentAddress.parse(
                "public.human.general.http.s.session.http-observer",
            ),
            target=target_pattern,
            verb="CAST-SUB",
            code=code,
            thread="thread::http::observe",
            body=None,
        )
        sub = await engine.handle(observe_msg)

        async def pump() -> None:
            try:
                async for msg in sub.messages(idle_timeout=0.5):
                    await ws.send_json(msg.to_dict())
            except Exception:
                log.exception("observe pump failed")

        pump_task = asyncio.create_task(pump())
        try:
            # We don't expect anything from the client; just keep the
            # connection open until they hang up.
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            pump_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await pump_task
            await sub.close()

    return app
