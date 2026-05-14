"""Tests for the FastAPI face — POST /query, /send, GET /agents, etc.

Uses httpx.ASGITransport against the in-process app (no uvicorn). Each
test builds its own AgentFactory + small agent roster so the FastAPI
endpoints have something real to talk to via the engine.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from httpx import ASGITransport, AsyncClient

from ahp.adapters.base import AHPAgent
from ahp.adapters.capability import Tool
from ahp.adapters.factory import AgentFactory
from ahp.adapters.resources import ResourceRegistry
from ahp.adapters.tool_registry import ToolRegistry
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.serve import build_app


# ── helpers ────────────────────────────────────────────────────────────


class _EchoAgent(AHPAgent):
    """Replies with ``echo:<body>``."""

    async def handle_message(self, message: Message):
        if not message.expects_response:
            return None
        return Message(
            source=self.address, target=message.source, verb="SEND",
            code=message.code, thread=message.thread,
            body=f"echo:{message.body}",
        )


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


@pytest.fixture
async def app_stack(stack):
    """Returns (app, agents, factory) wired and ready for an AsyncClient.

    The fixture itself manages agent lifecycle so the tests don't have
    to depend on FastAPI's lifespan firing under httpx.ASGITransport
    (it doesn't, by default). build_app() is called with agents=[]
    here — production users with uvicorn / hypercorn get the lifespan
    path automatically.
    """
    factory = AgentFactory(stack.engine)

    echo_addr = _addr("demo.collaborative.finance.equities.s.session.echo")
    echo = _EchoAgent(echo_addr, stack.engine, heartbeat_interval=0)

    factory.tools.register(
        lambda x: x, "demo", "tool", "*", "compute",
        operation="square_root", description="dummy square root",
    )
    factory.resources.register(
        lambda: {"client": "fake"},
        "demo", "kv", "finance", "equities", name="quotes-cache",
        description="fake quotes cache",
    )

    await echo.register()
    await echo.start()
    await asyncio.sleep(0.05)

    app = build_app(factory, agents=[], title="test")
    try:
        yield app, [echo], factory
    finally:
        await echo.stop()
        await echo.deregister()


# ── POST /query round-trip ─────────────────────────────────────────────


async def test_post_query_returns_echo(app_stack):
    app, _, _ = app_stack
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/query", json={
            "target": "demo.collaborative.finance.equities.s.session.echo",
            "body": "hello",
            "code": Code.INTERVIEW_TEXT,
            "thread": "thread::http",
            "timeout": 3.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["body"] == "echo:hello"
        assert data["target"] == "demo.collaborative.finance.equities.s.session.echo"


async def test_post_query_404ish_returns_504_when_no_responder(app_stack):
    app, _, _ = app_stack
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/query", json={
            "target": "demo.collaborative.finance.equities.s.session.nobody",
            "body": "hi", "thread": "thread::nope", "timeout": 0.2,
        })
        assert resp.status_code == 504


async def test_post_query_400_on_bad_target(app_stack):
    app, _, _ = app_stack
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/query", json={
            "target": "not-a-real-address",
            "body": "x", "thread": "t", "timeout": 0.5,
        })
        assert resp.status_code == 400


# ── GET /agents ────────────────────────────────────────────────────────


async def test_get_agents_lists_alive(app_stack):
    app, _, _ = app_stack
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/agents?alive_only=true")
        assert resp.status_code == 200
        agents = resp.json()
        uris = [a["address"] for a in agents]
        assert "demo.collaborative.finance.equities.s.session.echo" in uris


# ── GET /threads/{id} ─────────────────────────────────────────────────


async def test_get_thread_history(app_stack):
    app, _, _ = app_stack
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        # Drive a query first so the thread has entries.
        await client.post("/query", json={
            "target": "demo.collaborative.finance.equities.s.session.echo",
            "body": "history-test",
            "thread": "thread::for-history",
            "timeout": 3.0,
        })
        resp = await client.get("/threads/thread::for-history")
        assert resp.status_code == 200
        entries = resp.json()
        bodies = [e["body"] for e in entries]
        assert "history-test" in bodies
        assert any(b == "echo:history-test" for b in bodies)


# ── GET /tools ────────────────────────────────────────────────────────


async def test_get_tools(app_stack):
    app, _, _ = app_stack
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/tools")
        assert resp.status_code == 200
        tools = resp.json()
        assert {"address": "demo.tool.*.compute.square_root",
                "name": "square_root",
                "description": "dummy square root",
                "tags": []} in tools


# ── GET /resources ────────────────────────────────────────────────────


async def test_get_resources(app_stack):
    app, _, _ = app_stack
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.get("/resources")
        assert resp.status_code == 200
        resources = resp.json()
        assert {"address": "demo.kv.finance.equities.quotes-cache",
                "name": "quotes-cache",
                "description": "fake quotes cache"} in resources


# ── POST /send (broadcast) ────────────────────────────────────────────


async def test_post_send_cast_get_returns_list(app_stack):
    app, _, _ = app_stack
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test",
    ) as client:
        resp = await client.post("/send", json={
            "source": "demo.collaborative.finance.equities.s.session.alice",
            "target": "*.collaborative.finance.*.s.*.*",
            "target_kind": "pattern",
            "verb": "CAST-GET",
            "code": Code.INTERVIEW_TEXT,
            "thread": "thread::http::cast",
            "body": "all of you",
            "timeout": 3.0,
        })
        assert resp.status_code == 200
        replies = resp.json()
        assert isinstance(replies, list)
        # The echo agent matches *.collaborative.finance.*.s.*.* and should reply.
        bodies = [r["body"] for r in replies]
        assert "echo:all of you" in bodies


# ── WS /observe ────────────────────────────────────────────────────────
#
# The WebSocket endpoint is a thin pass-through to engine.handle("CAST-SUB"),
# which is exhaustively tested in test_engine.py. We can't reliably test
# the FastAPI mount in-process because starlette's TestClient runs its own
# event loop in a worker thread, and each FakeRedis client only sees
# messages published on its own loop — so the tap subscription set up by
# the WS handler doesn't observe traffic the test's async loop publishes.
# Real-Redis deployments don't have this problem; verify manually with:
#
#   uvicorn ahp.demo.serve:app --reload
#   websocat 'ws://localhost:8000/observe?pattern=*.*.*.*.*.*.*&code=*'


async def test_ws_observe_endpoint_mounts(app_stack):
    """Smoke: the /observe path is registered on the app."""
    app, _, _ = app_stack
    paths = [r.path for r in app.routes if hasattr(r, "path")]
    assert "/observe" in paths
