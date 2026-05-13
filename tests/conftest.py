"""Shared pytest fixtures for async Redis-backed tests."""

from __future__ import annotations

from typing import AsyncIterator

import pytest_asyncio


@pytest_asyncio.fixture
async def redis_client() -> AsyncIterator:
    """Fresh in-memory fakeredis client per test, decoded responses."""
    import fakeredis.aioredis  # type: ignore[import-not-found]

    client = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest_asyncio.fixture
async def stack(redis_client):
    """Bus + registry + cache + thread manager + engine, all sharing one Redis."""
    from ahp.core.compatibility import CompatibilityMatrix
    from ahp.engine.router import ProtocolEngine
    from ahp.engine.thread_manager import ThreadManager
    from ahp.registry.registry import AgentRegistry
    from ahp.transport.cache import ProtocolCache
    from ahp.transport.redis_bus import RedisBus

    bus = RedisBus(redis_client)
    registry = AgentRegistry(redis_client, heartbeat_ttl=30)
    cache = ProtocolCache(redis_client)
    matrix = CompatibilityMatrix()
    threads = ThreadManager(redis_client, bus)
    engine = ProtocolEngine(bus, registry, cache, matrix, threads, default_timeout=2.0)

    class Stack:
        pass

    s = Stack()
    s.redis = redis_client
    s.bus = bus
    s.registry = registry
    s.cache = cache
    s.matrix = matrix
    s.threads = threads
    s.engine = engine

    try:
        yield s
    finally:
        await bus.close()
