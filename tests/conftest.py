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
