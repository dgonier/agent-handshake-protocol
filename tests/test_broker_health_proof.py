"""Provider health-proof + unplanned-outage reputation tests.

Two rules under test:

1. A provider only appears on the menu once it has proven health
   (default registration counts as the first heartbeat; explicit
   ``prove_alive=False`` requires a separate heartbeat call).
2. A provider that disappears without a graceful deregister registers
   as an outage on the next ``check_compute_outages()`` sweep, which
   folds a ``timeout`` outcome into that provider's reputation record.
"""

from __future__ import annotations

import asyncio

import pytest

from ahp.broker import Broker
from ahp.broker.compute_registry import (
    PROVIDER_ALIVE_KEY,
    PROVIDER_LAST_SEEN_KEY,
    ComputeProviderRegistry,
)
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import REP_PENALTY_FAILURE


def _leaf(provider_id: str = "p-vllm") -> MenuLeaf:
    return MenuLeaf(
        provider_id=provider_id, tier="small", model="qwen2-5-7b",
        rate_per_1k_chars=0.0001,
    )


async def test_default_registration_proves_alive(redis_client):
    """A self-hosted provider that registers normally is alive
    immediately — its leaves are visible to the default
    ``list_leaves`` (which filters by liveness)."""
    broker = Broker(redis_client)
    await broker.register_compute_provider(
        ComputeProvider(provider_id="p-self-hosted"),
    )
    await broker.register_leaf(_leaf("p-self-hosted"))
    leaves = await broker.compute.list_leaves(only_alive_providers=True)
    assert any(l.provider_id == "p-self-hosted" for l in leaves)


async def test_prove_alive_false_keeps_leaves_hidden_until_heartbeat(redis_client):
    """Metadata-only registration (``prove_alive=False``) does NOT
    publish leaves to the live menu. They appear only after an
    explicit heartbeat — that's the "prove health" rule."""
    broker = Broker(redis_client)
    await broker.register_compute_provider(
        ComputeProvider(provider_id="p-remote"),
        prove_alive=False,
    )
    await broker.register_leaf(_leaf("p-remote"))

    hidden = await broker.compute.list_leaves(only_alive_providers=True)
    assert not any(l.provider_id == "p-remote" for l in hidden)

    # But it IS visible with the no-liveness filter — the metadata is
    # there, just not yet routable.
    raw = await broker.compute.list_leaves(only_alive_providers=False)
    assert any(l.provider_id == "p-remote" for l in raw)

    # Now prove health.
    assert await broker.heartbeat_compute_provider("p-remote") is True
    visible = await broker.compute.list_leaves(only_alive_providers=True)
    assert any(l.provider_id == "p-remote" for l in visible)


async def test_graceful_deregister_clears_last_seen(redis_client):
    """A clean deregister wipes the last-seen sentinel so the outage
    detector doesn't credit a phantom outage afterwards."""
    broker = Broker(redis_client)
    await broker.register_compute_provider(ComputeProvider(provider_id="p"))
    # Sentinel is set.
    assert await redis_client.exists(
        PROVIDER_LAST_SEEN_KEY.format(provider_id="p"),
    )
    await broker.deregister_compute_provider("p", graceful=True)
    # And cleared.
    assert not await redis_client.exists(
        PROVIDER_LAST_SEEN_KEY.format(provider_id="p"),
    )
    # No outage to detect.
    hits = await broker.check_compute_outages()
    assert hits == []


async def test_unplanned_outage_credits_reputation_hit(redis_client):
    """Heartbeat TTL expires without a graceful deregister →
    ``check_compute_outages()`` returns the provider id and the
    reputation entry has been moved down by ``REP_PENALTY_FAILURE``.
    """
    # Short TTL so the test doesn't sleep long.
    short_registry = ComputeProviderRegistry(redis_client, heartbeat_ttl=1)
    broker = Broker(redis_client)
    # Swap in the short-TTL registry so we don't wait 30s.
    broker.compute = short_registry

    await broker.register_compute_provider(ComputeProvider(provider_id="p-flaky"))
    # Confirm initial reputation is the default.
    initial = await broker.get_reputation("p-flaky")
    base_rep = initial.reputation if initial else 0.5

    # Wait past the TTL — the alive key expires, the sentinel stays.
    await asyncio.sleep(1.2)

    # Sanity: alive expired, sentinel survives.
    assert not await redis_client.exists(
        PROVIDER_ALIVE_KEY.format(provider_id="p-flaky"),
    )
    assert await redis_client.exists(
        PROVIDER_LAST_SEEN_KEY.format(provider_id="p-flaky"),
    )

    hits = await broker.check_compute_outages()
    assert "p-flaky" in hits

    updated = await broker.get_reputation("p-flaky")
    assert updated is not None
    # Reputation took a failure hit, not just a noop.
    assert updated.reputation < base_rep
    # Specifically the failure penalty (allowing tiny float slop).
    assert abs((base_rep - updated.reputation) - REP_PENALTY_FAILURE) < 1e-9
    assert updated.failed == 1

    # Idempotent: a second sweep finds nothing because the sentinel
    # was cleared by the first detection.
    again = await broker.check_compute_outages()
    assert again == []
