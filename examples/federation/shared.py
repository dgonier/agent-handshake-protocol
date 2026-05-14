"""Shared setup for the two federation nodes.

Both ``node_a.py`` and ``node_b.py`` import this module. They build an
identical ``AHP`` stack pointed at the same Redis URL — that's the
*entire* federation contract. Nothing in this file is special; it's
the standard library API.
"""

from __future__ import annotations

import os

import redis.asyncio as aioredis

from ahp.adapters import AgentFactory, GroupRegistry, HumanAgent
from ahp.core import AgentAddress
from ahp.engine import ProtocolEngine
from ahp.registry import (
    AddressClaimPolicy,
    AgentRegistry,
    Principal,
)
from ahp.transport import ProtocolCache, RedisBus


REDIS_URL = os.environ.get("AHP_REDIS_URL", "redis://localhost:6379/0")


# ── canonical addresses for the demo roster ────────────────────────────

ALICE_URI      = "tifin.collaborative.finance.equities.s.session.alice"
RESEARCHER_URI = "tifin.collaborative.finance.equities.s.session.researcher"
BULL_URI       = "tifin.adversarial.finance.equities.s.session.bull"
BEAR_URI       = "tifin.adversarial.finance.equities.s.session.bear"
HUMAN_URI      = "public.human.general.http.s.session.devin"


def build_stack(
    redis_url: str = REDIS_URL,
    *,
    principal: Principal | None = None,
):
    """Construct (client, bus, registry, cache, engine, factory) for one node.

    Both nodes call this with the same ``redis_url`` — they then share
    the registry, bus, cache, threads, and tap channel.

    When ``principal`` is supplied, an :class:`AddressClaimPolicy` is
    attached to the registry: this node can only register at addresses
    its principal's claims cover. Leave ``principal=None`` for the
    open default (any process can register anywhere — fine for
    single-tenant demos, NOT for multi-tenant production).
    """
    client = aioredis.from_url(redis_url, decode_responses=True)
    bus = RedisBus(client)
    registry = AgentRegistry(
        client,
        heartbeat_ttl=60,
        principal=principal,
        policy=AddressClaimPolicy() if principal is not None else None,
    )
    cache = ProtocolCache(client)
    engine = ProtocolEngine(bus, registry, cache, default_timeout=30.0)

    groups = GroupRegistry()
    groups.register("debaters",      "*.adversarial.finance.*.s.*.*")
    groups.register("research-team", "*.collaborative.finance.*.s.*.*")

    factory = AgentFactory(engine, groups=groups)
    return client, bus, registry, cache, engine, factory
