"""Tests for the runner's multi-compute (secondary-server) registration.

These don't run a full session — that would require a live Bedrock
endpoint. They exercise the small registration helpers that decide
*what* the broker knows about when a multi-compute demo is on.
"""

from __future__ import annotations

import asyncio

import pytest

from ahp.broker import Broker
from examples.viewer.runner import (
    _maybe_register_modal_vllm,
    _maybe_register_secondary_server,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Ensure each test starts from a known env state."""
    for k in (
        "AHP_MODAL_VLLM_URL",
        "AHP_MODAL_VLLM_MODEL",
        "AHP_MODAL_VLLM_TIER",
        "AHP_MODAL_VLLM_RATE",
        "AHP_MODAL_VLLM_PROVE_HEALTH",
        "AHP_SECONDARY_ORG",
        "AHP_SECONDARY_BASE_RATE",
        "AHP_DISABLE_SECONDARY_SERVER",
    ):
        monkeypatch.delenv(k, raising=False)


async def test_no_env_means_no_modal_no_secondary(redis_client):
    broker = Broker(redis_client)
    await _maybe_register_modal_vllm(broker)
    meta = await _maybe_register_secondary_server(broker, primary_org="acme")

    assert meta is None
    leaves = await broker.compute.list_leaves(only_alive_providers=False)
    assert not any(l.provider_id == "modal-vllm" for l in leaves)
    servers = await broker.servers.discover(alive_only=False)
    assert not any(s.org == "beta" for s in servers)


async def test_modal_url_registers_both_pieces_metadata_only(
    redis_client, monkeypatch,
):
    """With only the URL set: secondary server is registered, modal
    leaf is on the menu, but the modal leaf is NOT yet alive (no
    health proof has happened)."""
    monkeypatch.setenv("AHP_MODAL_VLLM_URL", "http://nowhere.invalid:8000/v1")
    monkeypatch.setenv("AHP_MODAL_VLLM_MODEL", "qwen2-5-7b")
    monkeypatch.setenv("AHP_MODAL_VLLM_TIER", "small")

    broker = Broker(redis_client)
    await _maybe_register_modal_vllm(broker)
    meta = await _maybe_register_secondary_server(broker, primary_org="tifin")

    # Secondary server is up.
    assert meta is not None
    assert meta.org == "beta"
    assert meta.compute_binding == "modal-vllm.small.qwen2-5-7b"

    # Modal leaf is on the menu (raw list)…
    raw = await broker.compute.list_leaves(only_alive_providers=False)
    assert any(l.provider_id == "modal-vllm" for l in raw)
    # …but NOT in the live menu — health unproven.
    live = await broker.compute.list_leaves(only_alive_providers=True)
    assert not any(l.provider_id == "modal-vllm" for l in live)


async def test_proven_health_makes_modal_leaf_live(redis_client, monkeypatch):
    """A subsequent heartbeat call from the user (or a successful
    health probe) flips the modal leaf into the live menu."""
    monkeypatch.setenv("AHP_MODAL_VLLM_URL", "http://nowhere.invalid:8000/v1")
    broker = Broker(redis_client)
    await _maybe_register_modal_vllm(broker)
    # Simulate the health probe having succeeded out-of-band.
    assert await broker.heartbeat_compute_provider("modal-vllm") is True

    live = await broker.compute.list_leaves(only_alive_providers=True)
    assert any(l.provider_id == "modal-vllm" for l in live)


async def test_secondary_disabled_by_explicit_env(redis_client, monkeypatch):
    monkeypatch.setenv("AHP_MODAL_VLLM_URL", "http://x")
    monkeypatch.setenv("AHP_DISABLE_SECONDARY_SERVER", "1")
    broker = Broker(redis_client)
    meta = await _maybe_register_secondary_server(broker, primary_org="acme")
    assert meta is None


async def test_secondary_org_collision_is_noop(redis_client, monkeypatch):
    """If the user picks the same org for primary and secondary, the
    helper must refuse rather than overwriting the primary server."""
    monkeypatch.setenv("AHP_MODAL_VLLM_URL", "http://x")
    monkeypatch.setenv("AHP_SECONDARY_ORG", "tifin")
    broker = Broker(redis_client)
    meta = await _maybe_register_secondary_server(broker, primary_org="tifin")
    assert meta is None
