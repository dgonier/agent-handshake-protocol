"""CLI tests for `ahp list-surveys` and `ahp vote`.

Both subcommands hit Redis through SurveyQueue / Broker. Tests use the
same monkeypatch-on-_connect_redis pattern as the other Redis-backed
CLI tests, and call the async workers directly to avoid the asyncio
nesting problem.
"""

from __future__ import annotations

import io
import time

import pytest

import ahp.cli
from ahp.broker import Broker, ServerMeta
from ahp.broker.surveys import SurveyQueue, SurveyRequest
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import (
    ReputationEntry,
    VISIBILITY_FULL_AT,
)


async def _arun(cmd: str, *argv: str) -> tuple[int, str]:
    parser = ahp.cli.build_parser()
    args = parser.parse_args([cmd, *argv])
    buf = io.StringIO()
    if cmd == "list-surveys":
        rc = await ahp.cli._list_surveys_async(args, buf)
    elif cmd == "vote":
        rc = await ahp.cli._vote_async(args, buf)
    else:
        raise AssertionError(f"unexpected cmd {cmd}")
    return rc, buf.getvalue()


# ── list-surveys ──────────────────────────────────────────────────────


async def test_list_surveys_empty(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, out = await _arun(
        "list-surveys", "--redis-url", "redis://test/0",
    )
    assert rc == 0
    assert "no pending surveys" in out


async def test_list_surveys_shows_pending(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    q = SurveyQueue(redis_client)
    now = time.time()
    await q.enqueue(SurveyRequest(
        survey_id="sv1234567890abcdef",
        kind="post_settlement",
        target_server="acme",
        surveyed_actor="you.human.x.y.s.session.alice",
        recipe="post_settlement:csat",
        settlement_id="msg:0001",
        reward=0.42,
        dispatch_at=now - 60,
        expires_at=now + 3600,
    ))
    rc, out = await _arun(
        "list-surveys", "--redis-url", "redis://test/0",
    )
    assert rc == 0
    assert "sv1234567890" in out
    assert "you.human.x.y.s.session.alice" in out
    assert "acme" in out
    assert "0.4200" in out


async def test_list_surveys_filters_by_actor(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    q = SurveyQueue(redis_client)
    now = time.time()
    await q.enqueue(SurveyRequest(
        survey_id="sv-alice", kind="post_settlement",
        target_server="acme",
        surveyed_actor="you.human.x.y.s.session.alice",
        recipe="r", settlement_id="msg:1", reward=0.1,
        dispatch_at=now - 60, expires_at=now + 3600,
    ))
    await q.enqueue(SurveyRequest(
        survey_id="sv-bob", kind="post_settlement",
        target_server="acme",
        surveyed_actor="you.human.x.y.s.session.bob",
        recipe="r", settlement_id="msg:2", reward=0.1,
        dispatch_at=now - 60, expires_at=now + 3600,
    ))
    rc, out = await _arun(
        "list-surveys", "--redis-url", "redis://test/0",
        "--for", "you.human.x.y.s.session.alice",
    )
    assert rc == 0
    assert "sv-alice" in out
    assert "sv-bob" not in out


# ── vote ──────────────────────────────────────────────────────────────


async def _seed_broker_with_survey(
    redis_client,
    *,
    actor: str = "you.human.x.y.s.session.caller",
    target: str = "acme",
    reward: float = 0.5,
) -> str:
    """Stand up a broker with one pending survey. Returns the survey_id."""
    broker = Broker(redis_client)
    await broker.register_server(ServerMeta(
        server_id=target, org=target, base_rate=0.0002,
        compute_binding=f"{target}.small.echo",
        supported_tiers=["small"],
    ))
    await broker.register_compute_provider(ComputeProvider(provider_id=target))
    await broker.register_leaf(MenuLeaf(
        provider_id=target, tier="small", model="echo",
        rate_per_1k_chars=0.0,
    ))
    await broker.set_reputation(ReputationEntry(
        owner=target, reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))
    await broker.wallet("__commons__").topup(20.0, reason="seed")

    request = SurveyRequest.new(
        kind="post_settlement",
        target_server=target,
        surveyed_actor=actor,
        recipe="post_settlement:csat",
        settlement_id="msg:42",
        reward=reward,
        delay_seconds=0.0,
    )
    await broker.surveys.enqueue(request)
    return request.survey_id


async def test_vote_credits_actor_and_updates_csat(
    redis_client, monkeypatch,
):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    actor = "you.human.x.y.s.session.caller"
    survey_id = await _seed_broker_with_survey(redis_client, actor=actor)
    actor_before = (await Broker(redis_client).wallet(actor).get_state()).balance

    rc, out = await _arun(
        "vote",
        "--redis-url", "redis://test/0",
        "--survey-id", survey_id,
        "--score", "0.8",
    )
    assert rc == 0
    assert "recorded vote" in out
    assert "new_balance" in out

    # Wallet credited.
    actor_after = (await Broker(redis_client).wallet(actor).get_state()).balance
    assert abs((actor_after - actor_before) - 0.5) < 1e-6

    # CSAT updated.
    rep = await Broker(redis_client).get_reputation("acme")
    assert rep is not None
    assert rep.csat_samples == 1
    assert abs(rep.csat - 0.8) < 1e-9


async def test_vote_1to5_scale_normalizes(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    survey_id = await _seed_broker_with_survey(redis_client)

    rc, _ = await _arun(
        "vote",
        "--redis-url", "redis://test/0",
        "--survey-id", survey_id,
        "--score", "5",
        "--scale", "1to5",
    )
    assert rc == 0
    rep = await Broker(redis_client).get_reputation("acme")
    # 5 on 1..5 → 1.0 normalized.
    assert abs(rep.csat - 1.0) < 1e-9


async def test_vote_out_of_range_score_errors(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    survey_id = await _seed_broker_with_survey(redis_client)

    rc, _ = await _arun(
        "vote",
        "--redis-url", "redis://test/0",
        "--survey-id", survey_id,
        "--score", "1.5",
    )
    assert rc == 2


async def test_vote_unknown_survey_id(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, _ = await _arun(
        "vote",
        "--redis-url", "redis://test/0",
        "--survey-id", "does-not-exist",
        "--score", "0.5",
    )
    assert rc == 2


async def test_vote_double_submit_errors(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    survey_id = await _seed_broker_with_survey(redis_client)
    rc, _ = await _arun(
        "vote",
        "--redis-url", "redis://test/0",
        "--survey-id", survey_id, "--score", "0.5",
    )
    assert rc == 0
    rc, _ = await _arun(
        "vote",
        "--redis-url", "redis://test/0",
        "--survey-id", survey_id, "--score", "0.9",
    )
    assert rc == 2


async def test_vote_allow_training_records_consent(redis_client, monkeypatch):
    """--allow-training propagates to the persisted response row."""
    from ahp.broker.surveys import SURVEY_RESPONSE_KEY, response_from_raw

    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    survey_id = await _seed_broker_with_survey(redis_client)
    rc, _ = await _arun(
        "vote",
        "--redis-url", "redis://test/0",
        "--survey-id", survey_id, "--score", "0.7",
        "--allow-training",
    )
    assert rc == 0

    raw = await redis_client.get(
        SURVEY_RESPONSE_KEY.format(survey_id=survey_id)
    )
    resp = response_from_raw(raw)
    assert resp.consent_training_export is True
