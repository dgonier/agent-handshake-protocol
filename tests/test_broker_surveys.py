"""Surveys: SurveyQueue + broker auto-enqueue tests.

The CSAT loop has three moving pieces that need to lock in:

1. SurveyQueue persistence — enqueue is idempotent on survey_id;
   list_pending filters by dispatch_at; submit_response records
   immutably + drops the queue entry.
2. Broker integration — submit_response credits the actor from
   commons and folds the score into the target server's CSAT via
   apply_csat. Opted-out servers are NOT enqueued.
3. Auto-enqueue at settlement — calculate_and_settle enqueues a
   survey when the settled pre_tax >= AHP_SURVEY_STAKES_THRESHOLD.
   Sub-threshold settlements do not.
"""

from __future__ import annotations

import time

import pytest

from ahp.broker import Broker, ServerMeta
from ahp.broker.surveys import (
    SURVEY_QUEUE_KEY,
    SURVEY_REQUEST_KEY,
    SURVEY_RESPONSE_INDEX,
    SURVEY_RESPONSE_KEY,
    SurveyKind,
    SurveyQueue,
    SurveyRequest,
    SurveyResponse,
)
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import (
    DEFAULT_REPUTATION,
    LAMBDA_CSAT,
    ReputationEntry,
    VISIBILITY_FULL_AT,
)


# ── SurveyQueue (no broker side effects) ──────────────────────────────


def _req(*, dispatch_at: float = 0.0, expires_at: float = 1e12, **overrides):
    base = dict(
        survey_id=overrides.pop("survey_id", "sv-test"),
        kind=overrides.pop("kind", "post_settlement"),
        target_server=overrides.pop("target_server", "acme"),
        surveyed_actor=overrides.pop("surveyed_actor", "you.human.x.y.s.session.caller"),
        recipe=overrides.pop("recipe", "post_settlement:csat"),
        settlement_id=overrides.pop("settlement_id", "msg:0001"),
        reward=overrides.pop("reward", 0.5),
        dispatch_at=dispatch_at,
        expires_at=expires_at,
    )
    base.update(overrides)
    return SurveyRequest(**base)


async def test_enqueue_is_idempotent(redis_client):
    q = SurveyQueue(redis_client)
    r1 = _req()
    assert await q.enqueue(r1) is True
    # Same survey_id, different reward — should NOT overwrite.
    r2 = _req(reward=99.0)
    assert await q.enqueue(r2) is False
    fetched = await q.get_request(r1.survey_id)
    assert fetched is not None and fetched.reward == 0.5


async def test_list_pending_filters_by_dispatch_time(redis_client):
    q = SurveyQueue(redis_client)
    now = time.time()
    await q.enqueue(_req(survey_id="due", dispatch_at=now - 60))
    await q.enqueue(_req(survey_id="future", dispatch_at=now + 3600))

    # Default — only due surveys.
    rows = await q.list_pending(now=now)
    assert [r.survey_id for r in rows] == ["due"]

    # include_future — both surfaces.
    both = await q.list_pending(now=now, include_future=True)
    assert {r.survey_id for r in both} == {"due", "future"}


async def test_list_pending_filters_by_actor(redis_client):
    q = SurveyQueue(redis_client)
    now = time.time()
    await q.enqueue(_req(
        survey_id="for-alice", dispatch_at=now - 60,
        surveyed_actor="you.human.x.y.s.session.alice",
    ))
    await q.enqueue(_req(
        survey_id="for-bob", dispatch_at=now - 60,
        surveyed_actor="you.human.x.y.s.session.bob",
    ))
    rows = await q.list_pending(
        now=now,
        surveyed_actor="you.human.x.y.s.session.alice",
    )
    assert [r.survey_id for r in rows] == ["for-alice"]


async def test_expired_request_is_dropped_on_list(redis_client):
    """A survey past its expires_at is removed from the queue + store
    on the next list_pending sweep."""
    q = SurveyQueue(redis_client)
    now = time.time()
    await q.enqueue(_req(
        survey_id="stale", dispatch_at=now - 7200, expires_at=now - 3600,
    ))
    rows = await q.list_pending(now=now)
    assert rows == []
    # And the request blob is gone.
    assert (await redis_client.get(
        SURVEY_REQUEST_KEY.format(survey_id="stale"),
    )) is None


async def test_submit_response_idempotent_and_removes_from_queue(
    redis_client,
):
    q = SurveyQueue(redis_client)
    await q.enqueue(_req(survey_id="sv1"))
    response = SurveyResponse(
        survey_id="sv1",
        surveyed_actor="you.human.x.y.s.session.caller",
        target_server="acme",
        recipe="post_settlement:csat",
        settlement_id="msg:0001",
        score=0.8,
    )
    assert await q.submit_response(response) is True
    # No longer in the pending queue.
    assert await q.list_pending(now=time.time() + 1) == []
    # Response indexed.
    assert await redis_client.sismember(SURVEY_RESPONSE_INDEX, "sv1")
    # Second submission is a no-op.
    assert await q.submit_response(response) is False


# ── Broker integration ───────────────────────────────────────────────


async def _seed_broker(redis_client) -> Broker:
    broker = Broker(redis_client)
    server = ServerMeta(
        server_id="acme", org="acme",
        base_rate=0.0002, compute_binding="acme.small.echo",
        supported_tiers=["small"],
    )
    await broker.register_server(server)
    await broker.register_compute_provider(ComputeProvider(provider_id="acme"))
    await broker.register_leaf(MenuLeaf(
        provider_id="acme", tier="small", model="echo",
        rate_per_1k_chars=0.0,
    ))
    await broker.set_reputation(ReputationEntry(
        owner="acme", reputation=0.9,
        completed_accepted=VISIBILITY_FULL_AT,
    ))
    # Seed commons so survey reward has somewhere to flow from.
    await broker.wallet("__commons__").topup(20.0, reason="test")
    return broker


async def test_submit_response_credits_actor_and_updates_csat(redis_client):
    broker = await _seed_broker(redis_client)
    actor = "you.human.x.y.s.session.caller"
    # Manually enqueue (skip auto-enqueue threshold gymnastics).
    request = SurveyRequest.new(
        kind="post_settlement",
        target_server="acme",
        surveyed_actor=actor,
        recipe="post_settlement:csat",
        settlement_id="msg:42",
        reward=0.5,
        delay_seconds=0.0,
    )
    await broker.surveys.enqueue(request)

    actor_before = (await broker.wallet(actor).get_state()).balance
    commons_before = (await broker.wallet("__commons__").get_state()).balance

    response = SurveyResponse(
        survey_id=request.survey_id,
        surveyed_actor=actor,
        target_server="acme",
        recipe=request.recipe,
        settlement_id=request.settlement_id,
        score=0.9,
    )
    assert await broker.submit_survey_response(response) is True

    # Wallets moved.
    actor_after = (await broker.wallet(actor).get_state()).balance
    commons_after = (await broker.wallet("__commons__").get_state()).balance
    assert abs((actor_after - actor_before) - 0.5) < 1e-6
    assert abs((commons_before - commons_after) - 0.5) < 1e-6

    # CSAT updated on the target server's reputation.
    rep = await broker.get_reputation("acme")
    assert rep is not None
    assert rep.csat_samples == 1
    # First sample replaces the default.
    assert abs(rep.csat - 0.9) < 1e-9


async def test_opted_out_server_isnt_surveyed(redis_client):
    """A server with survey_opt_in=False makes request_survey a no-op."""
    broker = await _seed_broker(redis_client)
    # Flip the consent.
    meta = await broker.servers.get("acme")
    assert meta is not None
    meta.__dict__["survey_opt_in"] = False  # frozen=False on ServerMeta
    await broker.register_server(meta)

    result = await broker.request_survey(
        kind="post_settlement",
        target_server="acme",
        surveyed_actor="you.human.x.y.s.session.caller",
        recipe="post_settlement:csat",
        settlement_id="msg:101",
        reward=0.5,
    )
    assert result is None
    # And nothing is on the queue.
    assert (await broker.surveys.list_pending(
        now=time.time() + 1, include_future=True,
    )) == []


async def test_auto_enqueue_fires_above_stakes_threshold(
    redis_client, monkeypatch,
):
    """A high-stakes settlement should auto-enqueue a survey."""
    monkeypatch.setenv("AHP_SURVEY_STAKES_THRESHOLD", "0.001")
    # Reward rate is the 5% default so the auto-enqueued reward is
    # easy to inspect.
    broker = await _seed_broker(redis_client)
    actor = "you.human.x.y.s.session.caller"
    # Place a hold (caller is the surveyed actor in this model).
    await broker.wallet(actor).topup(10.0, reason="seed")
    await broker.hold(caller=actor, amount=1.0, hold_id="msg:99")

    settlement = await broker.calculate_and_settle(
        caller=actor, hold_id="msg:99",
        server=await broker.servers.get("acme"),
        leaf=MenuLeaf(
            provider_id="acme", tier="small", model="echo",
            rate_per_1k_chars=0.0,
        ),
        response_chars=500, max_response_chars=1000,
        actual_latency_ms=500.0, completed_with_caller=0,
    )

    # pre_tax is non-trivial; a survey should be queued.
    pending = await broker.surveys.list_pending(include_future=True)
    assert len(pending) == 1
    s = pending[0]
    assert s.surveyed_actor == actor
    assert s.target_server == "acme"
    assert s.kind == "post_settlement"
    # Reward is 5% of pre_tax.
    assert abs(s.reward - settlement.pre_tax * 0.05) < 1e-6


async def test_no_auto_enqueue_below_threshold(redis_client, monkeypatch):
    """A sub-threshold settlement leaves the queue empty."""
    monkeypatch.setenv("AHP_SURVEY_STAKES_THRESHOLD", "1000.0")
    broker = await _seed_broker(redis_client)
    actor = "you.human.x.y.s.session.caller"
    await broker.wallet(actor).topup(10.0, reason="seed")
    # Hold needs to clear the actual computed pre_tax; size it
    # generously so the settlement succeeds and we can assert on the
    # post-settlement survey state.
    await broker.hold(caller=actor, amount=1.0, hold_id="msg:1")

    await broker.calculate_and_settle(
        caller=actor, hold_id="msg:1",
        server=await broker.servers.get("acme"),
        leaf=MenuLeaf(
            provider_id="acme", tier="small", model="echo",
            rate_per_1k_chars=0.0,
        ),
        response_chars=20, max_response_chars=100,
        actual_latency_ms=200.0, completed_with_caller=0,
    )

    assert (await broker.surveys.list_pending(include_future=True)) == []


async def test_response_consent_is_immutable_per_row(redis_client):
    """A SurveyResponse records consent state at submission, not at
    later read time. Re-reading the row must show the same flags
    even if the server's current consent has flipped."""
    broker = await _seed_broker(redis_client)
    actor = "you.human.x.y.s.session.caller"
    request = SurveyRequest.new(
        kind="post_settlement",
        target_server="acme", surveyed_actor=actor,
        recipe="post_settlement:csat",
        settlement_id="msg:7", reward=0.0,
        delay_seconds=0.0,
    )
    await broker.surveys.enqueue(request)
    response = SurveyResponse(
        survey_id=request.survey_id,
        surveyed_actor=actor,
        target_server="acme",
        recipe=request.recipe,
        settlement_id=request.settlement_id,
        score=0.7,
        consent_csat_routing=True,
        consent_training_export=True,  # opt-in at collection
    )
    assert await broker.submit_survey_response(response) is True

    # Flip the target server's training_data_opt_in OFF post-hoc.
    meta = await broker.servers.get("acme")
    meta.__dict__["training_data_opt_in"] = False
    await broker.register_server(meta)

    # Response row still has training_export=True (immutable per row).
    raw = await redis_client.get(
        SURVEY_RESPONSE_KEY.format(survey_id=request.survey_id),
    )
    from ahp.broker.surveys import response_from_raw
    fetched = response_from_raw(raw)
    assert fetched.consent_training_export is True
