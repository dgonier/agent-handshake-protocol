"""Tests for SurveyQueue.fire_due + export_corpus + the new CLI verbs.

Two clusters:

1. fire_due — idempotent dispatch via the bus, dry-run mode, expired
   surveys are abandoned not fired, the fired-set marker is cleaned
   up on submit_response.
2. Export — consent-filtered, anonymization, --since cutoff, JSONL
   round-trip.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
from pathlib import Path

import pytest

import ahp.cli
from ahp.broker import Broker, ServerMeta
from ahp.broker.surveys import (
    SURVEY_FIRED_INDEX,
    SURVEY_QUEUE_KEY,
    SURVEY_REQUEST_KEY,
    SurveyQueue,
    SurveyRequest,
    SurveyResponse,
    export_corpus,
    write_corpus_jsonl,
)
from ahp.core import AgentAddress, Code, Message
from ahp.economy.compute_provider import ComputeProvider, MenuLeaf
from ahp.economy.reputation import (
    ReputationEntry,
    VISIBILITY_FULL_AT,
)
from ahp.transport.redis_bus import RedisBus


def _req(
    *,
    survey_id: str = "sv-test",
    surveyed_actor: str = "you.human.x.y.s.session.alice",
    target_server: str = "acme",
    reward: float = 0.5,
    dispatch_at: float | None = None,
    expires_at: float | None = None,
    **overrides,
) -> SurveyRequest:
    now = time.time()
    return SurveyRequest(
        survey_id=survey_id,
        kind=overrides.pop("kind", "post_settlement"),
        target_server=target_server,
        surveyed_actor=surveyed_actor,
        recipe=overrides.pop("recipe", "post_settlement:csat"),
        settlement_id=overrides.pop("settlement_id", "msg:0001"),
        reward=reward,
        dispatch_at=dispatch_at if dispatch_at is not None else now - 60,
        expires_at=expires_at if expires_at is not None else now + 3600,
        **overrides,
    )


# ── fire_due: dry-run + dispatch ──────────────────────────────────────


async def test_fire_due_dry_run_returns_ready_without_marking(redis_client):
    q = SurveyQueue(redis_client)
    await q.enqueue(_req(survey_id="due-1"))
    await q.enqueue(_req(survey_id="due-2"))
    # bus=None → dry run.
    ready = await q.fire_due(bus=None)
    assert {r.survey_id for r in ready} == {"due-1", "due-2"}
    # Fired set is untouched.
    assert (await redis_client.scard(SURVEY_FIRED_INDEX)) == 0


async def test_fire_due_dispatches_human_observe(redis_client):
    """fire_due sends one HUMAN_OBSERVE per ready survey and marks
    each fired so a repeat sweep is a no-op."""
    q = SurveyQueue(redis_client)
    bus = RedisBus(redis_client)
    actor = "you.human.x.y.s.session.alice"
    await q.enqueue(_req(survey_id="sv-fire", surveyed_actor=actor))

    # Subscribe FIRST so we don't miss the publish.
    target = AgentAddress.parse(actor)
    sub = await bus.listen(target)
    try:
        dispatched = await q.fire_due(bus=bus, max_dispatch=10)
        assert [r.survey_id for r in dispatched] == ["sv-fire"]
        # Marker landed.
        assert await redis_client.sismember(SURVEY_FIRED_INDEX, "sv-fire")

        # The message published is HUMAN_OBSERVE with the right body.
        msg = await sub.get_one(timeout=2.0)
        assert msg is not None
        assert msg.code == Code.HUMAN_OBSERVE
        assert msg.verb == "SEND"
        assert msg.target == target
        body = msg.body
        assert body["kind"] == "survey"
        assert body["survey_id"] == "sv-fire"
        assert body["target_server"] == "acme"
        assert "ahp vote" in body["prompt"]
    finally:
        await sub.close()
        await bus.close()


async def test_fire_due_is_idempotent(redis_client):
    """Once a survey is fired, a second sweep doesn't re-dispatch it."""
    q = SurveyQueue(redis_client)
    bus = RedisBus(redis_client)
    await q.enqueue(_req(survey_id="sv-once"))
    try:
        first = await q.fire_due(bus=bus)
        assert len(first) == 1
        second = await q.fire_due(bus=bus)
        assert second == []
    finally:
        await bus.close()


async def test_fire_due_skips_future_surveys(redis_client):
    q = SurveyQueue(redis_client)
    await q.enqueue(_req(
        survey_id="sv-future",
        dispatch_at=time.time() + 3600,  # an hour from now
    ))
    ready = await q.fire_due(bus=None)
    assert ready == []


async def test_fire_due_drops_expired_surveys(redis_client):
    """A survey past expires_at is abandoned rather than fired."""
    q = SurveyQueue(redis_client)
    now = time.time()
    await q.enqueue(_req(
        survey_id="sv-stale",
        dispatch_at=now - 7200,
        expires_at=now - 3600,
    ))
    ready = await q.fire_due(bus=None)
    assert ready == []
    # Request blob and queue entry gone.
    assert await redis_client.get(
        SURVEY_REQUEST_KEY.format(survey_id="sv-stale"),
    ) is None


async def test_submit_response_clears_fired_marker(redis_client):
    """A vote on a fired survey cleans up SURVEY_FIRED_INDEX so the
    set doesn't grow unbounded."""
    q = SurveyQueue(redis_client)
    bus = RedisBus(redis_client)
    await q.enqueue(_req(survey_id="sv-clean"))
    try:
        await q.fire_due(bus=bus)
        assert await redis_client.sismember(SURVEY_FIRED_INDEX, "sv-clean")
        response = SurveyResponse(
            survey_id="sv-clean",
            surveyed_actor="you.human.x.y.s.session.alice",
            target_server="acme",
            recipe="post_settlement:csat",
            settlement_id="msg:0001",
            score=0.8,
        )
        await q.submit_response(response)
        assert not await redis_client.sismember(SURVEY_FIRED_INDEX, "sv-clean")
    finally:
        await bus.close()


async def test_fire_due_skips_invalid_actor_address(redis_client):
    """A survey whose surveyed_actor isn't a valid 7-field address is
    skipped without crashing the sweep — it stays in the queue
    (un-fired) so a fix-and-retry is possible."""
    q = SurveyQueue(redis_client)
    bus = RedisBus(redis_client)
    await q.enqueue(_req(
        survey_id="sv-bad-actor",
        surveyed_actor="not.a.valid.address",  # < 7 fields
    ))
    try:
        dispatched = await q.fire_due(bus=bus)
        assert dispatched == []
        # Not in the fired set — we never sent anything.
        assert not await redis_client.sismember(SURVEY_FIRED_INDEX, "sv-bad-actor")
    finally:
        await bus.close()


# ── export_corpus + write_corpus_jsonl ────────────────────────────────


async def _seed_responses(redis_client) -> list[SurveyResponse]:
    """Seed three responses: one with training opt-in, two without."""
    q = SurveyQueue(redis_client)
    responses = [
        SurveyResponse(
            survey_id="sv-train",
            surveyed_actor="you.human.x.y.s.session.alice",
            target_server="acme",
            recipe="post_settlement:csat",
            settlement_id="msg:1",
            score=0.9,
            free_text="great answer",
            consent_csat_routing=True,
            consent_training_export=True,
        ),
        SurveyResponse(
            survey_id="sv-no-train",
            surveyed_actor="you.human.x.y.s.session.bob",
            target_server="acme",
            recipe="post_settlement:csat",
            settlement_id="msg:2",
            score=0.4,
            free_text="meh",
            consent_csat_routing=True,
            consent_training_export=False,
        ),
        SurveyResponse(
            survey_id="sv-old-train",
            surveyed_actor="you.human.x.y.s.session.carol",
            target_server="beta",
            recipe="post_settlement:csat",
            settlement_id="msg:3",
            score=0.7,
            consent_training_export=True,
            collected_at=100.0,  # ancient
        ),
    ]
    # Submit each; q.submit_response handles persistence + index add.
    for r in responses:
        # Pre-seed a request so submit_response has something to read,
        # but bypass payment by passing broker=None.
        await q.enqueue(SurveyRequest(
            survey_id=r.survey_id,
            kind="post_settlement",
            target_server=r.target_server,
            surveyed_actor=r.surveyed_actor,
            recipe=r.recipe,
            settlement_id=r.settlement_id,
            reward=0.0,
            dispatch_at=time.time(),
            expires_at=time.time() + 3600,
        ))
        await q.submit_response(r, broker=None)
    return responses


async def test_export_only_emits_consenting_rows(redis_client):
    await _seed_responses(redis_client)
    rows = await export_corpus(redis_client)
    ids = {r.row_id for r in rows}
    assert "sv-train" in ids
    assert "sv-old-train" in ids
    assert "sv-no-train" not in ids
    # Free text is preserved on the consenting row.
    train_row = next(r for r in rows if r.row_id == "sv-train")
    assert train_row.free_text == "great answer"


async def test_export_anonymizes_by_default(redis_client):
    await _seed_responses(redis_client)
    rows = await export_corpus(redis_client)
    train_row = next(r for r in rows if r.row_id == "sv-train")
    assert train_row.actor_handle.startswith("act-")
    assert "alice" not in train_row.actor_handle


async def test_export_no_anonymize_keeps_raw(redis_client):
    await _seed_responses(redis_client)
    rows = await export_corpus(redis_client, anonymize=False)
    train_row = next(r for r in rows if r.row_id == "sv-train")
    assert "alice" in train_row.actor_handle


async def test_export_since_cutoff(redis_client):
    """The old row (collected_at=100) is filtered out by --since 1000."""
    await _seed_responses(redis_client)
    rows = await export_corpus(redis_client, since=1000.0)
    ids = {r.row_id for r in rows}
    assert "sv-train" in ids
    assert "sv-old-train" not in ids


async def test_write_corpus_jsonl_writes_valid_lines(
    redis_client, tmp_path: Path,
):
    await _seed_responses(redis_client)
    out_path = tmp_path / "export.jsonl"
    count = await write_corpus_jsonl(redis_client, out_path)
    assert count == 2  # train + old-train
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
    # Each line is parseable JSON with the row schema.
    for line in lines:
        row = json.loads(line)
        assert "row_id" in row
        assert "target_server" in row
        assert "score" in row
        assert row["row_id"] in {"sv-train", "sv-old-train"}


# ── CLI: fire-surveys + export-surveys ────────────────────────────────


async def _arun(cmd: str, *argv: str) -> tuple[int, str]:
    parser = ahp.cli.build_parser()
    args = parser.parse_args([cmd, *argv])
    buf = io.StringIO()
    if cmd == "fire-surveys":
        rc = await ahp.cli._fire_surveys_async(args, buf)
    elif cmd == "export-surveys":
        rc = await ahp.cli._export_surveys_async(args, buf)
    else:
        raise AssertionError(f"unexpected cmd {cmd}")
    return rc, buf.getvalue()


async def test_cli_fire_surveys_dry_run(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    q = SurveyQueue(redis_client)
    await q.enqueue(_req(survey_id="cli-dry-1"))
    await q.enqueue(_req(survey_id="cli-dry-2"))
    rc, out = await _arun(
        "fire-surveys",
        "--redis-url", "redis://test/0",
        "--dry-run",
    )
    assert rc == 0
    assert "dry-run" in out
    assert "cli-dry-1" in out
    assert "cli-dry-2" in out
    # Not marked fired.
    assert (await redis_client.scard(SURVEY_FIRED_INDEX)) == 0


async def test_cli_fire_surveys_empty(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    rc, out = await _arun(
        "fire-surveys", "--redis-url", "redis://test/0", "--dry-run",
    )
    assert rc == 0
    assert "no ready surveys" in out


async def test_cli_export_surveys_to_stdout(redis_client, monkeypatch):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    await _seed_responses(redis_client)
    rc, out = await _arun(
        "export-surveys", "--redis-url", "redis://test/0",
    )
    assert rc == 0
    lines = [line for line in out.splitlines() if line.strip()]
    assert len(lines) == 2  # train + old-train
    parsed = [json.loads(l) for l in lines]
    assert {p["row_id"] for p in parsed} == {"sv-train", "sv-old-train"}


async def test_cli_export_surveys_to_file(
    redis_client, monkeypatch, tmp_path: Path,
):
    monkeypatch.setattr(ahp.cli, "_connect_redis", lambda url: redis_client)
    await _seed_responses(redis_client)
    out_path = tmp_path / "exports" / "corpus.jsonl"
    rc, out = await _arun(
        "export-surveys",
        "--redis-url", "redis://test/0",
        "--out", str(out_path),
    )
    assert rc == 0
    assert "wrote 2 row(s)" in out
    assert out_path.is_file()
    lines = out_path.read_text().splitlines()
    assert len(lines) == 2
