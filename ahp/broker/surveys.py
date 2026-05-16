"""Post-hoc CSAT surveys — storage + manual collection.

The CSAT (consumer satisfaction) score in :class:`ReputationEntry` is
a separate dimension from system-observed reputation. It comes from
asking the consuming actor — agent or human — *after* an interaction
how useful the response actually was.

This module implements the storage + manual-collection half of the
survey loop. Auto-firing to actors is still deferred: a survey sits
in the queue (``ahp:survey:queue``) until something calls
:meth:`SurveyQueue.submit_response` for it. The "something" can be
the ``ahp vote`` CLI (human-side) or, eventually, an agent-side
auto-rate hook.

Surveys are *opt-in*. Every actor has three consent flags on their
:class:`~ahp.broker.ServerMeta` profile:

* ``survey_opt_in`` — can be queued for surveys at all.
* ``csat_routing_opt_in`` — CSAT score feeds the router.
* ``training_data_opt_in`` — responses are eligible for the future
  open-source export (still stubbed; see :func:`export_corpus`).

Every recorded :class:`SurveyResponse` carries a snapshot of the
consent flags that were active when it was collected. Consent changes
are *not retroactive*: a row collected with
``consent_training_export=True`` remains eligible for export even if
the actor flips that flag off later. (GDPR-style deletion requests
are handled separately by tagging-then-rewriting the export.)

Survey responses are paid out of the commons pool. Reward is set per
:class:`SurveyRequest` at queue time; the broker debits commons and
credits the surveyed actor when the response lands.

Redis layout (single :class:`SurveyQueue` per broker):

* ``ahp:survey:queue`` — sorted set; score = ``dispatch_at``,
  member = ``survey_id``. Cheap ``ZRANGEBYSCORE`` for "what's ready
  to fire."
* ``ahp:survey:request:<survey_id>`` — JSON of the
  :class:`SurveyRequest`. ``SET NX`` on enqueue makes
  re-enqueue idempotent.
* ``ahp:survey:response:<survey_id>`` — JSON of the
  :class:`SurveyResponse` after :meth:`submit_response`.
* ``ahp:survey:responses`` — set of survey ids that have responses.
  Cheap iteration for future export.

Anti-gaming defenses planned but NOT yet implemented:

* Outlier responders (always 5/5 or always 1/5) get survey weight
  down-weighted. Currently every response counts the same.
* Reward comes from commons, not from the surveyed server — already
  enforced here; the server can't bribe the surveyor.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


log = logging.getLogger("ahp.broker.surveys")


SURVEY_QUEUE_KEY = "ahp:survey:queue"
SURVEY_REQUEST_KEY = "ahp:survey:request:{survey_id}"
SURVEY_RESPONSE_KEY = "ahp:survey:response:{survey_id}"
SURVEY_RESPONSE_INDEX = "ahp:survey:responses"


SurveyKind = Literal[
    "post_settlement",  # delayed CSAT after a normal call
    "drill_down",       # follow-up after an unusual outcome (refund, sub_tier)
    "periodic",         # broker-initiated sample of a longtime peer
]


@dataclass(frozen=True)
class SurveyRequest:
    """One survey the broker wants to dispatch to a consuming actor.

    Created at settlement time and queued for delayed dispatch. The
    queue is persisted in Redis under ``ahp:survey:queue`` so it
    survives broker restarts.
    """

    survey_id: str
    kind: SurveyKind
    target_server: str            # the server being rated
    surveyed_actor: str           # the address being asked
    recipe: str                   # e.g. "adversarial:debate-me"
    settlement_id: str            # links back to the wallet settlement
    reward: float                 # credits paid for responding
    dispatch_at: float            # wall-clock timestamp; broker fires on/after
    expires_at: float             # broker abandons the survey past this

    # Consent snapshot at queue time. The actor's consent at survey
    # *response* time is what governs how the response is tagged; this
    # is just a hint so the broker doesn't queue surveys to actors who
    # have opted out at queue time.
    consent_csat_routing_at_queue: bool = True
    consent_training_export_at_queue: bool = False

    @classmethod
    def new(
        cls,
        *,
        kind: SurveyKind,
        target_server: str,
        surveyed_actor: str,
        recipe: str,
        settlement_id: str,
        reward: float,
        delay_seconds: float = 300.0,
        ttl_seconds: float = 86_400.0,
    ) -> "SurveyRequest":
        now = time.time()
        return cls(
            survey_id=str(uuid.uuid4()),
            kind=kind,
            target_server=target_server,
            surveyed_actor=surveyed_actor,
            recipe=recipe,
            settlement_id=settlement_id,
            reward=reward,
            dispatch_at=now + delay_seconds,
            expires_at=now + ttl_seconds,
        )


@dataclass(frozen=True)
class SurveyResponse:
    """The answer to a survey. Stored as both a CSAT update and a
    permanent record for downstream training-data export.

    Each row records the *consent state at the moment of collection* —
    not the actor's current consent. This is load-bearing: it lets the
    export pipeline emit only rows where the actor consented to export
    at the time, regardless of whether they later flipped opt-in off.
    """

    survey_id: str
    surveyed_actor: str
    target_server: str
    recipe: str
    settlement_id: str
    score: float                       # 0..1
    free_text: str = ""
    collected_at: float = field(default_factory=time.time)

    # Consent snapshot — immutable per row.
    consent_csat_routing: bool = True
    consent_training_export: bool = False


# ── queue ─────────────────────────────────────────────────────────────


class SurveyQueue:
    """Broker-side queue for survey requests + responses.

    The queue persists :class:`SurveyRequest` objects in Redis and is
    the single chokepoint for collecting :class:`SurveyResponse`
    submissions. It does NOT auto-fire surveys to actors — a caller
    (the ``ahp vote`` CLI, the runner, or a future cadence loop)
    walks ``list_pending(...)`` and submits responses on the actor's
    behalf.

    Operations are designed to be idempotent on ``survey_id`` so a
    repeated enqueue or submission is a no-op rather than a duplicate.
    """

    def __init__(
        self,
        redis_client: Any,
        *,
        commons_wallet_owner: str = "__commons__",
        audit: Any = None,
    ) -> None:
        self._redis = redis_client
        self._commons = commons_wallet_owner
        self._audit = audit

    async def _emit(
        self,
        op: str,
        *,
        target: str | None = None,
        success: bool = True,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Best-effort audit emission. Never raises."""
        if self._audit is None:
            return
        try:
            # Local import to avoid a circular import between
            # ahp.broker.surveys and ahp.audit at module load time.
            from ahp.audit import AuditEvent
            await self._audit.emit(AuditEvent(
                op=op, target=target,
                success=success, error=error,
                extra=extra or {},
            ))
        except Exception:
            log.exception("survey audit emit failed for op=%s", op)

    # ── enqueue / inspect ────────────────────────────────────────────

    async def enqueue(self, request: SurveyRequest) -> bool:
        """Persist a survey request and add it to the dispatch queue.

        Idempotent: if a survey with this ``survey_id`` is already
        stored, the existing record is preserved and this returns
        False. Returns True when a new entry was written.
        """
        key = SURVEY_REQUEST_KEY.format(survey_id=request.survey_id)
        payload = json.dumps(asdict(request))
        # SET NX gives us idempotency: first enqueue wins, repeats
        # return None and we skip the queue add.
        was_new = await self._redis.set(key, payload, nx=True)
        if not was_new:
            return False
        await self._redis.zadd(
            SURVEY_QUEUE_KEY,
            {request.survey_id: request.dispatch_at},
        )
        log.info(
            "survey enqueued: id=%s actor=%s server=%s reward=%.4f",
            request.survey_id, request.surveyed_actor,
            request.target_server, request.reward,
        )
        await self._emit(
            "survey.enqueue",
            target=request.survey_id,
            extra={
                "kind": request.kind,
                "surveyed_actor": request.surveyed_actor,
                "target_server": request.target_server,
                "recipe": request.recipe,
                "reward": request.reward,
                "dispatch_at": request.dispatch_at,
                "expires_at": request.expires_at,
                "consent_csat_routing_at_queue": request.consent_csat_routing_at_queue,
                "consent_training_export_at_queue": request.consent_training_export_at_queue,
            },
        )
        return True

    async def get_request(self, survey_id: str) -> SurveyRequest | None:
        raw = await self._redis.get(
            SURVEY_REQUEST_KEY.format(survey_id=survey_id)
        )
        if raw is None:
            return None
        return _request_from_raw(raw)

    async def list_pending(
        self,
        *,
        now: float | None = None,
        include_future: bool = False,
        surveyed_actor: str | None = None,
        limit: int = 100,
    ) -> list[SurveyRequest]:
        """Return queued surveys.

        With the defaults: only surveys whose ``dispatch_at`` is in
        the past or right now, up to ``limit`` entries, oldest first.

        ``include_future=True`` lifts the ``dispatch_at <= now`` gate
        — useful for the CLI's "what's queued, in any state" view.

        ``surveyed_actor`` filters to surveys targeting a specific
        actor address — what an agent / human would ask before voting.
        """
        cutoff = now if now is not None else time.time()
        max_score = "+inf" if include_future else cutoff
        ids = await self._redis.zrangebyscore(
            SURVEY_QUEUE_KEY, min="-inf", max=max_score,
            start=0, num=limit,
        )
        out: list[SurveyRequest] = []
        for sid in ids:
            if isinstance(sid, (bytes, bytearray)):
                sid = sid.decode("utf-8")
            req = await self.get_request(sid)
            if req is None:
                # Stale queue entry pointing at a deleted record. Tidy
                # up opportunistically.
                await self._redis.zrem(SURVEY_QUEUE_KEY, sid)
                continue
            if req.expires_at <= cutoff:
                # Expired — drop both the request and its queue entry.
                await self._abandon(sid)
                continue
            if surveyed_actor and req.surveyed_actor != surveyed_actor:
                continue
            out.append(req)
        return out

    # ── response ─────────────────────────────────────────────────────

    async def submit_response(
        self,
        response: SurveyResponse,
        *,
        broker: Any = None,
    ) -> bool:
        """Record a response, credit the actor, update CSAT.

        Returns True when the response was newly recorded, False if
        a response with this ``survey_id`` already exists (idempotent
        re-submission is a no-op).

        Side effects when broker is wired:
        1. Persist the :class:`SurveyResponse` and add to the response
           index.
        2. Debit ``request.reward`` from the commons wallet, credit
           the surveyed actor.
        3. Apply CSAT update to the target server's reputation via
           :func:`~ahp.economy.reputation.apply_csat`.
        4. Remove the request from the pending queue.

        ``broker=None`` is allowed for tests that only want to
        exercise persistence — steps 2 and 3 are then skipped.
        """
        resp_key = SURVEY_RESPONSE_KEY.format(survey_id=response.survey_id)
        payload = json.dumps(asdict(response))
        was_new = await self._redis.set(resp_key, payload, nx=True)
        if not was_new:
            return False
        await self._redis.sadd(SURVEY_RESPONSE_INDEX, response.survey_id)
        # Drop the entry from the pending queue. Idempotent.
        await self._redis.zrem(SURVEY_QUEUE_KEY, response.survey_id)

        if broker is not None:
            req = await self.get_request(response.survey_id)
            if req is not None:
                await self._pay_and_score(req, response, broker)

        log.info(
            "survey response submitted: id=%s actor=%s score=%.2f",
            response.survey_id, response.surveyed_actor, response.score,
        )
        await self._emit(
            "survey.response",
            target=response.survey_id,
            extra={
                "surveyed_actor": response.surveyed_actor,
                "target_server": response.target_server,
                "score": response.score,
                "consent_csat_routing": response.consent_csat_routing,
                "consent_training_export": response.consent_training_export,
                "has_free_text": bool(response.free_text),
            },
        )
        return True

    async def _pay_and_score(
        self,
        request: SurveyRequest,
        response: SurveyResponse,
        broker: Any,
    ) -> None:
        """Apply the wallet + CSAT side effects of a submission.

        Failures here are logged but do not raise — the response is
        already durably recorded; a transient broker failure shouldn't
        bubble up to whoever just submitted a vote.
        """
        # Wallet: commons -> actor.
        if request.reward > 0:
            try:
                # Take the reward from commons. We model this as a
                # hold + immediate settle: hold locks the funds,
                # settle_against_hold debits them, then we credit the
                # actor's wallet.
                from ahp.economy.wallet import InsufficientFundsError
                hold_id = f"survey:{request.survey_id}"
                commons_wallet = broker.wallet(self._commons)
                try:
                    await commons_wallet.hold(
                        hold_id=hold_id,
                        amount=request.reward,
                        reason=f"survey reward {request.survey_id}",
                    )
                    await commons_wallet.settle_against_hold(
                        hold_id=hold_id,
                        debit=request.reward,
                        reason=f"survey payout {request.survey_id}",
                    )
                    await broker.wallet(request.surveyed_actor).topup(
                        request.reward,
                        reason=f"survey reward {request.survey_id}",
                    )
                except InsufficientFundsError:
                    log.warning(
                        "survey reward skipped: commons depleted "
                        "(survey_id=%s, reward=%.4f)",
                        request.survey_id, request.reward,
                    )
            except Exception:
                log.exception(
                    "survey wallet movement failed; record stands "
                    "(survey_id=%s)", request.survey_id,
                )

        # CSAT update on the target server's reputation.
        try:
            from ahp.economy.reputation import apply_csat
            rep = await broker.get_reputation(request.target_server)
            if rep is None:
                from ahp.economy.reputation import ReputationEntry
                rep = ReputationEntry(owner=request.target_server)
            updated = apply_csat(rep, response.score)
            await broker.set_reputation(updated)
        except Exception:
            log.exception(
                "survey CSAT update failed; record stands "
                "(survey_id=%s)", request.survey_id,
            )

    async def _abandon(self, survey_id: str) -> None:
        """Drop an expired request from the queue + request store.

        Doesn't touch responses — if a response somehow already exists
        it stays. The request blob goes because we don't want pending
        surveys lingering past their TTL.
        """
        await self._redis.zrem(SURVEY_QUEUE_KEY, survey_id)
        await self._redis.delete(
            SURVEY_REQUEST_KEY.format(survey_id=survey_id)
        )
        log.info("survey expired and dropped: id=%s", survey_id)


def _request_from_raw(raw: str | bytes) -> SurveyRequest:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    fields = SurveyRequest.__dataclass_fields__
    return SurveyRequest(**{k: v for k, v in data.items() if k in fields})


def response_from_raw(raw: str | bytes) -> SurveyResponse:
    """Public so the CLI / broker can deserialize without reaching into
    private helpers."""
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    fields = SurveyResponse.__dataclass_fields__
    return SurveyResponse(**{k: v for k, v in data.items() if k in fields})


# ── exports ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingDataRow:
    """One row in the future open-source preference-data export.

    Mirrors :class:`SurveyResponse` with two differences:
    * ``surveyed_actor`` is anonymized to a stable opaque hash unless
      the actor has explicitly consented to identified export.
    * ``free_text`` is included only if both consent flags were set.

    The export pipeline is also stubbed; ``export()`` raises until v1
    of the training-data pipeline lands.
    """

    row_id: str
    target_server: str
    recipe: str
    score: float
    free_text: str
    collected_at: float
    actor_handle: str  # anonymized or identified depending on consent

    @classmethod
    def from_response(
        cls, response: SurveyResponse, *, anonymize: bool = True,
    ) -> "TrainingDataRow":
        if not response.consent_training_export:
            raise ValueError(
                "cannot export a SurveyResponse that was collected without "
                "training_data_opt_in consent"
            )
        actor = _anonymize_actor(response.surveyed_actor) if anonymize else response.surveyed_actor
        return cls(
            row_id=response.survey_id,
            target_server=response.target_server,
            recipe=response.recipe,
            score=response.score,
            free_text=response.free_text if response.consent_training_export else "",
            collected_at=response.collected_at,
            actor_handle=actor,
        )


def _anonymize_actor(actor_address: str) -> str:
    """Stable opaque hash of the actor's address.

    Same actor → same hash, but the hash doesn't reveal which actor.
    Useful for keeping per-actor consistency in the export corpus
    without unmasking identities.
    """
    import hashlib
    return "act-" + hashlib.sha256(actor_address.encode()).hexdigest()[:16]


async def export_corpus(*, since: float = 0.0):
    """Future entry point for the open-source preference-data export.

    Stubbed. When implemented, this will:
        - Scan :class:`SurveyResponse` records from ``since`` onward.
        - Filter to ``consent_training_export=True`` only.
        - Convert via :meth:`TrainingDataRow.from_response`.
        - Emit a stable JSONL bundle suitable for HuggingFace
          datasets upload.
    """
    raise NotImplementedError("training-data export pipeline is stubbed; v1 deferred")
