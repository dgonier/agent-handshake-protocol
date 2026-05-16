"""Post-hoc CSAT surveys — stubbed; integration shape only.

The CSAT (consumer satisfaction) score in :class:`ReputationEntry` is
a separate dimension from system-observed reputation. It comes from
asking the consuming actor — agent or human — *after* an interaction
how useful the response actually was.

This module defines the public surface area that the rest of the
broker codes against. The actual dispatch loop, queue persistence,
and per-row training-data export are deliberately not implemented yet
— they're significant operational work and the protocol can ship
without them. Reputation scores stay populated by settlement
verdicts; CSAT scores stay at their neutral default (0.5) until the
survey loop is implemented.

When the survey loop is built, the contract here is:

* Surveys are *opt-in*. Every actor has three consent flags on their
  :class:`~ahp.broker.ServerMeta` profile: ``survey_opt_in``,
  ``csat_routing_opt_in``, ``training_data_opt_in``.
* Survey responses are paid out of the commons pool. Reward amount
  is configurable per recipe.
* Every recorded :class:`SurveyResponse` carries a snapshot of the
  consent flags that were active when it was collected. Consent
  changes are *not retroactive*: a row collected with
  ``training_data_opt_in=True`` remains in the corpus even if the
  actor flips that flag off later. (GDPR-style deletion requests are
  handled separately by tagging-then-rewriting the export.)
* The training-data export pipeline filters by
  ``consent_training_export=True``. Anyone who never opted in never
  appears.

Sample rate scales with stakes. A 0.1-credit interaction probably
doesn't warrant a survey; a 10-credit one does. The broker can
probabilistically sample so high-value interactions are surveyed
~always and low-value ones rarely.

Anti-gaming defenses planned for the real implementation:

* Surveying actor's own track record matters: outlier responders
  (always 5/5 or always 1/5) get their survey weight down-weighted.
* Reward comes from commons, not from the surveyed server — so
  the server can't bribe the surveyor to inflate scores.

When the v1 surveys ship, this module is the public interface
that callers should already have been importing.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal


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


# ── interface stubs ───────────────────────────────────────────────────


class SurveyQueue:
    """Broker-side queue that schedules and dispatches surveys.

    Stub. When this ships, the queue will:
        - Persist :class:`SurveyRequest`-s in Redis under
          ``ahp:survey:queue`` keyed by ``dispatch_at``.
        - Read its consent snapshot from the surveyed actor's
          :class:`ServerMeta` at queue time.
        - Fire dispatch as ``Code.HUMAN_OBSERVE`` messages targeting
          the surveyed actor's address.
        - On response, credit the actor from the commons pool by
          ``request.reward`` and record :class:`SurveyResponse`.
        - Update the target server's CSAT via :func:`apply_csat`.
        - Log a :class:`~ahp.audit.AuditEvent` per state change.
    """

    def __init__(self, *args, **kwargs):
        self._stub = True

    async def enqueue(self, request: SurveyRequest) -> None:
        raise NotImplementedError("SurveyQueue is stubbed; v1 deferred")

    async def fire_due(self) -> int:
        raise NotImplementedError("SurveyQueue is stubbed; v1 deferred")

    async def submit_response(self, response: SurveyResponse) -> None:
        raise NotImplementedError("SurveyQueue is stubbed; v1 deferred")


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
