"""TeacherAgent — agent-as-judge that writes its findings to a knowledge graph.

The Teacher is a protocol participant whose job is to *evaluate* other
agents and persist the evaluations. Two interaction modes:

* ``teacher.judge`` — inbound message body is something to score against
  a :class:`Rubric`. The Teacher runs its ``judge_fn`` (sync or async),
  builds a :class:`Judgement`, writes a :class:`KGNode` to its KG
  resource, and replies with the judgement JSON.

* ``teacher.survey`` — inbound message body is a survey spec
  (``{"target": <pattern>, "code": <code>, "prompt": "..."}``). The
  Teacher broadcasts the prompt to the matching agents, collects
  responses, judges each one, and persists a per-respondent
  :class:`Judgement` linked to the surveyed agent.

The agent is intentionally LLM-agnostic — the ``judge_fn`` is a pure
callable. Plug in a rubric-driven LangChain chain, a deterministic
scorer, or a HEXIS-side judge by handing in the corresponding callable.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from ahp.adapters.base import AHPAgent
from ahp.adapters.capability import AgentProfile
from ahp.adapters.knowledge_graph import (
    KGEdge,
    KGNode,
    KnowledgeGraphBackend,
    build_kg_backend,
    node_id_for_agent,
    node_id_for_judgement,
    node_id_for_rubric,
)
from ahp.adapters.resources import ResourceRegistry
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


log = logging.getLogger(__name__)


# ── rubric / judgement shapes ─────────────────────────────────────────


@dataclass(frozen=True)
class Criterion:
    """A single scoring criterion in a :class:`Rubric`."""

    name: str
    description: str = ""
    weight: float = 1.0
    """Relative weight in the overall composite score."""


@dataclass(frozen=True)
class Rubric:
    """A named scoring rubric.

    The Teacher passes a rubric to its ``judge_fn`` along with the
    subject. The judge function decides how to interpret the criteria —
    they can be LLM-evaluated, regex-checked, or computed via HEXIS.
    """

    name: str
    description: str = ""
    criteria: tuple[Criterion, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Rubric":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            criteria=tuple(
                Criterion(
                    name=str(c["name"]),
                    description=str(c.get("description", "")),
                    weight=float(c.get("weight", 1.0)),
                )
                for c in data.get("criteria", [])
            ),
        )


@dataclass(frozen=True)
class Judgement:
    """The Teacher's verdict on a single piece of work.

    * ``per_criterion`` is a ``{name: score_in_[0,1]}`` map. Missing
      criteria get a zero contribution to the composite.
    * ``composite`` is the weighted average over the rubric's
      criteria. Computed automatically by :meth:`compose` if the
      ``judge_fn`` doesn't supply one.
    * ``rationale`` is the human-readable explanation that gets stored
      on the KG node.
    """

    subject: str
    rubric: str
    per_criterion: Mapping[str, float] = field(default_factory=dict)
    composite: float = 0.0
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    @classmethod
    def compose(
        cls,
        subject: str,
        rubric: Rubric,
        per_criterion: Mapping[str, float],
        *,
        rationale: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "Judgement":
        weights = {c.name: c.weight for c in rubric.criteria}
        total_weight = sum(weights.get(k, 0.0) for k in per_criterion) or 1.0
        composite = sum(
            per_criterion.get(k, 0.0) * weights.get(k, 0.0)
            for k in per_criterion
        ) / total_weight
        return cls(
            subject=subject,
            rubric=rubric.name,
            per_criterion=dict(per_criterion),
            composite=composite,
            rationale=rationale,
            metadata=dict(metadata or {}),
        )

    def to_props(self) -> dict[str, Any]:
        """Flatten for storage as :class:`KGNode` ``props``."""
        return {
            "subject": self.subject,
            "rubric": self.rubric,
            "composite": float(self.composite),
            "rationale": self.rationale,
            "per_criterion_json": json.dumps(dict(self.per_criterion)),
            "metadata_json": json.dumps(dict(self.metadata)),
            "ts": float(self.ts),
        }


# ── judge function signature ──────────────────────────────────────────


JudgeFn = Callable[
    [Rubric, Any],
    "Judgement | Awaitable[Judgement] | Mapping[str, float] | Awaitable[Mapping[str, float]]",
]
"""``judge_fn(rubric, subject_body) -> Judgement | per-criterion scores``.

If the callable returns a plain ``{criterion_name: score}`` mapping,
:meth:`Judgement.compose` is applied automatically using the rubric's
weights. Returning a full :class:`Judgement` lets the judge attach its
own composite + rationale.
"""


# ── teacher agent ─────────────────────────────────────────────────────


class TeacherAgent(AHPAgent):
    """Agent-as-judge that persists judgements into a knowledge graph.

    Lifecycle is identical to any other :class:`AHPAgent`. Override
    points:

    * ``judge_fn`` — the scoring callable. Sync or async. Returns either
      a full :class:`Judgement` or a per-criterion mapping.
    * ``rubric`` — the default :class:`Rubric` applied when an inbound
      message doesn't specify one. Inbound messages can override via the
      body's ``rubric`` key.
    * ``kg_backend`` — pre-bound :class:`KnowledgeGraphBackend`. If
      omitted, the factory builds one from the resource registry via
      :func:`~ahp.adapters.knowledge_graph.build_kg_backend`.
    """

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        *,
        rubric: Rubric,
        judge_fn: JudgeFn | None = None,
        kg_backend: KnowledgeGraphBackend | None = None,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(address, engine, metadata=metadata, **kwargs)
        self.rubric = rubric
        self.judge_fn = judge_fn or _default_zero_judge
        self.kg = kg_backend if kg_backend is not None else _make_default_kg()
        # Self-node + rubric-node materialize on registration so the
        # graph is queryable from the moment the Teacher boots.
        self._ensure_self_nodes()

    # ── inbound dispatch ──────────────────────────────────────────────

    async def handle_message(self, message: Message) -> Message | None:
        code = message.code
        if code == Code.TEACHER_JUDGE:
            judgement = await self._judge_one(message.body, message.source)
            self._persist_judgement(judgement, subject=message.source)
            return self._reply(message, judgement)
        if code == Code.TEACHER_SURVEY:
            findings = await self._run_survey(message)
            return self._reply(
                message,
                {"survey": findings, "rubric": self.rubric.name},
            )
        if code == Code.TEACHER_OBSERVE:
            # External party hands us a pre-formed observation to write.
            self._persist_observation(message.body, source=message.source)
            return self._reply(message, {"ok": True})
        return None

    # ── core operations ───────────────────────────────────────────────

    async def _judge_one(
        self, body: Any, subject: AgentAddress,
    ) -> Judgement:
        """Run the judge_fn against the body. Returns a :class:`Judgement`."""
        rubric = self._rubric_from_body(body, default=self.rubric)
        subject_payload = (
            body.get("subject", body) if isinstance(body, Mapping) else body
        )
        result = self.judge_fn(rubric, subject_payload)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Judgement):
            return result
        if isinstance(result, Mapping):
            return Judgement.compose(
                subject=str(subject),
                rubric=rubric,
                per_criterion={str(k): float(v) for k, v in result.items()},
            )
        raise TypeError(
            f"judge_fn returned {type(result).__name__}; "
            f"expected Judgement or Mapping[str, float]"
        )

    async def _run_survey(self, message: Message) -> list[dict[str, Any]]:
        spec = message.body if isinstance(message.body, Mapping) else {}
        target = spec.get("target")
        prompt = spec.get("prompt", "")
        code = spec.get("code", Code.COLLAB_REASON)
        if target is None:
            raise ValueError(
                "teacher.survey body must include a 'target' pattern"
            )
        pattern = (
            target if isinstance(target, AddressPattern)
            else AddressPattern.parse(str(target))
        )
        replies = await self.engine.handle(
            Message(
                source=self.address,
                target=pattern,
                verb="CAST-GET",
                code=code,
                thread=message.thread,
                body=prompt,
            ),
        )
        return await self._judge_replies(replies, message.thread)

    async def _judge_replies(
        self, replies: Any, thread: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for reply in _iter_replies(replies):
            respondent = getattr(reply, "source", None)
            body = getattr(reply, "body", reply)
            if respondent is None:
                continue
            judgement = await self._judge_one(body, respondent)
            self._persist_judgement(judgement, subject=respondent)
            rows.append({
                "respondent": str(respondent),
                "composite": float(judgement.composite),
                "rationale": judgement.rationale,
            })
        return rows

    # ── KG persistence ────────────────────────────────────────────────

    def _ensure_self_nodes(self) -> None:
        self.kg.write_node(KGNode(
            id=node_id_for_agent(self.address),
            kind="Agent",
            label=str(self.address),
            props={"role": "teacher"},
        ))
        self.kg.write_node(KGNode(
            id=node_id_for_rubric(self.rubric.name),
            kind="Rubric",
            label=self.rubric.name,
            props={
                "description": self.rubric.description,
                "criteria_json": json.dumps([
                    asdict(c) for c in self.rubric.criteria
                ]),
            },
        ))

    def _persist_judgement(
        self, judgement: Judgement, *, subject: AgentAddress,
    ) -> None:
        # Materialize subject + rubric nodes if we haven't seen them
        # before; MERGE-style semantics live in the backend.
        subject_id = node_id_for_agent(subject)
        self.kg.write_node(KGNode(
            id=subject_id, kind="Agent", label=str(subject),
        ))
        rubric_id = node_id_for_rubric(judgement.rubric)
        self.kg.write_node(KGNode(
            id=rubric_id, kind="Rubric", label=judgement.rubric,
        ))
        judgement_id = node_id_for_judgement(
            self.address, subject, judgement.ts,
        )
        self.kg.write_node(KGNode(
            id=judgement_id,
            kind="Judgement",
            label=f"{judgement.rubric}({judgement.composite:.2f})",
            props=judgement.to_props(),
        ))
        self.kg.write_edge(KGEdge(
            source_id=node_id_for_agent(self.address),
            target_id=judgement_id,
            kind="ISSUED",
        ))
        self.kg.write_edge(KGEdge(
            source_id=judgement_id,
            target_id=subject_id,
            kind="ABOUT",
        ))
        self.kg.write_edge(KGEdge(
            source_id=judgement_id,
            target_id=rubric_id,
            kind="USES_RUBRIC",
        ))

    def _persist_observation(self, body: Any, *, source: AgentAddress) -> None:
        payload = body if isinstance(body, Mapping) else {"text": str(body)}
        node = KGNode(
            id=f"observation::{source}::{time.time():.6f}",
            kind="Observation",
            label=str(payload.get("label", "")),
            props={
                "text": str(payload.get("text", payload)),
                "ts": time.time(),
            },
        )
        self.kg.write_node(node)
        self.kg.write_node(KGNode(
            id=node_id_for_agent(source), kind="Agent", label=str(source),
        ))
        self.kg.write_edge(KGEdge(
            source_id=node_id_for_agent(source),
            target_id=node.id,
            kind="REPORTED",
        ))

    # ── helpers ───────────────────────────────────────────────────────

    def _reply(self, request: Message, payload: Any) -> Message | None:
        if not request.expects_response:
            return None
        body: Any
        if isinstance(payload, Judgement):
            body = {
                "subject": payload.subject,
                "rubric": payload.rubric,
                "composite": payload.composite,
                "per_criterion": dict(payload.per_criterion),
                "rationale": payload.rationale,
                "ts": payload.ts,
            }
        else:
            body = payload
        return Message(
            source=self.address,
            target=request.source,
            verb="SEND",
            code=request.code,
            thread=request.thread,
            body=body,
        )

    @staticmethod
    def _rubric_from_body(body: Any, *, default: Rubric) -> Rubric:
        if isinstance(body, Mapping) and "rubric" in body:
            rub = body["rubric"]
            if isinstance(rub, Rubric):
                return rub
            if isinstance(rub, Mapping):
                return Rubric.from_dict(rub)
        return default

    # ── factory ───────────────────────────────────────────────────────

    @classmethod
    def from_profile(
        cls,
        address: AgentAddress,
        engine: ProtocolEngine,
        profile: AgentProfile,
        *,
        rubric: Rubric,
        judge_fn: JudgeFn | None = None,
        resources: ResourceRegistry | None = None,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> "TeacherAgent":
        """Build a TeacherAgent, resolving its KG backend by address.

        When ``resources`` is supplied, the registry is searched for a
        ``kg``-kind resource whose ``allowed_for`` matches ``address``.
        Otherwise the constructed agent falls back to an in-memory KG.
        """
        if resources is not None:
            kg_backend = build_kg_backend(resources, address)
        else:
            kg_backend = None
        return cls(
            address=address,
            engine=engine,
            rubric=rubric,
            judge_fn=judge_fn,
            kg_backend=kg_backend,
            metadata=metadata,
            **kwargs,
        )


# ── module helpers ────────────────────────────────────────────────────


def _default_zero_judge(rubric: Rubric, _subject: Any) -> Judgement:
    """Stub judge that returns zero on every criterion.

    Useful as a placeholder so a Teacher can boot without an external
    scorer wired. Real deployments pass a callable.
    """
    return Judgement.compose(
        subject="<unscored>",
        rubric=rubric,
        per_criterion={c.name: 0.0 for c in rubric.criteria},
        rationale="(default zero judge — no judge_fn configured)",
    )


def _make_default_kg() -> KnowledgeGraphBackend:
    from ahp.adapters.knowledge_graph import InMemoryKnowledgeGraph
    return InMemoryKnowledgeGraph()


def _iter_replies(replies: Any) -> list[Any]:
    """Normalize the engine's broadcast return value to a list of messages.

    ``ProtocolEngine.handle`` of a CAST-GET returns a list of Message-s
    in the common case; older shapes returned a dict keyed by address.
    Accept both — tests don't pin a specific shape.
    """
    if replies is None:
        return []
    if isinstance(replies, list):
        return replies
    if isinstance(replies, dict):
        return list(replies.values())
    return [replies]
