"""Tests for the TeacherAgent (agent-as-judge + KG persistence)."""

from __future__ import annotations

from typing import Any

import pytest

from ahp.adapters.knowledge_graph import (
    InMemoryKnowledgeGraph,
    node_id_for_agent,
    node_id_for_rubric,
)
from ahp.adapters.teacher_agent import (
    Criterion,
    Judgement,
    Rubric,
    TeacherAgent,
)
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message


def _addr(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


TEACHER = _addr("acme.teacher.finance.equities.s.session.judge1")
STUDENT = _addr("acme.adversarial.finance.equities.s.session.student1")
PEER = _addr("acme.adversarial.finance.equities.s.session.peer1")


def _rubric() -> Rubric:
    return Rubric(
        name="financial-reasoning",
        description="argument quality on equities research",
        criteria=(
            Criterion("evidence", "concrete data / citations", weight=2.0),
            Criterion("clarity", "is the argument easy to follow", weight=1.0),
        ),
    )


def _msg(
    code: str,
    body: Any,
    *,
    source: AgentAddress = STUDENT,
    target: AgentAddress = TEACHER,
    verb: str = "SEND-GET",
    reply: bool = True,
) -> Message:
    return Message(
        source=source, target=target, verb=verb,
        code=code, thread="thread::teacher::test", body=body,
        reply_to="ahp:reply:fake" if reply else None,
    )


# ── boot wiring ────────────────────────────────────────────────────────


async def test_teacher_writes_self_and_rubric_nodes_on_construct(stack):
    kg = InMemoryKnowledgeGraph()
    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        kg_backend=kg, heartbeat_interval=0,
    )
    assert teacher.kg.get_node(node_id_for_agent(TEACHER)) is not None
    rubric_node = teacher.kg.get_node(node_id_for_rubric("financial-reasoning"))
    assert rubric_node is not None
    assert "evidence" in rubric_node.props["criteria_json"]


async def test_default_judge_fn_returns_zero_scored_judgement(stack):
    kg = InMemoryKnowledgeGraph()
    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        kg_backend=kg, heartbeat_interval=0,
    )
    reply = await teacher.handle_message(
        _msg(Code.TEACHER_JUDGE, "Tesla looks fine"),
    )
    assert reply is not None
    assert reply.body["composite"] == 0.0
    assert "default zero judge" in reply.body["rationale"]


# ── teacher.judge dispatch ────────────────────────────────────────────


async def test_dict_returning_judge_fn_is_composed_with_weights(stack):
    kg = InMemoryKnowledgeGraph()

    def judge(rubric: Rubric, _body: Any) -> dict[str, float]:
        return {"evidence": 1.0, "clarity": 0.0}

    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        judge_fn=judge, kg_backend=kg, heartbeat_interval=0,
    )
    reply = await teacher.handle_message(
        _msg(Code.TEACHER_JUDGE, "Tesla has high P/E"),
    )
    # weighted average: (1.0*2 + 0.0*1) / (2+1) = 2/3
    assert reply.body["composite"] == pytest.approx(2 / 3)
    assert reply.body["rubric"] == "financial-reasoning"


async def test_async_judge_fn_is_awaited(stack):
    kg = InMemoryKnowledgeGraph()

    async def judge(rubric: Rubric, _body: Any) -> Judgement:
        return Judgement.compose(
            subject="async",
            rubric=rubric,
            per_criterion={"evidence": 0.5, "clarity": 0.5},
            rationale="reviewed by async judge",
        )

    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        judge_fn=judge, kg_backend=kg, heartbeat_interval=0,
    )
    reply = await teacher.handle_message(_msg(Code.TEACHER_JUDGE, "x"))
    assert reply.body["composite"] == pytest.approx(0.5)
    assert "async judge" in reply.body["rationale"]


async def test_judge_persists_judgement_to_kg(stack):
    kg = InMemoryKnowledgeGraph()

    def judge(_rubric, _body):
        return {"evidence": 1.0, "clarity": 1.0}

    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        judge_fn=judge, kg_backend=kg, heartbeat_interval=0,
    )
    await teacher.handle_message(_msg(Code.TEACHER_JUDGE, "great"))

    judgements = kg.list_nodes(kind="Judgement")
    assert len(judgements) == 1
    issued = kg.list_edges(source_id=node_id_for_agent(TEACHER), kind="ISSUED")
    assert len(issued) == 1
    about = kg.list_edges(target_id=node_id_for_agent(STUDENT), kind="ABOUT")
    assert len(about) == 1
    uses = kg.list_edges(
        target_id=node_id_for_rubric("financial-reasoning"),
        kind="USES_RUBRIC",
    )
    assert len(uses) == 1


async def test_judge_accepts_rubric_override_in_body(stack):
    kg = InMemoryKnowledgeGraph()

    def judge(rubric: Rubric, _body: Any) -> dict[str, float]:
        # Whatever rubric arrives, score all its criteria 1.0.
        return {c.name: 1.0 for c in rubric.criteria}

    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        judge_fn=judge, kg_backend=kg, heartbeat_interval=0,
    )
    custom = {
        "subject": "use this body",
        "rubric": {
            "name": "tiny",
            "criteria": [{"name": "ok", "weight": 1.0}],
        },
    }
    reply = await teacher.handle_message(_msg(Code.TEACHER_JUDGE, custom))
    assert reply.body["rubric"] == "tiny"
    assert reply.body["composite"] == pytest.approx(1.0)


async def test_silent_when_response_not_expected(stack):
    kg = InMemoryKnowledgeGraph()
    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        kg_backend=kg, heartbeat_interval=0,
    )
    silent = _msg(Code.TEACHER_JUDGE, "x", reply=False)
    silent_send = Message(
        source=silent.source, target=silent.target, verb="SEND",
        code=silent.code, thread=silent.thread, body=silent.body,
    )
    assert await teacher.handle_message(silent_send) is None
    # KG write still happens — persistence is not conditional on reply.
    assert kg.list_nodes(kind="Judgement")


# ── teacher.observe dispatch ──────────────────────────────────────────


async def test_observation_persisted_with_reporter_edge(stack):
    kg = InMemoryKnowledgeGraph()
    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        kg_backend=kg, heartbeat_interval=0,
    )
    body = {"label": "broker drift", "text": "saw a 3% gap on the quote feed"}
    reply = await teacher.handle_message(_msg(Code.TEACHER_OBSERVE, body))
    assert reply.body == {"ok": True}
    observations = kg.list_nodes(kind="Observation")
    assert len(observations) == 1
    edges = kg.list_edges(
        source_id=node_id_for_agent(STUDENT), kind="REPORTED",
    )
    assert len(edges) == 1


# ── teacher.survey dispatch ───────────────────────────────────────────


async def test_survey_judges_each_reply_and_persists(stack, monkeypatch):
    kg = InMemoryKnowledgeGraph()

    def judge(_rubric, body):
        # crude scorer: positive if the body mentions "evidence"
        score = 1.0 if isinstance(body, str) and "evidence" in body else 0.0
        return {"evidence": score, "clarity": score}

    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        judge_fn=judge, kg_backend=kg, heartbeat_interval=0,
    )

    canned = [
        Message(
            source=PEER, target=TEACHER, verb="SEND",
            code=Code.COLLAB_REASON, thread="thread::teacher::test",
            body="here is some evidence",
        ),
        Message(
            source=STUDENT, target=TEACHER, verb="SEND",
            code=Code.COLLAB_REASON, thread="thread::teacher::test",
            body="just an opinion",
        ),
    ]

    async def fake_handle(message: Message, **kwargs):
        return canned

    monkeypatch.setattr(stack.engine, "handle", fake_handle)

    spec = {
        "target": "acme.adversarial.*.*.*.*.*",
        "code": Code.COLLAB_REASON,
        "prompt": "argue for or against Tesla",
    }
    reply = await teacher.handle_message(_msg(Code.TEACHER_SURVEY, spec))
    findings = reply.body["survey"]
    assert len(findings) == 2
    by_respondent = {row["respondent"]: row["composite"] for row in findings}
    assert by_respondent[str(PEER)] == pytest.approx(1.0)
    assert by_respondent[str(STUDENT)] == pytest.approx(0.0)
    # KG: two Judgements + ABOUT edges for both respondents.
    assert len(kg.list_nodes(kind="Judgement")) == 2


async def test_survey_requires_target(stack):
    kg = InMemoryKnowledgeGraph()
    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        kg_backend=kg, heartbeat_interval=0,
    )
    with pytest.raises(ValueError, match="must include a 'target'"):
        await teacher.handle_message(_msg(Code.TEACHER_SURVEY, {"prompt": "x"}))


# ── unknown code is silently ignored ─────────────────────────────────


async def test_unknown_code_returns_none(stack):
    kg = InMemoryKnowledgeGraph()
    teacher = TeacherAgent(
        TEACHER, stack.engine, rubric=_rubric(),
        kg_backend=kg, heartbeat_interval=0,
    )
    assert await teacher.handle_message(_msg(Code.COLLAB_REASON, "x")) is None
