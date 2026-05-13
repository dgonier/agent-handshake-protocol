"""Tests for the Message envelope."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import LIFECYCLE_TTL, VALID_VERBS, Message
from ahp.core.pattern import AddressPattern


SOURCE_URI = "tifin.collaborative.finance.equities.s.session.alice"
TARGET_URI = "tifin.adversarial.finance.equities.s.session.bob"


def _src(uri: str = SOURCE_URI) -> AgentAddress:
    return AgentAddress.parse(uri)


def _tgt(uri: str = TARGET_URI) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── construction ────────────────────────────────────────────────────────


def test_minimal_construction_assigns_defaults():
    msg = Message(
        source=_src(), target=_tgt(), verb="SEND",
        code=Code.INTERVIEW_TEXT, thread="thread::topic", body="hi",
    )
    assert msg.message_id  # uuid populated
    assert msg.timestamp.tzinfo is timezone.utc
    # session lifecycle → 1h ttl
    assert msg.ttl == LIFECYCLE_TTL["session"]


def test_ttl_explicit_overrides_default():
    msg = Message(
        source=_src(), target=_tgt(), verb="SEND",
        code=Code.INTERVIEW_TEXT, thread="t", body=None, ttl=42,
    )
    assert msg.ttl == 42


def test_ttl_rejects_negative():
    with pytest.raises(ValueError):
        Message(
            source=_src(), target=_tgt(), verb="SEND",
            code=Code.INTERVIEW_TEXT, thread="t", body=None, ttl=-1,
        )


def test_ttl_zero_for_ephemeral_source():
    src = AgentAddress.parse("o.r.d.sd.s.ephemeral.i")
    msg = Message(
        source=src, target=_tgt(), verb="SEND",
        code=Code.INTERVIEW_TEXT, thread="t", body=None,
    )
    assert msg.ttl == 0


def test_naive_timestamp_normalized_to_utc():
    naive = datetime(2026, 1, 1, 12, 0, 0)
    msg = Message(
        source=_src(), target=_tgt(), verb="SEND",
        code=Code.INTERVIEW_TEXT, thread="t", body=None,
        timestamp=naive,
    )
    assert msg.timestamp.tzinfo is timezone.utc


# ── verb / target consistency ───────────────────────────────────────────


def test_all_verbs_accepted():
    for verb in VALID_VERBS:
        if verb in {"SEND", "SEND-GET"}:
            target = _tgt()
        else:
            target = AddressPattern.parse("*.*.*.*.*.*.*")
        Message(
            source=_src(), target=target, verb=verb,
            code=Code.INTERVIEW_TEXT, thread="t", body=None,
        )


def test_invalid_verb_rejected():
    with pytest.raises(ValueError, match="invalid verb"):
        Message(
            source=_src(), target=_tgt(), verb="YOLO",
            code=Code.INTERVIEW_TEXT, thread="t", body=None,
        )


def test_point_to_point_rejects_pattern_target():
    pat = AddressPattern.parse("*.*.*.*.*.*.*")
    with pytest.raises(ValueError, match="requires an AgentAddress"):
        Message(
            source=_src(), target=pat, verb="SEND",
            code=Code.INTERVIEW_TEXT, thread="t", body=None,
        )


def test_is_broadcast_flag():
    addr_msg = Message(
        source=_src(), target=_tgt(), verb="SEND",
        code=Code.INTERVIEW_TEXT, thread="t", body=None,
    )
    pat_msg = Message(
        source=_src(),
        target=AddressPattern.parse("*.*.*.*.*.*.*"),
        verb="CAST",
        code=Code.INTERVIEW_TEXT, thread="t", body=None,
    )
    assert not addr_msg.is_broadcast
    assert pat_msg.is_broadcast


def test_expects_response_flag():
    base = dict(source=_src(), code=Code.INTERVIEW_TEXT, thread="t", body=None)
    assert Message(target=_tgt(), verb="SEND-GET", **base).expects_response
    assert Message(
        target=AddressPattern.parse("*.*.*.*.*.*.*"),
        verb="CAST-GET", **base,
    ).expects_response
    assert not Message(target=_tgt(), verb="SEND", **base).expects_response


# ── validation: code / thread ───────────────────────────────────────────


@pytest.mark.parametrize("field,bad", [
    ("code", ""),
    ("thread", ""),
])
def test_empty_strings_rejected(field, bad):
    kwargs = dict(
        source=_src(), target=_tgt(), verb="SEND",
        code=Code.INTERVIEW_TEXT, thread="t", body=None,
    )
    kwargs[field] = bad
    with pytest.raises(ValueError):
        Message(**kwargs)


def test_source_must_be_address():
    with pytest.raises(TypeError):
        Message(
            source="not-an-address",  # type: ignore[arg-type]
            target=_tgt(), verb="SEND",
            code=Code.INTERVIEW_TEXT, thread="t", body=None,
        )


# ── serialization round-trip ────────────────────────────────────────────


def test_to_from_dict_point_to_point():
    original = Message(
        source=_src(), target=_tgt(), verb="SEND-GET",
        code=Code.INTERVIEW_SCHEMA, thread="thread::abc",
        body={"q": "what?"}, reply_to=None,
    )
    restored = Message.from_dict(original.to_dict())
    assert restored.source == original.source
    assert restored.target == original.target
    assert restored.verb == original.verb
    assert restored.code == original.code
    assert restored.thread == original.thread
    assert restored.body == original.body
    assert restored.ttl == original.ttl
    assert restored.message_id == original.message_id
    assert restored.timestamp == original.timestamp


def test_to_from_dict_broadcast():
    pat = AddressPattern.parse("*.adversarial.finance.*.s.*.*")
    original = Message(
        source=_src(), target=pat, verb="CAST-GET",
        code=Code.ADVERSARIAL_DEBATE, thread="thread::abc",
        body="argue",
    )
    restored = Message.from_dict(original.to_dict())
    assert isinstance(restored.target, AddressPattern)
    assert restored.target == pat
    assert restored.is_broadcast


def test_from_dict_infers_target_kind_from_verb_when_missing():
    src = _src()
    tgt = _tgt()
    raw = {
        "source": str(src), "target": str(tgt), "verb": "SEND",
        "code": Code.INTERVIEW_TEXT, "thread": "t", "body": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    msg = Message.from_dict(raw)
    assert isinstance(msg.target, AgentAddress)
    assert msg.target == tgt


def test_from_dict_broadcast_inference():
    src = _src()
    raw = {
        "source": str(src),
        "target": "*.adversarial.*.*.*.*.*",
        "verb": "CAST", "code": Code.ADVERSARIAL_DEBATE, "thread": "t",
        "body": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    msg = Message.from_dict(raw)
    assert isinstance(msg.target, AddressPattern)
