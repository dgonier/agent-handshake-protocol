"""Tests for the CompatibilityMatrix."""

from __future__ import annotations

import pytest

from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.compatibility import CompatibilityMatrix


def _addr(accept: str) -> AgentAddress:
    return AgentAddress.parse(f"o.r.d.sd.{accept}.session.i")


@pytest.fixture
def matrix() -> CompatibilityMatrix:
    return CompatibilityMatrix()


# ── tier lookup ─────────────────────────────────────────────────────────


def test_required_tiers_known_codes(matrix: CompatibilityMatrix):
    assert matrix.required_tiers(Code.INTERVIEW_TEXT) == {"s"}
    assert matrix.required_tiers(Code.INTERVIEW_SCHEMA) == {"j"}
    assert matrix.required_tiers(Code.INTERVIEW_EMBEDDINGS) == {"b", "e"}
    assert matrix.required_tiers(Code.COLLAB_MERGE) == {"b", "e"}


def test_required_tiers_unknown_falls_back(matrix: CompatibilityMatrix):
    assert matrix.required_tiers("custom.unknown.code") == {"s"}
    assert not matrix.is_known("custom.unknown.code")
    assert matrix.is_known(Code.INTERVIEW_TEXT)


def test_required_tiers_is_a_copy(matrix: CompatibilityMatrix):
    """Callers shouldn't be able to mutate the registry by mutating the result."""
    a = matrix.required_tiers(Code.INTERVIEW_TEXT)
    a.add("z")
    b = matrix.required_tiers(Code.INTERVIEW_TEXT)
    assert "z" not in b


def test_every_defined_code_has_requirements():
    for code in Code.all():
        assert code in CompatibilityMatrix.CODE_REQUIREMENTS, code


# ── routing decisions ───────────────────────────────────────────────────


def test_string_target_accepts_text_code(matrix: CompatibilityMatrix):
    src = _addr("s")
    assert matrix.can_route(src, _addr("s"), Code.INTERVIEW_TEXT)
    assert matrix.can_route(src, _addr("sj"), Code.INTERVIEW_TEXT)
    assert matrix.can_route(src, _addr("sjbe"), Code.INTERVIEW_TEXT)


def test_string_only_target_rejects_schema_code(matrix: CompatibilityMatrix):
    """interview.schema needs JSON — a string-only target can't receive it."""
    src = _addr("s")
    assert not matrix.can_route(src, _addr("s"), Code.INTERVIEW_SCHEMA)
    assert matrix.can_route(src, _addr("j"), Code.INTERVIEW_SCHEMA)
    assert matrix.can_route(src, _addr("sj"), Code.INTERVIEW_SCHEMA)


def test_embeddings_satisfied_by_either_tier(matrix: CompatibilityMatrix):
    src = _addr("e")
    assert matrix.can_route(src, _addr("b"), Code.INTERVIEW_EMBEDDINGS)
    assert matrix.can_route(src, _addr("e"), Code.INTERVIEW_EMBEDDINGS)
    assert matrix.can_route(src, _addr("be"), Code.INTERVIEW_EMBEDDINGS)
    assert not matrix.can_route(src, _addr("s"), Code.INTERVIEW_EMBEDDINGS)
    assert not matrix.can_route(src, _addr("j"), Code.INTERVIEW_EMBEDDINGS)
    assert not matrix.can_route(src, _addr("sj"), Code.INTERVIEW_EMBEDDINGS)


def test_human_codes_only_require_string(matrix: CompatibilityMatrix):
    src = _addr("s")
    for human_code in [
        Code.HUMAN_QUERY, Code.HUMAN_OBSERVE, Code.HUMAN_INTERVENE,
        Code.HUMAN_APPROVE, Code.HUMAN_EXPLAIN, Code.HUMAN_HALT,
    ]:
        assert matrix.can_route(src, _addr("s"), human_code)
        assert not matrix.can_route(src, _addr("j"), human_code)


def test_errors_universally_routable_to_string_targets(matrix: CompatibilityMatrix):
    src = _addr("s")
    error_codes = [v for k, v in vars(Code).items()
                   if k.startswith("ERROR_") and isinstance(v, str)]
    assert error_codes
    for ec in error_codes:
        assert matrix.can_route(src, _addr("s"), ec)


# ── filtering ───────────────────────────────────────────────────────────


def test_filter_targets(matrix: CompatibilityMatrix):
    src = _addr("s")
    targets = [_addr("s"), _addr("j"), _addr("sj"), _addr("be")]
    filtered = matrix.filter_targets(src, targets, Code.INTERVIEW_SCHEMA)
    # Only targets that accept "j" survive
    assert _addr("s") not in filtered
    assert _addr("be") not in filtered
    assert _addr("j") in filtered
    assert _addr("sj") in filtered


def test_filter_targets_empty(matrix: CompatibilityMatrix):
    src = _addr("s")
    targets = [_addr("s"), _addr("sj")]
    # interview.embeddings needs b or e; none of these have it
    assert matrix.filter_targets(src, targets, Code.INTERVIEW_EMBEDDINGS) == []


def test_filter_targets_unknown_code_treats_as_string(matrix: CompatibilityMatrix):
    src = _addr("s")
    targets = [_addr("s"), _addr("j"), _addr("be")]
    filtered = matrix.filter_targets(src, targets, "custom.x")
    assert _addr("s") in filtered
    assert _addr("j") not in filtered
    assert _addr("be") not in filtered
