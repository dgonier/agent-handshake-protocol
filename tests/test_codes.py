"""Tests for the Code constants and helpers."""

from __future__ import annotations

import pytest

from ahp.core.codes import Code


def test_codes_are_namespaced_strings():
    for code in Code.all():
        assert isinstance(code, str)
        assert "." in code, code


def test_families():
    families = {Code.family(c) for c in Code.all()}
    assert {"interview", "adversarial", "collaborative", "human", "error"} <= families


def test_family_extraction():
    assert Code.family(Code.INTERVIEW_TEXT) == "interview"
    assert Code.family(Code.ADVERSARIAL_DEBATE) == "adversarial"
    assert Code.family(Code.HUMAN_HALT) == "human"
    assert Code.family(Code.ERROR_TIMEOUT) == "error"


def test_family_rejects_empty():
    with pytest.raises(ValueError):
        Code.family("")


def test_is_error():
    assert Code.is_error(Code.ERROR_MALFORMED)
    assert Code.is_error(Code.ERROR_TIMEOUT)
    assert not Code.is_error(Code.INTERVIEW_TEXT)
    assert not Code.is_error(Code.COLLAB_CONSENSUS)


def test_matches_exact():
    assert Code.matches("interview.text", "interview.text")
    assert not Code.matches("interview.text", "interview.schema")


def test_matches_family_glob():
    assert Code.matches("interview.text", "interview.*")
    assert Code.matches("interview.embeddings", "interview.*")
    assert not Code.matches("adversarial.debate", "interview.*")


def test_matches_universal():
    for code in Code.all():
        assert Code.matches(code, "*")


def test_matches_no_partial_prefix():
    """`inter.*` should not match `interview.text`."""
    assert not Code.matches("interview.text", "inter.*")


def test_no_duplicate_code_values():
    codes = [v for k, v in vars(Code).items() if k.isupper() and isinstance(v, str)]
    assert len(codes) == len(set(codes)), "duplicate code values detected"
