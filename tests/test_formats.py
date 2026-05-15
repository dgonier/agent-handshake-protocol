"""Tests for the Format registry and recipe coverage."""

from __future__ import annotations

import pytest

from ahp.adapters.formats import (
    FORMATS,
    Format,
    FormatNotFoundError,
    get_format,
    list_formats,
)
from ahp.adapters.prompts import RECIPES


def test_at_least_15_recipes():
    """Sanity floor: we promised the user a richer recipe library."""
    assert len(RECIPES) >= 15


def test_at_least_11_formats():
    """We promised at least the documented format set."""
    assert len(FORMATS) >= 11


@pytest.mark.parametrize("fmt", list(FORMATS.values()), ids=lambda f: f.name)
def test_every_format_recipes_resolve(fmt: Format):
    """Every format's round1/round2/closing recipe key must be registered."""
    for key in (fmt.round1_recipe, fmt.round2_recipe, fmt.closing_recipe):
        if key is None:
            continue
        assert key in RECIPES, f"recipe {key!r} referenced by {fmt.name} missing"


def test_force_one_formats_match_expected():
    """interview-me, teach, interrogate are 1-on-1 by design."""
    one_on_one = {f.name for f in FORMATS.values() if f.count_strategy == "force_one"}
    assert one_on_one == {"interview-me", "teach", "interrogate"}


def test_skip_round2_only_on_interview_yall():
    """interview-yall is the only format that skips round 2 by design."""
    skips = {f.name for f in FORMATS.values() if f.round2_kind == "skip"}
    assert skips == {"interview-yall"}


def test_lookup_round_trip():
    f = get_format("debate")
    assert isinstance(f, Format)
    assert f.role == "adversarial"


def test_unknown_format_raises():
    with pytest.raises(FormatNotFoundError):
        get_format("does-not-exist")


def test_list_sorted_by_name():
    names = [f.name for f in list_formats()]
    assert names == sorted(names)
