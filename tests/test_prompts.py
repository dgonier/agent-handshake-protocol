"""Tests for the fixed dialog prompt recipe library."""

from __future__ import annotations

import pytest

from ahp.adapters.prompts import (
    RECIPES,
    Recipe,
    RecipeNotFoundError,
    get_recipe,
    list_recipes,
    render,
)


def test_known_recipes_present():
    keys = set(RECIPES.keys())
    assert {
        "adversarial:debate-me",
        "adversarial:debate-others",
        "interview:open",
        "interview:probe",
        "collaborative:reason",
    }.issubset(keys)


def test_get_recipe_round_trip():
    r = get_recipe("adversarial", "debate-me")
    assert isinstance(r, Recipe)
    assert r.role == "adversarial"
    assert r.mode == "debate-me"


def test_unknown_recipe_raises():
    with pytest.raises(RecipeNotFoundError):
        get_recipe("adversarial", "no-such-mode")


def test_render_debate_me_includes_question_and_system():
    text = render(
        "adversarial", "debate-me",
        system="You hold the inflation view.",
        question="What caused the Big Bang?",
    )
    assert "inflation" in text
    assert "Big Bang" in text
    assert "3 short sentences" in text


def test_render_debate_others_excludes_self_slug():
    text = render(
        "adversarial", "debate-others",
        system="You hold view A.",
        question="Q?",
        self_slug="me",
        others=[
            {"slug": "me", "body": "my own argument"},
            {"slug": "you", "body": "another's argument"},
        ],
    )
    assert "another's argument" in text
    assert "my own argument" not in text


def test_render_interview_probe_uses_prior():
    text = render(
        "interview", "probe",
        system="You are the witness.",
        topic="the incident",
        prior="I saw nothing.",
        follow_up="Were the lights on?",
    )
    assert "I saw nothing." in text
    assert "Were the lights on?" in text


def test_list_recipes_sorted_by_key():
    keys = [r.key for r in list_recipes()]
    assert keys == sorted(keys)
