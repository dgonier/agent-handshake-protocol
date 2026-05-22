"""Tests for compose_recipe + describe_recipe.

The helper turns an :class:`AgentProfile`'s recipe handles into a
ready-to-use chat model pointed at a Modal vLLM (or any OpenAI-
compatible) endpoint. These tests don't touch the network — they
inject a fake chat-model factory so the composition logic can be
verified in isolation.
"""

from __future__ import annotations

import pytest

from ahp.adapters import AgentProfile
from ahp.core.address import AgentAddress
from ahp.llm.recipe import (
    LoRAHandle,
    ModelHandle,
    RecipeError,
    compose_recipe,
    describe_recipe,
)


def _agent(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


class _FakeFactory:
    """Records the kwargs every compose call received so tests can
    assert on the composed identity without spinning up real
    langchain_openai infrastructure."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"_fake_chat_model": True, **kwargs}


# ── empty / no-recipe paths ──────────────────────────────────────────


def test_compose_returns_none_when_no_model_handle():
    """An agent profile with no ModelHandle yields None — the caller
    falls back to a hand-picked chat model."""
    profile = AgentProfile(address=_agent("o.r.d.sd.s.session.i"))
    factory = _FakeFactory()
    result = compose_recipe(profile, _chat_model_factory=factory)
    assert result is None
    assert factory.calls == []


# ── endpoint resolution ──────────────────────────────────────────────


def test_compose_uses_explicit_endpoint_from_extra():
    base = ModelHandle(
        name="qwen2-5-7b",
        extra={"endpoint": "https://you--vllm-qwen.modal.run/v1"},
    )
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"qwen2-5-7b": base},
    )
    factory = _FakeFactory()
    result = compose_recipe(profile, _chat_model_factory=factory)
    assert result is not None
    assert len(factory.calls) == 1
    call = factory.calls[0]
    assert call["base_url"] == "https://you--vllm-qwen.modal.run/v1"
    # No LoRAs → model_id is the base name.
    assert call["model"] == "qwen2-5-7b"


def test_compose_resolves_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("AHP_TEST_RECIPE_URL", "https://from-env.modal.run/v1")
    base = ModelHandle(
        name="qwen2-5-7b",
        extra={"endpoint_env": "AHP_TEST_RECIPE_URL"},
    )
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"qwen2-5-7b": base},
    )
    factory = _FakeFactory()
    compose_recipe(profile, _chat_model_factory=factory)
    assert factory.calls[0]["base_url"] == "https://from-env.modal.run/v1"


def test_compose_endpoint_literal_wins_over_env(monkeypatch):
    """A literal `endpoint` takes precedence over `endpoint_env`."""
    monkeypatch.setenv("AHP_TEST_RECIPE_URL", "https://from-env.modal.run/v1")
    base = ModelHandle(
        name="qwen2-5-7b",
        extra={
            "endpoint": "https://literal.modal.run/v1",
            "endpoint_env": "AHP_TEST_RECIPE_URL",
        },
    )
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"qwen2-5-7b": base},
    )
    factory = _FakeFactory()
    compose_recipe(profile, _chat_model_factory=factory)
    assert factory.calls[0]["base_url"] == "https://literal.modal.run/v1"


def test_compose_raises_recipe_error_when_no_endpoint():
    """A ModelHandle without endpoint metadata is a partial recipe —
    raise clearly rather than silently returning a misconfigured
    model."""
    base = ModelHandle(name="qwen2-5-7b")
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"qwen2-5-7b": base},
    )
    with pytest.raises(RecipeError, match="no endpoint configured"):
        compose_recipe(profile, _chat_model_factory=_FakeFactory())


def test_compose_recipe_error_when_env_var_unset(monkeypatch):
    """endpoint_env points at an unset env var → RecipeError."""
    monkeypatch.delenv("AHP_TEST_MISSING_URL", raising=False)
    base = ModelHandle(
        name="qwen2-5-7b",
        extra={"endpoint_env": "AHP_TEST_MISSING_URL"},
    )
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"qwen2-5-7b": base},
    )
    with pytest.raises(RecipeError):
        compose_recipe(profile, _chat_model_factory=_FakeFactory())


# ── LoRA routing ─────────────────────────────────────────────────────


def test_lora_name_becomes_request_model_id():
    """vLLM-style: multiple adapters served on one endpoint, routed by
    the `model` request param. The primary LoRA's name is what gets
    sent."""
    base = ModelHandle(
        name="qwen2-5-7b-base",
        extra={"endpoint": "https://x.modal.run/v1"},
    )
    lora = LoRAHandle(name="bearish-v2", weight=1.0)
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"qwen2-5-7b-base": base, "bearish-v2": lora},
    )
    factory = _FakeFactory()
    compose_recipe(profile, _chat_model_factory=factory)
    assert factory.calls[0]["model"] == "bearish-v2"


def test_compose_picks_highest_weight_lora():
    """When multiple LoRAs are present, the one with the highest
    weight becomes the primary served adapter."""
    base = ModelHandle(
        name="base", extra={"endpoint": "https://x.modal.run/v1"},
    )
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={
            "base": base,
            "low": LoRAHandle(name="low", weight=0.3),
            "high": LoRAHandle(name="high", weight=0.9),
            "mid": LoRAHandle(name="mid", weight=0.5),
        },
    )
    factory = _FakeFactory()
    compose_recipe(profile, _chat_model_factory=factory)
    assert factory.calls[0]["model"] == "high"


def test_compose_forwards_temperature_and_max_tokens():
    base = ModelHandle(
        name="base", extra={"endpoint": "https://x.modal.run/v1"},
    )
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"base": base},
    )
    factory = _FakeFactory()
    compose_recipe(
        profile,
        temperature=0.7, max_tokens=2048,
        _chat_model_factory=factory,
    )
    call = factory.calls[0]
    assert call["temperature"] == 0.7
    assert call["max_tokens"] == 2048


# ── describe_recipe ──────────────────────────────────────────────────


def test_describe_empty_recipe():
    profile = AgentProfile(address=_agent("o.r.d.sd.s.session.i"))
    desc = describe_recipe(profile)
    assert desc == {"base": None, "loras": [], "endpoint": None}


def test_describe_full_recipe():
    base = ModelHandle(
        name="qwen2-5-7b-base",
        repo_id="Qwen/Qwen2.5-7B",
        revision="main",
        extra={"endpoint": "https://x.modal.run/v1"},
    )
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={
            "qwen2-5-7b-base": base,
            "primary": LoRAHandle(
                name="primary", weight=0.9, repo_id="tifin/primary",
            ),
            "secondary": LoRAHandle(
                name="secondary", weight=0.4, repo_id="tifin/secondary",
            ),
        },
    )
    desc = describe_recipe(profile)
    assert desc["base"]["name"] == "qwen2-5-7b-base"
    assert desc["base"]["repo_id"] == "Qwen/Qwen2.5-7B"
    assert desc["endpoint"] == "https://x.modal.run/v1"
    assert desc["primary_lora"] == "primary"
    assert desc["model_id_for_request"] == "primary"
    # LoRAs sorted by name in find_loras.
    assert [l["name"] for l in desc["loras"]] == ["primary", "secondary"]


def test_describe_base_only_uses_base_name_for_request():
    """No LoRAs → describe says the base name is what gets sent."""
    base = ModelHandle(
        name="solo-model",
        extra={"endpoint": "https://x.modal.run/v1"},
    )
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"solo-model": base},
    )
    desc = describe_recipe(profile)
    assert desc["primary_lora"] is None
    assert desc["model_id_for_request"] == "solo-model"
