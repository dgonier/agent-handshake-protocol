"""Tests for ahp.llm.recipe — model/LoRA composition through the address layer.

Exercises the full path: register handles as @resource factories,
build a profile via AgentFactory.profile_for(addr), use find_model /
find_loras / recipe_summary to inspect.
"""

from __future__ import annotations

import pytest

from ahp.adapters import AgentFactory, AgentProfile, ResourceRegistry
from ahp.core.address import AgentAddress
from ahp.llm.recipe import (
    LoRAHandle,
    ModelHandle,
    all_recipe_handles,
    find_loras,
    find_model,
    recipe_summary,
)


def _agent(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── handle invariants ─────────────────────────────────────────────────


def test_model_handle_is_frozen_metadata():
    h = ModelHandle(name="llama3-8b", repo_id="meta-llama/Meta-Llama-3-8B-Instruct")
    with pytest.raises(Exception):  # FrozenInstanceError / similar
        h.name = "other"   # type: ignore[misc]


def test_lora_handle_defaults_weight_to_one():
    h = LoRAHandle(name="bearish", repo_id="tifin/bearish")
    assert h.weight == 1.0


# ── finders against an empty / minimal profile ────────────────────────


def test_find_model_returns_none_when_empty():
    profile = AgentProfile(address=_agent("o.r.d.sd.s.session.i"))
    assert find_model(profile) is None
    assert find_loras(profile) == []
    assert recipe_summary(profile) == ""


def test_finders_skip_non_recipe_resources():
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={
            "db-client": object(),     # not a recipe handle
            "vector-store": {"id": "x"},
        },
    )
    assert find_model(profile) is None
    assert find_loras(profile) == []


# ── finders against a populated profile ──────────────────────────────


def test_find_model_returns_first_model_handle():
    base = ModelHandle(name="llama3-8b")
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"llama3-8b": base, "db-client": object()},
    )
    assert find_model(profile) is base


def test_find_loras_returns_all_sorted_by_name():
    a = LoRAHandle(name="zeta-tone")
    b = LoRAHandle(name="alpha-tone")
    c = LoRAHandle(name="mid-tone")
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={"zeta-tone": a, "alpha-tone": b, "mid-tone": c},
    )
    loras = find_loras(profile)
    assert [l.name for l in loras] == ["alpha-tone", "mid-tone", "zeta-tone"]


def test_recipe_summary_formats_model_and_loras():
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={
            "llama3-8b": ModelHandle(name="llama3-8b",
                                     repo_id="meta-llama/Meta-Llama-3-8B-Instruct"),
            "bearish-v2": LoRAHandle(name="bearish-v2",
                                     repo_id="tifin/bearish-v2", weight=0.8),
        },
    )
    text = recipe_summary(profile)
    assert "Recipe:" in text
    assert "llama3-8b" in text
    assert "meta-llama/Meta-Llama-3-8B-Instruct" in text
    assert "bearish-v2" in text
    assert "weight=0.8" in text


def test_all_recipe_handles_orders_base_first():
    profile = AgentProfile(
        address=_agent("o.r.d.sd.s.session.i"),
        resources={
            "zeta": LoRAHandle(name="zeta"),
            "llama": ModelHandle(name="llama"),
            "alpha": LoRAHandle(name="alpha"),
        },
    )
    out = all_recipe_handles(profile)
    assert [h.name for h in out] == ["llama", "alpha", "zeta"]


# ── end-to-end: address-routed recipe assembly ───────────────────────


def test_recipe_composes_through_address_layer(stack):
    """An adversarial finance agent picks up its base + LoRAs by address alone."""
    resources = ResourceRegistry()

    @resources.resource("tifin", "model", "*", "*", name="llama3-8b")
    def make_base():
        return ModelHandle(
            name="llama3-8b",
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        )

    @resources.resource(
        "tifin", "lora", "finance", "*", name="bearish-v2",
        # ResourceAddress doesn't have a role field, so role-discriminated
        # LoRAs use an explicit allowed_for. This LoRA only applies to
        # adversarial finance agents.
        allowed_for="tifin.adversarial.finance.*.*.*.*",
    )
    def make_bearish():
        return LoRAHandle(name="bearish-v2", repo_id="tifin/bearish-v2")

    @resources.resource(
        "tifin", "lora", "finance", "*", name="cite-numbers",
        # All finance agents (any role) get this one.
        allowed_for="tifin.*.finance.*.*.*.*",
    )
    def make_cite():
        return LoRAHandle(name="cite-numbers", repo_id="tifin/cite-numbers")

    factory = AgentFactory(stack.engine, resources=resources)

    # An adversarial finance equities agent: matches base (domain=any),
    # bearish (role+domain), and cite-numbers (domain via explicit allowed_for).
    adv_profile = factory.profile_for(
        "tifin.adversarial.finance.equities.s.session.bull",
    )
    base = find_model(adv_profile)
    loras = find_loras(adv_profile)
    assert base is not None and base.name == "llama3-8b"
    assert sorted(l.name for l in loras) == ["bearish-v2", "cite-numbers"]

    # A collaborative finance agent: gets the base + cite-numbers but
    # NOT bearish-v2 (role doesn't match adversarial).
    collab_profile = factory.profile_for(
        "tifin.collaborative.finance.equities.s.session.alice",
    )
    base = find_model(collab_profile)
    loras = find_loras(collab_profile)
    assert base is not None
    assert [l.name for l in loras] == ["cite-numbers"]

    # A science agent: gets only the base (model is domain=any),
    # neither LoRA matches.
    sci_profile = factory.profile_for(
        "tifin.adversarial.science.biology.s.session.x",
    )
    assert find_model(sci_profile) is not None
    assert find_loras(sci_profile) == []


def test_recipe_summary_after_full_resolution(stack):
    resources = ResourceRegistry()

    @resources.resource("tifin", "model", "*", "*", name="llama3-8b")
    def make_base():
        return ModelHandle(
            name="llama3-8b",
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        )

    @resources.resource(
        "tifin", "lora", "finance", "*", name="bearish-v2",
        allowed_for="tifin.adversarial.finance.*.*.*.*",
    )
    def make_bearish():
        return LoRAHandle(name="bearish-v2", repo_id="tifin/bearish-v2")

    factory = AgentFactory(stack.engine, resources=resources)
    profile = factory.profile_for(
        "tifin.adversarial.finance.equities.s.session.bull",
    )
    text = recipe_summary(profile)
    assert "llama3-8b" in text
    assert "bearish-v2" in text
    assert text.startswith("Recipe:")
