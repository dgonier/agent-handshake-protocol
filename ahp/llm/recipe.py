"""LLM recipes — base model + LoRA composition over the address layer.

The idea: agent *recipes* (base model, LoRA adapters, generation
config) are first-class addressable resources, just like tools and
filesystems. Two new ``ResourceAddress.kind`` values, used by
convention:

* ``kind="model"`` — a base model handle. One per agent recipe.
  Address shape: ``{scope}.model.{domain}.{subdomain}.{name}``
  (use ``domain="*"`` / ``subdomain="*"`` when the base model is
  shared across all domains in the scope — the convention treats
  ``*`` as a real wildcard at match time).
* ``kind="lora"`` — a LoRA / adapter handle. Zero or more per recipe.
  Address shape: ``{scope}.lora.{domain}.{subdomain}.{name}``.

Decorate factory functions the same way you'd register any resource;
the factory should return a :class:`ModelHandle` / :class:`LoRAHandle`
instance. Loading the actual weights is the consumer's responsibility
(see ``ahp.llm.huggingface`` when that ships) — these handles are
pure metadata so the recipe is composable on machines that don't have
GPUs.

::

    from ahp.adapters import resource
    from ahp.llm.recipe import LoRAHandle, ModelHandle

    @resource("tifin", "model", "*", "*", name="llama3-8b")
    def make_base():
        return ModelHandle(
            name="llama3-8b",
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        )

    @resource(
        "tifin", "lora", "finance", "*", name="bearish-v2",
        # ResourceAddress doesn't carry a role field, so role-gated
        # LoRAs use an explicit allowed_for. The convention's default
        # projection only covers scope/domain/subdomain.
        allowed_for="tifin.adversarial.finance.*.*.*.*",
    )
    def make_bearish():
        return LoRAHandle(
            name="bearish-v2",
            repo_id="tifin/finance-bearish-v2",
            weight=1.0,
        )

The recipe an agent ends up with is determined by its address — an
adversarial finance agent picks up ``bearish-v2`` (matched via the
explicit ``allowed_for``), ``llama3-8b`` (matches ``domain="*"``,
``subdomain="*"``), and any other matching LoRAs automatically via
the standard :class:`ResourceRegistry` rules.

Consumers (a HuggingFaceAgent that actually loads + composes the
weights) use :func:`find_model`, :func:`find_loras`, and
:func:`recipe_summary` to introspect the profile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from ahp.adapters.capability import AgentProfile


@dataclass(frozen=True)
class ModelHandle:
    """Pure-metadata handle to a base chat model.

    The factory that produces this doesn't have to load the weights —
    that's the consumer's job. The handle just declares *which*
    weights and *how* to load them. Consumers (e.g. a
    ``HuggingFaceAgent`` that wraps ``transformers``) read these
    fields to call ``AutoModelForCausalLM.from_pretrained(...)``.
    """

    name: str
    repo_id: str | None = None
    revision: str | None = None
    quantization: str | None = None
    torch_dtype: str | None = None
    device_map: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LoRAHandle:
    """Pure-metadata handle to a LoRA / adapter."""

    name: str
    repo_id: str | None = None
    revision: str | None = None
    adapter_name: str | None = None
    weight: float = 1.0
    extra: Mapping[str, Any] = field(default_factory=dict)


# ── finders ────────────────────────────────────────────────────────────


def find_model(profile: AgentProfile) -> ModelHandle | None:
    """Return the first :class:`ModelHandle` in ``profile.resources``, or None.

    Recipes are expected to declare at most one base model per agent;
    if multiple are registered and matched, the dict-iteration order
    decides (Python preserves insertion order — registry insertion
    sequence + ``allowed_for`` filter). Two competing base models is
    a wiring bug — narrow your ``allowed_for`` patterns or give them
    different names.
    """
    for instance in profile.resources.values():
        if isinstance(instance, ModelHandle):
            return instance
    return None


def find_loras(profile: AgentProfile) -> list[LoRAHandle]:
    """Return every :class:`LoRAHandle` in ``profile.resources``, sorted by name.

    Stable sort by ``name`` (the resource's short name, which equals
    the address's ``name`` field) so the composition order is
    reproducible across processes — important when the order of LoRA
    application matters (it does, for stacked adapters with
    overlapping target modules).
    """
    return sorted(
        (v for v in profile.resources.values() if isinstance(v, LoRAHandle)),
        key=lambda h: h.name,
    )


def recipe_summary(
    profile: AgentProfile,
    *,
    header: str = "Recipe:",
) -> str:
    """A system-prompt-friendly summary of the agent's model + LoRA stack.

    Empty string when no model and no LoRAs are present. The deep /
    react agent builders concatenate this with the rest of the
    profile prompt so the LLM is aware of its own ingredients —
    helpful when LoRAs steer style/tone in non-obvious ways.
    """
    base = find_model(profile)
    loras = find_loras(profile)
    if base is None and not loras:
        return ""
    lines: list[str] = [header]
    if base is not None:
        suffix = f" ({base.repo_id})" if base.repo_id else ""
        lines.append(f"  base: {base.name}{suffix}")
    if loras:
        lines.append("  loras (composed in this order):")
        for lora in loras:
            suffix = f" weight={lora.weight}"
            if lora.repo_id:
                suffix = f" ({lora.repo_id})" + suffix
            lines.append(f"    - {lora.name}{suffix}")
    return "\n".join(lines)


def all_recipe_handles(profile: AgentProfile) -> list:
    """Every model / LoRA handle in the profile, base first then LoRAs by name."""
    out: list = []
    base = find_model(profile)
    if base is not None:
        out.append(base)
    out.extend(find_loras(profile))
    return out
