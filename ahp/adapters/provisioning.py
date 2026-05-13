"""Provisioning patterns — bulk-spawn N agents from a single spec.

Routing patterns (:class:`ahp.core.pattern.AddressPattern`) match
*existing* agents; provisioning patterns *create* them. They live in
separate types so the routing layer never has to reason about counts.

Per-field syntax (each of the 7 fields):

* Concrete token (``tifin``, ``adversarial``, ``s``, ...) — exact value.
* ``*`` — single auto-generated value (equivalent to ``1*``).
* ``N*`` (**prefix-N**, *max-count*) — caps spawn at N agents for this
  field by *cycling* values modulo N. Multiple prefix-N fields share
  one outer loop: total from this group = ``max(N_i)``.
* ``*N`` (**suffix-N**, *cross-join*) — Cartesian multiplier on top of
  the prefix-N group.
* ``N-*`` / ``*-N`` (**dash variants**) — same shapes, but
  ``-`` means "fresh only": ignore anything already in the registry.

Without a dash, the default is **reuse-then-top-up**: when materializing
against a registry, existing alive agents matching the spec's fixed
skeleton supply the first values for that field; if fewer than N are
found, fresh auto-generated names fill the rest.

Total spawn count =
``max(prefix N's, defaulting to 1) × prod(suffix N's)``.

Examples::

    4*.r.d.2*.s.session.*        →  4 agents       (subdomain cycles)
    *4.r.d.*2.s.session.*        →  8 agents       (Cartesian)
    *4.r.d.2*.s.session.*        →  8 agents       (4 orgs × 2 max-iters)
    4-*.r.d.2-*.s.session.*      →  4 fresh agents (no registry lookup)
    Nike.r.d.2*.s.session.*      →  2 agents       (only subdomain varies)

``N*`` / ``*N`` / dash variants are permitted only on the freely-named
fields (``org``, ``domain``, ``subdomain``, ``instance``). ``role``,
``accept``, and ``lifecycle`` must be concrete or plain ``*`` (which
expands to a pragmatic default for ``accept``/``lifecycle``).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal

from ahp.core.address import (
    ACCEPT_TIER_ORDER,
    VALID_LIFECYCLES,
    AgentAddress,
    _canonical_accept,
)
from ahp.core.pattern import AddressPattern

if TYPE_CHECKING:
    from ahp.registry.registry import AgentRegistry


_COUNTABLE_FIELDS: frozenset[str] = frozenset({"org", "domain", "subdomain", "instance"})

_CONSTRAINED_FIELDS: frozenset[str] = frozenset({"role", "accept", "lifecycle"})

_FIELD_ORDER: tuple[str, ...] = (
    "org", "role", "domain", "subdomain", "accept", "lifecycle", "instance",
)

# Regex captures the optional dash that flips reuse off.
_PREFIX_RE = re.compile(r"^(?P<n>\d+)(?P<dash>-?)\*$")
_SUFFIX_RE = re.compile(r"^\*(?P<dash>-?)(?P<n>\d+)$")


Kind = Literal["fixed", "max", "cross"]
FieldNamer = Callable[[str, int], str]
"""Auto-name strategy: ``namer(field_name, index) -> generated_token``."""


def default_namer(field_name: str, index: int) -> str:
    """Default: ``org0``, ``subdomain1``, ``instance2``, ..."""
    return f"{field_name}{index}"


@dataclass(frozen=True)
class ProvisioningField:
    """One position in a provisioning spec."""

    name: str
    kind: Kind
    count: int                  # always ≥ 1; 1 for fixed and plain *
    fixed: str | None = None    # set iff kind == "fixed"
    reuse: bool = True          # True = consult registry; False = always fresh

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(
                f"field {self.name!r}: count must be ≥ 1, got {self.count}"
            )
        if self.kind == "fixed" and self.fixed is None:
            raise ValueError(f"field {self.name!r}: kind 'fixed' requires a value")
        if self.kind != "fixed" and self.fixed is not None:
            raise ValueError(
                f"field {self.name!r}: only 'fixed' kind may carry a value"
            )

    def fresh_values(self, namer: FieldNamer) -> list[str]:
        """All-fresh value list (no registry interaction)."""
        if self.kind == "fixed":
            return [self.fixed]  # type: ignore[list-item]
        return [namer(self.name, i) for i in range(self.count)]


@dataclass(frozen=True)
class ProvisioningPattern:
    """A 7-field spec that materializes to one or more concrete addresses."""

    org: ProvisioningField
    role: ProvisioningField
    domain: ProvisioningField
    subdomain: ProvisioningField
    accept: ProvisioningField
    lifecycle: ProvisioningField
    instance: ProvisioningField

    # ── parsing ─────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, spec: str) -> "ProvisioningPattern":
        if not isinstance(spec, str):
            raise TypeError(f"spec must be str, got {type(spec).__name__}")
        spec = spec.strip()
        if "?" in spec:
            raise ValueError(
                f"provisioning patterns do not carry query params: {spec!r}"
            )
        parts = spec.split(".")
        if len(parts) != 7:
            raise ValueError(
                f"provisioning pattern must have exactly 7 dot-separated "
                f"fields, got {len(parts)}: {spec!r}"
            )
        fields: dict[str, ProvisioningField] = {}
        for name, raw in zip(_FIELD_ORDER, parts):
            fields[name] = cls._parse_field(name, raw)
        return cls(**fields)

    @classmethod
    def _parse_field(cls, name: str, raw: str) -> ProvisioningField:
        if not raw:
            raise ValueError(f"field {name!r}: empty value")

        m_pre = _PREFIX_RE.match(raw)
        m_suf = _SUFFIX_RE.match(raw)

        if m_pre or m_suf:
            if name not in _COUNTABLE_FIELDS:
                raise ValueError(
                    f"field {name!r} does not allow count syntax "
                    f"(allowed on {sorted(_COUNTABLE_FIELDS)}); got {raw!r}"
                )
            if m_pre:
                n = int(m_pre.group("n"))
                reuse = m_pre.group("dash") == ""
                return ProvisioningField(
                    name=name, kind="max", count=n, reuse=reuse,
                )
            n = int(m_suf.group("n"))  # type: ignore[union-attr]
            reuse = m_suf.group("dash") == ""  # type: ignore[union-attr]
            return ProvisioningField(
                name=name, kind="cross", count=n, reuse=reuse,
            )

        if raw == "*":
            if name == "role":
                raise ValueError("field 'role' must be concrete (no wildcards)")
            if name == "accept":
                return ProvisioningField(
                    name=name, kind="fixed", count=1, fixed="s", reuse=False,
                )
            if name == "lifecycle":
                return ProvisioningField(
                    name=name, kind="fixed", count=1, fixed="session", reuse=False,
                )
            return ProvisioningField(name=name, kind="max", count=1, reuse=True)

        # Concrete token — validate where applicable.
        if name == "accept":
            canon = _canonical_accept(raw)
            if canon != raw:
                raise ValueError(
                    f"accept must be in canonical tier order "
                    f"({ACCEPT_TIER_ORDER!r}): got {raw!r}, expected {canon!r}"
                )
        elif name == "lifecycle":
            if raw not in VALID_LIFECYCLES:
                raise ValueError(
                    f"lifecycle has invalid value {raw!r}; "
                    f"valid: {sorted(VALID_LIFECYCLES)}"
                )
        return ProvisioningField(
            name=name, kind="fixed", count=1, fixed=raw, reuse=False,
        )

    # ── inspection ──────────────────────────────────────────────────────

    @property
    def fields(self) -> tuple[ProvisioningField, ...]:
        return (
            self.org, self.role, self.domain, self.subdomain,
            self.accept, self.lifecycle, self.instance,
        )

    def total(self) -> int:
        """Number of addresses this spec materializes to (registry-agnostic)."""
        max_iters = self._max_iters()
        cross_total = 1
        for f in self.fields:
            if f.kind == "cross":
                cross_total *= f.count
        return max_iters * cross_total

    def _max_iters(self) -> int:
        counts = [f.count for f in self.fields if f.kind == "max"]
        return max(counts) if counts else 1

    def __str__(self) -> str:
        out = []
        for f in self.fields:
            if f.kind == "fixed":
                out.append(f.fixed)  # type: ignore[arg-type]
                continue
            dash = "" if f.reuse else "-"
            if f.kind == "max":
                out.append("*" if f.count == 1 else f"{f.count}{dash}*")
            else:  # cross
                out.append(f"*{dash}{f.count}")
        return ".".join(out)

    def skeleton(self) -> AddressPattern:
        """Routing pattern that matches existing agents conforming to fixed fields.

        Used by :meth:`materialize_async` to find registry-reusable values.
        """
        return AddressPattern(
            org=self.org.fixed if self.org.kind == "fixed" else "*",
            role=self.role.fixed if self.role.kind == "fixed" else "*",
            domain=self.domain.fixed if self.domain.kind == "fixed" else "*",
            subdomain=self.subdomain.fixed if self.subdomain.kind == "fixed" else "*",
            accept=self.accept.fixed if self.accept.kind == "fixed" else "*",
            lifecycle=self.lifecycle.fixed if self.lifecycle.kind == "fixed" else "*",
            instance=self.instance.fixed if self.instance.kind == "fixed" else "*",
        )

    # ── materialization ─────────────────────────────────────────────────

    def materialize(
        self,
        *,
        namer: FieldNamer = default_namer,
    ) -> list[AgentAddress]:
        """Build addresses without consulting any registry — all wildcards fresh."""
        value_lists = [f.fresh_values(namer) for f in self.fields]
        return self._combine(value_lists)

    async def materialize_async(
        self,
        *,
        registry: "AgentRegistry",
        namer: FieldNamer = default_namer,
    ) -> list[AgentAddress]:
        """Registry-aware materialization with tuple-level reuse.

        If every variable field is reuse-mode (no dash anywhere), alive
        agents matching :meth:`skeleton` are used whole — preserving
        their existing field correlations — up to ``self.total()``
        addresses. Any shortfall is filled with fresh addresses drawn
        from the tail of the all-fresh materialization, so the spec's
        cycling/cross-join structure still governs the new ones.

        If any variable field has ``reuse=False`` (i.e., a dash variant
        anywhere in the spec), reuse is skipped entirely and the result
        is the same as :meth:`materialize`.
        """
        target = self.total()

        # Any fresh-only variable field disables reuse globally — the
        # caller has signaled they want *new* agents, not the existing
        # ones whose values might not align with the fresh-name scheme.
        has_fresh_only = any(
            f.kind != "fixed" and not f.reuse for f in self.fields
        )
        if has_fresh_only:
            return self.materialize(namer=namer)

        matches = await registry.resolve(self.skeleton(), alive_only=True)
        # Deterministic order so tests + production agree on which existing
        # tuples are reused first.
        matches.sort(key=str)
        reused = matches[:target]

        shortfall = target - len(reused)
        if shortfall <= 0:
            return list(reused)

        # Generate the full fresh set, then take the LAST `shortfall`
        # addresses. Tail selection preserves the spec's cycling pattern
        # for the new slots without disturbing how reused tuples slot in.
        fresh_all = self.materialize(namer=namer)
        return list(reused) + fresh_all[-shortfall:]

    # ── internals ───────────────────────────────────────────────────────

    def _combine(self, value_lists: list[list[str]]) -> list[AgentAddress]:
        """Apply prefix-max-cycling + suffix-cross-join semantics over precomputed value lists."""
        max_iters = self._max_iters()

        max_indices = [i for i, f in enumerate(self.fields) if f.kind == "max"]
        cross_indices = [i for i, f in enumerate(self.fields) if f.kind == "cross"]
        fixed_indices = [i for i, f in enumerate(self.fields) if f.kind == "fixed"]

        # Effective max_iters: bounded by the largest actual value list (could
        # be smaller than the declared count if reuse mode supplied less than N).
        actual_max_iters = max(
            (len(value_lists[i]) for i in max_indices), default=1,
        )
        max_iters = max(max_iters, 1)
        max_iters = min(max_iters, actual_max_iters) if max_indices else max_iters

        addresses: list[AgentAddress] = []
        for max_iter in range(max_iters):
            base_slots: list[str | None] = [None] * 7
            for i in fixed_indices:
                base_slots[i] = value_lists[i][0]
            for i in max_indices:
                vals = value_lists[i]
                base_slots[i] = vals[max_iter % len(vals)]

            cross_value_lists = [value_lists[i] for i in cross_indices]
            for combo in itertools.product(*cross_value_lists) if cross_value_lists else [()]:
                slots = list(base_slots)
                for i, val in zip(cross_indices, combo):
                    slots[i] = val
                addresses.append(AgentAddress(
                    org=slots[0], role=slots[1], domain=slots[2],
                    subdomain=slots[3], accept=slots[4], lifecycle=slots[5],
                    instance=slots[6],
                ))
        return addresses
