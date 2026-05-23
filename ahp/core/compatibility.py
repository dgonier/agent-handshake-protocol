"""Format compatibility matrix between agents and interaction codes.

Each interaction code declares a set of *acceptable* output tiers. A target
agent can receive a message of a given code iff its accept set intersects
the code's required tier set. For example, ``interview.embeddings`` is
satisfied by either bytes (``b``) or embeddings (``e``).
"""

from __future__ import annotations

from ahp.core.address import AgentAddress
from ahp.core.codes import Code


class CompatibilityMatrix:
    """Static, data-driven mapping ``code → acceptable tier set``."""

    # A target must accept AT LEAST ONE of the tiers listed for the code.
    CODE_REQUIREMENTS: dict[str, set[str]] = {
        # ── interview ─────────────────────────────────────────────────
        Code.INTERVIEW_TEXT:         {"s"},
        Code.INTERVIEW_SCHEMA:       {"j"},
        Code.INTERVIEW_DATA:         {"j"},
        Code.INTERVIEW_EMBEDDINGS:   {"b", "e"},
        Code.INTERVIEW_SUFFICIENCY:  {"s", "j"},

        # ── adversarial ───────────────────────────────────────────────
        Code.ADVERSARIAL_CHALLENGE:  {"s"},
        Code.ADVERSARIAL_DEBATE:     {"s"},
        Code.ADVERSARIAL_AUDIT:      {"j"},
        Code.ADVERSARIAL_REDTEAM:    {"s"},

        # ── collaborative ─────────────────────────────────────────────
        Code.COLLAB_REASON:          {"s"},
        Code.COLLAB_DELEGATE:        {"s", "j"},
        Code.COLLAB_MERGE:           {"b", "e"},
        Code.COLLAB_CONSENSUS:       {"j"},

        # ── human ─────────────────────────────────────────────────────
        Code.HUMAN_QUERY:            {"s"},
        Code.HUMAN_OBSERVE:          {"s"},
        Code.HUMAN_INTERVENE:        {"s"},
        Code.HUMAN_APPROVE:          {"s"},
        Code.HUMAN_EXPLAIN:          {"s"},
        Code.HUMAN_HALT:             {"s"},

        # ── teacher ───────────────────────────────────────────────────
        # Survey / judge / observe payloads are JSON-shaped; the rubric
        # itself is a string the model reads. Both tiers are accepted.
        Code.TEACHER_JUDGE:          {"s", "j"},
        Code.TEACHER_SURVEY:         {"s", "j"},
        Code.TEACHER_OBSERVE:        {"s", "j"},
        Code.TEACHER_RUBRIC:         {"s", "j"},

        # ── knowledge graph ───────────────────────────────────────────
        # Reads/writes are JSON CRUD; embeddings travel as bytes/floats.
        Code.KG_WRITE:               {"j"},
        Code.KG_READ:                {"j"},
        Code.KG_QUERY:               {"j", "e"},

        # ── error ─────────────────────────────────────────────────────
        # Errors must always be deliverable, so they only require strings.
        Code.ERROR_MALFORMED:        {"s"},
        Code.ERROR_UNAUTHORIZED:     {"s"},
        Code.ERROR_OUT_OF_SCOPE:     {"s"},
        Code.ERROR_TIMEOUT:          {"s"},
        Code.ERROR_CONFLICT:         {"s"},
        Code.ERROR_INTERNAL:         {"s"},
        Code.ERROR_LOW_CONFIDENCE:   {"s"},
        Code.ERROR_BAD_UPSTREAM:     {"s"},
        Code.ERROR_OVERLOADED:       {"s"},
    }

    DEFAULT_REQUIREMENT: set[str] = {"s"}
    """Fallback when an unknown code is asked about — strings are universal."""

    # ── lookup ──────────────────────────────────────────────────────────

    @classmethod
    def required_tiers(cls, code: str) -> set[str]:
        """Tier set acceptable for the given code. Unknown codes fall back to ``{'s'}``."""
        return set(cls.CODE_REQUIREMENTS.get(code, cls.DEFAULT_REQUIREMENT))

    @classmethod
    def is_known(cls, code: str) -> bool:
        return code in cls.CODE_REQUIREMENTS

    # ── routing ─────────────────────────────────────────────────────────

    def can_route(
        self,
        source: AgentAddress,
        target: AgentAddress,
        code: str,
    ) -> bool:
        """True iff ``target`` can receive a ``code`` message from ``source``.

        Currently this only checks target accept ∩ code requirements; future
        extensions (auth, scoping) can layer on top.
        """
        tiers = self.required_tiers(code)
        return bool(set(target.accept) & tiers)

    def filter_targets(
        self,
        source: AgentAddress,
        targets: list[AgentAddress],
        code: str,
    ) -> list[AgentAddress]:
        """Subset of ``targets`` that can validly receive ``code`` from ``source``."""
        return [t for t in targets if self.can_route(source, t, code)]
