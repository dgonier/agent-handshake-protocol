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

        # ── information sources ───────────────────────────────────────
        # Higher-level abstraction over any backend. Tier requirements
        # are deliberately permissive on the text side ({"s", "j"})
        # because most info sources can serve either format; the
        # embedding variant requires bytes/embeddings so it routes only
        # to sources that can emit raw vectors.
        Code.INFO_QUERY:             {"s", "j"},
        Code.INFO_QUERY_EMBEDDING:   {"b", "e"},
        Code.INFO_LIST:              {"j"},
        Code.INFO_WRITE:             {"j"},

        # ── turn primitives ───────────────────────────────────────────
        # Conversational turn types used by the format taxonomy (24
        # game modes in ahp/adapters/formats.py). Most turns carry
        # text or JSON ({"s", "j"}); structured-only turns (COMMIT,
        # DECIDE, EXECUTE, REPORT, TRADE) require JSON because their
        # payload is by definition machine-readable.
        Code.TURN_ACK:                {"s", "j"},
        Code.TURN_ADVANCE:            {"s", "j"},
        Code.TURN_AMPLIFY:            {"s", "j"},
        Code.TURN_ANSWER:             {"s", "j"},
        Code.TURN_APORIA:             {"s", "j"},
        Code.TURN_APPEAL:             {"s", "j"},
        Code.TURN_ASK:                {"s", "j"},
        Code.TURN_ASSERT:             {"s", "j"},
        Code.TURN_BACK_OR_QUALIFY:    {"s", "j"},
        Code.TURN_BRIDGE:             {"s", "j"},
        Code.TURN_CARNIVALIZE:        {"s", "j"},
        Code.TURN_CHALLENGE_WARRANT:  {"s", "j"},
        Code.TURN_CHECK:              {"s", "j"},
        Code.TURN_CLAIM:              {"s", "j"},
        Code.TURN_CLARIFY:            {"s", "j"},
        Code.TURN_COMMIT:             {"j"},
        Code.TURN_CONFIRM:            {"s", "j"},
        Code.TURN_CONNECT:            {"s", "j"},
        Code.TURN_CONSOLIDATE:        {"s", "j"},
        Code.TURN_CONTEST:            {"s", "j"},
        Code.TURN_COUPLE:             {"s", "j"},
        Code.TURN_DECIDE:             {"j"},
        Code.TURN_DISCLOSE:           {"s", "j"},
        Code.TURN_DISCUSS:            {"s", "j"},
        Code.TURN_ELICIT:             {"s", "j"},
        Code.TURN_ENCOUNTER:          {"s", "j"},
        Code.TURN_EXECUTE:            {"j"},
        Code.TURN_EXPLORE_OPTIONS:    {"s", "j"},
        Code.TURN_EXPOSE_CONTRADICTION: {"s", "j"},
        Code.TURN_EXTEND:             {"s", "j"},
        Code.TURN_FUSE:               {"s", "j"},
        Code.TURN_HEAR_A:             {"s", "j"},
        Code.TURN_HEAR_B:             {"s", "j"},
        Code.TURN_HYPOTHESIZE:        {"s", "j"},
        Code.TURN_IDENTIFY_BINARY:    {"s", "j"},
        Code.TURN_INSTRUCT:           {"s", "j"},
        Code.TURN_INTEGRATE:          {"s", "j"},
        Code.TURN_INTERPRET:          {"s", "j"},
        Code.TURN_INTERROGATE:        {"s", "j"},
        Code.TURN_INTERRUPT:          {"s", "j"},
        Code.TURN_INVERT:             {"s", "j"},
        Code.TURN_LAYER:              {"s", "j"},
        Code.TURN_LEAVE_OPEN:         {"s", "j"},
        Code.TURN_LISTEN:             {"s", "j"},
        Code.TURN_MIRROR:             {"s", "j"},
        Code.TURN_MOVE:               {"s", "j"},
        Code.TURN_NEGATE:             {"s", "j"},
        Code.TURN_OFFER:              {"s", "j"},
        Code.TURN_OPEN:               {"s", "j"},
        Code.TURN_POSITION:           {"s", "j"},
        Code.TURN_PRESENT:            {"s", "j"},
        Code.TURN_PROBE:              {"s", "j"},
        Code.TURN_PROBLEMATIZE:       {"s", "j"},
        Code.TURN_PROPOSE:            {"s", "j"},
        Code.TURN_REBUT:              {"s", "j"},
        Code.TURN_RECALIBRATE:        {"s", "j"},
        Code.TURN_RECEIVE:            {"s", "j"},
        Code.TURN_RECIPROCATE:        {"s", "j"},
        Code.TURN_RECOGNIZE:          {"s", "j"},
        Code.TURN_RECONSTRUCT:        {"s", "j"},
        Code.TURN_RECURSE:            {"s", "j"},
        Code.TURN_REFLECT:            {"s", "j"},
        Code.TURN_REFRAME:            {"s", "j"},
        Code.TURN_REPORT:             {"j"},
        Code.TURN_RESPOND:            {"s", "j"},
        Code.TURN_RESPONSIBILITY:     {"s", "j"},
        Code.TURN_RETELL:             {"s", "j"},
        Code.TURN_RETERRITORIALIZE:   {"s", "j"},
        Code.TURN_REVISE:             {"s", "j"},
        Code.TURN_REVOICE:            {"s", "j"},
        Code.TURN_RIFF:               {"s", "j"},
        Code.TURN_RUPTURE:            {"s", "j"},
        Code.TURN_SCAFFOLD:           {"s", "j"},
        Code.TURN_SHOW_APORIA:        {"s", "j"},
        Code.TURN_STATE_INTEREST:     {"s", "j"},
        Code.TURN_SUPPORT:            {"s", "j"},
        Code.TURN_SURFACE_MISMATCH:   {"s", "j"},
        Code.TURN_SUSTAIN:            {"s", "j"},
        Code.TURN_SUSTAIN_TENSION:    {"s", "j"},
        Code.TURN_SYNTHESIZE:         {"s", "j"},
        Code.TURN_TELL:               {"s", "j"},
        Code.TURN_TEST:               {"s", "j"},
        Code.TURN_TRADE:              {"j"},
        Code.TURN_VALIDATE:           {"s", "j"},
        Code.TURN_VOICE:              {"s", "j"},
        Code.TURN_WEAVE:              {"s", "j"},
        Code.TURN_WEIGH:              {"s", "j"},
        Code.TURN_WITNESS:            {"s", "j"},

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
