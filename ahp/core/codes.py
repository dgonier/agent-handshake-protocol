"""Hierarchical, dot-delimited interaction codes.

Codes are grouped into families: ``interview.*``, ``adversarial.*``,
``collaborative.*``, ``human.*``, and ``error.*``. The hierarchy lets
agents subscribe at any granularity (e.g. ``interview.*`` or
``adversarial.debate``).
"""

from __future__ import annotations


class Code:
    """Canonical interaction code constants."""

    # ── interview ───────────────────────────────────────────────────────
    INTERVIEW_TEXT = "interview.text"
    INTERVIEW_SCHEMA = "interview.schema"
    INTERVIEW_DATA = "interview.data"
    INTERVIEW_EMBEDDINGS = "interview.embeddings"
    INTERVIEW_SUFFICIENCY = "interview.sufficiency"

    # ── adversarial ─────────────────────────────────────────────────────
    ADVERSARIAL_CHALLENGE = "adversarial.challenge"
    ADVERSARIAL_DEBATE = "adversarial.debate"
    ADVERSARIAL_AUDIT = "adversarial.audit"
    ADVERSARIAL_REDTEAM = "adversarial.redteam"

    # ── collaborative ───────────────────────────────────────────────────
    COLLAB_REASON = "collaborative.reason"
    COLLAB_DELEGATE = "collaborative.delegate"
    COLLAB_MERGE = "collaborative.merge"
    COLLAB_CONSENSUS = "collaborative.consensus"

    # ── human ───────────────────────────────────────────────────────────
    HUMAN_QUERY = "human.query"
    HUMAN_OBSERVE = "human.observe"
    HUMAN_INTERVENE = "human.intervene"
    HUMAN_APPROVE = "human.approve"
    HUMAN_EXPLAIN = "human.explain"
    HUMAN_HALT = "human.halt"

    # ── teacher ─────────────────────────────────────────────────────────
    TEACHER_JUDGE = "teacher.judge"
    TEACHER_SURVEY = "teacher.survey"
    TEACHER_OBSERVE = "teacher.observe"
    TEACHER_RUBRIC = "teacher.rubric"

    # ── knowledge graph ─────────────────────────────────────────────────
    KG_WRITE = "kg.write"
    KG_READ = "kg.read"
    KG_QUERY = "kg.query"

    # ── information sources ─────────────────────────────────────────────
    # Higher-level abstraction over any backend (KG, document store,
    # SQL, vector store). An info source is an addressable agent that
    # declares accept tiers; querying it is a normal SEND-GET with
    # one of these codes. The tier requirements drive Compatibility-
    # Matrix negotiation — a caller with accept='s' can hit info.query
    # but not info.query.embedding, and would route through a gateway
    # agent if it needs embedding-tier data.
    INFO_QUERY = "info.query"
    """Text-style query → text/JSON snippets back."""

    INFO_QUERY_EMBEDDING = "info.query.embedding"
    """Vector-style query → raw embedding-tier hits back."""

    INFO_LIST = "info.list"
    """List/enumerate available items in the source. JSON response."""

    INFO_WRITE = "info.write"
    """Write a document/node into the source. JSON response with id."""

    # ── turn primitives ─────────────────────────────────────────────────
    # Atomic turn types used by the format taxonomy (24 game modes; see
    # ahp/adapters/formats.py and docs/ahp-game-mode-taxonomy.md). Each
    # turn primitive is a Code so the engine's CompatibilityMatrix can
    # enforce tier requirements per-turn, and so callers can subscribe
    # to a turn-primitive glob (e.g. `turn.*` to tap every conversational
    # move). Most primitives are {s, j} (text or JSON); a few are
    # JSON-only because they carry structured payloads (DECIDE, COMMIT,
    # REPORT).
    #
    # The exhaustive list is the union of all primitives across the 24
    # formats. Several are reused across formats (ASSERT, ELICIT,
    # RESPOND, ACKNOWLEDGE, PROPOSE) — same code, different format
    # context.
    TURN_ACK              = "turn.acknowledge"
    TURN_ADVANCE          = "turn.advance"
    TURN_AMPLIFY          = "turn.amplify"
    TURN_ANSWER           = "turn.answer"
    TURN_APORIA           = "turn.aporia"
    TURN_APPEAL           = "turn.appeal"
    TURN_ASK              = "turn.ask"
    TURN_ASSERT           = "turn.assert"
    TURN_BACK_OR_QUALIFY  = "turn.back-or-qualify"
    TURN_BRIDGE           = "turn.bridge"
    TURN_CARNIVALIZE      = "turn.carnivalize"
    TURN_CHALLENGE_WARRANT = "turn.challenge-warrant"
    TURN_CHECK            = "turn.check"
    TURN_CLAIM            = "turn.claim"
    TURN_CLARIFY          = "turn.clarify"
    TURN_COMMIT           = "turn.commit"
    TURN_CONFIRM          = "turn.confirm"
    TURN_CONNECT          = "turn.connect"
    TURN_CONSOLIDATE      = "turn.consolidate"
    TURN_CONTEST          = "turn.contest"
    TURN_COUPLE           = "turn.couple"
    TURN_DECIDE           = "turn.decide"
    TURN_DISCLOSE         = "turn.disclose"
    TURN_DISCUSS          = "turn.discuss"
    TURN_ELICIT           = "turn.elicit"
    TURN_ENCOUNTER        = "turn.encounter"
    TURN_EXECUTE          = "turn.execute"
    TURN_EXPLORE_OPTIONS  = "turn.explore-options"
    TURN_EXPOSE_CONTRADICTION = "turn.expose-contradiction"
    TURN_EXTEND           = "turn.extend"
    TURN_FUSE             = "turn.fuse"
    TURN_HEAR_A           = "turn.hear-a"
    TURN_HEAR_B           = "turn.hear-b"
    TURN_HYPOTHESIZE      = "turn.hypothesize"
    TURN_IDENTIFY_BINARY  = "turn.identify-binary"
    TURN_INSTRUCT         = "turn.instruct"
    TURN_INTEGRATE        = "turn.integrate"
    TURN_INTERPRET        = "turn.interpret"
    TURN_INTERROGATE      = "turn.interrogate"
    TURN_INTERRUPT        = "turn.interrupt"
    TURN_INVERT           = "turn.invert"
    TURN_LAYER            = "turn.layer"
    TURN_LEAVE_OPEN       = "turn.leave-open"
    TURN_LISTEN           = "turn.listen"
    TURN_MIRROR           = "turn.mirror"
    TURN_MOVE             = "turn.move"
    TURN_NEGATE           = "turn.negate"
    TURN_OFFER            = "turn.offer"
    TURN_OPEN             = "turn.open"
    TURN_POSITION         = "turn.position"
    TURN_PRESENT          = "turn.present"
    TURN_PROBE            = "turn.probe"
    TURN_PROBLEMATIZE     = "turn.problematize"
    TURN_PROPOSE          = "turn.propose"
    TURN_REBUT            = "turn.rebut"
    TURN_RECALIBRATE      = "turn.recalibrate"
    TURN_RECEIVE          = "turn.receive"
    TURN_RECIPROCATE      = "turn.reciprocate"
    TURN_RECOGNIZE        = "turn.recognize"
    TURN_RECONSTRUCT      = "turn.reconstruct"
    TURN_RECURSE          = "turn.recurse"
    TURN_REFLECT          = "turn.reflect"
    TURN_REFRAME          = "turn.reframe"
    TURN_REPORT           = "turn.report"
    TURN_RESPOND          = "turn.respond"
    TURN_RESPONSIBILITY   = "turn.responsibility"
    TURN_RETELL           = "turn.retell"
    TURN_RETERRITORIALIZE = "turn.reterritorialize"
    TURN_REVISE           = "turn.revise"
    TURN_REVOICE          = "turn.revoice"
    TURN_RIFF             = "turn.riff"
    TURN_RUPTURE          = "turn.rupture"
    TURN_SCAFFOLD         = "turn.scaffold"
    TURN_SHOW_APORIA      = "turn.show-aporia"
    TURN_STATE_INTEREST   = "turn.state-interest"
    TURN_SUPPORT          = "turn.support"
    TURN_SURFACE_MISMATCH = "turn.surface-mismatch"
    TURN_SUSTAIN          = "turn.sustain"
    TURN_SUSTAIN_TENSION  = "turn.sustain-tension"
    TURN_SYNTHESIZE       = "turn.synthesize"
    TURN_TELL             = "turn.tell"
    TURN_TEST             = "turn.test"
    TURN_TRADE            = "turn.trade"
    TURN_VALIDATE         = "turn.validate"
    TURN_VOICE            = "turn.voice"
    TURN_WEAVE            = "turn.weave"
    TURN_WEIGH            = "turn.weigh"
    TURN_WITNESS          = "turn.witness"

    # ── error ───────────────────────────────────────────────────────────
    ERROR_MALFORMED = "error.malformed"
    ERROR_UNAUTHORIZED = "error.unauthorized"
    ERROR_OUT_OF_SCOPE = "error.scope"
    ERROR_TIMEOUT = "error.timeout"
    ERROR_CONFLICT = "error.conflict"
    ERROR_INTERNAL = "error.internal"
    ERROR_LOW_CONFIDENCE = "error.confidence"
    ERROR_BAD_UPSTREAM = "error.upstream"
    ERROR_OVERLOADED = "error.overloaded"

    # ── helpers ─────────────────────────────────────────────────────────

    @classmethod
    def all(cls) -> frozenset[str]:
        """Every defined code as a frozenset."""
        return frozenset(
            v for k, v in vars(cls).items()
            if k.isupper() and isinstance(v, str)
        )

    @staticmethod
    def family(code: str) -> str:
        """Top-level family of a code (everything before the first dot).

        ``Code.family("interview.embeddings") == "interview"``
        """
        if not isinstance(code, str) or not code:
            raise ValueError(f"code must be a non-empty string, got {code!r}")
        return code.split(".", 1)[0]

    @staticmethod
    def is_error(code: str) -> bool:
        return Code.family(code) == "error"

    @staticmethod
    def matches(code: str, selector: str) -> bool:
        """Hierarchical match. ``selector`` may end in ``.*`` for a family glob.

        ``Code.matches("interview.text", "interview.*") == True``
        ``Code.matches("interview.text", "interview.text") == True``
        ``Code.matches("interview.text", "adversarial.*") == False``
        """
        if selector == "*":
            return True
        if selector.endswith(".*"):
            prefix = selector[:-2]
            return code == prefix or code.startswith(prefix + ".")
        return code == selector
