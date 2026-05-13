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
