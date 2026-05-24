"""FormatAgent — thin wrapper that wires a format's LangGraph state
machine onto an :class:`AHPAgent`.

A FormatAgent declares which formats it participates in via the
class attribute :attr:`supported_formats`. For each declared format,
the wrapper:

* Validates at instantiation that every turn primitive in the format
  has a corresponding overridden hook on the subclass. Missing hooks
  surface as a ``TypeError`` before the agent starts handling
  messages — fail loud, not silently echo.
* Routes inbound messages whose ``Message.format`` matches a
  supported format to the right ``on_<turn>`` method (chosen by the
  message's code, which is the turn primitive).
* Persists per-(thread, format) state via the LangGraph checkpointer
  protocol. Callers supply the checkpointer; in-tree we default to
  the in-memory :class:`MemorySaver` so the wrapper works without a
  Redis dependency. Production deployments hand in a
  :class:`langgraph.checkpoint.redis.RedisSaver` constructed from
  the AHP redis client.

Author API — subclass and override:

    from ahp.adapters import FormatAgent
    from ahp.core import Message

    class MyResponder(FormatAgent):
        supported_formats = ("information-exchange",)

        async def on_ask(self, message: Message) -> Message | None:
            # answer the question
            ...

        async def on_clarify(self, message: Message) -> Message | None:
            # respond to clarification request
            ...

The base class provides default ``on_<turn>`` methods that raise
``NotImplementedError`` for every turn primitive in the taxonomy. A
subclass overrides only the turns it actually handles for its
declared roles; the contract check enforces that the right set is
overridden for each declared format.

The wrapper doesn't dispatch turns the subclass doesn't claim
support for — it returns ``None`` (drops the message silently) so a
format-tagged but unsupported turn doesn't accidentally hit a
no-op base method.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, ClassVar

from ahp.adapters.base import AHPAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


log = logging.getLogger("ahp.adapters.format_agent")


def _turn_code_to_method(code: str) -> str:
    """Turn primitive code → method name.

    ``turn.ask`` → ``on_ask``
    ``turn.back-or-qualify`` → ``on_back_or_qualify``
    ``turn.show-aporia`` → ``on_show_aporia``
    """
    if not code.startswith("turn."):
        raise ValueError(f"not a turn primitive code: {code!r}")
    stem = code[len("turn."):]
    return "on_" + stem.replace("-", "_")


class _NotImplementedSentinel:
    """Marks a base-class hook that hasn't been overridden.

    Easier to detect than checking ``method.__qualname__`` because
    subclassing semantics with dataclasses + decorators sometimes
    rewrite qualnames in ways that fool the obvious check.
    """


_BASE_HOOK_SENTINEL = _NotImplementedSentinel()


def _make_base_hook(turn_code: str) -> Callable[..., Any]:
    """Build a `on_<turn>` method that raises NotImplementedError.

    The hook is tagged with ``_BASE_HOOK_SENTINEL`` so the contract
    check can tell base-class hooks from subclass overrides.
    """
    async def _hook(self: "FormatAgent", message: Message) -> Message | None:
        raise NotImplementedError(
            f"{type(self).__name__} declared format support that "
            f"requires {turn_code!r} but didn't override the matching "
            f"on_* method. Subclass must provide an async method "
            f"named {_turn_code_to_method(turn_code)!r}."
        )

    _hook._ahp_base_hook = _BASE_HOOK_SENTINEL  # type: ignore[attr-defined]
    _hook._ahp_turn_code = turn_code  # type: ignore[attr-defined]
    return _hook


def _all_turn_codes() -> list[str]:
    """Every TURN_* code declared on Code. Cached lazily so adding a
    new turn primitive surfaces here without code changes elsewhere."""
    return [
        v for k, v in vars(Code).items()
        if k.startswith("TURN_") and isinstance(v, str)
    ]


class FormatAgent(AHPAgent):
    """Thin :class:`AHPAgent` subclass wiring per-format turn dispatch.

    Subclass and:

    1. Set ``supported_formats: ClassVar[tuple[str, ...]]`` to the
       format names this agent plays.
    2. Override the ``on_<turn>`` async methods for whichever turn
       primitives the agent handles. The contract check at
       instantiation verifies the overrides are present.

    The wrapper handles routing (`handle_message` dispatches by
    inbound code) but does not impose a state machine — that's the
    format's :attr:`graph_builder`'s job. A subclass can use the
    checkpointer (via ``self._checkpointer``) directly when its hooks
    need to read/write conversation state per (thread, format).
    """

    supported_formats: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        *,
        checkpointer: Any = None,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> None:
        # Contract check FIRST so a misconfigured subclass fails at
        # construction time, not at first message arrival.
        self._validate_format_contracts()

        # Default to an in-memory LangGraph checkpointer. Production
        # deployments pass a RedisSaver constructed from the AHP
        # Redis client. We import lazily so the wrapper is usable
        # even when LangGraph isn't installed (the import will fail
        # only when a caller actually invokes a graph).
        if checkpointer is None:
            try:
                from langgraph.checkpoint.memory import MemorySaver
                checkpointer = MemorySaver()
            except ImportError:
                # Tests / deployments without LangGraph proceed with
                # None — graph invocations will fail clearly later
                # rather than silently no-op.
                checkpointer = None
        self._checkpointer = checkpointer

        # Stamp format participation into AgentMeta so list-agents can
        # filter / inspect which formats an agent supports.
        meta = metadata or AgentMeta()
        existing_extra = dict(meta.extra) if meta.extra else {}
        existing_extra["formats"] = {
            "supported": list(self.supported_formats),
        }
        meta.extra = existing_extra
        if "format-agent" not in meta.capabilities:
            meta.capabilities = list(meta.capabilities) + ["format-agent"]

        super().__init__(address, engine, metadata=meta, **kwargs)

    # ── contract check ────────────────────────────────────────────────

    @classmethod
    def _validate_format_contracts(cls) -> None:
        """Verify every turn primitive in every supported format has
        a non-base hook on the subclass.

        Raises :class:`TypeError` if a required hook is missing. The
        message names the format, the missing turn, and the method
        name the subclass should provide.

        Resolution: lazy import of FORMATS keeps ahp.adapters.format_agent
        free of a hard dep on the formats module at class-definition
        time.
        """
        if not cls.supported_formats:
            return  # nothing to validate

        from ahp.adapters.formats import FORMATS

        missing: list[tuple[str, str, str]] = []
        for fmt_name in cls.supported_formats:
            fmt = FORMATS.get(fmt_name)
            if fmt is None:
                raise TypeError(
                    f"{cls.__name__}.supported_formats lists "
                    f"{fmt_name!r} but no such format is registered "
                    f"in ahp.adapters.FORMATS"
                )
            # legacy_session formats don't have turn_primitives so
            # there's nothing to verify on the subclass.
            if fmt.recipe_kind == "legacy_session":
                continue
            for turn in fmt.turn_primitives:
                method_name = _turn_code_to_method(turn)
                method = getattr(cls, method_name, None)
                # Method missing or still the base sentinel hook =
                # not overridden.
                if method is None or getattr(
                    method, "_ahp_base_hook", None,
                ) is _BASE_HOOK_SENTINEL:
                    missing.append((fmt_name, turn, method_name))

        if missing:
            lines = ["FormatAgent contract violation:"]
            for fmt_name, turn, method_name in missing:
                lines.append(
                    f"  format {fmt_name!r} requires turn {turn!r} → "
                    f"subclass must define `async def {method_name}"
                    f"(self, message)`"
                )
            raise TypeError("\n".join(lines))

    # ── dispatch ─────────────────────────────────────────────────────

    async def handle_message(self, message: Message) -> Message | None:
        """Route by turn primitive when the message declares a format.

        When ``message.format`` is None or not in
        :attr:`supported_formats`, the wrapper drops the message
        silently (returns None). The engine's _check_format already
        verified format validity; a mismatch here would be a routing
        bug, not a subclass concern.

        When the message's code isn't a turn primitive at all
        (e.g. some legacy code accidentally tagged with a format),
        the wrapper also drops silently — a FormatAgent isn't the
        right consumer for non-turn traffic.
        """
        fmt_name = message.format
        if fmt_name is None or fmt_name not in self.supported_formats:
            return None
        if not message.code.startswith("turn."):
            return None
        method_name = _turn_code_to_method(message.code)
        method = getattr(self, method_name, None)
        if method is None:
            return None
        return await method(message)


# ── dynamically attach base hooks for every turn primitive ──────────


def _install_base_hooks(cls: type) -> None:
    """Attach a NotImplementedError sentinel hook for every TURN_*
    code that doesn't already have one on the class. Called once at
    module load — see below."""
    for turn in _all_turn_codes():
        method_name = _turn_code_to_method(turn)
        if not hasattr(cls, method_name):
            setattr(cls, method_name, _make_base_hook(turn))


_install_base_hooks(FormatAgent)
