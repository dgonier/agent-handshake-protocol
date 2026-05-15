"""ProtocolEngine — the verb dispatcher.

The engine is the outbound gate for every protocol message. It:

1. Validates the envelope's verb/target shape.
2. Checks the response cache for GET-style verbs against concrete targets.
3. Resolves :class:`AddressPattern` targets through the registry.
4. Filters resolved targets through the :class:`CompatibilityMatrix`.
5. Dispatches via the :class:`RedisBus`.
6. Caches the response of cacheable GET requests.
7. Returns a verb-appropriate result.

Inbound delivery is the bus's job — agents subscribe to their inbox via
``bus.listen()`` or ``bus.consume()``.

Return shapes by verb:

================  ====================
``SEND``          ``int`` delivery count
``SEND-GET``      ``Message | None`` (or cached ``Message``)
``CAST``          ``int`` total deliveries across resolved targets
``CAST-GET``      ``list[Message]``
``CAST-SUB``      :class:`Subscription` over matching tap traffic
``INVALIDATE``    ``int`` cache entries invalidated
================  ====================
"""

from __future__ import annotations

import logging
from typing import Any

from ahp.audit import AuditEvent, AuditSink
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.compatibility import CompatibilityMatrix
from ahp.core.message import Message
from ahp.core.pattern import AddressPattern
from ahp.engine.errors import (
    IncompatibleTargetError,
    InvalidTargetTypeError,
    ProtocolError,
    UnauthorizedError,
)
from ahp.engine.scope import ScopePolicy
from ahp.engine.thread_manager import ThreadManager
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus, Subscription


log = logging.getLogger(__name__)

DEFAULT_TIMEOUT: float = 5.0
"""Default seconds to wait for SEND-GET / CAST-GET responses."""


class ProtocolEngine:
    """Routes :class:`Message` envelopes through the AHP stack."""

    def __init__(
        self,
        bus: RedisBus,
        registry: AgentRegistry,
        cache: ProtocolCache,
        matrix: CompatibilityMatrix | None = None,
        threads: ThreadManager | None = None,
        *,
        default_timeout: float = DEFAULT_TIMEOUT,
        groups: Any = None,
        scope: ScopePolicy | None = None,
        audit: AuditSink | None = None,
    ) -> None:
        self.bus = bus
        self.registry = registry
        self.cache = cache
        self.matrix = matrix or CompatibilityMatrix()
        self.threads = threads or ThreadManager(bus.redis, bus)
        self.default_timeout = default_timeout
        # Group registry is set lazily by the factory if one was supplied.
        # Kept here so adapters can resolve a group name without needing a
        # factory reference.
        self.groups = groups
        # Optional access-control layer. Open-default — None means
        # everyone can reach everyone (current behavior preserved).
        self.scope = scope
        # Optional audit sink for dispatch outcomes.
        self.audit = audit

    # ── entry point ─────────────────────────────────────────────────────

    async def handle(
        self,
        message: Message,
        *,
        timeout: float | None = None,
        max_responses: int | None = None,
    ) -> Any:
        """Dispatch a single outbound message. See module docstring for return shapes."""
        timeout = self.default_timeout if timeout is None else timeout

        dispatch = {
            "SEND": self._handle_send,
            "SEND-GET": self._handle_send_get,
            "CAST": self._handle_cast,
            "CAST-GET": self._handle_cast_get,
            "CAST-SUB": self._handle_cast_sub,
            "INVALIDATE": self._handle_invalidate,
        }.get(message.verb)
        if dispatch is None:  # pragma: no cover — Message rejects invalid verbs
            raise ProtocolError(f"unhandled verb: {message.verb!r}")

        kwargs: dict[str, Any] = {}
        if message.verb in {"SEND-GET", "CAST-GET"}:
            kwargs["timeout"] = timeout
        if message.verb == "CAST-GET":
            kwargs["max_responses"] = max_responses

        try:
            result = await dispatch(message, **kwargs)
        except Exception as exc:
            await self._emit_message(message, success=False, error=_short_err(exc))
            raise
        await self._emit_message(message, success=True, result=result)
        return result

    # ── thread convenience ──────────────────────────────────────────────

    async def spawn_thread(self, topic: str, initiator: AgentAddress) -> str:
        thread = await self.threads.create(topic, initiator)
        return thread.thread_id

    async def join_thread(self, thread_id: str, agent: AgentAddress) -> None:
        await self.threads.join(thread_id, agent)

    # ── verb handlers ───────────────────────────────────────────────────

    async def _handle_send(self, message: Message) -> int:
        target = self._require_address(message)
        self._check_compatibility(message.source, target, message.code)
        self._check_scope(message.source, target, message.code)
        if not await self.registry.is_alive(target):
            log.debug("SEND to %s — target not alive", target)
            return 0
        return await self.bus.send(message)

    async def _handle_send_get(
        self, message: Message, *, timeout: float,
    ) -> Message | None:
        target = self._require_address(message)

        cached = await self.cache.get(message)
        if cached is not None:
            return cached

        self._check_compatibility(message.source, target, message.code)
        self._check_scope(message.source, target, message.code)
        if not await self.registry.is_alive(target):
            return None

        response = await self.bus.send_get(message, timeout=timeout)
        if response is not None:
            await self.cache.put(message, response)
        return response

    async def _handle_cast(self, message: Message) -> int:
        targets = await self._resolve_for_broadcast(message)
        if not targets:
            return 0
        return await self.bus.cast(message, targets)

    async def _handle_cast_get(
        self,
        message: Message,
        *,
        timeout: float,
        max_responses: int | None,
    ) -> list[Message]:
        targets = await self._resolve_for_broadcast(message)
        if not targets:
            return []
        return await self.bus.cast_get(
            message, targets, timeout=timeout, max_responses=max_responses,
        )

    async def _handle_cast_sub(self, message: Message) -> Subscription:
        """Open a long-lived subscription on the bus's tap channel.

        The returned :class:`Subscription` yields every message whose code
        matches ``message.code`` (treated as a hierarchical glob, so
        ``"interview.*"`` matches any interview verb) AND whose
        target/source matches ``message.target`` when the latter is an
        :class:`AddressPattern`. A concrete :class:`AgentAddress` target
        is treated as "subscribe to messages addressed to exactly this
        address."

        The caller owns the subscription's lifetime — call
        :meth:`Subscription.close` when done.
        """
        target = message.target
        code_glob = message.code

        if isinstance(target, AddressPattern):
            pattern: AddressPattern | None = target
            exact: AgentAddress | None = None
        elif isinstance(target, AgentAddress):
            pattern = None
            exact = target
        else:  # pragma: no cover — Message rejects bad targets
            raise InvalidTargetTypeError(
                "CAST-SUB target must be an AgentAddress or AddressPattern"
            )

        def predicate(msg: Message) -> bool:
            if not Code.matches(msg.code, code_glob):
                return False
            if exact is not None:
                if isinstance(msg.target, AgentAddress) and msg.target == exact:
                    return True
                return False
            assert pattern is not None
            # Match against the concrete target if there is one; otherwise
            # match the source (so we observe broadcasts emitted *by* agents
            # in the pattern, since broadcast targets carry no concrete
            # destination).
            if isinstance(msg.target, AgentAddress):
                return pattern.matches(msg.target)
            return pattern.matches(msg.source)

        return await self.bus.tap_subscribe(predicate=predicate)

    async def _handle_invalidate(self, message: Message) -> int:
        if not isinstance(message.target, AddressPattern):
            raise InvalidTargetTypeError(
                "INVALIDATE requires an AddressPattern target"
            )
        params: dict[str, str] | None = None
        if isinstance(message.body, dict):
            raw_params = message.body.get("params")
            if isinstance(raw_params, dict):
                params = {str(k): str(v) for k, v in raw_params.items()}
        return await self.cache.invalidate(message.target, params=params)

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _require_address(message: Message) -> AgentAddress:
        if not isinstance(message.target, AgentAddress):
            raise InvalidTargetTypeError(
                f"verb {message.verb!r} requires an AgentAddress target"
            )
        return message.target

    def _check_compatibility(
        self,
        source: AgentAddress,
        target: AgentAddress,
        code: str,
    ) -> None:
        if not self.matrix.can_route(source, target, code):
            tiers = self.matrix.required_tiers(code)
            raise IncompatibleTargetError(
                f"target {target} accept={target.accept!r} cannot receive "
                f"code {code!r} (required: any of {sorted(tiers)})"
            )

    def _check_scope(
        self,
        source: AgentAddress,
        target: AgentAddress,
        code: str,
    ) -> None:
        """Strict gate for point-to-point verbs. Raises on denial.

        With no policy set, this is a no-op (open default).
        """
        if self.scope is None:
            return
        if not self.scope.is_allowed(source, target, code):
            raise UnauthorizedError(
                f"source {source} is not permitted to reach {target} "
                f"for code {code!r} under the active ScopePolicy"
            )

    async def _emit_message(
        self,
        message: Message,
        *,
        success: bool,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        if self.audit is None:
            return
        extra: dict[str, Any] = {}
        if success:
            # Best-effort outcome summary per verb.
            if message.verb in {"SEND", "CAST", "INVALIDATE"}:
                if isinstance(result, int):
                    extra["count"] = result
            elif message.verb == "SEND-GET":
                extra["hit"] = result is not None
            elif message.verb == "CAST-GET":
                if isinstance(result, list):
                    extra["responses"] = len(result)
        op = {
            "SEND": "engine.send",
            "SEND-GET": "engine.send_get",
            "CAST": "engine.cast",
            "CAST-GET": "engine.cast_get",
            "CAST-SUB": "engine.cast_sub",
            "INVALIDATE": "engine.invalidate",
        }.get(message.verb, f"engine.{message.verb.lower()}")
        principal = (
            self.registry.principal.id
            if self.registry.principal is not None
            else None
        )
        await self.audit.emit(AuditEvent(
            op=op,
            principal=principal,
            source=str(message.source),
            target=str(message.target),
            code=message.code,
            verb=message.verb,
            success=success,
            error=error,
            extra=extra,
        ))

    async def _resolve_for_broadcast(self, message: Message) -> list[AgentAddress]:
        """Resolve a broadcast message's target through registry + matrix + scope.

        Accepts either an :class:`AddressPattern` (normal case) or a
        concrete :class:`AgentAddress` (degenerate single-target broadcast).
        Scope policy is applied as a silent filter — unauthorized targets
        are dropped from the resolved set, mirroring how compatibility
        filtering already works for broadcasts.
        """
        if isinstance(message.target, AgentAddress):
            if not await self.registry.is_alive(message.target):
                return []
            if not self.matrix.can_route(message.source, message.target, message.code):
                return []
            if self.scope is not None and not self.scope.is_allowed(
                message.source, message.target, message.code,
            ):
                return []
            return [message.target]

        # Pattern: registry returns alive matches; matrix + scope filter.
        candidates = await self.registry.resolve(message.target, alive_only=True)
        candidates = self.matrix.filter_targets(
            message.source, candidates, message.code,
        )
        if self.scope is not None:
            candidates = self.scope.filter_targets(
                message.source, candidates, message.code,
            )
        return candidates


def _short_err(exc: BaseException) -> str:
    name = type(exc).__name__
    msg = str(exc)
    return f"{name}: {msg}" if msg else name
