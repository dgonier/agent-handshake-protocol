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
from ahp.economy.pricing import DEFAULT_EXPECTED_RESPONSE_CHARS
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
        broker: Any = None,
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
        # Optional broker for the economic plane. When None, the engine
        # dispatches without any wallet / settlement work — preserves
        # backward compatibility with existing demos that haven't yet
        # opted into the economy. When present, SEND-GET runs the full
        # hold-and-settle pipeline.
        self.broker = broker

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

        # Economic plane: if a broker is wired, run hold → dispatch →
        # settle. Otherwise pass through to the bus directly. The
        # broker path uses the message's source and target as the two
        # wallet endpoints (agent-to-agent settlement); compute slice
        # is paid based on whatever leaf the target's owning server
        # has registered (or None for self-hosted).
        if self.broker is None:
            response = await self.bus.send_get(message, timeout=timeout)
        else:
            response = await self._broker_send_get(
                message, target=target, timeout=timeout,
            )

        if response is not None:
            await self.cache.put(message, response)
        return response

    async def _broker_send_get(
        self,
        message: Message,
        *,
        target: AgentAddress,
        timeout: float,
    ) -> Message | None:
        """SEND-GET with full economic settlement around the dispatch."""
        from ahp.economy.wallet import InsufficientFundsError

        prompt_chars = len(str(message.body)) if message.body is not None else 0
        # Conservative max-response estimate when caller didn't specify;
        # recipes in the runner do pass max_response_chars in the body,
        # so prefer that when present.
        max_resp = DEFAULT_EXPECTED_RESPONSE_CHARS
        if isinstance(message.body, dict):
            try:
                max_resp = int(message.body.get(
                    "max_response_chars", DEFAULT_EXPECTED_RESPONSE_CHARS,
                ))
            except (TypeError, ValueError):
                pass

        # Best-effort target tier: derive from accept set. A target
        # accepting only 's' is treated as small; otherwise default to
        # small. Tier is a routing input to the broker; the actual
        # model selection is the responding server's concern.
        tier = "small"

        # Look up the target's owning server. Convention: the server
        # whose org == target.org. If no such server exists, we still
        # dispatch — broker bills against the agent directly.
        owning_servers = await self.broker.servers.discover(alive_only=True)
        owning_server = next(
            (s for s in owning_servers if s.org == target.org), None,
        )
        if owning_server is None:
            # No registered server. Skip economic settlement; dispatch
            # raw. (This matches the no-broker pass-through.)
            return await self.bus.send_get(message, timeout=timeout)

        # Look up the compute leaf this server is bound to.
        leaves = await self.broker.compute.list_leaves(
            only_alive_providers=True,
        )
        from ahp.economy.compute_provider import best_leaf
        chosen_leaf = best_leaf(
            owning_server.compute_binding,
            leaves,
            rank_by=owning_server.compute_ranking,  # type: ignore[arg-type]
        )

        # Estimate the hold using the formula's hold estimator.
        from ahp.economy.pricing import estimate_hold
        hold = estimate_hold(
            base_rate=owning_server.base_rate,
            tier=tier,
            prompt_chars=prompt_chars,
            max_response_chars=max_resp,
            leaf_rate_per_1k_chars=(
                chosen_leaf.rate_per_1k_chars if chosen_leaf else 0.0
            ),
        )

        caller_wallet_owner = str(message.source)
        hold_id = f"msg:{message.message_id}"
        try:
            await self.broker.hold(
                caller=caller_wallet_owner,
                amount=hold.amount,
                hold_id=hold_id,
                reason=f"send_get to {target}",
            )
        except InsufficientFundsError:
            log.warning(
                "broker: %s has insufficient funds to call %s (need %.4f)",
                caller_wallet_owner, target, hold.amount,
            )
            return None

        # Dispatch over the bus, with hold in place.
        import time as _time
        dispatch_started = _time.monotonic()
        try:
            response = await self.bus.send_get(message, timeout=timeout)
        except Exception:
            await self.broker.refund(
                caller=caller_wallet_owner, hold_id=hold_id,
                reason="dispatch exception",
            )
            raise

        if response is None:
            # Timeout or no live target — refund the hold.
            await self.broker.refund(
                caller=caller_wallet_owner, hold_id=hold_id,
                reason="no response (timeout or target dead)",
            )
            return None

        # Settle. The responding agent earns; their owning server's
        # bound compute leaf gets the compute slice; tax flows to
        # broker + commons.
        latency_ms = (_time.monotonic() - dispatch_started) * 1000.0
        response_chars = (
            len(str(response.body)) if response.body is not None else 0
        )
        try:
            await self.broker.calculate_and_settle(
                caller=caller_wallet_owner,
                hold_id=hold_id,
                server=owning_server,
                leaf=chosen_leaf,
                response_chars=response_chars,
                max_response_chars=max_resp,
                actual_latency_ms=latency_ms,
                completed_with_caller=0,  # TODO: track per-pair counts
                tier_verdict="matched",
                responder=response.source,
            )
        except Exception:
            log.exception("broker settlement failed; refunding hold")
            try:
                await self.broker.refund(
                    caller=caller_wallet_owner, hold_id=hold_id,
                    reason="settlement raised",
                )
            except Exception:
                pass

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
        if self.broker is None:
            return await self.bus.cast_get(
                message, targets,
                timeout=timeout, max_responses=max_responses,
            )
        return await self._broker_cast_get(
            message, targets=targets,
            timeout=timeout, max_responses=max_responses,
        )

    async def _broker_cast_get(
        self,
        message: Message,
        *,
        targets: list[AgentAddress],
        timeout: float,
        max_responses: int | None,
    ) -> list[Message]:
        """CAST-GET with per-target hold + per-response settlement.

        One hold per resolved target — each can settle or refund
        independently. Targets that don't respond by ``timeout`` have
        their hold refunded; responders settle exactly like the SEND-GET
        path would.

        Insufficient-funds behavior: each per-target hold is attempted
        independently; targets whose hold can't be placed are dropped
        from the broadcast. This is the right tradeoff for adversarial
        debates where the caller has finite credits — a partial
        broadcast is preferable to a thrown exception that loses every
        response.
        """
        from ahp.economy.compute_provider import best_leaf
        from ahp.economy.pricing import estimate_hold
        from ahp.economy.wallet import InsufficientFundsError

        prompt_chars = len(str(message.body)) if message.body is not None else 0
        max_resp = DEFAULT_EXPECTED_RESPONSE_CHARS
        if isinstance(message.body, dict):
            try:
                max_resp = int(message.body.get(
                    "max_response_chars", DEFAULT_EXPECTED_RESPONSE_CHARS,
                ))
            except (TypeError, ValueError):
                pass

        caller_wallet_owner = str(message.source)

        # Pre-cache server + leaf decisions per org so two agents from
        # the same org don't trigger duplicate broker scans.
        owning_servers = await self.broker.servers.discover(alive_only=True)
        servers_by_org: dict[str, Any] = {s.org: s for s in owning_servers}
        leaves = await self.broker.compute.list_leaves(only_alive_providers=True)
        leaf_by_server_id: dict[str, Any] = {}

        # Per-target context: hold_id, server, leaf. Targets that fail
        # the hold (no server, or insufficient funds) are pruned from
        # the broadcast.
        held: dict[AgentAddress, dict[str, Any]] = {}
        deliverable_targets: list[AgentAddress] = []
        for target in targets:
            owning_server = servers_by_org.get(target.org)
            if owning_server is None:
                # No owning server: fall back to free dispatch (the
                # no-broker behavior) — agent is in the network but
                # nobody is billing for it.
                deliverable_targets.append(target)
                continue
            if owning_server.server_id not in leaf_by_server_id:
                leaf_by_server_id[owning_server.server_id] = best_leaf(
                    owning_server.compute_binding,
                    leaves,
                    rank_by=owning_server.compute_ranking,  # type: ignore[arg-type]
                )
            chosen_leaf = leaf_by_server_id[owning_server.server_id]
            hold = estimate_hold(
                base_rate=owning_server.base_rate,
                tier="small",
                prompt_chars=prompt_chars,
                max_response_chars=max_resp,
                leaf_rate_per_1k_chars=(
                    chosen_leaf.rate_per_1k_chars if chosen_leaf else 0.0
                ),
            )
            hold_id = f"msg:{message.message_id}:t:{target}"
            try:
                await self.broker.hold(
                    caller=caller_wallet_owner,
                    amount=hold.amount,
                    hold_id=hold_id,
                    reason=f"cast_get to {target}",
                )
            except InsufficientFundsError:
                log.warning(
                    "broker: %s out of funds for %s mid-broadcast — skipping",
                    caller_wallet_owner, target,
                )
                continue
            held[target] = {
                "hold_id": hold_id,
                "server": owning_server,
                "leaf": chosen_leaf,
            }
            deliverable_targets.append(target)

        if not deliverable_targets:
            # Nothing to send. Any held entries would already be empty
            # since no targets cleared the hold step.
            return []

        # Dispatch.
        import time as _time
        dispatch_started = _time.monotonic()
        try:
            responses = await self.bus.cast_get(
                message, deliverable_targets,
                timeout=timeout, max_responses=max_responses,
            )
        except Exception:
            # Bus error: refund every hold we placed.
            for ctx in held.values():
                try:
                    await self.broker.refund(
                        caller=caller_wallet_owner,
                        hold_id=ctx["hold_id"],
                        reason="cast_get dispatch raised",
                    )
                except Exception:
                    pass
            raise

        # Settle each response against the right per-target hold.
        responded: set[str] = set()
        for response in responses:
            source_str = str(response.source)
            if source_str in responded:
                continue  # one settlement per responder
            responded.add(source_str)
            ctx = held.get(response.source)
            if ctx is None:
                # A response from a target we didn't hold against (e.g.
                # a no-owning-server target). No settlement — the
                # message was a freebie.
                continue
            latency_ms = (_time.monotonic() - dispatch_started) * 1000.0
            response_chars = (
                len(str(response.body)) if response.body is not None else 0
            )
            try:
                await self.broker.calculate_and_settle(
                    caller=caller_wallet_owner,
                    hold_id=ctx["hold_id"],
                    server=ctx["server"],
                    leaf=ctx["leaf"],
                    response_chars=response_chars,
                    max_response_chars=max_resp,
                    actual_latency_ms=latency_ms,
                    completed_with_caller=0,
                    tier_verdict="matched",
                    responder=response.source,
                )
            except Exception:
                log.exception(
                    "cast_get settlement failed for %s; refunding hold",
                    response.source,
                )
                try:
                    await self.broker.refund(
                        caller=caller_wallet_owner,
                        hold_id=ctx["hold_id"],
                        reason="settlement raised",
                    )
                except Exception:
                    pass

        # Refund holds for targets that didn't respond.
        for target, ctx in held.items():
            if str(target) in responded:
                continue
            try:
                await self.broker.refund(
                    caller=caller_wallet_owner,
                    hold_id=ctx["hold_id"],
                    reason="no response (cast_get timeout or target silent)",
                )
            except Exception:
                pass

        return responses

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
