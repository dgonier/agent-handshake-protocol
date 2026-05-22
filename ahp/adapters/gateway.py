"""Gateway agents — tier translators between accept tiers.

A gateway agent is an :class:`AHPAgent` whose job is to receive
messages in one accept tier and emit messages in another tier. The
canonical example is an ``e→s`` gateway: it accepts embedding-tier
messages (because its ``accept`` field includes ``e``), translates
the embedding payload into a human-readable summary, and either
returns the summary as a reply or relays it onward to a string-tier
target.

Why this matters
----------------

The protocol's :class:`~ahp.core.CompatibilityMatrix` lets a code
require, say, ``{"b", "e"}`` — only agents that accept bytes or
embeddings can receive it. Without gateways, a human observer (whose
session address accepts ``s`` only) is permanently locked out of
embedding-tier traffic. With gateways, the human can target an
``e→s`` gateway and read the translated stream.

The gateway is *intentionally* a normal AHPAgent — same registration,
same broker settlement, same reputation. It earns credits for the
translation work like any other agent. That keeps the economy honest:
translation is computational work and gets paid for.

Patterns
--------

Two natural shapes that this module supports:

* **Translate-and-respond.** Gateway receives a SEND-GET, runs
  :meth:`translate`, returns the translated result as the reply.
  One-shot; the caller asked the gateway directly. This is the
  default ``handle_message`` path here.

* **Translate-and-relay.** Caller embeds a ``relay_to`` field in the
  request body. The gateway translates, forwards to the relay target,
  awaits its response, translates the response back, and returns
  that. Available via :class:`RelayingGatewayAgent`.

The :meth:`translate` hook is intentionally async — production
translators will call LLMs.

Subclasses ship in this module for the two most useful directions:

* :class:`EmbeddingToTextGateway` — describes embedding payloads in a
  human-readable summary. The default implementation is a stand-in:
  it reports payload length and a hash. Real deployments override
  :meth:`translate` with a chat-model-driven describer.
* :class:`JsonToTextGateway` — pretty-prints JSON. Useful for human
  observers consuming JSON-tier audit / consensus traffic.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ahp.adapters.base import AHPAgent
from ahp.core.address import AcceptTier, AgentAddress
from ahp.core.message import Message
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


class GatewayAgent(AHPAgent):
    """Base class for tier-translating agents.

    Subclasses override :meth:`translate`. The default
    :meth:`handle_message` calls ``translate`` and returns the result
    as the reply, preserving thread + code so the caller's
    conversation isn't fragmented.

    The agent's :attr:`accept` field must include the ``source_tier``
    (otherwise the protocol routes the message away from it). The
    ``output_tier`` is *advertised* via the agent's :attr:`AgentMeta`
    ``extra`` map, so downstream callers can discover which gateway
    translates which direction without hardcoded URLs.

    Discovery convention:

        agent.metadata.extra["gateway"] = {
            "source_tier": "e",
            "output_tier": "s",
        }

    A ``list-agents --tag gateway`` filter (or a custom
    discover-by-extra query) can surface every gateway in scope.
    """

    SOURCE_TIER: str = AcceptTier.STRING
    """Override in subclasses — the tier the gateway *receives*."""

    OUTPUT_TIER: str = AcceptTier.STRING
    """Override in subclasses — the tier the gateway *emits*."""

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        *,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> None:
        if self.SOURCE_TIER not in address.accept:
            raise ValueError(
                f"gateway {address} declares SOURCE_TIER={self.SOURCE_TIER!r} "
                f"but its accept field {address.accept!r} doesn't include it — "
                "the protocol would never route source-tier messages here"
            )
        # Stamp the gateway direction into AgentMeta so a discovery
        # query can find this gateway by tier pair.
        meta = metadata or AgentMeta()
        existing_extra = dict(meta.extra) if meta.extra else {}
        existing_extra["gateway"] = {
            "source_tier": self.SOURCE_TIER,
            "output_tier": self.OUTPUT_TIER,
        }
        meta.extra = existing_extra
        # Add a "gateway" capability tag so list-tools / list-agents
        # filters can surface it.
        if "gateway" not in meta.capabilities:
            meta.capabilities = list(meta.capabilities) + ["gateway"]
        super().__init__(address, engine, metadata=meta, **kwargs)

    # ── must override ────────────────────────────────────────────────

    async def translate(self, message: Message) -> Any:
        """Convert ``message`` into the gateway's output tier.

        Receives the full inbound :class:`Message`; returns whatever
        body shape is appropriate for ``OUTPUT_TIER``. The default
        :meth:`handle_message` wraps the returned body into a reply
        message back to the source.

        Subclasses must implement this. The default raises so an
        accidental subclass without an override fails loudly rather
        than echoing the un-translated payload.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override translate()"
        )

    # ── default handler ──────────────────────────────────────────────

    async def handle_message(self, message: Message) -> Message | None:
        """Translate and reply.

        The reply preserves ``code`` + ``thread`` so the caller's
        conversation context survives. Errors during translation are
        re-raised; the base AHPAgent will record the failure and the
        broker settlement (if wired) will refund the hold.
        """
        translated_body = await self.translate(message)
        return Message(
            source=self.address,
            target=message.source,
            verb="SEND",
            code=message.code,
            thread=message.thread,
            body=translated_body,
        )


# ── concrete stand-ins ───────────────────────────────────────────────


class EmbeddingToTextGateway(GatewayAgent):
    """``e → s`` gateway. Stand-in: describes payload shape, not content.

    Real deployments override :meth:`translate` with a chat model
    that can actually describe what the embedding represents (often
    by retrieving nearest neighbors from a labeled corpus and
    summarizing). This default exists so the contract is verifiable
    end-to-end and so a deployment that wires the gateway *before*
    swapping in a real translator still gets a response with the
    right tier, not a NotImplementedError.
    """

    SOURCE_TIER = AcceptTier.EMBEDDINGS
    OUTPUT_TIER = AcceptTier.STRING

    async def translate(self, message: Message) -> str:
        body = message.body
        if isinstance(body, (bytes, bytearray)):
            digest = hashlib.sha256(body).hexdigest()[:12]
            return (
                f"[embedding payload: {len(body)} bytes, "
                f"sha256={digest}; stand-in describer]"
            )
        if isinstance(body, list):
            # Probably a vector of floats.
            try:
                length = len(body)
                head = ", ".join(f"{float(v):.3f}" for v in body[:4])
                return (
                    f"[embedding vector: dim={length}, "
                    f"first4=[{head}{', ...' if length > 4 else ''}]; "
                    "stand-in describer]"
                )
            except (TypeError, ValueError):
                pass
        # Anything else: opaque-but-truthful.
        return f"[embedding payload of type {type(body).__name__}; stand-in describer]"


class JsonToTextGateway(GatewayAgent):
    """``j → s`` gateway. Pretty-prints JSON for human observers.

    Useful for piping JSON-tier audit / consensus messages into a
    human-readable feed. The output is deterministic so multiple
    observers see identical text — no LLM in the loop.
    """

    SOURCE_TIER = AcceptTier.JSON
    OUTPUT_TIER = AcceptTier.STRING

    async def translate(self, message: Message) -> str:
        body = message.body
        try:
            return json.dumps(body, indent=2, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(body)


# ── relaying variant ─────────────────────────────────────────────────


class RelayingGatewayAgent(GatewayAgent):
    """A gateway that translates AND forwards.

    Pattern: caller sends ``{"relay_to": "addr-string", ...payload}``
    in the body. The gateway translates the payload, forwards a new
    SEND-GET to the relay target, translates the response back, and
    returns that as the reply.

    Useful when a string-tier client wants to query a JSON-tier
    service: client → ``j→s`` gateway → JSON service → gateway →
    client. The gateway "speaks" both tiers.

    Subclasses override :meth:`translate_request` and
    :meth:`translate_response` separately because the directions are
    often asymmetric (request usually contains a short prompt, response
    is the long payload).
    """

    async def translate_request(self, message: Message) -> Any:
        """Output-tier shape of the request. Default: same as
        :meth:`translate` (forward-direction)."""
        return await self.translate(message)

    async def translate_response(
        self,
        response: Message,
        original: Message,
    ) -> Any:
        """Source-tier shape of the relayed response. Default: pass
        through unchanged. Override when reverse translation is
        needed (e.g. JSON service returns JSON and the client wants
        a string summary).
        """
        return response.body

    async def handle_message(self, message: Message) -> Message | None:
        relay_to = None
        if isinstance(message.body, dict):
            relay_to = message.body.get("relay_to")

        if not relay_to:
            # No relay target — fall back to the one-shot translation
            # behavior.
            return await super().handle_message(message)

        try:
            relay_target = AgentAddress.parse(relay_to)
        except (ValueError, TypeError):
            return Message(
                source=self.address, target=message.source,
                verb="SEND", code=message.code, thread=message.thread,
                body=f"[gateway error: invalid relay_to {relay_to!r}]",
            )

        translated = await self.translate_request(message)
        relayed = Message(
            source=self.address,
            target=relay_target,
            verb="SEND-GET",
            code=message.code,
            thread=message.thread,
            body=translated,
        )
        response = await self.engine.handle(relayed, timeout=30.0)
        if response is None:
            return Message(
                source=self.address, target=message.source,
                verb="SEND", code=message.code, thread=message.thread,
                body="[gateway: relay target did not respond]",
            )

        translated_back = await self.translate_response(response, message)
        return Message(
            source=self.address, target=message.source,
            verb="SEND", code=message.code, thread=message.thread,
            body=translated_back,
        )
