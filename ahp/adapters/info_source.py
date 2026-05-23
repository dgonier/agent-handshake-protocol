"""Information-source agents — addressable, queryable knowledge bases.

An information source is an :class:`AHPAgent` whose job is to answer
``info.*`` queries against a backing store. It can be backed by a
document KB, a vector store, a SQL database, a knowledge graph, an
external API — the *kind* of backend is abstracted away. What's
visible on the network is:

* A normal 7-field :class:`AgentAddress` (convention: role contains
  ``data-source`` or similar; the demos use ``role="data-source"``
  but nothing enforces it).
* The agent's :attr:`accept` field declaring which data tiers it can
  emit (``s`` for text snippets, ``j`` for structured records,
  ``e`` for raw embeddings, etc).
* The :class:`CompatibilityMatrix` requirements on the ``info.*``
  codes — callers' messages are tier-checked at routing time just
  like any other protocol traffic.

Querying an info source is a normal ``SEND-GET`` with a code from
the ``info.*`` family. The broker bills the call. If a caller and
source disagree on tier (e.g. caller wants text but source emits
embeddings), the existing :class:`~ahp.adapters.gateway.GatewayAgent`
pattern bridges — no parallel protocol.

Subclasses ship in this module for the two most common backings:

* :class:`StaticDocumentSource` — a frozen dict of ``name → text``,
  naive keyword search. Useful for policy docs, FAQs, regulation
  text — small, stable, hand-curated corpora.
* :class:`KGBackedSource` — wraps a :class:`KnowledgeGraphBackend`.
  Handles both ``info.query`` (returns text node summaries) and
  ``info.query.embedding`` (returns raw similarity hits). Real
  deployments wire Neo4j here.

Discovery convention: the constructor stamps
``capabilities=["info-source"]`` and an ``extra["info_source"]``
descriptor into :class:`AgentMeta`, so ``ahp list-agents`` filters
can surface every info source in scope. The descriptor records the
declared output tiers and the backend label.
"""

from __future__ import annotations

from typing import Any

from ahp.adapters.base import AHPAgent
from ahp.core.address import AgentAddress
from ahp.core.codes import Code
from ahp.core.message import Message
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentMeta


class InfoSourceAgent(AHPAgent):
    """Base class for queryable, addressable information sources.

    Subclasses set :attr:`OUTPUT_TIERS` (the tiers this source can
    emit) and override one or more of :meth:`handle_query`,
    :meth:`handle_query_embedding`, :meth:`handle_list`,
    :meth:`handle_write`. The default :meth:`handle_message`
    dispatches on the inbound code and routes to the right hook.

    The constructor validates that the agent's address ``accept``
    field includes at least one of the declared output tiers — a
    source declaring it can emit embeddings (``"e"``) must accept
    embeddings; otherwise the protocol would route embedding-tier
    requests away from it.

    Subclasses should also override :attr:`BACKEND_LABEL` so the
    discovery descriptor records what kind of store backs the source
    ("static-document", "neo4j", "sqlite", "vector", ...). Pure
    metadata; doesn't affect protocol behavior.
    """

    OUTPUT_TIERS: set[str] = {"s"}
    """Tiers this source can emit. Override in subclasses."""

    BACKEND_LABEL: str = "abstract"
    """Short label for what backs this source. Pure metadata."""

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        *,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> None:
        if not (self.OUTPUT_TIERS & set(address.accept)):
            raise ValueError(
                f"info source {address} declares OUTPUT_TIERS="
                f"{sorted(self.OUTPUT_TIERS)} but its accept field "
                f"{address.accept!r} doesn't include any of them. The "
                "protocol would route output-tier requests away from "
                "this agent — fix the address or the OUTPUT_TIERS."
            )
        meta = metadata or AgentMeta()
        existing_extra = dict(meta.extra) if meta.extra else {}
        existing_extra["info_source"] = {
            "output_tiers": sorted(self.OUTPUT_TIERS),
            "backend": self.BACKEND_LABEL,
        }
        meta.extra = existing_extra
        if "info-source" not in meta.capabilities:
            meta.capabilities = list(meta.capabilities) + ["info-source"]
        super().__init__(address, engine, metadata=meta, **kwargs)

    # ── default handler: dispatch on code family ─────────────────────

    async def handle_message(self, message: Message) -> Message | None:
        """Route by code to the right handler.

        Returns ``None`` (drops the message silently) for codes outside
        the info family — keeps the source from accidentally responding
        to e.g. ``human.query`` messages addressed to its address.
        """
        if message.code == Code.INFO_QUERY:
            body = await self.handle_query(message)
        elif message.code == Code.INFO_QUERY_EMBEDDING:
            body = await self.handle_query_embedding(message)
        elif message.code == Code.INFO_LIST:
            body = await self.handle_list(message)
        elif message.code == Code.INFO_WRITE:
            body = await self.handle_write(message)
        else:
            return None
        if body is None:
            return None
        return Message(
            source=self.address,
            target=message.source,
            verb="SEND",
            code=message.code,
            thread=message.thread,
            body=body,
        )

    # ── subclass hooks (default: NotImplemented) ─────────────────────

    async def handle_query(self, message: Message) -> Any:
        """Handle a text-style query. Return the body (str or dict) the
        caller will see, or ``None`` to drop.

        Subclasses override. Default raises so a misconfigured subclass
        fails loudly rather than silently echoing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} doesn't handle info.query — "
            "override handle_query() or don't claim INFO_QUERY support"
        )

    async def handle_query_embedding(self, message: Message) -> Any:
        """Handle a vector-similarity query. Inbound body is the query
        vector (list of floats); return a list of hits."""
        raise NotImplementedError(
            f"{type(self).__name__} doesn't handle info.query.embedding"
        )

    async def handle_list(self, message: Message) -> Any:
        """Handle a list/enumerate request. Inbound body may carry
        filter params; return a list of items."""
        raise NotImplementedError(
            f"{type(self).__name__} doesn't handle info.list"
        )

    async def handle_write(self, message: Message) -> Any:
        """Handle a write request. Most info sources are read-only;
        override only when the source is mutable."""
        raise NotImplementedError(
            f"{type(self).__name__} is read-only (handle_write not overridden)"
        )


# ── concrete: StaticDocumentSource ───────────────────────────────────


class StaticDocumentSource(InfoSourceAgent):
    """Naive keyword-search over a frozen dict of named documents.

    Each document is ``(name, text)``. A query returns up to
    ``top_k`` documents whose text contains the most query terms,
    scored by raw term-overlap. Deterministic, dependency-free,
    perfect for policy docs / FAQs / regulation text — small, stable,
    hand-curated corpora where building a real index is overkill.

    Output tier is ``s`` (text snippets). Real production sources
    would emit ``j`` to carry structured metadata alongside; this
    stand-in stays text-only to keep the contract minimal.
    """

    OUTPUT_TIERS = {"s"}
    BACKEND_LABEL = "static-document"

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        documents: dict[str, str],
        *,
        default_top_k: int = 3,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(address, engine, metadata=metadata, **kwargs)
        self._documents = dict(documents)
        self._default_top_k = default_top_k

    async def handle_query(self, message: Message) -> Any:
        query = self._extract_query(message)
        top_k = self._extract_top_k(message)
        terms = [t.lower() for t in query.split() if t]
        scored: list[tuple[int, str, str]] = []
        for name, text in self._documents.items():
            text_lower = text.lower()
            score = sum(text_lower.count(term) for term in terms)
            if score > 0:
                scored.append((score, name, text))
        scored.sort(key=lambda x: (-x[0], x[1]))
        snippets = [
            f"[{name}] {text}" for _score, name, text in scored[:top_k]
        ]
        return "\n\n".join(snippets) if snippets else (
            f"(no matches in {self.BACKEND_LABEL} source for {query!r})"
        )

    async def handle_list(self, message: Message) -> Any:
        return {"documents": sorted(self._documents.keys())}

    def _extract_query(self, message: Message) -> str:
        body = message.body
        if isinstance(body, str):
            return body
        if isinstance(body, dict):
            return str(body.get("query") or body.get("q") or "")
        return str(body)

    def _extract_top_k(self, message: Message) -> int:
        if isinstance(message.body, dict):
            try:
                return int(message.body.get("top_k", self._default_top_k))
            except (TypeError, ValueError):
                pass
        return self._default_top_k


# ── concrete: KGBackedSource ─────────────────────────────────────────


class KGBackedSource(InfoSourceAgent):
    """Info source backed by a :class:`KnowledgeGraphBackend`.

    Handles both ``info.query`` (text) and ``info.query.embedding``
    (vector). The text path falls back to a node-name keyword search
    because in-memory KGs don't carry a full-text index; production
    Neo4j-backed deployments would override :meth:`handle_query` to
    use Cypher's CONTAINS / fulltext index.

    Output tiers: ``s`` (text summaries) and ``e`` (raw embeddings).
    The agent's address must accept both — the constructor's
    OUTPUT_TIERS check enforces this.
    """

    OUTPUT_TIERS = {"s", "e"}
    BACKEND_LABEL = "knowledge-graph"

    def __init__(
        self,
        address: AgentAddress,
        engine: ProtocolEngine,
        backend: Any,  # KnowledgeGraphBackend protocol
        *,
        kind_filter: str | None = None,
        default_top_k: int = 5,
        metadata: AgentMeta | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(address, engine, metadata=metadata, **kwargs)
        self._backend = backend
        self._kind_filter = kind_filter
        self._default_top_k = default_top_k

    async def handle_query(self, message: Message) -> Any:
        """Keyword-match against node labels + props.

        Naive (linear scan); real Neo4j deployments override with a
        Cypher fulltext query. Returns the matched nodes' label +
        any string-valued props as a joined text block.
        """
        query = (
            message.body if isinstance(message.body, str)
            else str(
                (message.body or {}).get("query", "")
                if isinstance(message.body, dict) else message.body
            )
        )
        terms = [t.lower() for t in query.split() if t]
        nodes = self._backend.list_nodes(kind=self._kind_filter)
        scored: list[tuple[int, Any]] = []
        for node in nodes:
            # Build a haystack from the label and any string-valued
            # props. KGNode doesn't enforce a "summary" field, so we
            # look at whatever string props the caller stashed.
            prop_strs = [
                str(v) for v in (node.props or {}).values()
                if isinstance(v, str)
            ]
            haystack = " ".join([node.label, *prop_strs]).lower()
            score = sum(haystack.count(term) for term in terms)
            if score > 0:
                scored.append((score, node))
        scored.sort(key=lambda x: -x[0])
        top = scored[:self._default_top_k]
        if not top:
            return f"(no KG matches for {query!r})"
        return "\n\n".join(
            f"[{n.label or n.id}] " + " ".join(
                str(v) for v in (n.props or {}).values()
                if isinstance(v, str)
            )
            for _score, n in top
        )

    async def handle_query_embedding(self, message: Message) -> Any:
        """Vector similarity via the backend's query_by_similarity.

        Inbound body is the query vector (list of floats). Returns a
        list of ``{node_id, name, score}`` dicts so callers can render
        as needed; the raw embeddings stay on the source side.
        """
        body = message.body
        vector: list[float] = []
        top_k = self._default_top_k
        if isinstance(body, list):
            vector = [float(x) for x in body]
        elif isinstance(body, dict):
            raw = body.get("embedding") or body.get("vector") or []
            try:
                vector = [float(x) for x in raw]
            except (TypeError, ValueError):
                vector = []
            try:
                top_k = int(body.get("top_k", top_k))
            except (TypeError, ValueError):
                pass
        if not vector:
            return {"hits": [], "error": "no vector in request body"}
        hits = self._backend.query_by_similarity(
            vector, top_k=top_k, kind=self._kind_filter,
        )
        return {
            "hits": [
                {
                    "node_id": h.node.id,
                    "label": h.node.label,
                    "kind": h.node.kind,
                    "score": round(float(h.score), 6),
                }
                for h in hits
            ],
        }

    async def handle_list(self, message: Message) -> Any:
        kind = self._kind_filter
        if isinstance(message.body, dict):
            kind = message.body.get("kind", kind)
        nodes = self._backend.list_nodes(kind=kind)
        return {
            "nodes": [
                {"node_id": n.id, "label": n.label, "kind": n.kind}
                for n in nodes
            ],
        }
