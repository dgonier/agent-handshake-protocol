"""Research tools — web search, fetch, summarize.

The first one, :func:`search_tavily`, is registered at the global
tool scope ``*.api.*.research.search_tavily`` so any agent in any org
can call it. The ``TAVILY_API_KEY`` is read from the process
environment on first use; if the key isn't present the call returns a
clearly-marked error string rather than raising — the LLM gets a
helpful message rather than an exception.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from ahp.adapters.tool_registry import tool


log = logging.getLogger(__name__)


TAVILY_URL = "https://api.tavily.com/search"
DEFAULT_MAX_RESULTS = 5
DEFAULT_SEARCH_DEPTH = "basic"  # or "advanced" — costs more credits
DEFAULT_TIMEOUT_S = 15.0


@dataclass(frozen=True)
class _Result:
    title: str
    url: str
    content: str
    score: float = 0.0


def _api_key() -> str | None:
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return None
    return key.strip() or None


@lru_cache(maxsize=512)
def _cached_search(
    query: str, max_results: int, search_depth: str,
) -> str:
    """Sync HTTP call; cached for the lifetime of the process.

    Returns a JSON-encoded string so the LLM gets stable text it can
    quote back. Uses ``urllib`` from the stdlib to avoid pulling in
    ``requests`` as a runtime dep.
    """
    key = _api_key()
    if key is None:
        return json.dumps({
            "error": "TAVILY_API_KEY not set in the container env",
        })

    import urllib.error
    import urllib.request

    payload = json.dumps({
        "api_key": key,
        "query": query,
        "search_depth": search_depth,
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        TAVILY_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_TIMEOUT_S) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        return json.dumps({
            "error": f"tavily HTTP {e.code}: {body or e.reason}",
        })
    except Exception as e:
        return json.dumps({
            "error": f"{type(e).__name__}: {e}",
        })

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return json.dumps({"error": "tavily returned non-JSON response"})

    results = []
    for item in data.get("results", [])[:max_results]:
        results.append({
            "title": item.get("title", "").strip(),
            "url": item.get("url", "").strip(),
            "content": (item.get("content") or "").strip(),
            "score": item.get("score", 0.0),
        })
    return json.dumps({"query": query, "results": results}, indent=2)


@tool(
    scope="*",
    kind="api",
    role="*",
    category="research",
    operation="search_tavily",
    description=(
        "Search the web via Tavily and return the top results as JSON. "
        "Each result has title, url, content (a 1-2 sentence snippet), "
        "and score (0-1). Use this when you need recent or external "
        "information you don't already know. Returns at most 5 results. "
        "If the API key is missing the response will contain an "
        "'error' field — don't pretend otherwise."
    ),
)
async def search_tavily(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> str:
    """Web search via Tavily.

    Args:
        query: The natural-language search query.
        max_results: Cap on the number of results returned (1-10).
    """
    if not query or not query.strip():
        return json.dumps({"error": "empty query"})
    max_results = max(1, min(10, int(max_results)))
    return await asyncio.to_thread(
        _cached_search, query.strip(), max_results, DEFAULT_SEARCH_DEPTH,
    )
