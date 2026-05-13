"""End-to-end test for the finance-analysis demo.

Verifies that:

* the deep researcher fans out to bull + bear + data and composes a single brief;
* the data agent's fundamentals appear in the response;
* a second identical query short-circuits at the response cache;
* cache hit is substantially faster than the cold path.
"""

from __future__ import annotations

import pytest

from ahp.demo.finance_analysis import (
    BEAR_URI,
    BULL_URI,
    DATA_URI,
    HUMAN_URI,
    RESEARCHER_URI,
    run,
)


async def test_demo_cold_query_synthesizes_all_views(redis_client):
    result = await run(redis_client=redis_client, question="Tesla")

    assert result.first_reply is not None
    body = result.first_reply.body

    # The composed brief mentions data, bull, and bear sections.
    assert "Fundamentals:" in body
    assert "Bull view:" in body
    assert "Bear view:" in body

    # Data agent's canned fixtures are present.
    assert "Revenue $96B" in body
    # Bull and bear stubs each contribute their flavor.
    assert "Bull case for Tesla" in body
    assert "Bear case for Tesla" in body


async def test_demo_warm_query_hits_cache(redis_client):
    result = await run(redis_client=redis_client, question="Tesla")

    # Both replies present and identical-bodied.
    assert result.first_reply is not None
    assert result.second_reply is not None
    assert result.first_reply.body == result.second_reply.body

    # Cache hit should be at least an order of magnitude faster than the cold
    # path. (Generous bound to keep this stable on slow CI.)
    assert result.second_seconds < result.first_seconds * 0.5
    assert result.second_seconds < 0.05  # under 50ms


async def test_demo_unknown_ticker_falls_back_gracefully(redis_client):
    """Data agent reports "no fixtures" for unknown tickers; pipeline still composes."""
    result = await run(redis_client=redis_client, question="Unknown")
    assert result.first_reply is not None
    assert "no fixtures for 'Unknown'" in result.first_reply.body
    # Bull/bear still produce their canned cases for any ticker string.
    assert "Bull case for Unknown" in result.first_reply.body
    assert "Bear case for Unknown" in result.first_reply.body


async def test_demo_human_view_records_observations(redis_client):
    """The HumanAgent observation callback captures cold + warm announcements."""
    result = await run(redis_client=redis_client, question="Tesla")
    # Two section markers from the demo plus two reply bodies.
    cold = next(
        (s for s in result.human_view if "asks the researcher (cold)" in s), None
    )
    warm = next(
        (s for s in result.human_view if "asks the same question (warm)" in s),
        None,
    )
    assert cold is not None
    assert warm is not None
    # The composed brief shows up at least twice (once cold, once warm).
    briefs = [s for s in result.human_view if "Analysis for Tesla" in s]
    assert len(briefs) == 2


async def test_demo_agents_registered_under_expected_uris(redis_client):
    """After the demo runs, the registry should have seen all five canonical URIs."""
    from ahp.core.address import AgentAddress
    from ahp.registry.registry import AgentRegistry

    await run(redis_client=redis_client, question="Tesla")
    registry = AgentRegistry(redis_client)

    # Demo deregisters via stop() but doesn't deregister explicitly; entries
    # remain. (Liveness markers may have expired, but registry hash persists.)
    for uri in [HUMAN_URI, RESEARCHER_URI, BULL_URI, BEAR_URI, DATA_URI]:
        meta = await registry.get(AgentAddress.parse(uri))
        assert meta is not None, f"missing registry entry for {uri}"
