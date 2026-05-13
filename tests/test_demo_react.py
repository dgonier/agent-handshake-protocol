"""Smoke tests for the LLM-backed finance-react demo.

Most assertions run against a fake chat model so they pass in CI
without Bedrock. The one test that actually hits AWS is gated behind a
credential check + the ``AHP_RUN_BEDROCK`` env var, so it stays opt-in.
"""

from __future__ import annotations

import os
import warnings

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ahp.demo.finance_react import run
from ahp.llm import has_aws_credentials


pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    "ignore::PendingDeprecationWarning",
)


class _ToolableFakeChat(FakeListChatModel):
    """FakeListChatModel + a no-op bind_tools so create_react_agent accepts it."""

    def bind_tools(self, tools, **kwargs):  # type: ignore[override]
        return self


async def test_react_demo_runs_with_fake_model(redis_client):
    """Same shape as the stub demo's end-to-end test, but with ReactAgent + fakes."""
    # The fake cycles through these responses for bull / bear / etc. With
    # only two adversarial agents we just need two scripted replies; the
    # FakeListChatModel cycles if asked for more.
    model = _ToolableFakeChat(responses=[
        "BULL bullets: 1) moat 2) execution 3) TAM 4) cash 5) optionality",
        "BEAR bullets: 1) regulation 2) margins 3) competition 4) macro 5) valuation",
    ])

    result = await run(
        redis_client=redis_client, model=model, question="Tesla",
    )

    assert result.first_reply is not None
    body = result.first_reply.body
    assert "Fundamentals:" in body
    assert "Bull view:" in body
    assert "Bear view:" in body
    assert "Revenue $96B" in body
    # Fake answers landed somewhere in the brief.
    assert "BULL bullets" in body or "BEAR bullets" in body


async def test_react_demo_warm_query_hits_cache(redis_client):
    model = _ToolableFakeChat(responses=[
        "BULL: x", "BEAR: y",
        # No further responses — the second pass should not consume any.
    ])
    result = await run(
        redis_client=redis_client, model=model, question="Tesla",
    )
    assert result.first_reply is not None
    assert result.second_reply is not None
    assert result.first_reply.body == result.second_reply.body
    assert result.second_seconds < max(result.first_seconds, 0.01)


def test_run_without_aws_credentials_raises(monkeypatch):
    """If no model is passed and no creds are discoverable, run() should fail loudly."""
    import ahp.demo.finance_react as fr

    monkeypatch.setattr(fr, "has_aws_credentials", lambda: False)
    with pytest.raises(RuntimeError, match="no AWS credentials"):
        import asyncio
        asyncio.run(fr.run(question="Tesla"))


@pytest.mark.skipif(
    not has_aws_credentials() or os.environ.get("AHP_RUN_BEDROCK") != "1",
    reason="set AWS_PROFILE (or env vars) and AHP_RUN_BEDROCK=1 to hit real Bedrock",
)
async def test_react_demo_against_real_bedrock(redis_client):  # pragma: no cover
    """Live smoke test against AWS Bedrock. Off by default."""
    result = await run(redis_client=redis_client, question="Tesla")
    assert result.first_reply is not None
    assert "Fundamentals:" in result.first_reply.body
