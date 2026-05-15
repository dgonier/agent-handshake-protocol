"""Tests for the SLM-driven Inviter."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ahp.adapters.inviter import AgentInvitation, Inviter


@dataclass
class _FakeResp:
    content: str


class _ScriptedModel:
    """Returns the next scripted string each .invoke() call."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if not self._replies:
            raise RuntimeError("scripted model exhausted")
        return _FakeResp(content=self._replies.pop(0))


GOOD = '''
{
  "agents": [
    {"slug": "inflation", "system": "You hold inflation drove the Big Bang."},
    {"slug": "cyclic", "system": "You hold the universe is cyclic."},
    {"slug": "quantum", "system": "You hold it's a quantum fluctuation."},
    {"slug": "simulation", "system": "You hold we are simulated."}
  ]
}
'''


async def test_invite_happy_path():
    model = _ScriptedModel([GOOD])
    inviter = Inviter(model)
    slate = await inviter.invite(
        domain="science", subdomain="astrophysics",
        topic="What caused the Big Bang?", count=4,
        mode_hint="adversarial debate",
    )
    assert len(slate) == 4
    assert all(isinstance(x, AgentInvitation) for x in slate)
    assert [x.slug for x in slate] == ["inflation", "cyclic", "quantum", "simulation"]
    # The mode hint and topic should both appear in the prompt.
    assert "adversarial debate" in model.prompts[0]
    assert "Big Bang" in model.prompts[0]


async def test_invite_handles_fenced_json():
    fenced = "Here you go:\n```json\n" + GOOD + "\n```\nDone."
    model = _ScriptedModel([fenced])
    inviter = Inviter(model)
    slate = await inviter.invite(
        domain="science", subdomain="astrophysics",
        topic="x", count=4,
    )
    assert len(slate) == 4


async def test_invite_handles_prose_wrapped_json():
    prose = "Sure! " + GOOD + " — hope that helps."
    model = _ScriptedModel([prose])
    inviter = Inviter(model)
    slate = await inviter.invite(
        domain="science", subdomain="astrophysics",
        topic="x", count=4,
    )
    assert len(slate) == 4


async def test_invite_retries_on_garbage_then_succeeds():
    model = _ScriptedModel(["not json at all", GOOD])
    inviter = Inviter(model, max_retries=2)
    slate = await inviter.invite(
        domain="science", subdomain="astrophysics",
        topic="x", count=4,
    )
    assert len(slate) == 4
    # The retry prompt was tightened.
    assert "STRICT JSON" in model.prompts[1]


async def test_invite_raises_when_retries_exhausted():
    model = _ScriptedModel(["garbage1", "garbage2", "garbage3"])
    inviter = Inviter(model, max_retries=2)
    with pytest.raises(RuntimeError, match="exhausted retries"):
        await inviter.invite(
            domain="science", subdomain="astrophysics",
            topic="x", count=4,
        )


async def test_invite_rejects_wrong_count():
    too_few = '''{"agents":[{"slug":"a","system":"s"}]}'''
    # Three garbage tries → all return same too-few payload.
    model = _ScriptedModel([too_few] * 3)
    inviter = Inviter(model, max_retries=2)
    with pytest.raises(RuntimeError, match="exhausted retries"):
        await inviter.invite(
            domain="science", subdomain="astrophysics",
            topic="x", count=4,
        )


async def test_invite_rejects_duplicate_slug():
    dup = '''{"agents":[
        {"slug":"a","system":"s1"},
        {"slug":"a","system":"s2"}
    ]}'''
    model = _ScriptedModel([dup] * 3)
    inviter = Inviter(model, max_retries=2)
    with pytest.raises(RuntimeError):
        await inviter.invite(
            domain="science", subdomain="astrophysics",
            topic="x", count=2,
        )


async def test_invite_rejects_bad_slug_format():
    bad = '''{"agents":[
        {"slug":"Has Spaces","system":"s1"},
        {"slug":"ok-slug","system":"s2"}
    ]}'''
    model = _ScriptedModel([bad] * 3)
    inviter = Inviter(model, max_retries=2)
    with pytest.raises(RuntimeError):
        await inviter.invite(
            domain="science", subdomain="astrophysics",
            topic="x", count=2,
        )


def test_invite_rejects_zero_count():
    inviter = Inviter(_ScriptedModel([]))
    import asyncio
    with pytest.raises(ValueError):
        asyncio.run(inviter.invite(
            domain="x", subdomain="y", topic="z", count=0,
        ))
