"""Tests for the address-keyed ResourceRegistry + @resource decorator."""

from __future__ import annotations

import pytest

from ahp.adapters.resources import ResourceRegistry
from ahp.adapters.tool_address import ResourceAddress
from ahp.core.address import AgentAddress


def _agent(uri: str) -> AgentAddress:
    return AgentAddress.parse(uri)


# ── decorator: class or callable as factory ────────────────────────────


def test_decorate_a_class():
    reg = ResourceRegistry()

    @reg.resource("tifin", "fs", "finance", "documents")
    class FinanceDocs:
        def __init__(self):
            self.root = "/data/finance"

    bindings = list(reg.bindings())
    assert len(bindings) == 1
    assert bindings[0].address.name == "FinanceDocs"
    inst = reg.get("tifin.fs.finance.documents.FinanceDocs")
    assert inst.root == "/data/finance"


def test_decorate_a_factory_function_with_explicit_name():
    reg = ResourceRegistry()

    @reg.resource("tifin", "vector", "finance", "filings", name="sec-edgar")
    def make_vec():
        return {"client": "fake-chroma"}

    inst = reg.get("tifin.vector.finance.filings.sec-edgar")
    assert inst == {"client": "fake-chroma"}


def test_duplicate_address_rejected():
    reg = ResourceRegistry()

    @reg.resource("tifin", "fs", "finance", "documents", name="docs")
    def make_docs(): return object()

    with pytest.raises(ValueError, match="already registered"):
        @reg.resource("tifin", "fs", "finance", "documents", name="docs")
        def make_docs2(): return object()


# ── lazy instantiation + memoization ──────────────────────────────────


def test_lazy_instantiation():
    reg = ResourceRegistry()
    calls = {"n": 0}

    @reg.resource("tifin", "fs", "finance", "documents", name="docs")
    def make_docs():
        calls["n"] += 1
        return object()

    # Registration alone should not construct.
    assert calls["n"] == 0
    inst1 = reg.get("tifin.fs.finance.documents.docs")
    assert calls["n"] == 1
    inst2 = reg.get("tifin.fs.finance.documents.docs")
    assert calls["n"] == 1                      # memoized
    assert inst1 is inst2


# ── access scope ─────────────────────────────────────────────────────


def test_default_allowed_for_filters_by_domain_subdomain():
    reg = ResourceRegistry()

    @reg.resource("tifin", "fs", "finance", "documents", name="fin-docs")
    def make_docs(): return {"who": "fin"}

    @reg.resource("tifin", "fs", "science", "papers", name="sci-papers")
    def make_papers(): return {"who": "sci"}

    fin_agent = _agent("tifin.adversarial.finance.documents.s.session.f")
    sci_agent = _agent("tifin.collaborative.science.papers.s.session.x")

    fin_res = reg.for_address(fin_agent)
    sci_res = reg.for_address(sci_agent)
    assert list(fin_res.keys()) == ["fin-docs"]
    assert list(sci_res.keys()) == ["sci-papers"]


def test_explicit_allowed_for_override():
    reg = ResourceRegistry()

    @reg.resource("tifin", "fs", "finance", "documents", name="restricted",
                  allowed_for="tifin.adversarial.*.*.*.*.*")
    def make_secret(): return {"secret": True}

    adv = _agent("tifin.adversarial.finance.documents.s.session.f")
    collab = _agent("tifin.collaborative.finance.documents.s.session.f")
    assert "restricted" in reg.for_address(adv)
    assert "restricted" not in reg.for_address(collab)


# ── cleanup ─────────────────────────────────────────────────────────


async def test_close_all_calls_cleanups():
    reg = ResourceRegistry()
    cleaned: list[str] = []

    @reg.resource("tifin", "fs", "x", "y", name="r1",
                  cleanup=lambda inst: cleaned.append(inst["id"]))
    def r1(): return {"id": "r1"}

    @reg.resource("tifin", "fs", "x", "y", name="r2",
                  cleanup=lambda inst: cleaned.append(inst["id"]))
    def r2(): return {"id": "r2"}

    reg.get("tifin.fs.x.y.r1")
    reg.get("tifin.fs.x.y.r2")
    await reg.close_all()

    # Reverse-construction order: r2 then r1.
    assert cleaned == ["r2", "r1"]


async def test_close_all_handles_async_cleanup():
    reg = ResourceRegistry()
    cleaned: list[str] = []

    async def acleanup(inst):
        cleaned.append(inst["id"])

    @reg.resource("tifin", "vector", "x", "y", name="vec", cleanup=acleanup)
    def make_vec(): return {"id": "vec"}

    reg.get("tifin.vector.x.y.vec")
    await reg.close_all()
    assert cleaned == ["vec"]


async def test_auto_cleanup_from_aclose():
    reg = ResourceRegistry()
    closed: list[bool] = []

    class Client:
        async def aclose(self):
            closed.append(True)

    @reg.resource("tifin", "db", "x", "y")
    class _DBClient(Client):
        def __init__(self): pass

    reg.get("tifin.db.x.y._DBClient")
    await reg.close_all()
    assert closed == [True]


async def test_close_all_clears_instances_so_can_close_again():
    reg = ResourceRegistry()

    @reg.resource("tifin", "fs", "x", "y", name="r")
    def make_r(): return {}

    reg.get("tifin.fs.x.y.r")
    await reg.close_all()
    # No instances left, second close is a no-op.
    await reg.close_all()


def test_cannot_unregister_live_resource():
    reg = ResourceRegistry()

    @reg.resource("tifin", "fs", "x", "y", name="r")
    def make_r(): return {}

    reg.get("tifin.fs.x.y.r")  # instantiate
    with pytest.raises(RuntimeError, match="cannot unregister"):
        reg.unregister("tifin.fs.x.y.r")
