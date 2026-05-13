# AHP — Agentic Handshake Protocol

A typed, addressable, format-aware messaging protocol for heterogeneous
agents. Agents are identified by structured URIs, exchange messages
tagged with hierarchical interaction codes, and negotiate payload format
through a single shared compatibility matrix.

This repository currently implements **Phase 1 — core primitives**
(`ahp.core`). Transport, registry, engine, and adapters land in later
phases.

## Status

| Phase | Module | State |
|-------|--------|-------|
| 1 | `ahp.core` (addresses, patterns, codes, messages, compatibility) | implemented |
| 2 | `ahp.transport`, `ahp.registry` | not started |
| 3 | `ahp.engine` | not started |
| 4 | `ahp.adapters` (LangGraph, DSPy, deep agent, human) | not started |
| 5 | `ahp.demo` | not started |

## Install

```bash
pip install -e ".[test]"
```

`ahp.core` has zero runtime dependencies — only the test extras pull in
`pytest` and `hypothesis`.

## Address Format

```
{org}.{role}.{domain}.{subdomain}.{accept}.{lifecycle}.{instance}?{params}
```

Example: `tifin.adversarial.finance.projections.j.longterm.frank?stock=Tesla`

| Field | Meaning |
|-------|---------|
| `org` | Namespace (`tifin`, `public`, `user-devin`) |
| `role` | `adversarial`, `collaborative`, `interview`, `human`, … |
| `domain` | Top-level subject (`finance`, `science`, `general`) |
| `subdomain` | Specialization (`projections`, `biology`) |
| `accept` | Payload tier set in canonical order (`s`, `j`, `b`, `e` → string, JSON, bytes, embeddings); combinations like `sj`, `sjbe` |
| `lifecycle` | `longterm` / `session` / `ephemeral` / `stale-ok` (drives cache TTL) |
| `instance` | Agent identity within the same role/domain |
| `params` | URL-encoded query string |

```python
from ahp.core import AgentAddress

addr = AgentAddress.parse("tifin.adversarial.finance.projections.j.longterm.frank?stock=Tesla")
addr.accepts("j")               # True
addr.cache_key("interview.text")  # deterministic SHA-256
```

## Patterns

Wildcard patterns route broadcast messages. Accept fields use **subset
semantics** — a pattern accept of `sj` requires the target to accept at
least both `s` and `j`.

```python
from ahp.core import AddressPattern

pat = AddressPattern.parse("*.adversarial.science.*.s.*.*")
pat.matches(addr)
```

## Interaction Codes

Hierarchical, dot-delimited code constants live on `Code`:

```python
from ahp.core import Code

Code.INTERVIEW_TEXT          # "interview.text"
Code.ADVERSARIAL_DEBATE      # "adversarial.debate"
Code.family(Code.HUMAN_HALT) # "human"
Code.matches("interview.text", "interview.*")  # True
```

## Messages

```python
from ahp.core import AgentAddress, AddressPattern, Code, Message

src = AgentAddress.parse("demo.collaborative.finance.equities.s.session.alice")
pat = AddressPattern.parse("*.adversarial.finance.*.s.*.*")

msg = Message(
    source=src,
    target=pat,
    verb="CAST-GET",
    code=Code.ADVERSARIAL_DEBATE,
    thread="thread::tesla-12m",
    body="Make the case against Tesla.",
)
msg.is_broadcast       # True
msg.expects_response   # True
msg.ttl                # 3600 (derived from "session" lifecycle)
restored = Message.from_dict(msg.to_dict())
```

Valid verbs: `SEND`, `SEND-GET`, `CAST`, `CAST-GET`, `CAST-SUB`,
`INVALIDATE`.

## Compatibility Matrix

```python
from ahp.core import CompatibilityMatrix, Code, AgentAddress

m = CompatibilityMatrix()
m.required_tiers(Code.INTERVIEW_EMBEDDINGS)   # {"b", "e"}
m.can_route(src, AgentAddress.parse("o.r.d.sd.j.session.i"), Code.INTERVIEW_SCHEMA)  # True
```

A target satisfies a code if its `accept` set intersects the code's
required tier set (any-of semantics).

## Tests

```bash
pytest
```

101 tests cover address round-trips, pattern semantics, code helpers,
message envelope validation/serialization, and the compatibility matrix,
plus Hypothesis property tests for parse/serialize stability.

## Layout

```
ahp/
└── core/
    ├── address.py        AgentAddress
    ├── pattern.py        AddressPattern
    ├── codes.py          Code constants + family helpers
    ├── message.py        Message envelope + verbs + TTL table
    └── compatibility.py  CompatibilityMatrix
tests/
    ├── test_address.py
    ├── test_pattern.py
    ├── test_codes.py
    ├── test_message.py
    └── test_compatibility.py
```
