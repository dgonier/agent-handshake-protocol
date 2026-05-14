# AHP × FastAPI example

A complete FastAPI service built **on top of** the `ahp` package — not
inside it. Copy this directory into your own project as a starting
point.

## What this shows

* How to wire `AgentFactory` + agents inside a FastAPI lifespan so the
  HTTP process IS the AHP runtime.
* How HTTP handlers can self-call the protocol via the engine the same
  way intra-process agents do.
* A live WebSocket tap (`/observe`) that streams CAST-SUB traffic.

## Layout

```
examples/fastapi_serve/
├── server.py          # generic build_app(factory, agents=...) factory
├── app.py             # runnable uvicorn entry — wires the finance demo
├── requirements.txt   # FastAPI + uvicorn + httpx (NOT ahp deps)
└── README.md          # this file
```

`server.py` is the only piece that talks to FastAPI directly. It takes
an already-built `AgentFactory` and exposes the protocol over
HTTP/WebSocket — agnostic to which agents you've spun up.

`app.py` is the demo wiring: it builds a stub or Bedrock-driven
finance stack and hands it to `build_app(...)`.

## Run

```bash
pip install -e ../../ ".[deepagents,aws]"   # the library + LLM extras
pip install -r requirements.txt             # FastAPI extras

# Deterministic (no LLM):
uvicorn app:app --reload

# Bedrock-driven (needs AWS credentials):
AHP_DEMO_VARIANT=react uvicorn app:app
```

Then:

```bash
# Ask the researcher a question
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"target": "demo.collaborative.finance.equities.s.session.researcher",
       "body": "Tesla", "thread": "thread::devin", "timeout": 30}'

# See every adversarial.* message live
websocat 'ws://localhost:8000/observe?pattern=*.adversarial.*.*.*.*.*&code=adversarial.*'

# Inspect registered tools / resources
curl http://localhost:8000/tools
curl http://localhost:8000/resources

# Read the thread history
curl http://localhost:8000/threads/thread::devin
```

## Endpoints

| Verb | Path | Body / Query | Purpose |
|------|------|------|---------|
| `POST` | `/query` | `{target, body, code?, thread?, timeout?}` | SEND-GET to a single target |
| `POST` | `/send` | full `{source, target, target_kind, verb, code, ...}` | arbitrary AHP message |
| `GET` | `/agents` | `?alive_only=true` | list registered agents |
| `GET` | `/threads/{id}` | `?tier_filter=s&count=20` | thread history |
| `GET` | `/tools` | — | tool addresses |
| `GET` | `/resources` | — | resource addresses |
| `WS` | `/observe` | `?pattern=...&code=...` | live tap stream |

## Why FastAPI lives here, not inside `ahp/`

`ahp` is the protocol library — addressing, routing, registries,
adapters, engine. FastAPI is one of many possible front-ends; putting
it inside the package would force every user of `ahp` to think about
HTTP. Keep the library framework-agnostic; let consumers (this
directory, or your own service) wire the HTTP layer they want.
