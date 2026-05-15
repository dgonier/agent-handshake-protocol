"""FastAPI debate viewer — mobile-first browser UI for the AHP demo.

Three routes:

* ``GET /``           — landing page; show the most recent debate run,
                        provide a form to kick off a new one.
* ``POST /run``       — kick off a new debate (background task).
                        Redirects back to ``/``.
* ``GET /audit``      — render the audit-event timeline from the most
                        recent run.
* ``GET /runs``       — list all persisted runs.
* ``GET /runs/{id}``  — show one specific persisted run's debate page.
* ``GET /healthz``    — liveness for docker compose.

Persistence lives in ``$VIEWER_DATA_DIR`` (default
``/data``). Each run produces one JSON file
``<id>.json`` written atomically.

Redis: pass ``REDIS_URL=redis://redis:6379/0`` to point at the compose
service. Without it the runner falls back to fakeredis.

This is an *example*, not a production service:
* No auth (LAN-only intended).
* Concurrent runs are serialized (one global lock).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

import ahp
from ahp.core.address import AgentAddress
from ahp.registry.registry import AgentRegistry
from examples.viewer.runner import DebateResult, run_debate


log = logging.getLogger("ahp.viewer")
logging.basicConfig(level=logging.INFO)


# ── secret loading ────────────────────────────────────────────────────


# Keys we *want* to pull out of the bind-mounted frontend .env.
# Important: NOT AWS_* — those would corrupt the boto3 credential chain
# (the frontend uses AWS_KEY_ID/AWS_SECRET_ACCESS_KEY, which doesn't
# match boto3's expected AWS_ACCESS_KEY_ID, and a half-set state breaks
# ChatBedrockConverse). AWS creds come from ~/.aws via the mount.
_IMPORTED_SECRET_KEYS = {
    "TAVILY_API_KEY",
    "BRAVE_API_KEY",
    "SERPER_API_KEY",
    "EXA_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
}


def _load_dotenv_file(path: str) -> int:
    """Selective load: import only keys we want into os.environ.

    Doesn't override anything already set; ignores comments, blanks,
    and any key NOT in :data:`_IMPORTED_SECRET_KEYS`. The frontend
    .env carries AWS_* with names that don't match boto3 expectations
    — leaking those into env breaks Bedrock cred resolution.
    """
    if not os.path.isfile(path):
        log.warning("no secret file at %s — TAVILY_API_KEY etc will be empty", path)
        return 0
    count = 0
    skipped: list[str] = []
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                if k not in _IMPORTED_SECRET_KEYS:
                    skipped.append(k)
                    continue
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
                    count += 1
    except Exception:
        log.exception("could not load %s", path)
        return 0
    log.info(
        "loaded %d allowlisted keys from %s; skipped %d non-allowlisted",
        count, path, len(skipped),
    )
    return count


_LOADED_SECRETS = _load_dotenv_file("/secrets/frontend.env")


# ── config ────────────────────────────────────────────────────────────


DATA_DIR = Path(os.environ.get("VIEWER_DATA_DIR", "./data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.environ.get("REDIS_URL")  # falls back to fakeredis
CLOUDWATCH_GROUP = os.environ.get(
    "VIEWER_CLOUDWATCH_GROUP", "/ahp/astrophysics-demo"
)
# Set to empty string to disable CloudWatch.
if CLOUDWATCH_GROUP == "":
    CLOUDWATCH_GROUP = None  # type: ignore[assignment]

DEFAULT_TOPIC = os.environ.get("VIEWER_DEFAULT_TOPIC", "What caused the Big Bang?")
DEFAULT_ORG = os.environ.get("VIEWER_DEFAULT_ORG", "tifin")
DEFAULT_DOMAIN = os.environ.get("VIEWER_DEFAULT_DOMAIN", "science")
DEFAULT_SUBDOMAIN = os.environ.get("VIEWER_DEFAULT_SUBDOMAIN", "astrophysics")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
)


# ── app + templates ───────────────────────────────────────────────────


HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(HERE / "templates"))


def _common_ctx() -> dict[str, Any]:
    from ahp.adapters import list_formats
    return {
        "version": ahp.__version__,
        "now": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "default_topic": DEFAULT_TOPIC,
        "default_org": DEFAULT_ORG,
        "default_domain": DEFAULT_DOMAIN,
        "default_subdomain": DEFAULT_SUBDOMAIN,
        "default_format": "debate",
        "formats": list_formats(),
        "model_id": MODEL_ID,
        "model_label": _model_label(MODEL_ID),
    }


def _model_label(model_id: str) -> str:
    """Friendly short name for the header pill."""
    tail = model_id.split(".")[-1].split(":")[0]
    return tail.replace("-v1", "").replace("-20251001", "")


# ── in-memory state ───────────────────────────────────────────────────


class _State:
    def __init__(self) -> None:
        self.latest: DebateResult | None = None
        self.running_since: float | None = None
        self.last_error: str | None = None
        self.lock = asyncio.Lock()


state = _State()


# ── persistence ───────────────────────────────────────────────────────


def _persist(result: DebateResult) -> str:
    run_id = datetime.fromtimestamp(result.started_at, timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    path = DATA_DIR / f"{run_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result.to_dict(), indent=2))
    tmp.replace(path)
    return run_id


def _load_run(run_id: str) -> DebateResult | None:
    path = DATA_DIR / f"{run_id}.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    # Reconstruct nested dataclasses lightly — we only need attribute access.
    from examples.viewer.runner import AgentTurn

    def _coerce_turn(t: dict) -> AgentTurn:
        t.setdefault("round_name", "")
        t.setdefault("tool_calls", [])
        return AgentTurn(**t)

    data["round1"] = [_coerce_turn(t) for t in data.get("round1", [])]
    data["round2"] = [_coerce_turn(t) for t in data.get("round2", [])]
    data["closing"] = [_coerce_turn(t) for t in data.get("closing", [])]
    # Back-fill fields added after older runs were saved.
    data.setdefault("org", "tifin")
    data.setdefault("format", "debate")
    data.setdefault("elapsed_closing", 0.0)
    return DebateResult(**data)


def _list_runs() -> list[dict[str, Any]]:
    out = []
    for path in sorted(DATA_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        ts = data.get("started_at", 0)
        out.append({
            "id": path.stem,
            "topic": data.get("topic", "?"),
            "subdomain": data.get("subdomain", "?"),
            "count": data.get("count", 0),
            "events": len(data.get("audit_events", [])),
            "when": datetime.fromtimestamp(ts, timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            ) if ts else "?",
        })
    return out


# ── background runner ─────────────────────────────────────────────────


async def _run_once(
    topic: str,
    format: str,
    org: str,
    domain: str,
    subdomain: str,
    count: int,
) -> None:
    async with state.lock:
        state.running_since = time.time()
        state.last_error = None
        try:
            result = await run_debate(
                topic=topic, format=format,
                org=org, domain=domain, subdomain=subdomain,
                count=count, redis_url=REDIS_URL,
                cloudwatch_group=CLOUDWATCH_GROUP,
            )
            state.latest = result
            _persist(result)
            log.info("debate done: %s (%d round1, %d round2)",
                     topic, len(result.round1), len(result.round2))
        except Exception as e:
            log.exception("debate failed")
            state.last_error = f"{type(e).__name__}: {e}"
        finally:
            state.running_since = None


# ── live registry view ────────────────────────────────────────────────


async def _live_agents() -> list[dict[str, Any]]:
    """Snapshot the live AgentRegistry — what's currently alive in Redis.

    Returns one row per agent with the fields the UI cares about. Best-
    effort: if Redis is unreachable, returns an empty list rather than
    breaking the page.
    """
    if not REDIS_URL:
        return []
    try:
        import redis.asyncio as aioredis  # type: ignore[import-not-found]
    except ImportError:
        return []
    client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        registry = AgentRegistry(client, heartbeat_ttl=60)
        # alive_only=True scans the registry hash + checks each
        # liveness key. Cheap at our scale.
        addresses = await registry.list_all(alive_only=True)
        return [_address_row(a) for a in addresses]
    except Exception:
        log.exception("live registry snapshot failed")
        return []
    finally:
        try:
            await client.aclose()
        except AttributeError:
            await client.close()


def _address_row(addr: AgentAddress) -> dict[str, Any]:
    return {
        "address": str(addr),
        "org": addr.org,
        "role": addr.role,
        "domain": addr.domain,
        "subdomain": addr.subdomain,
        "accept": addr.accept,
        "lifecycle": addr.lifecycle,
        "instance": addr.instance,
    }


# ── routes ────────────────────────────────────────────────────────────


app = FastAPI(title="AHP Debate Viewer")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    ctx = _common_ctx() | {
        "page": "debate",
        "result": state.latest,
        "running": state.running_since is not None,
        "running_for": (
            int(time.time() - state.running_since)
            if state.running_since else 0
        ),
        "error": state.last_error,
        "live_agents": await _live_agents(),
    }
    return templates.TemplateResponse(request, "debate.html", ctx)


@app.get("/api/agents")
async def api_agents():
    return {"agents": await _live_agents()}


@app.post("/run")
async def kick_run(
    background_tasks: BackgroundTasks,
    topic: str = Form(...),
    format: str = Form("debate"),
    org: str = Form("tifin"),
    domain: str = Form("science"),
    subdomain: str = Form("astrophysics"),
    count: int = Form(4),
):
    if state.running_since is not None:
        raise HTTPException(409, "another debate is already running")
    if count < 1 or count > 6:
        raise HTTPException(400, "count must be between 1 and 6")
    from ahp.adapters import FORMATS as _FORMATS
    if format not in _FORMATS:
        raise HTTPException(400, f"unknown format {format!r}")
    for name, val in [("org", org), ("domain", domain), ("subdomain", subdomain)]:
        if not val.strip() or not all(c.isalnum() or c == "-" for c in val.strip()):
            raise HTTPException(400, f"{name} must be alphanumeric / dash")
    background_tasks.add_task(
        _run_once,
        topic.strip(), format, org.strip(),
        domain.strip(), subdomain.strip(), count,
    )
    return RedirectResponse("/", status_code=303)


@app.get("/audit", response_class=HTMLResponse)
async def audit(request: Request):
    ctx = _common_ctx() | {"page": "audit", "result": state.latest}
    return templates.TemplateResponse(request, "audit.html", ctx)


@app.get("/runs", response_class=HTMLResponse)
async def runs(request: Request):
    ctx = _common_ctx() | {"page": "runs", "runs": _list_runs()}
    return templates.TemplateResponse(request, "runs.html", ctx)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(request: Request, run_id: str):
    result = _load_run(run_id)
    if result is None:
        raise HTTPException(404, "no such run")
    ctx = _common_ctx() | {
        "page": "runs",
        "result": result,
        "running": False,
        "running_for": 0,
    }
    return templates.TemplateResponse(request, "debate.html", ctx)


@app.get("/api/latest")
async def api_latest():
    """JSON snapshot of the most recent run — handy for scripting."""
    if state.latest is None:
        return JSONResponse({"latest": None}, status_code=404)
    return state.latest.to_dict()


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "version": ahp.__version__,
        "running": state.running_since is not None,
        "secrets_loaded": _LOADED_SECRETS,
        "tavily_key_present": bool(os.environ.get("TAVILY_API_KEY")),
        "model_id": MODEL_ID,
    }
