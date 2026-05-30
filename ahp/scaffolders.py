"""Project-template scaffolders for the AHP CLI.

``ahp new <thing> --name <X> [--depth skeleton|starter]`` generates a
boilerplate file in a convention path (``./tools/``,
``./integrations/``, ``./agents/``).

Three scaffolders:

* :func:`scaffold_tool`        — ``./tools/{name}.py``
* :func:`scaffold_integration` — ``./integrations/{name}.py``
* :func:`scaffold_agent`       — ``./agents/{name}.py``

Each writes one file, refuses to overwrite an existing file unless
``force=True``, and returns the absolute path written. The CLI
(:mod:`ahp.cli`) wires arg-parsing on top.

Two depths:

* ``skeleton`` — bare scaffold. Every hook raises
  ``NotImplementedError`` with a clear message naming what to
  override. Doesn't run end-to-end; doesn't even register. The
  author has to fill every method to get a working agent.
  Use case: "I know exactly what I want; just give me the right shape."

* ``starter`` (default) — fully working agent with sensible defaults.
  For agents: handles its declared format end-to-end with templated
  responses, self-registers, runs under ``python -m agents.{name}``.
  For tools: a working dict-lookup / echo implementation.
  For integrations: ping endpoint actually pings.
  Use case: "Give me something that runs so I can study it and extend."

The agent scaffolder also writes a ``__main__`` block so the
generated module is runnable as ``python -m agents.{name}``:
registers with Redis, opens the consumer loop, waits for Ctrl-C.
Registration via the CLI (``ahp register agent``) is intentionally
separate so the metadata write and the worker host stay independent.

The ``formatagent`` agent kind is special: it scaffolds a
:class:`~ahp.adapters.FormatAgent` subclass with hooks for the
declared format's turn primitives. The contract check at construction
verifies every required hook is overridden.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal


Depth = Literal["skeleton", "starter"]
"""Two depths the scaffolders support.

* ``skeleton`` — every hook raises NotImplementedError; minimal
  imports; no end-to-end runnable behavior.
* ``starter`` — fully working defaults; runs end-to-end without
  edits; safe to register and serve.
"""


# ── name validation ───────────────────────────────────────────────────


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
"""Permitted snake_case names: lowercase, digits, underscores, ≤64 chars.

Snake-case because the file becomes a Python module name. The CLI
normalizes kebab-case input (``find-nearby-restaurants``) by replacing
hyphens with underscores before calling these scaffolders, so users
can type whichever feels natural.
"""


def normalize_name(name: str) -> str:
    """Lowercase + replace hyphens / spaces with underscores.

    Doesn't validate beyond that — the caller does the regex check so
    error messages can name the original input rather than the
    normalized form.
    """
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def validate_name(name: str) -> None:
    """Raise :class:`ValueError` if ``name`` isn't a valid module slug."""
    if not _NAME_RE.match(name):
        raise ValueError(
            f"invalid name {name!r}: must be lowercase snake_case, "
            "start with a letter, ≤64 chars (letters, digits, underscores)"
        )


# ── file writer ───────────────────────────────────────────────────────


def _write(path: Path, text: str, *, force: bool) -> Path:
    """Write ``text`` to ``path``, creating parent dirs.

    Refuses to overwrite when the file exists and ``force`` is False —
    raises :class:`FileExistsError` so the caller can format the
    diagnostic. The scaffolders are deterministic, so an overwrite
    refusal almost always means the user is repeating themselves.
    """
    path = path.resolve()
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite {path} (pass --force to overwrite)"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


# ── tool scaffold ─────────────────────────────────────────────────────


_TOOL_TEMPLATES: dict[str, str] = {
    "skeleton": '''\
"""Tool: {name} (skeleton).

Registered at ``{scope}.{kind}.{role}.{category}.{name}`` — every agent
whose address matches the convention (``{scope}.{role}.*.*.*.*.*``)
will see this tool when the module is imported.

Skeleton depth: the function body raises ``NotImplementedError``.
Replace with a real implementation before relying on this tool.
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import tool


@tool({scope!r}, {kind!r}, {role!r}, {category!r}, name={name!r})
def {name}({signature}) -> Any:
    """{summary}"""
    raise NotImplementedError("tool {name} is not implemented yet")
''',
    "starter": '''\
"""Tool: {name} (starter — works out of the box).

Registered at ``{scope}.{kind}.{role}.{category}.{name}`` — every agent
whose address matches the convention (``{scope}.{role}.*.*.*.*.*``)
will see this tool when the module is imported.

Starter depth: the function returns a templated response that's
useful as a sanity check (callable, type-correct, deterministic).
Replace the body when you wire it to your real backend.
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import tool


@tool({scope!r}, {kind!r}, {role!r}, {category!r}, name={name!r})
def {name}({signature}) -> dict[str, Any]:
    """{summary}"""
    # Working stub: echoes the inputs so callers can verify wiring.
    # Replace with a real implementation when you have one.
    return {{
        "tool": {name!r},
        "scope": {scope!r},
        "echo_query": query,
        "result_count": 0,
        "results": [],
        "note": "starter stub — replace this body with real logic",
    }}
''',
}


def scaffold_tool(
    *,
    name: str,
    scope: str = "tifin",
    kind: str = "api",
    role: str = "*",
    category: str = "search",
    signature: str = "query: str",
    summary: str | None = None,
    depth: Depth = "starter",
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Write a tool stub to ``{out_dir}/tools/{name}.py``.

    The tool's address is ``{scope}.{kind}.{role}.{category}.{name}``.
    ``role='*'`` is the convention-friendly default — any agent role
    in the same scope can use the tool. Tighten only when the tool is
    role-specific.

    ``depth='starter'`` (default) writes a working echo implementation;
    ``depth='skeleton'`` writes a NotImplementedError body. The
    starter version requires the signature to include a ``query``
    parameter (the default ``signature='query: str'`` satisfies
    this); the skeleton has no body constraint.
    """
    validate_name(name)
    if depth not in _TOOL_TEMPLATES:
        raise ValueError(
            f"unknown depth {depth!r}; expected one of "
            f"{sorted(_TOOL_TEMPLATES)}"
        )
    out_dir = (out_dir or Path.cwd()).resolve()
    path = out_dir / "tools" / f"{name}.py"
    text = _TOOL_TEMPLATES[depth].format(
        name=name, scope=scope, kind=kind, role=role,
        category=category, signature=signature,
        summary=summary or f"{name} tool — describe what it does.",
    )
    return _write(path, text, force=force)


# ── integration scaffold ──────────────────────────────────────────────


IntegrationKind = Literal["api_key", "oauth", "webhook"]


# Two depths per integration kind. Keys: (kind, depth).
_INTEGRATION_TEMPLATES: dict[tuple[IntegrationKind, Depth], str] = {
    ("api_key", "skeleton"): '''\
"""Integration: {name} (api_key auth, skeleton).

External service wrapper for {name}. Skeleton depth — every tool
in this file raises NotImplementedError until you wire it.
"""

from __future__ import annotations

import os
from typing import Any

from ahp.adapters import tool


_ENV_KEY = "{env_key}"


def _api_key() -> str:
    key = os.environ.get(_ENV_KEY)
    if not key:
        raise RuntimeError(
            f"set {{_ENV_KEY!r}} to call {name}"
        )
    return key


@tool({scope!r}, "api", "*", {name!r}, name="{name}_ping")
def {name}_ping() -> dict[str, Any]:
    """Sanity-check that the API key env var is present."""
    raise NotImplementedError("{name}_ping needs implementation")
''',
    ("api_key", "starter"): '''\
"""Integration: {name} (api_key auth, starter — runnable as-is).

External service wrapper for {name}. Loads its credential from an
environment variable and exposes ``{name}_ping`` (sanity check) and
``{name}_search`` (templated stub returning empty results so callers
can verify wiring without a real backend).

When the module is imported, the @tool decorator registers each
operation under the ``{scope}.api.*.{name}.*`` address space.
"""

from __future__ import annotations

import os
from typing import Any

from ahp.adapters import tool


# Where the API key lives in the env. Convention: NAME_API_KEY upper.
_ENV_KEY = "{env_key}"


def _api_key() -> str | None:
    """Read the API key. Starter returns None when unset so calls
    can degrade to templated responses rather than crash; production
    code typically raises instead."""
    return os.environ.get(_ENV_KEY)


@tool({scope!r}, "api", "*", {name!r}, name="{name}_ping")
def {name}_ping() -> dict[str, Any]:
    """Sanity-check the integration is wired (no external call)."""
    return {{
        "integration": {name!r},
        "env_key": _ENV_KEY,
        "env_key_present": bool(_api_key()),
        "status": "ok",
    }}


@tool({scope!r}, "api", "*", {name!r}, name="{name}_search")
def {name}_search(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search {name}. Starter returns an empty result set with the
    inputs echoed so you can verify the call shape without a backend."""
    return {{
        "integration": {name!r},
        "query": query,
        "top_k": top_k,
        "results": [],
        "note": "starter stub — wire to {name}'s real search endpoint",
    }}
''',
    ("oauth", "skeleton"): '''\
"""Integration: {name} (oauth, skeleton — every hook unwired).

OAuth flow scaffolding is intentionally left as TODOs.
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import tool


def _client() -> Any:
    raise NotImplementedError("OAuth client for {name} is not configured")


@tool({scope!r}, "api", "*", {name!r}, name="{name}_whoami")
def {name}_whoami() -> dict[str, Any]:
    raise NotImplementedError("hook up to {name} OAuth")
''',
    ("oauth", "starter"): '''\
"""Integration: {name} (oauth, starter — runnable stubs).

OAuth flow scaffolding is intentionally left as TODOs (drop in
authlib / oauthlib / your provider's SDK). The starter depth wires
``{name}_whoami`` to return a clear "not yet authenticated"
response so you can verify the registration without crashing.

Suggested env-var layout:

    {ENVUPPER}_CLIENT_ID
    {ENVUPPER}_CLIENT_SECRET
    {ENVUPPER}_REDIRECT_URI
    {ENVUPPER}_REFRESH_TOKEN
"""

from __future__ import annotations

import os
from typing import Any

from ahp.adapters import tool


def _client_configured() -> bool:
    """Check whether the OAuth client env vars are all set."""
    needed = ("{ENVUPPER}_CLIENT_ID", "{ENVUPPER}_CLIENT_SECRET")
    return all(os.environ.get(k) for k in needed)


@tool({scope!r}, "api", "*", {name!r}, name="{name}_whoami")
def {name}_whoami() -> dict[str, Any]:
    """Report OAuth wiring status. Starter doesn't actually hit the
    OAuth provider — replace with a real /me call when you wire it."""
    return {{
        "integration": {name!r},
        "configured": _client_configured(),
        "identity": None,
        "note": "starter stub — wire to {name}'s real OAuth /me endpoint",
    }}
''',
    ("webhook", "skeleton"): '''\
"""Integration: {name} (inbound webhook, skeleton).

Webhooks are inbound: an external service POSTs to a URL you host.
Skeleton — verify_signature and event handler both raise.
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import tool


_SECRET_ENV = "{ENVUPPER}_WEBHOOK_SECRET"


def verify_signature(payload: bytes, header_sig: str) -> bool:
    raise NotImplementedError("verify_signature for {name} is not implemented")


@tool({scope!r}, "api", "*", {name!r}, name="{name}_consume_event")
def {name}_consume_event(payload: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError("event handler for {name} is not implemented")
''',
    ("webhook", "starter"): '''\
"""Integration: {name} (inbound webhook, starter — runnable stubs).

Webhooks are inbound: an external service POSTs to a URL you host.
This module holds signature verification, parsing, and AHP-side
dispatch logic that turns the webhook payload into a message on the
protocol.

The starter depth wires ``{name}_consume_event`` to acknowledge any
payload by echoing it; ``verify_signature`` returns False by default
(fail closed) until you implement the real scheme.

This scaffold doesn't ship a server — the FastAPI viewer example
(see ``examples/fastapi_serve/``) is the recommended pattern.
"""

from __future__ import annotations

import os
from typing import Any

from ahp.adapters import tool


_SECRET_ENV = "{ENVUPPER}_WEBHOOK_SECRET"


def verify_signature(payload: bytes, header_sig: str) -> bool:
    """Default: fail closed. Replace with HMAC-SHA256 of the body with
    the shared secret (or whatever scheme {name} uses)."""
    return False


@tool({scope!r}, "api", "*", {name!r}, name="{name}_consume_event")
def {name}_consume_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Acknowledge a verified webhook event. Starter echoes the
    payload back so you can verify wiring. Replace with logic that
    converts the payload into a protocol message."""
    return {{
        "integration": {name!r},
        "ack": True,
        "secret_env_present": bool(os.environ.get(_SECRET_ENV)),
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "note": "starter stub — wire to a real protocol-side dispatch",
    }}
''',
}


def scaffold_integration(
    *,
    name: str,
    kind: IntegrationKind = "api_key",
    scope: str = "tifin",
    depth: Depth = "starter",
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Write an integration stub to ``{out_dir}/integrations/{name}.py``.

    ``kind`` picks one of three flavors (api_key, oauth, webhook);
    ``depth`` picks how much of the implementation ships:

    * ``starter`` (default) — runnable stubs that return templated
      responses so callers can verify wiring without a backend.
    * ``skeleton`` — every hook raises NotImplementedError.

    api_key starter also adds a ``{name}_search`` tool alongside the
    ping; oauth starter wires a configuration check; webhook starter
    has ``verify_signature`` fail-closed by default. These are
    intentionally not opinionated about which OAuth library / HMAC
    scheme to use — the starter just makes the contract verifiable.
    """
    validate_name(name)
    key = (kind, depth)
    if key not in _INTEGRATION_TEMPLATES:
        valid_kinds = sorted({k for k, _ in _INTEGRATION_TEMPLATES})
        valid_depths = sorted({d for _, d in _INTEGRATION_TEMPLATES})
        raise ValueError(
            f"no integration template for kind={kind!r} depth={depth!r}; "
            f"kinds: {valid_kinds}; depths: {valid_depths}"
        )
    out_dir = (out_dir or Path.cwd()).resolve()
    path = out_dir / "integrations" / f"{name}.py"
    text = _INTEGRATION_TEMPLATES[key].format(
        name=name, scope=scope,
        env_key=f"{name.upper()}_API_KEY",
        ENVUPPER=name.upper(),
    )
    return _write(path, text, force=force)


# ── agent scaffold ────────────────────────────────────────────────────


AgentKind = Literal["simple", "react", "deepagent", "formatagent"]


# Two depths per agent kind. Existing simple/react/deepagent starter
# templates are the originals; the skeleton variants below strip
# everything down to NotImplementedError stubs. The formatagent kind
# is new in this phase and supports both depths from the start.
_AGENT_TEMPLATES: dict[tuple[AgentKind, Depth], str] = {
    ("simple", "starter"): '''\
"""Agent: {name} ({role} in {scope}).

A minimal :class:`ahp.adapters.base.AHPAgent` subclass that you can
extend in place. Override :meth:`handle_message` to define behavior.

To run::

    python -m agents.{name}

Menu visibility (separate, broker-side)::

    python -m ahp register agent  --name {name} ...   # write the durable record
    python -m ahp start agent     --name {name} ...   # become visible on the menu
    python -m ahp stop agent      --name {name} ...   # hide without stopping the process
    python -m ahp deregister agent --name {name} ...  # remove the record entirely
"""

from __future__ import annotations

import asyncio
import os

from ahp.adapters.base import AHPAgent
from ahp.core import AgentAddress, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


SCOPE = {scope!r}
ROLE = {role!r}
INSTANCE = {name!r}


class {class_name}(AHPAgent):
    """Replace this docstring with what the agent does."""

    async def handle_message(self, message: Message) -> Message | None:
        # TODO: implement
        body = message.body if isinstance(message.body, dict) else {{"text": str(message.body)}}
        return Message(
            source=self.address, target=message.source,
            verb="SEND", code=message.code,
            body={{"text": f"{name} received: {{body!r}}"}},
            thread=message.thread,
        )


async def main() -> None:
    redis_url = os.environ.get("AHP_REDIS_URL", "redis://localhost:6379/0")
    import redis.asyncio as aioredis
    redis = aioredis.from_url(redis_url, decode_responses=True)

    bus = RedisBus(redis)
    registry = AgentRegistry(redis, heartbeat_ttl=60)
    cache = ProtocolCache(redis)
    engine = ProtocolEngine(bus, registry, cache, CompatibilityMatrix())

    address = AgentAddress.parse(
        f"{{SCOPE}}.{{ROLE}}.example.example.s.session.{{INSTANCE}}"
    )
    agent = {class_name}(address=address, engine=engine)
    await agent.register()
    await agent.start()
    print(f"{{address}} alive — Ctrl-C to stop")
    try:
        await asyncio.Event().wait()
    finally:
        await agent.stop()
        await agent.deregister()
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
''',
    ("react", "starter"): '''\
"""Agent: {name} (ReAct loop, langgraph-backed).

Builds an :class:`ahp.adapters.react_agent.ReactAgent` from an
:class:`ahp.adapters.AgentProfile`. Tools registered to the address
(via ``@tool(...)``) are auto-bound. Persona is read from the
profile or supplied here.

Requires ``pip install ahp[langgraph]`` (or ``ahp[deepagents]``).

To run::

    python -m agents.{name}
"""

from __future__ import annotations

import asyncio
import os

from ahp.adapters import AgentProfile
from ahp.adapters.react_agent import ReactAgent
from ahp.core import AgentAddress
from ahp.core.compatibility import CompatibilityMatrix
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


SCOPE = {scope!r}
ROLE = {role!r}
INSTANCE = {name!r}


def _model():
    """Return a chat model. Defaults to Bedrock; swap freely.

    See ``ahp.llm.bedrock`` / ``ahp.llm.openrouter`` for helpers.
    """
    from ahp.llm.bedrock import bedrock_chat_model
    return bedrock_chat_model(temperature=0.3, max_tokens=512)


async def main() -> None:
    redis_url = os.environ.get("AHP_REDIS_URL", "redis://localhost:6379/0")
    import redis.asyncio as aioredis
    redis = aioredis.from_url(redis_url, decode_responses=True)

    bus = RedisBus(redis)
    registry = AgentRegistry(redis, heartbeat_ttl=60)
    cache = ProtocolCache(redis)
    engine = ProtocolEngine(bus, registry, cache, CompatibilityMatrix())

    address = AgentAddress.parse(
        f"{{SCOPE}}.{{ROLE}}.example.example.s.session.{{INSTANCE}}"
    )
    profile = AgentProfile(
        agent_kind="react",
        prompt="You are {name}. TODO: replace with a real persona.",
    )
    agent = ReactAgent.from_profile(
        address=address, engine=engine, profile=profile, model=_model(),
    )
    await agent.register()
    await agent.start()
    print(f"{{address}} alive — Ctrl-C to stop")
    try:
        await asyncio.Event().wait()
    finally:
        await agent.stop()
        await agent.deregister()
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
''',
    ("deepagent", "starter"): '''\
"""Agent: {name} (deepagent, with planning + subagents).

Wraps :class:`ahp.adapters.deep_agent.DeepAgent` — the deepagents
``create_deep_agent`` graph behind an AHP-shaped address. Use this when
you want planning + tool-using subagents rather than a flat ReAct loop.

Requires ``pip install ahp[deepagents]``.

To run::

    python -m agents.{name}
"""

from __future__ import annotations

import asyncio
import os

from ahp.adapters import AgentProfile
from ahp.adapters.deep_agent import DeepAgent
from ahp.core import AgentAddress
from ahp.core.compatibility import CompatibilityMatrix
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


SCOPE = {scope!r}
ROLE = {role!r}
INSTANCE = {name!r}


def _model():
    from ahp.llm.bedrock import bedrock_chat_model
    return bedrock_chat_model(temperature=0.3, max_tokens=512)


async def main() -> None:
    redis_url = os.environ.get("AHP_REDIS_URL", "redis://localhost:6379/0")
    import redis.asyncio as aioredis
    redis = aioredis.from_url(redis_url, decode_responses=True)

    bus = RedisBus(redis)
    registry = AgentRegistry(redis, heartbeat_ttl=60)
    cache = ProtocolCache(redis)
    engine = ProtocolEngine(bus, registry, cache, CompatibilityMatrix())

    address = AgentAddress.parse(
        f"{{SCOPE}}.{{ROLE}}.example.example.s.session.{{INSTANCE}}"
    )
    profile = AgentProfile(
        agent_kind="deepagent",
        prompt="You are {name}. TODO: replace with a real persona.",
    )
    agent = DeepAgent.from_profile(
        address=address, engine=engine, profile=profile, model=_model(),
    )
    await agent.register()
    await agent.start()
    print(f"{{address}} alive — Ctrl-C to stop")
    try:
        await asyncio.Event().wait()
    finally:
        await agent.stop()
        await agent.deregister()
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
''',
    ("simple", "skeleton"): '''\
"""Agent: {name} ({role} in {scope}) — skeleton.

A minimal :class:`ahp.adapters.base.AHPAgent` subclass. Override
:meth:`handle_message` to define behavior; without it the agent
will raise on every inbound message.

To run (once you've implemented the handler)::

    python -m agents.{name}
"""

from __future__ import annotations

from ahp.adapters.base import AHPAgent
from ahp.core import Message


SCOPE = {scope!r}
ROLE = {role!r}
INSTANCE = {name!r}


class {class_name}(AHPAgent):
    """TODO: replace this docstring with what the agent does."""

    async def handle_message(self, message: Message) -> Message | None:
        raise NotImplementedError(
            f"{class_name}.handle_message must be overridden"
        )
''',
    ("react", "skeleton"): '''\
"""Agent: {name} (ReAct, langgraph-backed) — skeleton.

Skeleton depth: declares the class + persona stub but does not wire
the chat model or the run loop. Replace the `_model` factory and the
persona before this is usable.
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import AgentProfile
from ahp.adapters.react_agent import ReactAgent
from ahp.core import AgentAddress
from ahp.engine.router import ProtocolEngine


SCOPE = {scope!r}
ROLE = {role!r}
INSTANCE = {name!r}


def _model() -> Any:
    """Return a LangChain-compatible chat model."""
    raise NotImplementedError("supply a chat model for {name}")


def _persona() -> str:
    """Return the system prompt that defines this agent's behavior."""
    raise NotImplementedError("write a persona for {name}")


def build(engine: ProtocolEngine) -> ReactAgent:
    address = AgentAddress.parse(
        f"{{SCOPE}}.{{ROLE}}.example.example.s.session.{{INSTANCE}}"
    )
    profile = AgentProfile(agent_kind="react", prompt=_persona())
    return ReactAgent.from_profile(
        address=address, engine=engine, profile=profile, model=_model(),
    )
''',
    ("deepagent", "skeleton"): '''\
"""Agent: {name} (deepagent) — skeleton.

Skeleton depth: declares the class + persona stub; does not wire
the chat model or run loop. Replace `_model` and `_persona` before
use.
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import AgentProfile
from ahp.adapters.deep_agent import DeepAgent
from ahp.core import AgentAddress
from ahp.engine.router import ProtocolEngine


SCOPE = {scope!r}
ROLE = {role!r}
INSTANCE = {name!r}


def _model() -> Any:
    raise NotImplementedError("supply a chat model for {name}")


def _persona() -> str:
    raise NotImplementedError("write a persona for {name}")


def build(engine: ProtocolEngine) -> DeepAgent:
    address = AgentAddress.parse(
        f"{{SCOPE}}.{{ROLE}}.example.example.s.session.{{INSTANCE}}"
    )
    profile = AgentProfile(agent_kind="deepagent", prompt=_persona())
    return DeepAgent.from_profile(
        address=address, engine=engine, profile=profile, model=_model(),
    )
''',
    ("formatagent", "skeleton"): '''\
"""Agent: {name} ({role} in {scope}) — FormatAgent skeleton.

A :class:`~ahp.adapters.FormatAgent` subclass declaring participation
in the {format!r} format. The wrapper's contract check verifies every
required ``on_<turn>`` hook is overridden at instantiation. Skeleton
depth: every hook raises NotImplementedError. Fill them in before
running.
"""

from __future__ import annotations

from ahp.adapters import FormatAgent, get_format
from ahp.core import Message


SCOPE = {scope!r}
ROLE = {role!r}
INSTANCE = {name!r}
FORMAT_NAME = {format!r}


class {class_name}(FormatAgent):
    """TODO: describe what this agent does in the {format!r} format."""

    supported_formats = (FORMAT_NAME,)

    # The required hooks below were generated from
    # get_format({format!r}).turn_primitives. Override each before
    # the contract check passes at construction.
{format_hooks_skeleton}
''',
    ("formatagent", "starter"): '''\
"""Agent: {name} ({role} in {scope}) — FormatAgent starter.

A :class:`~ahp.adapters.FormatAgent` subclass that participates in
the {format!r} format. Each ``on_<turn>`` hook returns a templated
response shaped like the format's contract so the agent runs
end-to-end without an LLM. Replace each body with real logic when
you wire your model.

To run::

    python -m agents.{name}
"""

from __future__ import annotations

import asyncio
import os

from ahp.adapters import FormatAgent
from ahp.core import AgentAddress, Code, Message
from ahp.core.compatibility import CompatibilityMatrix
from ahp.engine.router import ProtocolEngine
from ahp.registry.registry import AgentRegistry
from ahp.transport.cache import ProtocolCache
from ahp.transport.redis_bus import RedisBus


SCOPE = {scope!r}
ROLE = {role!r}
INSTANCE = {name!r}
FORMAT_NAME = {format!r}


class {class_name}(FormatAgent):
    """Starter agent for the {format!r} format. Each hook returns a
    templated reply so the conversation runs end-to-end without a
    real LLM. Replace each body when you have one."""

    supported_formats = (FORMAT_NAME,)

{format_hooks_starter}


async def main() -> None:
    redis_url = os.environ.get("AHP_REDIS_URL", "redis://localhost:6379/0")
    import redis.asyncio as aioredis
    redis = aioredis.from_url(redis_url, decode_responses=True)

    bus = RedisBus(redis)
    registry = AgentRegistry(redis, heartbeat_ttl=60)
    cache = ProtocolCache(redis)
    engine = ProtocolEngine(bus, registry, cache, CompatibilityMatrix())

    address = AgentAddress.parse(
        f"{{SCOPE}}.{{ROLE}}.example.example.s.session.{{INSTANCE}}"
    )
    agent = {class_name}(address=address, engine=engine)
    await agent.register()
    await agent.start()
    print(f"{{address}} alive (format={{FORMAT_NAME}}) — Ctrl-C to stop")
    try:
        await asyncio.Event().wait()
    finally:
        await agent.stop()
        await agent.deregister()
        await bus.close()


if __name__ == "__main__":
    asyncio.run(main())
''',
}


def _camel(name: str) -> str:
    """Turn ``restaurant_finder`` into ``RestaurantFinder`` for the class name."""
    return "".join(part.capitalize() or "_" for part in name.split("_")) + "Agent"


def _turn_to_hook_name(turn_code: str) -> str:
    """``turn.ask`` -> ``on_ask``; ``turn.back-or-qualify`` -> ``on_back_or_qualify``."""
    stem = turn_code[len("turn."):] if turn_code.startswith("turn.") else turn_code
    return "on_" + stem.replace("-", "_")


def _format_hooks(format_name: str, depth: Depth) -> str:
    """Generate the body of ``on_<turn>`` hooks for a FormatAgent
    subclass declaring support for ``format_name``.

    Each hook gets a docstring naming the turn primitive and a body
    matching the depth: NotImplementedError for skeleton, a templated
    reply for starter.
    """
    # Local import to avoid a circular import at module load.
    from ahp.adapters.formats import FORMATS
    fmt = FORMATS.get(format_name)
    if fmt is None:
        raise ValueError(
            f"unknown format {format_name!r}; cannot scaffold a "
            f"formatagent against it"
        )
    if fmt.recipe_kind != "turn_sequence":
        raise ValueError(
            f"format {format_name!r} is recipe_kind={fmt.recipe_kind!r}; "
            "formatagent scaffolds only support turn_sequence formats"
        )

    lines: list[str] = []
    for turn in fmt.turn_primitives:
        method = _turn_to_hook_name(turn)
        lines.append(f"    async def {method}(self, message: Message) -> Message | None:")
        lines.append(f'        """Handle a {turn!r} turn."""')
        if depth == "skeleton":
            lines.append(
                f'        raise NotImplementedError("{method} not implemented")'
            )
        else:  # starter
            lines.append(
                f"        # Templated starter reply; replace with real logic."
            )
            lines.append(
                "        return Message("
            )
            lines.append("            source=self.address,")
            lines.append("            target=message.source,")
            lines.append('            verb="SEND",')
            lines.append("            code=message.code,")
            lines.append("            thread=message.thread,")
            lines.append("            format=message.format,")
            lines.append(
                f'            body=f"{{INSTANCE}} handled {turn!r}: {{message.body!r}}",'
            )
            lines.append("        )")
        lines.append("")  # blank line between methods
    return "\n".join(lines)


def scaffold_agent(
    *,
    name: str,
    kind: AgentKind = "simple",
    scope: str = "tifin",
    role: str = "researcher",
    format: str | None = None,
    depth: Depth = "starter",
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Write an agent stub to ``{out_dir}/agents/{name}.py``.

    ``kind`` picks the framework:

    * ``simple`` — bare :class:`AHPAgent` subclass.
    * ``react`` — LangGraph ReAct loop via
      :class:`ahp.adapters.react_agent.ReactAgent`.
    * ``deepagent`` — deepagents graph via
      :class:`ahp.adapters.deep_agent.DeepAgent`.
    * ``formatagent`` — :class:`~ahp.adapters.FormatAgent` subclass
      with one ``on_<turn>`` hook per turn primitive in the declared
      format. Requires ``format=`` to be supplied.

    ``depth`` picks how much of the body ships:

    * ``starter`` (default) — fully runnable. The agent self-registers,
      opens its consumer loop, and returns templated responses so the
      end-to-end path works without an LLM.
    * ``skeleton`` — bare scaffold. Hooks raise NotImplementedError;
      the file isn't runnable until they're filled in.

    The generated file is runnable as ``python -m agents.{name}`` once
    its dependencies are installed and ``AHP_REDIS_URL`` points at a
    live Redis. Registration (via the CLI's ``register agent`` command)
    is a separate step.
    """
    validate_name(name)
    key = (kind, depth)
    if key not in _AGENT_TEMPLATES:
        kinds = sorted({k for k, _ in _AGENT_TEMPLATES})
        depths = sorted({d for _, d in _AGENT_TEMPLATES})
        raise ValueError(
            f"no agent template for kind={kind!r} depth={depth!r}; "
            f"kinds: {kinds}; depths: {depths}"
        )
    if kind == "formatagent" and not format:
        raise ValueError(
            "kind='formatagent' requires --format <format-name> "
            "(e.g. --format information-exchange)"
        )

    fmt_args: dict[str, str] = {}
    if kind == "formatagent":
        assert format is not None  # guarded above
        if depth == "skeleton":
            fmt_args["format_hooks_skeleton"] = _format_hooks(format, "skeleton")
        else:
            fmt_args["format_hooks_starter"] = _format_hooks(format, "starter")
        fmt_args["format"] = format

    out_dir = (out_dir or Path.cwd()).resolve()
    path = out_dir / "agents" / f"{name}.py"
    text = _AGENT_TEMPLATES[key].format(
        name=name, scope=scope, role=role, class_name=_camel(name),
        **fmt_args,
    )
    return _write(path, text, force=force)
