"""Project-template scaffolders for the AHP CLI.

``ahp new <thing> --name <X>`` generates a boilerplate file in a
convention path (``./tools/``, ``./integrations/``, ``./agents/``).
Files are intentionally minimal — the goal is to drop you into a
typed, importable starting point, not a finished implementation.

Three scaffolders:

* :func:`scaffold_tool`        — ``./tools/{name}.py``
* :func:`scaffold_integration` — ``./integrations/{name}.py``
* :func:`scaffold_agent`       — ``./agents/{name}.py``

Each writes one file, refuses to overwrite an existing file unless
``force=True``, and returns the absolute path written. The CLI
(:mod:`ahp.cli`) wires arg-parsing on top.

The agent scaffolder also writes a tiny ``__main__`` block so the
generated module is runnable as ``python -m agents.{name}``: it
imports its own decorators, registers with Redis, opens the consumer
loop, and waits for Ctrl-C. Registration via the CLI
(``ahp register agent``) is intentionally separate from this — see
:func:`ahp.cli.cmd_register_agent` — so the metadata write and the
worker host stay independent.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal


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


_TOOL_TEMPLATE = '''\
"""Tool: {name}.

Registered at ``{scope}.{kind}.{role}.{category}.{name}`` — every agent
whose address matches the convention (``{scope}.{role}.*.*.*.*.*``)
will see this tool when the module is imported.

Run the CLI to confirm registration:

    python -m ahp list-tools -m tools.{name} --for {scope}.{role}.example.example.s.session.x
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import tool


@tool({scope!r}, {kind!r}, {role!r}, {category!r}, name={name!r})
def {name}({signature}) -> Any:
    """{summary}"""
    # TODO: implement
    raise NotImplementedError("tool {name} is not implemented yet")
'''


def scaffold_tool(
    *,
    name: str,
    scope: str = "tifin",
    kind: str = "api",
    role: str = "*",
    category: str = "search",
    signature: str = "query: str",
    summary: str | None = None,
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Write a tool stub to ``{out_dir}/tools/{name}.py``.

    The tool's address is ``{scope}.{kind}.{role}.{category}.{name}``.
    ``role='*'`` is the convention-friendly default — any agent role
    in the same scope can use the tool. Tighten only when the tool is
    role-specific.
    """
    validate_name(name)
    out_dir = (out_dir or Path.cwd()).resolve()
    path = out_dir / "tools" / f"{name}.py"
    text = _TOOL_TEMPLATE.format(
        name=name, scope=scope, kind=kind, role=role,
        category=category, signature=signature,
        summary=summary or f"{name} tool — describe what it does.",
    )
    return _write(path, text, force=force)


# ── integration scaffold ──────────────────────────────────────────────


IntegrationKind = Literal["api_key", "oauth", "webhook"]


_INTEGRATION_TEMPLATES: dict[IntegrationKind, str] = {
    "api_key": '''\
"""Integration: {name} (api_key auth).

External service wrapper for {name}. Loads its credential from an
environment variable and exposes a small typed client + the tool
functions that need it.

TODO: replace the stubs below with real calls. When the module is
imported, the ``@tool`` decorator registers the operations under the
``{scope}.api.*.{name}`` address space.
"""

from __future__ import annotations

import os
from typing import Any

from ahp.adapters import tool


# Where the API key lives in the env. Convention: NAME_API_KEY upper.
_ENV_KEY = "{env_key}"


def _api_key() -> str:
    """Look up the API key or raise. Called lazily by each tool."""
    key = os.environ.get(_ENV_KEY)
    if not key:
        raise RuntimeError(
            f"set {{_ENV_KEY!r}} to call {name} (see ./integrations/{name}.py)"
        )
    return key


@tool({scope!r}, "api", "*", {name!r}, name="{name}_ping")
def {name}_ping() -> dict[str, Any]:
    """Sanity-check the integration is wired (no external call)."""
    return {{"integration": "{name}", "env_key_present": bool(os.environ.get(_ENV_KEY))}}


# TODO: add more tools, each with @tool({scope!r}, "api", "*", {name!r}, name="...").
''',
    "oauth": '''\
"""Integration: {name} (oauth auth — STUB).

OAuth flow scaffolding is intentionally left as TODOs. There's no
opinionated wiring this phase — drop in whichever OAuth client you
prefer (authlib, oauthlib, your provider's SDK). When you wire it,
the tool functions in this file will be discoverable to agents in
scope ``{scope}`` automatically once the module is imported.

Suggested env-var layout (override below as needed):

    {ENVUPPER}_CLIENT_ID
    {ENVUPPER}_CLIENT_SECRET
    {ENVUPPER}_REDIRECT_URI
    {ENVUPPER}_REFRESH_TOKEN
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import tool


# TODO: implement OAuth flow — auth URL, callback, token refresh, storage.
def _client() -> Any:
    """Return an authenticated client. TODO: implement."""
    raise NotImplementedError("OAuth client for {name} is not configured")


@tool({scope!r}, "api", "*", {name!r}, name="{name}_whoami")
def {name}_whoami() -> dict[str, Any]:
    """Verify the OAuth identity. TODO: replace with a real call."""
    raise NotImplementedError("hook this up to the {name} OAuth flow")
''',
    "webhook": '''\
"""Integration: {name} (inbound webhook receiver — STUB).

Webhooks are *inbound*: an external service POSTs to a URL you host.
This module is the place to keep the verification (signature header
check), parsing, and any AHP-side dispatch logic that turns the
webhook payload into a message on the protocol.

This scaffold doesn't ship a server — the FastAPI viewer example
(see ``examples/fastapi_serve/``) is the recommended pattern.
"""

from __future__ import annotations

from typing import Any

from ahp.adapters import tool


# TODO: paste the secret env var name here, e.g. "{ENVUPPER}_WEBHOOK_SECRET".
_SECRET_ENV = "{ENVUPPER}_WEBHOOK_SECRET"


def verify_signature(payload: bytes, header_sig: str) -> bool:
    """Verify the webhook signature using ``_SECRET_ENV``.

    TODO: implement the provider's actual scheme (HMAC-SHA256 of the
    body with the shared secret, in most cases). Fail closed.
    """
    raise NotImplementedError("verify_signature for {name} is not implemented")


@tool({scope!r}, "api", "*", {name!r}, name="{name}_consume_event")
def {name}_consume_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert a verified webhook payload into an AHP-side action."""
    raise NotImplementedError("event handler for {name} is not implemented")
''',
}


def scaffold_integration(
    *,
    name: str,
    kind: IntegrationKind = "api_key",
    scope: str = "tifin",
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Write an integration stub to ``{out_dir}/integrations/{name}.py``.

    ``kind`` picks one of three templates:

    * ``api_key`` — env-var credential, two stub tools.
    * ``oauth`` — TODO scaffolding (no opinionated flow; left to the
      user per the design call).
    * ``webhook`` — inbound webhook receiver scaffolding with a
      signature verification stub.
    """
    validate_name(name)
    if kind not in _INTEGRATION_TEMPLATES:
        raise ValueError(
            f"unknown integration kind {kind!r}; "
            f"expected one of {sorted(_INTEGRATION_TEMPLATES)}"
        )
    out_dir = (out_dir or Path.cwd()).resolve()
    path = out_dir / "integrations" / f"{name}.py"
    text = _INTEGRATION_TEMPLATES[kind].format(
        name=name, scope=scope,
        env_key=f"{name.upper()}_API_KEY",
        ENVUPPER=name.upper(),
    )
    return _write(path, text, force=force)


# ── agent scaffold ────────────────────────────────────────────────────


AgentKind = Literal["simple", "react", "deepagent"]


_AGENT_TEMPLATES: dict[AgentKind, str] = {
    "simple": '''\
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
    "react": '''\
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
    "deepagent": '''\
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
}


def _camel(name: str) -> str:
    """Turn ``restaurant_finder`` into ``RestaurantFinder`` for the class name."""
    return "".join(part.capitalize() or "_" for part in name.split("_")) + "Agent"


def scaffold_agent(
    *,
    name: str,
    kind: AgentKind = "simple",
    scope: str = "tifin",
    role: str = "researcher",
    out_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """Write an agent stub to ``{out_dir}/agents/{name}.py``.

    ``kind`` picks the framework:

    * ``simple`` — bare :class:`AHPAgent` subclass with a stub
      ``handle_message``.
    * ``react`` — LangGraph ReAct loop via
      :class:`ahp.adapters.react_agent.ReactAgent`.
    * ``deepagent`` — deepagents graph via
      :class:`ahp.adapters.deep_agent.DeepAgent`.

    The generated file is runnable as ``python -m agents.{name}`` once
    its dependencies are installed and ``AHP_REDIS_URL`` points at a
    live Redis. Registration (via the CLI's ``register agent`` command)
    is a separate step.
    """
    validate_name(name)
    if kind not in _AGENT_TEMPLATES:
        raise ValueError(
            f"unknown agent kind {kind!r}; "
            f"expected one of {sorted(_AGENT_TEMPLATES)}"
        )
    out_dir = (out_dir or Path.cwd()).resolve()
    path = out_dir / "agents" / f"{name}.py"
    text = _AGENT_TEMPLATES[kind].format(
        name=name, scope=scope, role=role, class_name=_camel(name),
    )
    return _write(path, text, force=force)
