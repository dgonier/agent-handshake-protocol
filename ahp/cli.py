"""Command-line interface for inspecting and scaffolding AHP entities.

Run via the module entry point::

    python -m ahp <command> [options]

Subcommands:

* ``list-tools``       — registered tools (optionally filtered by agent address / tags)
* ``list-resources``   — registered resources
* ``list-groups``      — named address-pattern groups
* ``list-skills``      — registered skill playbooks (workflow + suggested addresses)
* ``profile``          — what tools / resources / prompt an agent address resolves to
* ``template``         — print a starter tool / resource module to stdout
* ``scaffold``         — write a starter tool / resource module to a file

Importing user modules
----------------------

Tools and resources only appear in the listings if their module has been
imported (their ``@tool`` / ``@resource`` decorators run on import).
Pass ``--module/-m DOTTED.PATH`` one or more times to import modules
before the command runs::

    python -m ahp list-tools -m my_project.tools -m my_project.db
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import textwrap
import time

log = logging.getLogger("ahp.cli")
from pathlib import Path
from typing import Sequence, TextIO

from ahp.adapters import (
    DEFAULT_GROUP_REGISTRY,
    DEFAULT_RESOURCE_REGISTRY,
    DEFAULT_SKILL_REGISTRY,
    DEFAULT_TOOL_REGISTRY,
)
from ahp.adapters.factory import AgentFactory
from ahp.core.address import AgentAddress
from ahp.core.pattern import AddressPattern
from ahp.engine.router import ProtocolEngine


# ── module loading ────────────────────────────────────────────────────


def _load_modules(modules: list[str]) -> bool:
    """Import each dotted module path so its decorators register entries.

    Returns False on first failure (with diagnostic on stderr); the
    caller surfaces the error code so ``main()`` returns cleanly to
    test harnesses rather than raising ``SystemExit``.
    """
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception as exc:
            print(f"failed to import {name!r}: {exc}", file=sys.stderr)
            return False
    return True


# ── list-tools ────────────────────────────────────────────────────────


def cmd_list_tools(args: argparse.Namespace, out: TextIO) -> int:
    if not _load_modules(args.module or []): return 2
    registry = DEFAULT_TOOL_REGISTRY

    rows: list[tuple[str, str, str, str]] = []
    if args.for_addr:
        addr = AgentAddress.parse(args.for_addr)
        tags = set(args.tag) if args.tag else None
        bindings = registry.bindings_for_address(addr, tags=tags)
    else:
        bindings = list(registry.bindings())
        if args.tag:
            wanted = set(args.tag)
            bindings = [b for b in bindings if wanted & b.tags]

    for b in bindings:
        rows.append((
            str(b.address),
            b.tool.name,
            ",".join(sorted(b.tags)) or "-",
            (b.tool.description or "").splitlines()[0][:60],
        ))

    if not rows:
        print("(no tools registered)", file=out)
        return 0

    _render_table(out, ["address", "name", "tags", "description"], rows)
    return 0


# ── list-resources ────────────────────────────────────────────────────


def cmd_list_resources(args: argparse.Namespace, out: TextIO) -> int:
    if not _load_modules(args.module or []): return 2
    registry = DEFAULT_RESOURCE_REGISTRY

    if args.for_addr:
        addr = AgentAddress.parse(args.for_addr)
        bindings = [
            b for b in registry.bindings()
            if b.allowed_for.matches(addr)
        ]
    else:
        bindings = list(registry.bindings())

    if not bindings:
        print("(no resources registered)", file=out)
        return 0

    rows = [
        (str(b.address), b.address.kind, b.description[:60] or "-")
        for b in bindings
    ]
    _render_table(out, ["address", "kind", "description"], rows)
    return 0


# ── list-skills ───────────────────────────────────────────────────────


def cmd_list_skills(args: argparse.Namespace, out: TextIO) -> int:
    """List skills registered to :data:`DEFAULT_SKILL_REGISTRY`.

    Each row shows the skill's resource-address, name, tag set, and
    how many addresses are in each of the four suggested_* bundles
    (tools / specialists / loras / info-sources). Filterable by
    --for ADDR (only skills visible to that agent) and --tag.
    """
    if not _load_modules(args.module or []): return 2
    registry = DEFAULT_SKILL_REGISTRY

    if args.for_addr:
        addr = AgentAddress.parse(args.for_addr)
        tags = set(args.tag) if args.tag else None
        bindings = registry.bindings_for_address(addr, tags=tags)
    else:
        bindings = list(registry.bindings())
        if args.tag:
            wanted = set(args.tag)
            bindings = [b for b in bindings if wanted & b.tags]

    if not bindings:
        print("(no skills registered)", file=out)
        return 0

    rows: list[tuple[str, ...]] = []
    for b in bindings:
        sk = b.skill
        bundle_sizes = (
            f"t={len(sk.suggested_tools)} "
            f"s={len(sk.suggested_specialists)} "
            f"l={len(sk.suggested_loras)} "
            f"i={len(sk.suggested_information_sources)}"
        )
        graph_marker = "yes" if sk.graph is not None else "-"
        rows.append((
            str(b.address),
            sk.name,
            ",".join(sorted(b.tags)) or "-",
            graph_marker,
            bundle_sizes,
            (sk.description or "").splitlines()[0][:60],
        ))

    _render_table(
        out,
        ["address", "name", "tags", "graph", "bundles", "description"],
        rows,
    )
    return 0


# ── list-groups ───────────────────────────────────────────────────────


def cmd_list_groups(args: argparse.Namespace, out: TextIO) -> int:
    if not _load_modules(args.module or []): return 2
    groups = DEFAULT_GROUP_REGISTRY

    if len(groups) == 0:
        print("(no groups registered)", file=out)
        return 0

    rows = [
        (g.name, str(g.pattern), g.description[:60] or "-")
        for g in groups.groups()
    ]
    _render_table(out, ["name", "pattern", "description"], rows)
    return 0


# ── profile ───────────────────────────────────────────────────────────


def cmd_profile(args: argparse.Namespace, out: TextIO) -> int:
    """Show the resolved profile for an agent address.

    Synthesizes a throw-away factory pointing at the default tool /
    resource / capability / group registries — no engine connection
    required. This is purely an inspection command.
    """
    if not _load_modules(args.module or []): return 2

    address = AgentAddress.parse(args.address)
    factory = AgentFactory(
        # The factory needs an engine but we never dispatch — pass a stub
        # that satisfies the constructor without connecting to Redis.
        engine=_NullEngine(),
        tools=DEFAULT_TOOL_REGISTRY,
        resources=DEFAULT_RESOURCE_REGISTRY,
        groups=DEFAULT_GROUP_REGISTRY,
    )
    try:
        profile = factory.profile_for(address)
    except Exception as exc:
        print(f"failed to resolve profile for {address}: {exc}", file=sys.stderr)
        return 2

    print(f"agent address:  {address}", file=out)
    print(f"agent kind:     {profile.agent_kind}", file=out)
    print(file=out)

    print("tools:", file=out)
    if profile.tools:
        for t in profile.tools:
            desc = (t.description or "").splitlines()[0][:60]
            print(f"  - {t.name:32s} {desc}", file=out)
    else:
        print("  (none)", file=out)

    print(file=out)
    print("skills:", file=out)
    if profile.skills:
        for s in profile.skills:
            print(f"  - {s.name:32s} {s.description[:60]}", file=out)
    else:
        print("  (none)", file=out)

    print(file=out)
    print("resources:", file=out)
    if profile.resources:
        for name in sorted(profile.resources):
            print(f"  - {name}", file=out)
    else:
        print("  (none)", file=out)

    print(file=out)
    if profile.prompt:
        print("prompt:", file=out)
        for line in profile.prompt.splitlines():
            print(f"  {line}", file=out)
    return 0


# ── template / scaffold ───────────────────────────────────────────────


_TOOL_TEMPLATE = """\
\"\"\"Tools registered at scope={scope} kind={kind} role={role} category={category}.\"\"\"

from ahp.adapters import tool


@tool({scope!r}, {kind!r}, {role!r}, {category!r})
def {operation}({signature}):
    \"\"\"{summary}\"\"\"
    # TODO: implement
    raise NotImplementedError
"""


_RESOURCE_TEMPLATE = """\
\"\"\"Resource registered at scope={scope} kind={kind} domain={domain} subdomain={subdomain}.\"\"\"

from ahp.adapters import resource


@resource({scope!r}, {kind!r}, {domain!r}, {subdomain!r}, name={name!r})
def {factory_name}():
    \"\"\"{summary}\"\"\"
    # TODO: construct + return the resource instance
    raise NotImplementedError
"""


def _tool_template_text(args: argparse.Namespace) -> str:
    sig_lines = (args.signature or "").strip()
    return _TOOL_TEMPLATE.format(
        scope=args.scope, kind=args.kind, role=args.role,
        category=args.category, operation=args.operation,
        signature=sig_lines or "**kwargs",
        summary=args.summary or "One-line description.",
    )


def _resource_template_text(args: argparse.Namespace) -> str:
    return _RESOURCE_TEMPLATE.format(
        scope=args.scope, kind=args.kind, domain=args.domain,
        subdomain=args.subdomain, name=args.name,
        factory_name=args.factory_name or f"make_{args.name.replace('-', '_')}",
        summary=args.summary or "One-line description.",
    )


def cmd_template(args: argparse.Namespace, out: TextIO) -> int:
    if args.target == "tool":
        out.write(_tool_template_text(args))
    elif args.target == "resource":
        out.write(_resource_template_text(args))
    return 0


def cmd_scaffold(args: argparse.Namespace, out: TextIO) -> int:
    text = (
        _tool_template_text(args) if args.target == "tool"
        else _resource_template_text(args)
    )
    path = Path(args.output).resolve()
    if path.exists() and not args.force:
        print(
            f"refusing to overwrite {path} (pass --force to overwrite)",
            file=sys.stderr,
        )
        return 2
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"wrote {path}", file=out)
    return 0


# ── helpers ───────────────────────────────────────────────────────────


class _NullEngine:
    """Stub engine that satisfies AgentFactory's constructor for offline inspection."""

    # AgentFactory only reads these attributes during construction —
    # never actually dispatches when used purely via profile_for().
    groups = None
    scope = None


def _render_table(
    out: TextIO,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> None:
    widths = [
        max(len(str(h)), *(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers), file=out)
    print(fmt.format(*("-" * w for w in widths)), file=out)
    for row in rows:
        print(fmt.format(*row), file=out)


# ── list-agents (live Redis) ─────────────────────────────────────────


def _connect_redis(url: str):
    """Construct an async Redis client. Module-level so tests can swap it.

    Tests that want to use fakeredis monkey-patch this name; the
    production code path imports redis.asyncio lazily so the library
    has no hard dep on the redis client unless this command is used.
    """
    import redis.asyncio as aioredis
    return aioredis.from_url(url, decode_responses=True)


async def _list_agents_async(args: argparse.Namespace, out: TextIO) -> int:
    """The actual async work for the list-agents command."""
    from ahp.registry import AgentMeta, AgentRegistry

    client = _connect_redis(args.redis_url)
    registry = AgentRegistry(client)
    try:
        pattern = (
            AddressPattern.parse(args.pattern)
            if args.pattern else AddressPattern.all()
        )
        if args.all:
            all_addrs = await registry.list_all(alive_only=False)
            addresses = [a for a in all_addrs if pattern.matches(a)]
        else:
            addresses = await registry.resolve(pattern, alive_only=True)

        if not addresses:
            scope_note = "(no matching agents)" if args.pattern else "(registry is empty)"
            print(scope_note, file=out)
            return 0

        rows: list[tuple[str, str, str, str, str]] = []
        for addr in sorted(addresses, key=str):
            meta = await registry.get(addr) or AgentMeta()
            alive = await registry.is_alive(addr)
            rows.append((
                str(addr),
                "alive" if alive else "stale",
                ",".join(meta.capabilities) or "-",
                f"{meta.reputation:.2f}",
                (meta.description or "-")[:60],
            ))
        _render_table(
            out,
            ["address", "status", "capabilities", "rep", "description"],
            rows,
        )
        return 0
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


def cmd_list_agents(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_list_agents_async(args, out))


# ── new / register / start / stop / deregister ────────────────────────


def _build_address(args: argparse.Namespace) -> AgentAddress:
    """Compose an :class:`AgentAddress` from the registration-command args.

    Used by ``register``, ``start``, ``stop``, and ``deregister``.
    Centralized so the seven-field convention stays consistent and
    error messages cite the same defaults everywhere.
    """
    name = args.name.strip().lower().replace("-", "_").replace(" ", "_")
    return AgentAddress.parse(
        f"{args.scope}.{args.role}.{args.domain}.{args.subdomain}."
        f"{args.accept}.{args.lifecycle}.{name}"
    )


def cmd_new(args: argparse.Namespace, out: TextIO) -> int:
    """Top-level dispatcher for ``ahp new <kind>``.

    Routes to one of three scaffolders. The first positional is the
    kind (``tool``, ``integration``, ``agent``); the rest of the args
    are kind-specific.
    """
    from ahp import scaffolders
    try:
        if args.kind == "tool":
            path = scaffolders.scaffold_tool(
                name=scaffolders.normalize_name(args.name),
                scope=args.scope, kind=args.tool_kind,
                role=args.role, category=args.category,
                signature=args.signature or "query: str",
                summary=args.summary,
                out_dir=Path(args.path) if args.path else None,
                force=args.force,
            )
        elif args.kind == "integration":
            path = scaffolders.scaffold_integration(
                name=scaffolders.normalize_name(args.name),
                kind=args.type,
                scope=args.scope,
                out_dir=Path(args.path) if args.path else None,
                force=args.force,
            )
        elif args.kind == "agent":
            path = scaffolders.scaffold_agent(
                name=scaffolders.normalize_name(args.name),
                kind=args.type,
                scope=args.scope, role=args.role,
                out_dir=Path(args.path) if args.path else None,
                force=args.force,
            )
        else:  # pragma: no cover — argparse enforces choices
            print(f"unknown new kind: {args.kind!r}", file=sys.stderr)
            return 2
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"wrote {path}", file=out)
    return 0


async def _register_agent_async(args: argparse.Namespace, out: TextIO) -> int:
    """``ahp register agent`` — write the durable AgentMeta record.

    Does NOT mark alive. The agent stays invisible to pattern
    resolution until ``ahp start agent`` flips its heartbeat. Use this
    so durable claim is separable from menu visibility — matching the
    broker-as-source-of-truth model.
    """
    from ahp.registry.registry import AgentMeta, AgentRegistry
    try:
        address = _build_address(args)
    except ValueError as exc:
        print(f"invalid agent address: {exc}", file=sys.stderr)
        return 2
    meta = AgentMeta(
        capabilities=list(args.capability or []),
        description=args.description,
    )
    client = _connect_redis(args.redis_url)
    registry = AgentRegistry(client)
    try:
        await registry.register(address, meta, mark_alive=False)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
    print(f"registered {address}", file=out)
    return 0


async def _start_agent_async(args: argparse.Namespace, out: TextIO) -> int:
    """``ahp start agent`` — mark visible on the menu.

    Sets the liveness key on a previously-registered agent. The agent's
    process (wherever it's hosted) continues to run untouched; this is
    purely a broker-side advertise flip.
    """
    from ahp.registry.registry import AgentRegistry
    try:
        address = _build_address(args)
    except ValueError as exc:
        print(f"invalid agent address: {exc}", file=sys.stderr)
        return 2
    client = _connect_redis(args.redis_url)
    registry = AgentRegistry(client)
    try:
        ok = await registry.heartbeat(address)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
    if not ok:
        print(
            f"no record for {address} — run "
            f"`ahp register agent ...` first",
            file=sys.stderr,
        )
        return 2
    print(f"visible: {address}", file=out)
    return 0


async def _stop_agent_async(args: argparse.Namespace, out: TextIO) -> int:
    """``ahp stop agent`` — hide from the menu, keep the durable record.

    The agent process may still be running; this only clears the
    liveness key. Bring it back later with ``ahp start agent``.
    """
    from ahp.registry.registry import AgentRegistry
    try:
        address = _build_address(args)
    except ValueError as exc:
        print(f"invalid agent address: {exc}", file=sys.stderr)
        return 2
    client = _connect_redis(args.redis_url)
    registry = AgentRegistry(client)
    try:
        had_record = await registry.hide(address)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
    if not had_record:
        print(
            f"no record for {address} (nothing to hide)",
            file=sys.stderr,
        )
        return 2
    print(f"hidden: {address}", file=out)
    return 0


async def _deregister_agent_async(args: argparse.Namespace, out: TextIO) -> int:
    """``ahp deregister agent`` — remove the record entirely.

    Strong form of stop: drops the durable AgentMeta AND the liveness
    key. The agent must re-``register`` before it can be made visible
    again.
    """
    from ahp.registry.registry import AgentRegistry
    try:
        address = _build_address(args)
    except ValueError as exc:
        print(f"invalid agent address: {exc}", file=sys.stderr)
        return 2
    client = _connect_redis(args.redis_url)
    registry = AgentRegistry(client)
    try:
        await registry.deregister(address)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
    print(f"deregistered {address}", file=out)
    return 0


def cmd_register_agent(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_register_agent_async(args, out))


# ── surveys: list-surveys / vote ──────────────────────────────────────


async def _list_surveys_async(args: argparse.Namespace, out: TextIO) -> int:
    """List pending surveys from the broker queue.

    Defaults to surveys whose ``dispatch_at`` has already passed.
    ``--include-future`` shows queued-but-not-yet-due. ``--for ADDR``
    filters to surveys targeting a specific surveyed actor.
    """
    from ahp.broker.surveys import SurveyQueue

    client = _connect_redis(args.redis_url)
    queue = SurveyQueue(client)
    try:
        pending = await queue.list_pending(
            include_future=args.include_future,
            surveyed_actor=args.for_addr,
            limit=args.limit,
        )
    finally:
        try:
            await client.aclose()
        except Exception:
            pass

    if not pending:
        print(
            "(no pending surveys)"
            if not args.for_addr
            else f"(no pending surveys for {args.for_addr})",
            file=out,
        )
        return 0

    rows = [
        (
            req.survey_id[:12],
            req.kind,
            req.surveyed_actor,
            req.target_server,
            f"{req.reward:.4f}",
            time.strftime("%H:%M:%S", time.localtime(req.dispatch_at)),
        )
        for req in pending
    ]
    _render_table(
        out,
        ["survey_id", "kind", "actor", "server", "reward", "due"],
        rows,
    )
    return 0


def cmd_list_surveys(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_list_surveys_async(args, out))


async def _vote_async(args: argparse.Namespace, out: TextIO) -> int:
    """Submit a :class:`SurveyResponse` for a queued survey.

    The score is clamped to ``[0, 1]`` before submission; pass a value
    in that range or use ``--score`` with a 1..5 scale and ``--scale``
    will normalize for you. ``--allow-training`` records the
    actor-consent-at-collection-time flag — defaults False because
    training-data opt-in is intended to be explicit.

    Wallet + CSAT side effects fold into the broker. On success this
    prints the actor's new wallet balance (after the survey reward
    credit).
    """
    from ahp.broker import Broker
    from ahp.broker.surveys import SurveyResponse

    score = float(args.score)
    if args.scale == "1to5":
        if not (1.0 <= score <= 5.0):
            print(
                f"score must be in [1, 5] when --scale 1to5 "
                f"(got {score})",
                file=sys.stderr,
            )
            return 2
        score = (score - 1.0) / 4.0
    elif args.scale == "zero_to_one":
        if not (0.0 <= score <= 1.0):
            print(
                f"score must be in [0, 1] when --scale zero_to_one "
                f"(got {score})",
                file=sys.stderr,
            )
            return 2
    else:  # pragma: no cover — argparse enforces
        print(f"unknown scale {args.scale!r}", file=sys.stderr)
        return 2

    client = _connect_redis(args.redis_url)
    broker = Broker(client)
    try:
        # Reach the queue's request record so we have target_server,
        # actor, recipe, settlement_id — the response must echo them
        # for the row to be useful later.
        request = await broker.surveys.get_request(args.survey_id)
        if request is None:
            print(
                f"no such survey: {args.survey_id!r}",
                file=sys.stderr,
            )
            return 2
        response = SurveyResponse(
            survey_id=request.survey_id,
            surveyed_actor=request.surveyed_actor,
            target_server=request.target_server,
            recipe=request.recipe,
            settlement_id=request.settlement_id,
            score=score,
            free_text=args.free_text or "",
            consent_csat_routing=not args.deny_csat,
            consent_training_export=args.allow_training,
        )
        was_new = await broker.submit_survey_response(response)
        if not was_new:
            print(
                f"survey {args.survey_id} already has a response on file",
                file=sys.stderr,
            )
            return 2
        balance = (await broker.wallet(
            request.surveyed_actor,
        ).get_state()).balance
        print(
            f"recorded vote for {args.survey_id}: "
            f"score={score:.2f} actor={request.surveyed_actor} "
            f"new_balance={balance:.4f}",
            file=out,
        )
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
    return 0


def cmd_vote(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_vote_async(args, out))


# ── fire-surveys / export-surveys ─────────────────────────────────────


async def _fire_surveys_async(args: argparse.Namespace, out: TextIO) -> int:
    """Dispatch ready surveys to their surveyed actors.

    Reads from the same queue as ``list-surveys``. With ``--dry-run``
    just reports what would fire; without, sends a ``Code.HUMAN_OBSERVE``
    SEND message per ready survey via the bus. Idempotent — a
    survey is only fired once until its response is submitted.
    """
    from ahp.broker import Broker
    from ahp.transport.redis_bus import RedisBus

    client = _connect_redis(args.redis_url)
    broker = Broker(client)
    bus = RedisBus(client)
    try:
        if args.dry_run:
            dispatched = await broker.surveys.fire_due(bus=None)
            mode = "dry-run"
        else:
            dispatched = await broker.surveys.fire_due(
                bus=bus, max_dispatch=args.limit,
            )
            mode = "fired"
        if not dispatched:
            print(f"({mode}: no ready surveys)", file=out)
            return 0
        for req in dispatched:
            print(
                f"{mode}  {req.survey_id[:12]}  "
                f"actor={req.surveyed_actor}  "
                f"server={req.target_server}  "
                f"reward={req.reward:.4f}",
                file=out,
            )
    finally:
        try:
            await bus.close()
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass
    return 0


def cmd_fire_surveys(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_fire_surveys_async(args, out))


async def _export_surveys_async(args: argparse.Namespace, out: TextIO) -> int:
    """Dump consenting :class:`SurveyResponse` rows to a JSONL file."""
    from ahp.broker.surveys import export_corpus, write_corpus_jsonl

    client = _connect_redis(args.redis_url)
    try:
        if args.out:
            count = await write_corpus_jsonl(
                client, args.out,
                since=args.since, anonymize=not args.no_anonymize,
            )
            print(
                f"wrote {count} row(s) to {args.out} "
                f"(since={args.since}, anonymize={not args.no_anonymize})",
                file=out,
            )
        else:
            rows = await export_corpus(
                client, since=args.since, anonymize=not args.no_anonymize,
            )
            from dataclasses import asdict as _asdict
            for row in rows:
                print(json.dumps(_asdict(row), default=str), file=out)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
    return 0


def cmd_export_surveys(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_export_surveys_async(args, out))


# ── tap / send / describe-agent ───────────────────────────────────────


def _format_tap_line(message: Any) -> str:
    """One terse line per tap message, e.g.::

        2026-05-16T08:54:12  CAST  tifin.adv.science.astro.s.session.bull → *.adv.science.astro.*.*.*  cast.interview.text  body...
    """
    when = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.localtime(getattr(message, "sent_at", time.time()))
    )
    body = message.body
    if isinstance(body, (dict, list)):
        try:
            body_repr = json.dumps(body, default=str)
        except Exception:
            body_repr = repr(body)
    else:
        body_repr = str(body)
    if len(body_repr) > 80:
        body_repr = body_repr[:77] + "..."
    return (
        f"{when}  {message.verb:8s}  "
        f"{message.source} -> {message.target}  "
        f"{message.code}  {body_repr}"
    )


async def _tap_async(args: argparse.Namespace, out: TextIO) -> int:
    """Stream every message on the bus's tap channel, with filters."""
    from ahp.core.codes import Code
    from ahp.core.address import AgentAddress
    from ahp.core.pattern import AddressPattern
    from ahp.transport.redis_bus import RedisBus

    # Build a client-side predicate from the filter args. We use the
    # bus's predicate hook so we don't redundantly evaluate inside the
    # CLI loop.
    code_glob = args.code
    src_pattern: AddressPattern | None = (
        AddressPattern.parse(args.source) if args.source else None
    )
    tgt_pattern: AddressPattern | None = (
        AddressPattern.parse(args.target) if args.target else None
    )

    def predicate(msg: Any) -> bool:
        if code_glob and not Code.matches(msg.code, code_glob):
            return False
        if src_pattern is not None and not src_pattern.matches(msg.source):
            return False
        if tgt_pattern is not None:
            t = msg.target
            if isinstance(t, AgentAddress):
                if not tgt_pattern.matches(t):
                    return False
            elif isinstance(t, AddressPattern):
                # An outbound pattern is fuzzy; match if the two patterns
                # share *any* match by comparing string forms exactly.
                if str(t) != str(tgt_pattern):
                    return False
        return True

    client = _connect_redis(args.redis_url)
    bus = RedisBus(client)
    seen = 0
    try:
        sub = await bus.tap_subscribe(
            predicate=predicate if (
                code_glob or src_pattern or tgt_pattern
            ) else None,
        )
        print(
            "tap: streaming bus events (Ctrl-C to stop)…",
            file=out, flush=True,
        )
        async with sub:
            async for msg in sub.messages(idle_timeout=0.5):
                print(_format_tap_line(msg), file=out, flush=True)
                seen += 1
                if args.limit and seen >= args.limit:
                    break
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await bus.close()
        except Exception:
            pass
    return 0


def cmd_tap(args: argparse.Namespace, out: TextIO) -> int:
    try:
        return asyncio.run(_tap_async(args, out))
    except KeyboardInterrupt:
        return 0


# ── send ──────────────────────────────────────────────────────────────


async def _send_async(args: argparse.Namespace, out: TextIO) -> int:
    """Fire a one-off SEND or SEND-GET via :class:`ProtocolEngine`."""
    from ahp.core import AddressPattern, Message
    from ahp.core.address import AgentAddress
    from ahp.core.compatibility import CompatibilityMatrix
    from ahp.engine.router import ProtocolEngine
    from ahp.registry.registry import AgentRegistry
    from ahp.transport.cache import ProtocolCache
    from ahp.transport.redis_bus import RedisBus

    # Source defaults to a throwaway human address so we don't litter
    # the registry; thread defaults to a fresh per-call id.
    source = AgentAddress.parse(
        args.source
        or f"you.cli.x.y.s.ephemeral.send-{int(time.time()*1000)}"
    )
    target_str = args.target
    target: Any
    if "*" in target_str:
        target = AddressPattern.parse(target_str)
    else:
        target = AgentAddress.parse(target_str)

    # Try to parse body as JSON; fall back to raw string. Lets callers
    # ship either: `--body '{"q": "hi"}'` or `--body "plain text"`.
    body: Any
    try:
        body = json.loads(args.body)
    except (json.JSONDecodeError, TypeError):
        body = args.body

    thread = args.thread or f"cli::send::{int(time.time()*1000)}"

    client = _connect_redis(args.redis_url)
    bus = RedisBus(client)
    registry = AgentRegistry(client)
    cache = ProtocolCache(client)
    engine = ProtocolEngine(
        bus, registry, cache, CompatibilityMatrix(),
        default_timeout=args.timeout,
    )

    # Need a source address registered for SEND-GET so the engine can
    # route the reply back. Register-then-deregister for the call's
    # lifetime.
    registered = False
    if args.get:
        try:
            await registry.register(source)
            registered = True
        except Exception:
            log.warning("could not register source address %s", source)

    try:
        verb = "SEND-GET" if args.get else "SEND"
        msg = Message(
            source=source, target=target,
            code=args.code, verb=verb,
            body=body, thread=thread,
        )
        result = await engine.handle(msg, timeout=args.timeout)
        if args.get:
            if result is None:
                print("(no response within timeout)", file=out)
                return 0
            print(_format_tap_line(result), file=out)
        else:
            print(
                f"sent to {target} (delivered to {result} subscriber(s))",
                file=out,
            )
    finally:
        if registered:
            try:
                await registry.deregister(source)
            except Exception:
                pass
        try:
            await bus.close()
        except Exception:
            pass
    return 0


def cmd_send(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_send_async(args, out))


# ── describe-agent ────────────────────────────────────────────────────


async def _describe_agent_async(args: argparse.Namespace, out: TextIO) -> int:
    """Pretty-print everything we know about one agent.

    Pulls four planes:
      * registry: address, alive status, AgentMeta
      * reputation: ReputationEntry from broker (if any)
      * server binding: owning ServerMeta + best matching MenuLeaf
      * (audit events sourced from the in-memory sink only when one is
        wired into the running engine — not addressable from the CLI;
        skipped here)
    """
    from ahp.broker import Broker
    from ahp.core import AgentAddress
    from ahp.economy.compute_provider import best_leaf
    from ahp.registry.registry import AgentRegistry

    try:
        address = AgentAddress.parse(args.address)
    except ValueError as exc:
        print(f"invalid agent address: {exc}", file=sys.stderr)
        return 2

    client = _connect_redis(args.redis_url)
    registry = AgentRegistry(client)
    broker = Broker(client)
    try:
        meta = await registry.get(address)
        alive = await registry.is_alive(address)

        print(f"address:        {address}", file=out)
        print(
            f"status:         {'alive' if alive else 'stale or unknown'}",
            file=out,
        )

        if meta is None:
            print("(no registry metadata — agent is not registered)",
                  file=out)
            return 0

        print(
            f"capabilities:   {', '.join(meta.capabilities) if meta.capabilities else '-'}",
            file=out,
        )
        print(f"reputation*:    {meta.reputation:.2f}    "
              f"(*from AgentMeta; the broker tracks a richer record below)",
              file=out)
        if meta.description:
            print(f"description:    {meta.description}", file=out)
        if meta.health_endpoint:
            print(f"health_endpt:   {meta.health_endpoint}", file=out)
        if meta.extra:
            print(f"extra:          {json.dumps(meta.extra, default=str)}",
                  file=out)
        if meta.registered_at:
            print(
                "registered:     "
                + time.strftime(
                    "%Y-%m-%dT%H:%M:%S",
                    time.localtime(meta.registered_at),
                ),
                file=out,
            )

        # Broker plane — reputation + owning server.
        rep = await broker.get_reputation(str(address))
        if rep is not None:
            print(file=out)
            print("broker reputation:", file=out)
            print(f"  reputation:   {rep.reputation:.3f}", file=out)
            print(f"  completed:    {rep.completed_accepted}/{rep.completed_total}"
                  f"  failed: {rep.failed}", file=out)
            print(f"  csat:         {rep.csat:.3f}  ({rep.csat_samples} samples)",
                  file=out)
            print(f"  avg_latency:  {rep.avg_latency_ms:.0f}ms", file=out)
            print(f"  avg_overage:  {rep.avg_overage:.3f}", file=out)

        owning_servers = await broker.servers.discover(alive_only=False)
        owning = next((s for s in owning_servers if s.org == address.org), None)
        if owning is not None:
            print(file=out)
            print("owning server:", file=out)
            print(f"  server_id:    {owning.server_id}", file=out)
            print(f"  base_rate:    {owning.base_rate}", file=out)
            print(f"  binding:      {owning.compute_binding}", file=out)
            print(f"  rank_by:      {owning.compute_ranking}", file=out)
            leaves = await broker.compute.list_leaves(only_alive_providers=True)
            chosen = best_leaf(
                owning.compute_binding, leaves,
                rank_by=owning.compute_ranking,  # type: ignore[arg-type]
            )
            if chosen is not None:
                print(
                    f"  best leaf:    {chosen.address}  "
                    f"@ {chosen.rate_per_1k_chars}/1k chars  "
                    f"({chosen.latency_p95_ms:.0f}ms p95)",
                    file=out,
                )
            else:
                print("  best leaf:    (no matching live leaf)", file=out)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
    return 0


def cmd_describe_agent(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_describe_agent_async(args, out))


# ── audit-tail ────────────────────────────────────────────────────────


def _format_audit_line(stream_id: str, payload: dict[str, Any]) -> str:
    """One terse line per audit event."""
    when = time.strftime(
        "%Y-%m-%dT%H:%M:%S",
        time.localtime(payload.get("timestamp") or time.time()),
    )
    op = payload.get("op", "?")
    target = payload.get("target") or "-"
    success = payload.get("success", True)
    suffix = "" if success else f"  ERROR={payload.get('error') or ''}"
    extra = payload.get("extra") or {}
    extra_repr = ""
    if extra:
        try:
            extra_repr = "  " + json.dumps(extra, default=str)
        except Exception:
            extra_repr = "  " + str(extra)
        if len(extra_repr) > 120:
            extra_repr = extra_repr[:117] + "..."
    return f"{when}  {stream_id}  {op:32s}  target={target}{suffix}{extra_repr}"


async def _audit_tail_async(args: argparse.Namespace, out: TextIO) -> int:
    """Read audit events from a Redis Streams key produced by
    :class:`RedisStreamAuditSink`.

    Two modes:
      * one-shot (default): XRANGE from --since (default '-' = oldest)
        up to --limit entries, print, exit.
      * --follow: after the initial range, keep XREADing for new
        entries with a BLOCK timeout. Ctrl-C exits.

    --op accepts a hierarchical glob (e.g. ``broker.*``,
    ``survey.response``) and filters client-side.
    """
    from ahp.audit import DEFAULT_REDIS_AUDIT_STREAM
    from ahp.core.codes import Code

    client = _connect_redis(args.redis_url)
    stream_key = args.stream or DEFAULT_REDIS_AUDIT_STREAM
    seen = 0
    last_id: str = args.since or "-"

    def _matches(payload: dict[str, Any]) -> bool:
        if args.op and not Code.matches(payload.get("op", ""), args.op):
            return False
        if args.target_contains and args.target_contains not in (
            payload.get("target") or ""
        ):
            return False
        return True

    try:
        # Initial range: from `since` (or oldest) forward.
        entries = await client.xrange(
            stream_key, min=last_id, max="+",
            count=args.limit if args.limit else None,
        )
        for entry_id, fields in entries:
            if isinstance(entry_id, (bytes, bytearray)):
                entry_id = entry_id.decode("utf-8")
            raw = fields.get("data")
            if raw is None:
                continue
            if isinstance(raw, (bytes, bytearray)):
                raw = raw.decode("utf-8")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not _matches(payload):
                continue
            print(_format_audit_line(entry_id, payload), file=out, flush=True)
            seen += 1
            last_id = entry_id
            if args.limit and seen >= args.limit:
                return 0

        if not args.follow:
            if seen == 0:
                print(
                    f"(no audit entries on {stream_key})",
                    file=out,
                )
            return 0

        # Follow mode: tail forever via XREAD with BLOCK.
        print(
            f"tail: following {stream_key} (Ctrl-C to stop)…",
            file=out, flush=True,
        )
        # XREAD wants the "next entry after this id"; '$' means "only
        # new entries from now", which is what we want after the
        # initial range.
        cursor = last_id if last_id != "-" else "$"
        while True:
            try:
                resp = await client.xread(
                    {stream_key: cursor},
                    block=1000,  # 1s block; loop to allow Ctrl-C.
                    count=50,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("xread failed; backing off 500ms")
                await asyncio.sleep(0.5)
                continue
            if not resp:
                continue
            for _stream, entries in resp:
                for entry_id, fields in entries:
                    if isinstance(entry_id, (bytes, bytearray)):
                        entry_id = entry_id.decode("utf-8")
                    raw = fields.get("data")
                    if raw is None:
                        continue
                    if isinstance(raw, (bytes, bytearray)):
                        raw = raw.decode("utf-8")
                    try:
                        payload = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    cursor = entry_id
                    if not _matches(payload):
                        continue
                    print(
                        _format_audit_line(entry_id, payload),
                        file=out, flush=True,
                    )
                    seen += 1
                    if args.limit and seen >= args.limit:
                        return 0
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await client.aclose()
        except Exception:
            pass
    return 0


def cmd_audit_tail(args: argparse.Namespace, out: TextIO) -> int:
    try:
        return asyncio.run(_audit_tail_async(args, out))
    except KeyboardInterrupt:
        return 0


def cmd_start_agent(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_start_agent_async(args, out))


def cmd_stop_agent(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_stop_agent_async(args, out))


def cmd_deregister_agent(args: argparse.Namespace, out: TextIO) -> int:
    return asyncio.run(_deregister_agent_async(args, out))


# ── argparse setup ────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ahp",
        description="Inspect and scaffold AHP entities.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            tools and resources only appear in listings if their module
            has been imported. Use -m DOTTED.PATH (one or more times) to
            import user modules before running a command.
        """),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # list-tools
    p_lt = sub.add_parser("list-tools", help="list registered tools")
    p_lt.add_argument("-m", "--module", action="append", default=[],
                      help="dotted module to import (may be repeated)")
    p_lt.add_argument("--for", dest="for_addr",
                      help="filter to tools visible to this agent address")
    p_lt.add_argument("--tag", action="append", default=[],
                      help="filter by tag (may be repeated; ANY-of)")
    p_lt.set_defaults(func=cmd_list_tools)

    # list-resources
    p_lr = sub.add_parser("list-resources", help="list registered resources")
    p_lr.add_argument("-m", "--module", action="append", default=[])
    p_lr.add_argument("--for", dest="for_addr",
                      help="filter to resources visible to this agent address")
    p_lr.set_defaults(func=cmd_list_resources)

    # list-groups
    p_lg = sub.add_parser("list-groups", help="list named address-pattern groups")
    p_lg.add_argument("-m", "--module", action="append", default=[])
    p_lg.set_defaults(func=cmd_list_groups)

    # list-skills
    p_lsk = sub.add_parser(
        "list-skills",
        help="list registered skill playbooks (workflow + suggested addresses)",
    )
    p_lsk.add_argument(
        "-m", "--module", action="append", default=[],
        help="dotted module to import (may be repeated)",
    )
    p_lsk.add_argument(
        "--for", dest="for_addr",
        help="filter to skills visible to this agent address",
    )
    p_lsk.add_argument(
        "--tag", action="append", default=[],
        help="filter by tag (may be repeated; ANY-of)",
    )
    p_lsk.set_defaults(func=cmd_list_skills)

    # list-agents (live Redis)
    p_la = sub.add_parser(
        "list-agents",
        help="query a live registry over Redis for currently-registered agents",
    )
    default_url = os.environ.get("AHP_REDIS_URL", "redis://localhost:6379/0")
    p_la.add_argument(
        "--redis-url",
        default=default_url,
        help=f"Redis URL to query (default: {default_url}, "
             f"or $AHP_REDIS_URL when set)",
    )
    p_la.add_argument(
        "--pattern",
        help="AddressPattern to filter results (default: every alive agent)",
    )
    p_la.add_argument(
        "--all", action="store_true",
        help="include registered agents whose liveness marker has expired",
    )
    p_la.set_defaults(func=cmd_list_agents)

    # profile
    p_pr = sub.add_parser("profile",
                          help="show the resolved AgentProfile for an address")
    p_pr.add_argument("address", help="agent address URI")
    p_pr.add_argument("-m", "--module", action="append", default=[])
    p_pr.set_defaults(func=cmd_profile)

    # template
    p_tp = sub.add_parser("template", help="print a starter tool/resource module")
    _add_template_args(p_tp)
    p_tp.set_defaults(func=cmd_template)

    # scaffold
    p_sc = sub.add_parser("scaffold", help="write a starter tool/resource module to a file")
    _add_template_args(p_sc)
    p_sc.add_argument("-o", "--output", required=True, help="destination path")
    p_sc.add_argument("-f", "--force", action="store_true",
                      help="overwrite the destination if it exists")
    p_sc.set_defaults(func=cmd_scaffold)

    # new <kind> — Django-style project scaffolder
    p_new = sub.add_parser(
        "new",
        help="scaffold a new tool, integration, or agent into the project tree",
    )
    new_sub = p_new.add_subparsers(dest="kind", required=True)

    # new tool
    p_new_tool = new_sub.add_parser(
        "tool", help="write ./tools/<name>.py — a @tool-decorated stub"
    )
    p_new_tool.add_argument("--name", required=True,
                            help="tool function name (snake_case)")
    p_new_tool.add_argument("--scope", default="tifin")
    p_new_tool.add_argument("--tool-kind", dest="tool_kind", default="api",
                            help="tool address `kind` field (api, db, fs, ...)")
    p_new_tool.add_argument("--role", default="*",
                            help="tool address `role` field — default `*` "
                                 "(any role in scope can use it)")
    p_new_tool.add_argument("--category", default="search",
                            help="tool address `category` field")
    p_new_tool.add_argument("--signature",
                            help="Python signature for the function body, "
                                 "e.g. 'query: str, top_k: int = 5'")
    p_new_tool.add_argument("--summary",
                            help="one-line docstring for the generated function")
    p_new_tool.add_argument("--path",
                            help="project root override (default: cwd)")
    p_new_tool.add_argument("-f", "--force", action="store_true",
                            help="overwrite if the target file exists")
    p_new_tool.set_defaults(func=cmd_new)

    # new integration
    p_new_int = new_sub.add_parser(
        "integration",
        help="write ./integrations/<name>.py — external-service wrapper",
    )
    p_new_int.add_argument("--name", required=True,
                           help="integration name (snake_case)")
    p_new_int.add_argument("--type", default="api_key",
                           choices=["api_key", "oauth", "webhook"],
                           help="auth pattern; oauth scaffold is "
                                "intentionally a stub")
    p_new_int.add_argument("--scope", default="tifin")
    p_new_int.add_argument("--path",
                           help="project root override (default: cwd)")
    p_new_int.add_argument("-f", "--force", action="store_true",
                           help="overwrite if the target file exists")
    p_new_int.set_defaults(func=cmd_new)

    # new agent
    p_new_agent = new_sub.add_parser(
        "agent",
        help="write ./agents/<name>.py — a runnable agent module",
    )
    p_new_agent.add_argument("--name", required=True,
                             help="agent name (snake_case)")
    p_new_agent.add_argument("--type", default="simple",
                             choices=["simple", "react", "deepagent"],
                             help="framework for the agent body")
    p_new_agent.add_argument("--scope", default="tifin")
    p_new_agent.add_argument("--role", default="researcher")
    p_new_agent.add_argument("--path",
                             help="project root override (default: cwd)")
    p_new_agent.add_argument("-f", "--force", action="store_true",
                             help="overwrite if the target file exists")
    p_new_agent.set_defaults(func=cmd_new)

    # register / start / stop / deregister — share the address-build args
    def _add_address_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--name", required=True,
                       help="agent instance name (snake_case)")
        p.add_argument("--scope", default="tifin",
                       help="agent address `org` field (a.k.a. scope)")
        p.add_argument("--role", default="researcher")
        p.add_argument("--domain", default="example")
        p.add_argument("--subdomain", default="example")
        p.add_argument("--accept", default="s",
                       help="accept-tier glob; `s` matches structured text")
        p.add_argument("--lifecycle", default="session",
                       help="session | longterm | ephemeral | stale-ok")
        default_url = os.environ.get(
            "AHP_REDIS_URL", "redis://localhost:6379/0",
        )
        p.add_argument(
            "--redis-url", default=default_url,
            help=f"Redis URL (default: {default_url}, "
                 f"or $AHP_REDIS_URL when set)",
        )

    # register agent — durable record only
    p_reg = sub.add_parser(
        "register",
        help="durable agent registration (does NOT make visible)",
    )
    reg_sub = p_reg.add_subparsers(dest="entity", required=True)
    p_reg_a = reg_sub.add_parser(
        "agent",
        help="write the durable AgentMeta to Redis; agent stays hidden "
             "until `ahp start agent` flips its heartbeat",
    )
    _add_address_args(p_reg_a)
    p_reg_a.add_argument("--capability", action="append", default=[],
                         help="tag this agent with a capability "
                              "(may be repeated)")
    p_reg_a.add_argument("--description", help="short human description")
    p_reg_a.set_defaults(func=cmd_register_agent)

    # start agent — make visible
    p_start = sub.add_parser(
        "start",
        help="advertise a registered agent on the menu (broker-side)",
    )
    start_sub = p_start.add_subparsers(dest="entity", required=True)
    p_start_a = start_sub.add_parser(
        "agent",
        help="heartbeat a registered agent so pattern resolution finds it",
    )
    _add_address_args(p_start_a)
    p_start_a.set_defaults(func=cmd_start_agent)

    # stop agent — hide
    p_stop = sub.add_parser(
        "stop",
        help="hide an agent from the menu (the agent process keeps running)",
    )
    stop_sub = p_stop.add_subparsers(dest="entity", required=True)
    p_stop_a = stop_sub.add_parser(
        "agent",
        help="clear an agent's heartbeat (keeps the durable record)",
    )
    _add_address_args(p_stop_a)
    p_stop_a.set_defaults(func=cmd_stop_agent)

    # deregister agent — remove the record entirely
    p_dereg = sub.add_parser(
        "deregister",
        help="remove an agent's durable record (strong form of stop)",
    )
    dereg_sub = p_dereg.add_subparsers(dest="entity", required=True)
    p_dereg_a = dereg_sub.add_parser(
        "agent",
        help="drop the AgentMeta hash entry AND the heartbeat key",
    )
    _add_address_args(p_dereg_a)
    p_dereg_a.set_defaults(func=cmd_deregister_agent)

    # list-surveys — read pending surveys from the broker queue
    p_ls = sub.add_parser(
        "list-surveys",
        help="list pending surveys queued by the broker",
    )
    default_url = os.environ.get(
        "AHP_REDIS_URL", "redis://localhost:6379/0",
    )
    p_ls.add_argument(
        "--redis-url", default=default_url,
        help=f"Redis URL (default: {default_url}, "
             f"or $AHP_REDIS_URL when set)",
    )
    p_ls.add_argument(
        "--for", dest="for_addr",
        help="filter to surveys whose surveyed_actor is this address",
    )
    p_ls.add_argument(
        "--include-future", action="store_true",
        help="also list surveys whose dispatch_at is still in the future",
    )
    p_ls.add_argument(
        "--limit", type=int, default=100,
        help="max rows to return (default: 100)",
    )
    p_ls.set_defaults(func=cmd_list_surveys)

    # vote — submit a SurveyResponse
    p_vote = sub.add_parser(
        "vote",
        help="submit a response for a queued survey",
    )
    p_vote.add_argument(
        "--redis-url", default=default_url,
        help=f"Redis URL (default: {default_url})",
    )
    p_vote.add_argument(
        "--survey-id", required=True,
        help="survey_id from `ahp list-surveys`",
    )
    p_vote.add_argument(
        "--score", required=True, type=float,
        help="rating; range depends on --scale",
    )
    p_vote.add_argument(
        "--scale", default="zero_to_one",
        choices=["zero_to_one", "1to5"],
        help="how to interpret --score (default: zero_to_one)",
    )
    p_vote.add_argument(
        "--free-text", default="",
        help="optional comment recorded with the response",
    )
    p_vote.add_argument(
        "--allow-training", action="store_true",
        help="record consent_training_export=True on this response "
             "(default: False; training-data opt-in is explicit)",
    )
    p_vote.add_argument(
        "--deny-csat", action="store_true",
        help="record consent_csat_routing=False on this response "
             "(default: consent is granted)",
    )
    p_vote.set_defaults(func=cmd_vote)

    # fire-surveys — dispatch ready surveys to surveyed actors
    p_fs = sub.add_parser(
        "fire-surveys",
        help="dispatch ready surveys as HUMAN_OBSERVE messages on the bus",
    )
    p_fs.add_argument(
        "--redis-url", default=default_url,
        help=f"Redis URL (default: {default_url})",
    )
    p_fs.add_argument(
        "--dry-run", action="store_true",
        help="report what would be dispatched without sending anything",
    )
    p_fs.add_argument(
        "--limit", type=int, default=50,
        help="max surveys to dispatch in this sweep (default: 50)",
    )
    p_fs.set_defaults(func=cmd_fire_surveys)

    # export-surveys — consenting training-data rows as JSONL
    p_es = sub.add_parser(
        "export-surveys",
        help="export consenting survey responses as JSONL training data",
    )
    p_es.add_argument(
        "--redis-url", default=default_url,
        help=f"Redis URL (default: {default_url})",
    )
    p_es.add_argument(
        "--out",
        help="write to this path (default: stream to stdout)",
    )
    p_es.add_argument(
        "--since", type=float, default=0.0,
        help="wall-clock cutoff (seconds since epoch); only export "
             "responses with collected_at >= this value",
    )
    p_es.add_argument(
        "--no-anonymize", action="store_true",
        help="emit raw actor addresses instead of stable opaque hashes "
             "(internal exports only; default anonymizes)",
    )
    p_es.set_defaults(func=cmd_export_surveys)

    # tap — live event stream
    p_tap = sub.add_parser(
        "tap",
        help="live-stream the bus's tap channel (Ctrl-C to stop)",
    )
    p_tap.add_argument(
        "--redis-url", default=default_url,
        help=f"Redis URL (default: {default_url})",
    )
    p_tap.add_argument(
        "--code",
        help="filter to messages whose code matches this glob "
             "(e.g. 'cast.interview.*', 'send.send_get.*')",
    )
    p_tap.add_argument(
        "--source",
        help="filter to messages whose source matches this AddressPattern",
    )
    p_tap.add_argument(
        "--target",
        help="filter to messages whose target matches this AddressPattern",
    )
    p_tap.add_argument(
        "--limit", type=int, default=0,
        help="stop after N matching messages (default: stream until Ctrl-C)",
    )
    p_tap.set_defaults(func=cmd_tap)

    # send — one-off message
    p_send = sub.add_parser(
        "send",
        help="send a one-off SEND (or SEND-GET with --get) message",
    )
    p_send.add_argument(
        "--redis-url", default=default_url,
        help=f"Redis URL (default: {default_url})",
    )
    p_send.add_argument(
        "--target", required=True,
        help="target AgentAddress or AddressPattern (e.g. "
             "'tifin.researcher.x.y.s.session.alice' or "
             "'*.researcher.*.*.*.*.*')",
    )
    p_send.add_argument(
        "--code", required=True,
        help="message code (e.g. 'send.send_get.interview.text')",
    )
    p_send.add_argument(
        "--body", default="",
        help="message body — parsed as JSON if it parses, otherwise "
             "sent as a raw string",
    )
    p_send.add_argument(
        "--source",
        help="source AgentAddress (default: a fresh ephemeral CLI address)",
    )
    p_send.add_argument(
        "--get", action="store_true",
        help="use SEND-GET and print the response (default: fire-and-forget SEND)",
    )
    p_send.add_argument(
        "--timeout", type=float, default=10.0,
        help="--get timeout in seconds (default: 10)",
    )
    p_send.add_argument(
        "--thread", help="thread id (default: a fresh per-call id)",
    )
    p_send.set_defaults(func=cmd_send)

    # describe-agent — read-only per-agent view
    p_desc = sub.add_parser(
        "describe-agent",
        help="show full per-agent state: registry, reputation, server, leaf",
    )
    p_desc.add_argument(
        "address",
        help="agent address URI (7 dot-separated fields)",
    )
    p_desc.add_argument(
        "--redis-url", default=default_url,
        help=f"Redis URL (default: {default_url})",
    )
    p_desc.set_defaults(func=cmd_describe_agent)

    # audit-tail — read from RedisStreamAuditSink
    p_at = sub.add_parser(
        "audit-tail",
        help="read audit events from a RedisStreamAuditSink-backed stream",
    )
    p_at.add_argument(
        "--redis-url", default=default_url,
        help=f"Redis URL (default: {default_url})",
    )
    p_at.add_argument(
        "--stream",
        help="stream key (default: ahp:audit:stream — "
             "DEFAULT_REDIS_AUDIT_STREAM)",
    )
    p_at.add_argument(
        "--since",
        help="start from this stream id (default: '-' = from oldest)",
    )
    p_at.add_argument(
        "--op",
        help="filter to events whose op matches this hierarchical glob "
             "(e.g. 'broker.*', 'survey.response', 'registry.*')",
    )
    p_at.add_argument(
        "--target-contains", dest="target_contains",
        help="filter to events whose target field contains this substring",
    )
    p_at.add_argument(
        "--limit", type=int, default=100,
        help="max entries to print (default: 100; 0 = unlimited)",
    )
    p_at.add_argument(
        "--follow", action="store_true",
        help="after the initial range, tail for new entries via XREAD",
    )
    p_at.set_defaults(func=cmd_audit_tail)

    return parser


def _add_template_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("target", choices=("tool", "resource"),
                   help="what to generate")
    # tool fields
    p.add_argument("--scope", default="tifin")
    p.add_argument("--kind", default="db")
    p.add_argument("--role", default="*",
                   help="tool only; default `*` (any role in scope)")
    p.add_argument("--category", default="crud",
                   help="tool only")
    p.add_argument("--operation",
                   help="tool only; defaults to the function name")
    p.add_argument("--signature",
                   help="tool only; comma-separated parameter list "
                        "(e.g. 'table: str, row_id: str, fields: dict')")
    # resource fields
    p.add_argument("--domain", default="finance",
                   help="resource only")
    p.add_argument("--subdomain", default="equities",
                   help="resource only")
    p.add_argument("--name", default="my_resource",
                   help="resource only; the resource's short name")
    p.add_argument("--factory-name", dest="factory_name",
                   help="resource only; the decorated function name")
    p.add_argument("--summary",
                   help="docstring summary line for the generated function")


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = out or sys.stdout
    if not hasattr(args, "func"):
        parser.print_help(file=out)
        return 1
    if args.cmd in ("template", "scaffold"):
        if args.target == "tool" and not args.operation:
            args.operation = "my_tool"
    return args.func(args, out)


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(main())
