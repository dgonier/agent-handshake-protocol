"""Command-line interface for inspecting and scaffolding AHP entities.

Run via the module entry point::

    python -m ahp <command> [options]

Subcommands:

* ``list-tools``       — registered tools (optionally filtered by agent address / tags)
* ``list-resources``   — registered resources
* ``list-groups``      — named address-pattern groups
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
import importlib
import sys
import textwrap
from pathlib import Path
from typing import Sequence, TextIO

from ahp.adapters import (
    DEFAULT_GROUP_REGISTRY,
    DEFAULT_RESOURCE_REGISTRY,
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
