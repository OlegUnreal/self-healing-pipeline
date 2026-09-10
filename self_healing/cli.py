"""CLI for demos and repairing a real workspace."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .config import load_settings
from .demo import stub_proposer
from .workspace import resolve_workspace


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="self_healing",
        description="Repair a failing Python workspace with a sandboxed agent.",
    )
    p.add_argument("--demo", action="store_true", help="stub diff-loop demo (no key)")
    p.add_argument("--demo-llm", action="store_true", help="live LLM diff-loop demo")
    p.add_argument("--demo-tools", action="store_true", help="scripted tool-loop demo")
    p.add_argument("--demo-graph", action="store_true", help="scripted graph-loop demo")
    p.add_argument(
        "--src",
        type=Path,
        help="source file *or* directory (jail root) to repair",
    )
    p.add_argument(
        "--test",
        help="path to a test file, or a Python snippet that must exit 0",
    )
    p.add_argument(
        "--mode",
        choices=("diff", "tools", "graph"),
        default="tools",
        help="orchestrator used with --src/--test (default: tools)",
    )
    p.add_argument(
        "--planner",
        choices=("auto", "stub", "openai"),
        default="auto",
        help="auto = openai if OPENAI_API_KEY else stub",
    )
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--stream", action="store_true", help="print graph/tool events")
    p.add_argument("--checkpoint", type=Path, help="sqlite checkpoint path (LangGraph extra)")
    p.add_argument("--thread-id", default="heal")
    p.add_argument(
        "--approve-mutations",
        action="store_true",
        help="require confirmation before write_file / apply_patch",
    )
    p.add_argument("--yes", action="store_true", help="auto-approve mutations")
    p.add_argument("--use-langgraph", action="store_true", help="force LangGraph runtime if installed")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _load_test(raw: str) -> str:
    path = Path(raw)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return raw


def _choose_backend(kind: str, settings) -> str:
    if kind == "stub":
        return "stub"
    if kind == "openai":
        return "openai"
    return "openai" if settings.openai_api_key else "stub"


def _mutation_approver(yes: bool) -> Callable[[dict[str, Any]], bool]:
    def approve(call: dict[str, Any]) -> bool:
        if yes:
            return True
        if not sys.stdin.isatty():
            print(f"rejecting {call.get('name')} (non-interactive; pass --yes)", file=sys.stderr)
            return False
        name = call.get("name")
        args = call.get("arguments") or {}
        preview = str(args)[:240]
        answer = input(f"approve {name} {preview}? [y/N] ").strip().lower()
        return answer in {"y", "yes"}

    return approve


def _print_report(report) -> int:
    for a in report:
        extra = f" tools={a.tools}" if a.tools else ""
        err = f" err={a.error}" if a.error else ""
        print(f"attempt {a.n}: passed={a.passed}{extra}{err}")
    if report.tool_trace:
        print("tool_trace:", [t["tool"] for t in report.tool_trace])
    print("success:", report.success)
    if report.final_source:
        print("final source:\n", report.final_source)
    if report.error and not report.success:
        print("error:", report.error, file=sys.stderr)
    return 0 if report.success else 2


def run_workspace(args: argparse.Namespace) -> int:
    from .agent import heal, heal_with_tools
    from .graph import heal_with_graph, make_checkpointer
    from .tools import Workspace

    settings = load_settings()
    try:
        source, root = resolve_workspace(args.src)
    except FileNotFoundError as e:
        print(f"fatal: {e}", file=sys.stderr)
        return 2
    test = _load_test(args.test)
    backend = _choose_backend(args.planner, settings)
    max_attempts = args.max_attempts or settings.max_attempts
    max_steps = args.max_steps or settings.max_tool_steps
    ws = Workspace(root, max_file_bytes=settings.max_file_bytes)

    on_event = None
    if args.stream:
        def on_event(node: str, state: dict[str, Any]) -> None:  # noqa: F811
            flag = "PASS" if state.get("passed") else state.get("failure_class") or "..."
            print(f"[{node}] steps={state.get('steps', 0)} {flag}")

    if args.mode == "diff":
        if backend == "openai":
            from .llm import make_openai_proposer

            proposer = make_openai_proposer(model=settings.model)
        else:
            proposer = stub_proposer
        report = heal(source, test, proposer, max_attempts=max_attempts, cwd=root)
        return _print_report(report)

    if backend == "openai":
        from .llm import make_openai_planner

        planner = make_openai_planner(model=settings.model)
    else:
        from .demo_tools import detect_scripted_fix, make_scripted_planner

        fix = detect_scripted_fix(root)
        if fix is None:
            print(
                "fatal: stub planner has no fixture for this workspace "
                "(known: examples/add.py, examples/pkg/). Use --planner openai.",
                file=sys.stderr,
            )
            return 2
        planner = make_scripted_planner(fix, test_code=test)

    if args.mode == "tools":
        report = heal_with_tools(source, test, planner, max_steps=max_steps, workspace=ws)
        return _print_report(report)

    checkpointer = None
    if args.use_langgraph:
        checkpointer = make_checkpointer(args.checkpoint, required=bool(args.checkpoint))

    report = heal_with_graph(
        source,
        test,
        planner,
        max_steps=max_steps,
        workspace=ws,
        approve_mutations=args.approve_mutations,
        mutation_approver=_mutation_approver(args.yes) if args.approve_mutations else None,
        use_langgraph=args.use_langgraph,
        checkpointer=checkpointer,
        thread_id=args.thread_id,
        on_event=on_event,
    )
    return _print_report(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        import logging

        logging.getLogger("self_healing").setLevel(logging.DEBUG)

    demo_flags = [args.demo, args.demo_llm, args.demo_tools, args.demo_graph]
    if sum(bool(x) for x in demo_flags) > 1:
        print("fatal: pick one demo flag", file=sys.stderr)
        return 2

    try:
        if args.demo_llm:
            from .demo_llm import main as run
            run()
            return 0
        if args.demo_tools:
            from .demo_tools import main as run
            run()
            return 0
        if args.demo_graph:
            from .demo_graph import main as run
            run()
            return 0
        if args.demo:
            from .demo import main as run
            run()
            return 0
        if args.src and args.test:
            return run_workspace(args)
    except BrokenPipeError:
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"fatal: {e}", file=sys.stderr)
        return 1

    parser.print_help()
    return 2
