"""Offline demo of the graph orchestrator (no API key, no LangGraph extra)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from .demo_tools import BROKEN, TEST, scripted_planner
from .graph import heal_with_graph, langgraph_available


def main() -> None:
    print("langgraph extra:", "yes" if langgraph_available() else "no (local interpreter)")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "add.py"
        src.write_text(BROKEN)
        report = heal_with_graph(
            src,
            TEST,
            scripted_planner,
            max_steps=8,
            on_event=lambda node, state: print(
                f"  node={node:<9} passed={state.get('passed')} "
                f"class={state.get('failure_class') or '-'} steps={state.get('steps')}"
            ),
        )
        print("tools used:", [t["tool"] for t in report.tool_trace])
        print("success:", report.success)
        print("final source:\n", report.final_source)


if __name__ == "__main__":
    main()
