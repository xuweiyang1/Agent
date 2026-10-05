"""Run the W4 planner end to end, offline, and show where it branches.

The demo is written to make the graph's behaviour visible rather than to
print a nice plan. It runs the same request twice with different approvals:

* approved -> the run finishes, with the notes showing the search -> weather
  -> price -> decide -> approve path;
* rejected -> the plan is refused, its destination is excluded, and the loop
  re-enters search with a rewritten query.

Between them, the checkpoint is exercised for real: the second invocation
resumes the state written by the first, which is what ``thread_id`` is for.
No API key, no network -- every tool call is W2's offline stub.

    python scripts/demo_w4.py

The human decision is recorded with ``update_state`` rather than passed as a
resume value, because the gate is a static breakpoint. ``nodes.py`` has the
longer explanation of why the tidier ``interrupt()`` form needs Python 3.11.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.build import build_graph
from graph.checkpoint import thread_config
from graph.plan import render
from graph.state import initial_state

REQUEST = "plan a sunny weekend trip somewhere walkable"


async def run(thread: str, *, approve: bool) -> dict:
    app = build_graph()
    config = thread_config(thread)

    paused = await app.ainvoke(initial_state(REQUEST, nights=2, budget=9000.0), config)
    _show("paused for approval", paused, app=app, config=config)

    # The person answers. Recording it in the checkpoint is the hand-off --
    # the node itself only reads this field when the graph is resumed.
    app.update_state(config, {"approved": approve})
    resumed = await app.ainvoke(None, config)
    _show("approved" if approve else "rejected", resumed, app=app, config=config)
    return resumed


def _show(label: str, state: dict, *, app=None, config=None) -> None:
    print(f"\n===== {label} =====")
    itinerary = state.get("itinerary") or {}
    if itinerary:
        print(render(itinerary))
    print("approved:", state.get("approved"), "done:", state.get("done"))
    if app is not None:
        print("paused before:", app.get_state(config).next)
    print("path:", " -> ".join(a.get("step", "?") for a in state.get("attempts", [])))
    print("notes:")
    for note in state.get("notes", []):
        print("  -", note)


async def main() -> None:
    print("########## run 1: approve the draft ##########")
    await run("demo-approved", approve=True)
    print("\n\n########## run 2: reject it, and watch it re-plan ##########")
    await run("demo-rejected", approve=False)


if __name__ == "__main__":
    asyncio.run(main())
