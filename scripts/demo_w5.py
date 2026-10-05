"""Show the three memory layers working, offline.

The demo is ordered to make one claim per section, because "it has memory" is
not checkable and "this preference survived a reopen" is:

1. **Working memory** -- an intermediate result is kept out of the transcript
   and the budget bounds it.
2. **Session memory** -- turns overflow the window, get folded into a bounded
   summary, and stay findable through recall. This is the answer to "the
   conversation got long".
3. **Long-term memory** -- preferences and decisions are written to a file,
   and a *new object over that file* reads them back. That second object is
   what "across sessions" means; an in-process dict would prove nothing.
4. **The W4 bridge** -- a real planner run hydrates into session memory from
   its checkpoint, rather than being copied into a second history.

    python scripts/demo_w5.py

No API key and no network: the summariser is the offline one, and every
number printed is computed, not asserted.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.build import build_graph
from graph.checkpoint import thread_config
from graph.state import initial_state
from memory import build_memory, session_from_checkpoint, truncating_summarizer


def header(text: str) -> None:
    print(f"\n=== {text} ===")


def working_demo() -> None:
    header("1. working memory: bounded, and the bound is visible")
    _, _, working = build_memory()
    for step in ("search", "weather", "fx", "weather", "search"):
        # Each note is a plausible tool result: long enough that only a few
        # fit in the budget, which is the situation the layer exists for.
        working.note(step, f"{step} result: " + ("detail the model may not need " * 40))
    working.note("final", "1880.87 CNY")
    print(f"  notes made   : 6")
    print(f"  entries kept : {len(working)}")
    print(f"  tokens held  : {working.tokens()} (budget 1200)")
    print(f"  evicted      : {working.evicted}  <- non-zero is the budget working")
    print(f"  latest       : {working.recent()[-1].text}")
    print(f"  sources kept : {sorted({e.source for e in working.recent()})}")


def session_demo() -> None:
    header("2. session memory: window + summary + recall")
    _, session, _ = build_memory(window=4, summarize=truncating_summarizer())
    for i in range(12):
        session.add("user", f"I am weighing option {i} for the weekend")
        session.add("assistant", f"option {i} noted, with its tradeoffs")
    print(f"  turns total   : {len(session)}")
    print(f"  in window     : {[t.index for t in session.recent()]}  (verbatim)")
    summary_tokens = max(1, (len(session.summary) + 3) // 4)
    print(f"  summarised    : {session.summarized_tokens} tokens -> ~{summary_tokens} tokens")
    print(f"  compression   : {session.compression:.2f}  (< 1.0 means it shrank)")
    print(f"  prompt tokens : {session.tokens()}  <- what the next turn actually costs")

    hits = session.recall("option 2 tradeoffs")
    print(f"  recall('option 2 tradeoffs') -> {[h.doc_id for h in hits]}")
    if hits:
        turn = session.turn_for(hits[0].doc_id)
        print(f"    resolved back to turn {turn.index}: {turn.text}")


def longterm_demo(tmp: Path) -> None:
    header("3. long-term memory: survives the process")
    path = tmp / "memory.json"
    longterm, _, _ = build_memory(path=path)

    longterm.remember_preference("seat", "aisle")
    longterm.remember_preference("hotel", "quiet, away from nightlife")
    longterm.remember_decision("Lisbon", "rejected", reason="over budget", session="s1")
    longterm.remember_fact("Enjoyed the Time Out Market food hall in Lisbon.", session="s1")
    print(f"  wrote {len(longterm)} records to {path.name}")

    # A new object over the same file: the process-restart stand-in.
    reopened, _, _ = build_memory(path=path)
    print(f"  reopened: {len(reopened)} records")
    print(f"    preference('seat') -> {reopened.preference('seat')!r}")
    print(f"    preference('seating') -> {reopened.preference('seating', '(no exact match)')!r}")

    print("  preferences are exact, recall is semantic:")
    for hit in reopened.recall("food hall market", mark_used=False):
        print(f"    {hit.doc_id} -> {reopened.semantic.record_of(hit.doc_id).text}")

    print("  what a prompt would carry for a question about food:")
    for line in reopened.context_for("food market").splitlines():
        print(f"    {line}")


async def bridge_demo() -> None:
    header("4. the W4 bridge: session memory IS the checkpoint")
    app = build_graph()
    config = thread_config("demo-w5")
    await app.ainvoke(initial_state("plan a sunny weekend trip", nights=2, budget=9000.0), config)
    app.update_state(config, {"approved": True})
    await app.ainvoke(None, config)

    session = session_from_checkpoint(app, config, window=3, summarize=truncating_summarizer())
    print(f"  planner notes hydrated as turns: {len(session)}")
    print(f"  window: {[t.index for t in session.recent()]}")
    print(f"  summarised from the run's own record: {len(session.summary)} chars")
    hits = session.recall("forecast")
    if hits:
        turn = session.turn_for(hits[0].doc_id)
        print(f"  recall('forecast') -> {turn.text}")


async def main() -> None:
    working_demo()
    session_demo()
    with tempfile.TemporaryDirectory() as tmp:
        longterm_demo(Path(tmp))
    await bridge_demo()


if __name__ == "__main__":
    asyncio.run(main())
