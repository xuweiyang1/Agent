"""Run the whole chain once, and print the evidence at every joint.

    python scripts/demo_chain.py
    python scripts/demo_chain.py --save eval/chain-run.json

The demo is deliberately the same shape as ``ChainRun.render``: nine steps, in
execution order, each printing what it received and what it produced. That is
what makes this different from seven separate demos -- a reader can follow one
value from the board's pixels to a calendar event, and see where each step got
its input.

Three things are printed that a "did it work" demo would leave out:

- **The request text, next to what was perceived.** The request names no city,
  no budget and no date. Reading the two side by side is the cheapest possible
  check that the multimodal claim is real rather than decorative.
- **What memory changed.** The run seeds a rejection this user recorded, and
  the demo shows the avoid list it produced -- memory that is never read is a
  diary, not a memory.
- **The divergence guard, on demand.** ``--board-city`` renames the board's
  destination, which is how the stop-before-acting path is demonstrated
  without editing any code.

No API key, no network. The only paid-looking step is perception, and it is
priced from the image's actual pixel count.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import BOARD_LINES, BOARD_MEANINGS, run_chain, save_report


def header(text: str) -> None:
    print(f"\n=== {text} ===")


def parse(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {value!r}") from exc


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The end-to-end assistant chain")
    parser.add_argument("--today", type=parse, default=date(2026, 10, 5), help="YYYY-MM-DD; makes the run reproducible")
    parser.add_argument("--board-city", default=None, help="override the city written on the board")
    parser.add_argument("--scale", type=int, default=1, help="draw the board this much larger")
    parser.add_argument("--save", default=None, help="write the run as JSON")
    options = parser.parse_args(argv)

    lines = list(BOARD_LINES)
    meanings = dict(BOARD_MEANINGS)
    if options.board_city:
        # Rewriting the label changes what is drawn; the reader has to find the
        # new city in the pixels like any other value.
        old = next(label for label in lines if label.startswith("Destination"))
        new = f"Destination - {options.board_city}"
        lines = [new if line == old else line for line in lines]
        meanings = {new if label == old else label: value for label, value in meanings.items()}
        meanings[new] = options.board_city

    header("the request, and what the board actually says")
    print(f"  request (intent only): {'plan a weekend trip'}")
    for line in lines:
        print(f"  board row             : {line}")

    run = await run_chain(
        today=options.today,
        board_lines=lines,
        meanings=meanings,
        scale=options.scale,
    )

    header("the chain, step by step")
    print(run.render())

    header("verdict")
    print(f"  all steps ok    : {run.ok}")
    print(f"  perceived       : {run.perceived}")
    print(f"  answer          : {run.answer}")
    print(f"  artifacts       : {', '.join(sorted(run.artifacts)) or '(none)'}")
    print(f"  todo items      : {len(run.todos)}")
    if run.event:
        print(f"  first booked    : {run.event.get('start')} -> {run.event.get('end')}")
    if run.divergence:
        print(f"  divergence      : {run.divergence}")

    if options.save:
        target = save_report(run, options.save)
        print(f"\nsaved: {target}")
    return 0 if run.ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
