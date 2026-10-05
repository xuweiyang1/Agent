"""Run the single-vs-multi experiment and print, or save, the numbers.

    python scripts/rag_agents.py
    python scripts/rag_agents.py --save eval/multi-agent.json
    python scripts/rag_agents.py --capacity 5 --max-size 14

The week's acceptance criterion is a break-even point rather than a
preference, so the output is a table and a computed point: the smallest brief
where the team's coverage overtakes the single agent's. Offline and free, so
it can be re-run whenever the mechanism changes -- a results table you cannot
re-run is a table that goes stale.

The second table is the control for the ROADMAP's P2 rule. Same team, same
briefs, only the handoff policy swapped from summaries to full transcripts, so
the factor between the two columns is what "keep handoffs small" is worth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Single vs multi-agent break-even")
    parser.add_argument("--capacity", type=int, default=3, help="items one drafting pass can state")
    parser.add_argument("--max-iterations", type=int, default=2, help="review/revision rounds")
    parser.add_argument("--max-size", type=int, default=10, help="largest brief to test")
    parser.add_argument("--save", default=None, help="write the report as JSON")
    options = parser.parse_args(argv)

    report = run_experiment(
        sizes=tuple(range(1, options.max_size + 1)),
        capacity=options.capacity,
        max_iterations=options.max_iterations,
    )
    print(report.render())

    if options.save:
        target = Path(options.save)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nsaved: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
