"""Run the naive RAG baseline and print, or save, its numbers.

    python scripts/rag_baseline.py
    python scripts/rag_baseline.py --save eval/rag-baseline.json
    python scripts/rag_baseline.py --compare eval/rag-baseline.json

Offline and free: the generator is the extractive one, so this costs nothing
to run as often as needed. That matters more than it sounds -- a baseline you
hesitate to re-run is a baseline that goes stale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.naive import ExtractiveGenerator
from retrieval.report import run_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Naive RAG baseline")
    parser.add_argument("--k", type=int, default=5, help="chunks retrieved per question")
    parser.add_argument("--size", type=int, default=512, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=64, help="chunk overlap in characters")
    parser.add_argument("--save", default=None, help="write the report as JSON")
    parser.add_argument("--compare", default=None, help="diff against a saved report")
    parser.add_argument("--show-tasks", action="store_true", help="print every task's row")
    options = parser.parse_args(argv)

    from retrieval.corpus_rag import build_index_for

    index = build_index_for(size=options.size, overlap=options.overlap)
    report = run_baseline(index=index, generate=ExtractiveGenerator(), k=options.k)

    print(f"corpus: {len(index)} chunks, k={options.k}, size={options.size}, overlap={options.overlap}")
    print(report.render())

    if options.show_tasks:
        print("\nper task:")
        for row in report.per_task:
            mark = "ok " if row["answer_ok"] else "FAIL"
            missing = f" missing={row['missing']}" if row.get("missing") else ""
            print(f"  {row['id']:12} {mark} rank={row['expected_rank']} cov={row['coverage']:.2f}{missing}")

    if options.save:
        target = Path(options.save)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\nsaved: {target}")

    if options.compare:
        previous = json.loads(Path(options.compare).read_text(encoding="utf-8"))
        print("\nchanges vs", options.compare)
        _diff(previous, report.to_dict())

    return 0


def _diff(previous: dict, current: dict) -> None:
    """Print metric movement, and name the tasks whose verdict changed.

    Naming tasks is the useful part: a hit rate moving from 0.90 to 0.84 says
    something changed, and a line saying *which* task flipped says what.
    """
    fields = ("hit_rate", "mrr", "coverage", "answer_accuracy")
    for field in fields:
        before = previous["overall"].get(field)
        after = current["overall"].get(field)
        if before != after:
            delta = (after or 0) - (before or 0)
            print(f"  {field:16} {before} -> {after}  ({delta:+.4f})")

    was = {row["id"]: row["answer_ok"] for row in previous.get("per_task", [])}
    now = {row["id"]: row["answer_ok"] for row in current.get("per_task", [])}
    for task_id in sorted(set(was) | set(now)):
        if was.get(task_id) != now.get(task_id):
            direction = "fixed" if now.get(task_id) else "BROKE"
            print(f"  {task_id:12} {direction}")


if __name__ == "__main__":
    raise SystemExit(main())
