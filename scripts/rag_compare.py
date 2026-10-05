"""Run the naive-vs-agentic comparison and print, or save, the report.

    python scripts/rag_compare.py
    python scripts/rag_compare.py --save eval/rag-agentic.json
    python scripts/rag_compare.py --show-tasks

Offline and free, for the same reason the baseline is: a comparison you
hesitate to re-run is a comparison that goes stale. The generator is the
extractive one, so the only thing being measured is retrieval behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from retrieval.compare import compare
from retrieval.corpus_rag import build_index_for


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Naive vs agentic retrieval")
    parser.add_argument("--k", type=int, default=5, help="chunks per retrieval")
    parser.add_argument("--size", type=int, default=512, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=64, help="chunk overlap in characters")
    parser.add_argument("--max-retrievals", type=int, default=3, help="cap on retrievals per task")
    parser.add_argument("--save", default=None, help="write the report as JSON")
    parser.add_argument("--show-tasks", action="store_true", help="print every task's row")
    options = parser.parse_args(argv)

    index = build_index_for(size=options.size, overlap=options.overlap)
    report = compare(index=index, k=options.k, max_retrievals=options.max_retrievals)

    print(f"corpus: {len(index)} chunks, k={options.k}, size={options.size}, overlap={options.overlap}")
    print(report.render())

    if options.show_tasks:
        print("\nper task:")
        naive = {r.id: r for r in report.naive}
        for row in report.agentic:
            was = naive[row.id]
            print(
                f"  {row.id:12} hit {str(was.hit):5} -> {str(row.hit):5}"
                f"  coverage {was.coverage:4.2f} -> {row.coverage:4.2f}"
                f"  answer {str(was.answer_ok):5} -> {str(row.answer_ok):5}"
                f"  retrievals {was.retrievals} -> {row.retrievals}"
                f"  bottleneck={row.bottleneck}"
            )

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
