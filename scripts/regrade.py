"""Re-grade saved reports with the current grader, without calling the model.

    py scripts\regrade.py eval\report-v1.json eval\report-v2.json

A grading bug should not cost money to discover. The reports already contain
every answer, so when the grader changes, the honest move is to re-score the
same answers and report the corrected numbers, rather than to re-run the
benchmark and quietly hide that the earlier verdicts were wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentloop.eval import grade, load_tasks


def regrade(path: Path) -> tuple[int, int, list[tuple[str, str]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    tasks = {t.id: t for t in load_tasks()}
    changes: list[tuple[str, str]] = []
    passed = 0

    for row in report["results"]:
        task = tasks[row["id"]]
        was = row["passed"]
        now, failures = grade(task, row["answer"])
        row["passed"] = now
        row["failures"] = failures
        if now:
            passed += 1
        if was != now:
            changes.append((row["id"], "FIXED" if now else "BROKEN"))

    report["regraded"] = True
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return passed, len(report["results"]), changes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()

    for name in args.reports:
        path = Path(name)
        if not path.exists():
            print(f"error: not found: {path}", file=sys.stderr)
            return 2
        passed, total, changes = regrade(path)
        print(f"{path.name:24} {passed}/{total} passed", end="")
        if changes:
            print("   " + ", ".join(f"{tid} {kind}" for tid, kind in changes))
        else:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())