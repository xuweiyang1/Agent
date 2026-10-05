"""Fail when the documentation has drifted from the code.

The rule "update the docs when you change the code" is a good rule and an
unenforceable one. This repository already produced the evidence: one
machine's agent concluded W3 had not been done, because nothing in the repo
said what was finished. A rule nobody checks is a rule that decays.

So the checkable claims are checked, and nothing else:

1. **The current test total.** README, AGENTS.md and ROADMAP each state a
   number a reader will rely on. ROADMAP also keeps one total per week as a
   historical record, so only its newest entry is held to the current count;
   the older ones only have to stay below it.
2. **That documented entry points exist.** A README naming
   ``scripts/demo_w2.py`` is making a promise; a rename breaks it into a
   confusing FileNotFoundError.
3. **That the two progress tables agree.** AGENTS.md and ROADMAP each carry
   one, and two that disagree are worse than either alone, because the reader
   cannot tell which is stale.

Historical numbers ("68 tests added, 182 total" under W3) are deliberately not
checked. They were true when written and are meant to stay put; rewriting them
would destroy the record.

**This script must not run the test suite.** An earlier version did, and
``tests/test_docs.py`` runs this script -- so the suite invoked itself, one
process per nesting level, until the machine was covered in pythons. The test
count is therefore counted from the source, not observed from a run. The suite
already fails on its own if a test is broken; this check only needs to know
how many there are.

    python scripts/check_docs.py
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (file, regex with one capture group holding the *current* total)
CURRENT_TOTAL_CLAIMS: tuple[tuple[str, str], ...] = (
    ("README.md", r"should print `OK` with (\d+) tests"),
    ("AGENTS.md", r"应该 (\d+) 项全过"),
    ("docs/ROADMAP.md", r"全量 (\d+) 项离线通过"),
)

# Paths the README promises, and that a newcomer is told to use.
ENTRY_POINTS: tuple[str, ...] = (
    "demo.py",
    "scripts/demo_w2.py",
    "scripts/demo_w3.py",
    "scripts/rag_baseline.py",
    "scripts/check_docs.py",
    "requirements.txt",
    "AGENTS.md",
    "docs/ROADMAP.md",
    "docs/WORKFLOW.md",
)

# Files whose pattern legitimately matches several totals, newest last.
LATEST_TOTAL_CLAIMS: frozenset[str] = frozenset({"docs/ROADMAP.md"})

WEEK_PATTERN = re.compile(r"\bW(\d+(?:\.\d+)?)\b")


class Drift(Exception):
    """A documentation claim that no longer matches the repository."""


def count_tests() -> int:
    """Count test methods by parsing the test files.

    Parsed rather than executed: a file that fails to import still contains
    the tests it declares, and this number is compared against documentation
    that describes intent. Executing the suite here would also be the
    recursion described in the module docstring.
    """
    total = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("test_"):
                        total += 1
    if total == 0:
        raise Drift("found no test methods; the parser or the layout changed")
    return total


def check_total_claims(actual: int) -> list[str]:
    problems: list[str] = []
    for relative, pattern in CURRENT_TOTAL_CLAIMS:
        text = (ROOT / relative).read_text(encoding="utf-8")
        found = [int(n) for n in re.findall(pattern, text)]
        if not found:
            problems.append(
                f"{relative}: no current test total matched {pattern!r}; "
                "if the wording changed, update CURRENT_TOTAL_CLAIMS"
            )
            continue
        # ROADMAP records one total per week, so only its newest entry is the
        # current number; the rest are history and merely have to increase.
        if relative in LATEST_TOTAL_CLAIMS:
            stale = [n for n in found[:-1] if n >= found[-1]]
            if stale:
                problems.append(
                    f"{relative}: historical totals {stale} are not below the latest {found[-1]}"
                )
            if found[-1] != actual:
                problems.append(f"{relative}: latest total is {found[-1]}, actual is {actual}")
            continue
        wrong = sorted({n for n in found if n != actual})
        if wrong:
            problems.append(f"{relative}: claims {wrong} tests, actual is {actual}")
    return problems


def check_entry_points() -> list[str]:
    return [
        f"{relative}: documented but missing"
        for relative in ENTRY_POINTS
        if not (ROOT / relative).exists()
    ]


def check_progress_tables() -> list[str]:
    """The two progress tables must name the same finished weeks."""
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs" / "ROADMAP.md").read_text(encoding="utf-8")

    agents_done = {
        f"W{match}"
        for line in agents.splitlines()
        if "✅" in line
        for match in WEEK_PATTERN.findall(line)
    }
    roadmap_done = {
        f"W{match}"
        for line in roadmap.splitlines()
        if "已完成" in line
        for match in WEEK_PATTERN.findall(line)
    }

    problems: list[str] = []
    if only_agents := agents_done - roadmap_done:
        problems.append(f"AGENTS.md marks {sorted(only_agents)} done; ROADMAP does not")
    if only_roadmap := roadmap_done - agents_done:
        problems.append(f"ROADMAP marks {sorted(only_roadmap)} done; AGENTS.md does not")
    if not agents_done and not roadmap_done:
        problems.append("no finished week found in either table; the check found nothing")
    return problems


def main() -> int:
    try:
        actual = count_tests()
    except Drift as exc:
        print(f"FAIL  {exc}")
        return 1

    problems = check_total_claims(actual) + check_entry_points() + check_progress_tables()
    if problems:
        print(f"FAIL  docs have drifted from the code ({len(problems)} problem(s)):")
        for problem in problems:
            print(f"  - {problem}")
        print("\nUpdate README.md / AGENTS.md / docs/ROADMAP.md, then re-run.")
        return 1

    print(
        f"OK    docs match the code: {actual} tests, "
        f"{len(ENTRY_POINTS)} entry points, progress tables agree"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
