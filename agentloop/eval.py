"""The evaluation set: fixed tasks with checkable answers.

Grading is deterministic on purpose. An LLM judge would be a second model
whose behaviour drifts, and a benchmark whose score moves when nothing in
this repository changed is worthless. Each task therefore declares plain
string checks over the final answer, and the categories map to the failure
modes worth separating:

- lookup        one fact, one search should find it
- multi_hop     needs two entries combined
- paraphrase    a query phrased unlike the corpus, to test retrieval
- negative      the corpus has no answer, so the honest reply is a refusal
- robustness    an impossible request, to test error handling instead of retrieval
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

TASKS_PATH = Path(__file__).resolve().parent.parent / "eval" / "tasks.jsonl"


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    question: str
    must_contain: list[str] = field(default_factory=list)
    # A group is satisfied when any one of its alternatives appears.
    must_contain_any: list[list[str]] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    max_turns: int = 8
    notes: str = ""

    @property
    def checks(self) -> int:
        return len(self.must_contain) + len(self.must_contain_any) + len(self.must_not_contain)


@dataclass
class TaskResult:
    task: Task
    answer: str
    turns: int
    tool_calls: int
    failed_calls: int
    retries: int
    tokens: int
    passed: bool
    failures: list[str] = field(default_factory=list)


def load_tasks(path: Path | None = None) -> list[Task]:
    source = path or TASKS_PATH
    tasks: list[Task] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        tasks.append(Task(**json.loads(line)))
    return tasks


def grade(task: Task, answer: str) -> tuple[bool, list[str]]:
    """Check an answer against the task's assertions."""
    lowered = answer.lower()
    failures: list[str] = []

    for needle in task.must_contain:
        if needle.lower() not in lowered:
            failures.append(f"missing {needle!r}")

    for group in task.must_contain_any:
        if not any(option.lower() in lowered for option in group):
            failures.append(f"missing any of {group!r}")

    for needle in task.must_not_contain:
        if needle.lower() in lowered:
            failures.append(f"contains forbidden {needle!r}")

    return (not failures, failures)