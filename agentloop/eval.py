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
import re
import unicodedata
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


_EMPHASIS = re.compile(r"[*_`]+")
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Fold formatting away before matching, keeping the words intact.

    Markdown emphasis is the reason this exists. A real answer said
    "as *data*, never as instructions", which is correct, and the grader
    rejected it because it was looking for the literal string "as data".
    Emphasis characters change how text looks, never what it means, so they
    are removed here; a false negative from formatting would otherwise make
    the whole benchmark untrustworthy, and the temptation would be to loosen
    the assertions instead, which hides real failures.
    """
    folded = unicodedata.normalize("NFKC", text)
    # Normalise the dash characters a model mixes freely, so a check written
    # with "-" matches an answer written with an en dash or an em dash.
    folded = folded.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    folded = folded.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    folded = folded.replace("\u2018", "'").replace("\u2019", "'")
    folded = folded.replace("\u201c", '"').replace("\u201d", '"')
    folded = _EMPHASIS.sub(" ", folded)
    # Removing emphasis can leave a space before punctuation ("data , never"),
    # which would defeat a check written against the rendered text.
    folded = re.sub(r"\s+([,.;:!?])", r"\1", folded)
    return _WHITESPACE.sub(" ", folded).strip().lower()


def grade(task: Task, answer: str) -> tuple[bool, list[str]]:
    """Check an answer against the task's assertions."""
    lowered = normalize(answer)
    failures: list[str] = []

    for needle in task.must_contain:
        if normalize(needle) not in lowered:
            failures.append(f"missing {needle!r}")

    for group in task.must_contain_any:
        if not any(normalize(option) in lowered for option in group):
            failures.append(f"missing any of {group!r}")

    for needle in task.must_not_contain:
        if normalize(needle) in lowered:
            failures.append(f"contains forbidden {needle!r}")

    return (not failures, failures)
