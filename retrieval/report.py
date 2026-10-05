"""Baseline numbers: hit rate, MRR, and answer accuracy per category.

The shape of this report is the deliverable. It reports per category, not as
one number, because a single aggregate hides the thing worth knowing: the
naive pipeline is expected to hold up on ``lookup`` and ``paraphrase`` and to
break on ``multi_hop`` and ``negative``. Averaging those into "72% passing"
would bury the finding that motivates W6.

Two metrics, because they answer different questions:

- **hit rate** -- did the expected document appear at all? A retriever that
  finds the right entry at rank 5 is still usable.
- **MRR** -- how high did it rank? A retriever that always finds it at rank 5
  has a perfect hit rate and a terrible MRR, and the mean rank is what a
  context budget actually pays for.

Answer accuracy is reported separately from retrieval, because a failure can
come from either stage and conflating them makes both undiagnosable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .corpus_rag import (
    EXPECTED_WEAK,
    RagTask,
    RetrievalResult,
    build_index_for,
    load_tasks,
    retrieve,
)
from .naive import ExtractiveGenerator, NaiveAnswer, estimate_tokens, naive_rag


@dataclass
class CategoryScore:
    """One category's retrieval and answer numbers."""

    category: str
    tasks: int = 0
    scored: int = 0  # tasks that have an expected document
    hits: int = 0
    rr_sum: float = 0.0
    coverage_sum: float = 0.0
    answers_passed: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.scored if self.scored else 0.0

    @property
    def mrr(self) -> float:
        return self.rr_sum / self.scored if self.scored else 0.0

    @property
    def coverage(self) -> float:
        """Mean fraction of required documents actually retrieved.

        Reported next to hit rate because for a multi-hop task the difference
        between 0.5 and 1.0 is the difference between half the evidence and an
        answer -- and a binary hit rate collapses that to a single miss.
        """
        return self.coverage_sum / self.scored if self.scored else 0.0

    @property
    def answer_accuracy(self) -> float:
        return self.answers_passed / self.tasks if self.tasks else 0.0

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "tasks": self.tasks,
            "scored": self.scored,
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "coverage": round(self.coverage, 4),
            "answer_accuracy": round(self.answer_accuracy, 4),
        }


@dataclass
class BaselineReport:
    """The whole run, kept as data so a later change can be compared to it."""

    per_category: dict[str, CategoryScore] = field(default_factory=dict)
    per_task: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    generation_calls: int = 0

    @property
    def hit_rate(self) -> float:
        scored = sum(s.scored for s in self.per_category.values())
        hits = sum(s.hits for s in self.per_category.values())
        return hits / scored if scored else 0.0

    @property
    def mrr(self) -> float:
        scored = sum(s.scored for s in self.per_category.values())
        return sum(s.rr_sum for s in self.per_category.values()) / scored if scored else 0.0

    @property
    def answer_accuracy(self) -> float:
        total = sum(s.tasks for s in self.per_category.values())
        passed = sum(s.answers_passed for s in self.per_category.values())
        return passed / total if total else 0.0

    @property
    def coverage(self) -> float:
        scored = sum(s.scored for s in self.per_category.values())
        if not scored:
            return 0.0
        return sum(s.coverage_sum for s in self.per_category.values()) / scored

    def to_dict(self) -> dict:
        return {
            "overall": {
                "hit_rate": round(self.hit_rate, 4),
                "coverage": round(self.coverage, 4),
                "mrr": round(self.mrr, 4),
                "answer_accuracy": round(self.answer_accuracy, 4),
                "prompt_tokens": self.prompt_tokens,
                "generation_calls": self.generation_calls,
            },
            "per_category": [s.to_dict() for s in self.per_category.values()],
            "per_task": self.per_task,
        }

    def render(self) -> str:
        """A plain-text table, because that is what goes in a commit message."""
        lines = [
            f"{'category':12} {'tasks':>5} {'hit':>6} {'cov':>6} {'mrr':>6} {'answer':>7}",
            "-" * 50,
        ]
        for name in sorted(self.per_category):
            score = self.per_category[name]
            lines.append(
                f"{name:12} {score.tasks:>5} {score.hit_rate:>6.2f} "
                f"{score.coverage:>6.2f} {score.mrr:>6.2f} {score.answer_accuracy:>7.2f}"
            )
        lines.append("-" * 50)
        lines.append(
            f"{'overall':12} {sum(s.tasks for s in self.per_category.values()):>5} "
            f"{self.hit_rate:>6.2f} {self.coverage:>6.2f} {self.mrr:>6.2f} "
            f"{self.answer_accuracy:>7.2f}"
        )
        lines.append(f"\nprompt tokens: {self.prompt_tokens}  generation calls: {self.generation_calls}")
        return "\n".join(lines)


def run_baseline(
    tasks: Sequence[RagTask] | None = None,
    *,
    index=None,
    generate: Callable[[str], str] | None = None,
    k: int = 5,
    grade: Callable[[RagTask, str], tuple[bool, list[str]]] | None = None,
) -> BaselineReport:
    """Run the naive pipeline over the task set and collect the numbers.

    The grader is injected rather than imported so this module does not depend
    on W1's eval internals. The default is W1's grader, which is the point:
    both projects grade with the same function, so a difference between two
    reports is a difference in the system, not in the measuring stick.
    """
    task_list = list(tasks if tasks is not None else load_tasks())
    idx = index if index is not None else build_index_for()
    generator = generate if generate is not None else ExtractiveGenerator()
    grader = grade if grade is not None else _default_grader()

    report = BaselineReport()
    for task in task_list:
        hits = retrieve(idx, task.question, k=k)
        result = RetrievalResult(
            task_id=task.id,
            category=task.category,
            hits=hits,
            expected_docs=task.expected_docs,
        )
        answer = naive_rag(task, idx, generator, k=k)

        passed, failures = _grade(grader, task, answer.answer)

        score = report.per_category.setdefault(task.category, CategoryScore(task.category))
        score.tasks += 1
        if task.expected_docs:
            score.scored += 1
            score.rr_sum += result.reciprocal_rank
            score.coverage_sum += result.coverage
            if result.hit:
                score.hits += 1
        if passed:
            score.answers_passed += 1

        report.prompt_tokens += answer.prompt_tokens
        report.generation_calls += answer.calls
        report.per_task.append(
            {
                "id": task.id,
                "category": task.category,
                "expected_docs": list(task.expected_docs),
                "expected_rank": result.expected_rank,
                "coverage": round(result.coverage, 4),
                "missing": result.missing,
                "hit": result.hit,
                "sources": result.sources()[:3],
                "answer_ok": passed,
                "failures": failures,
                "prompt_tokens": answer.prompt_tokens,
            }
        )
    return report


def _grade(grader, task: RagTask, answer: str) -> tuple[bool, list[str]]:
    """Call the grader with whichever task shape it expects.

    W1's grader reads the fields by name, and ``RagTask`` has the same names,
    so the same function accepts both. This exists so a mismatch surfaces as a
    clear error rather than a silent zero.
    """
    try:
        return grader(task, answer)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise TypeError(f"grader does not accept the task shape: {exc}") from exc


def _default_grader():
    from agentloop.eval import grade

    return grade


def weak_categories(report: BaselineReport) -> list[str]:
    """Categories that scored badly, so the baseline can assert its own gaps.

    Used by the tests to pin the intended outcome: the naive pipeline should
    be weak where it is structurally weak, and a baseline that accidentally
    passes everything is a broken benchmark, not a good system.
    """
    return [
        name
        for name, score in report.per_category.items()
        if score.answer_accuracy < 0.5
    ]


__all__ = [
    "BaselineReport",
    "CategoryScore",
    "run_baseline",
    "weak_categories",
]
