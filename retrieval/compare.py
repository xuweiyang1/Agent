"""Naive vs agentic, measured on the same tasks with the same generator.

The comparison is the deliverable, so the thing this module protects is that
the comparison is *fair*. Three decisions do that work:

- **One task set, one index, one generator.** Both runs go through the same
  objects, so a difference in the numbers is a difference in retrieval
  behaviour. Swapping the generator between runs would let a token change be
  explained by the generator and make the whole report useless.
- **The same metrics, computed the same way.** Hit rate, coverage and MRR are
  W3.5's definitions, reused rather than re-derived, so the agentic row sits
  next to the baseline row without a conversion.
- **Cost is reported per run, not per system.** Retrievals and prompt tokens
  are counted for both, because "better" without "at what price" is half a
  claim.

What the report says, and what it does not
------------------------------------------

The retrieval-side numbers move and the answer-side number does not:

    retrieval   hit 0.90 -> 1.00, coverage 0.95 -> 1.00  (multi_hop 0.83 -> 1.00)
    cost        retrievals 1.00 -> 1.92, tokens 3921 -> 4274
    end to end  answer accuracy 0.92 -> 0.92

The last line is not a null result, it is a *limitation of the measuring
instrument*, and reporting it without saying so would be dishonest. Answer
accuracy is measured with an offline extractive generator that answers from
the single best-matching sentence -- it cannot combine two documents, and the
task W6 fixes needs exactly that. So ``multi-01`` now retrieves both required
entries (coverage 0.50 -> 1.00, the W3.5 report's named defect) and still
fails on the answer, because the stand-in generator reads one sentence.

That is why each row carries a ``bottleneck``: ``retrieval`` when an expected
document never arrived, ``generation`` when the evidence arrived and the
answer still failed, and ``-`` when the task passed. It turns "no improvement"
into "the retrieval half improved and this generator cannot show it", which is
the true statement, and it names the tasks so the claim is checkable.

With a real model generator the answer column is where the win would show.
That is a paid measurement, so it is not the one in this report -- and the
report says so rather than implying the offline one generalises.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from agentloop.eval import grade

from .agentic import MAX_RETRIEVALS, AgenticRetriever, AgenticRun
from .corpus_rag import RagTask, RetrievalResult, build_index_for, load_tasks, retrieve
from .index import BM25Index
from .naive import ExtractiveGenerator
from .tools import RetrieverTool


@dataclass
class RunRow:
    """One task's result under one strategy."""

    id: str
    category: str
    answer_ok: bool
    hit: bool
    coverage: float
    expected_rank: int | None
    retrievals: int
    prompt_tokens: int
    sources: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def bottleneck(self) -> str:
        """Which stage failed, when one did.

        The distinction the whole report turns on. ``retrieval`` means the
        evidence never arrived; ``generation`` means it did and the answer
        still did not use it. Collapsing the two into "failed" would make a
        retrieval fix invisible whenever the generator is the bottleneck.
        """
        if self.answer_ok:
            return "-"
        # A negative task has no expected document, so a failed answer there is
        # an abstention problem, not a retrieval one.
        if self.category == "negative":
            return "abstention"
        return "generation" if self.hit else "retrieval"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "category": self.category,
            "answer_ok": self.answer_ok,
            "hit": self.hit,
            "coverage": round(self.coverage, 4),
            "expected_rank": self.expected_rank,
            "retrievals": self.retrievals,
            "prompt_tokens": self.prompt_tokens,
            "sources": self.sources,
            "missing": self.missing,
            "bottleneck": self.bottleneck,
        }


@dataclass
class ComparisonReport:
    """Both strategies, plus the per-category and per-task detail."""

    naive: list[RunRow]
    agentic: list[RunRow]
    max_retrievals: int = MAX_RETRIEVALS

    # -- aggregate helpers -------------------------------------------------

    def _overall(self, rows: Sequence[RunRow]) -> dict:
        total = len(rows)
        return {
            "tasks": total,
            "answer_accuracy": round(sum(1 for r in rows if r.answer_ok) / total, 4) if total else 0.0,
            "retrievals": round(sum(r.retrievals for r in rows) / total, 3) if total else 0.0,
            "prompt_tokens": sum(r.prompt_tokens for r in rows),
        }

    def _retrieval(self, rows: Sequence[RunRow]) -> dict:
        scored = [r for r in rows if r.category != "negative"]
        if not scored:
            return {"hit_rate": 0.0, "coverage": 0.0, "mrr": 0.0}
        return {
            "hit_rate": round(sum(1 for r in scored if r.hit) / len(scored), 4),
            "coverage": round(sum(r.coverage for r in scored) / len(scored), 4),
            "mrr": round(
                sum((1.0 / r.expected_rank) if r.expected_rank else 0.0 for r in scored) / len(scored),
                4,
            ),
        }

    def by_category(self, side: str) -> dict:
        rows = self.agentic if side == "agentic" else self.naive
        groups: dict[str, list[RunRow]] = {}
        for row in rows:
            groups.setdefault(row.category, []).append(row)
        return {
            name: {**self._overall(group), **self._retrieval(group)}
            for name, group in sorted(groups.items())
        }

    def retrieval_fixed(self) -> list[dict]:
        """Tasks whose *retrieval* verdict changed, named.

        A coverage average moving from 0.95 to 1.00 says something happened;
        naming the task says what. This is the list for the commit message.
        """
        before = {r.id: r for r in self.naive}
        changed: list[dict] = []
        for after in self.agentic:
            was = before.get(after.id)
            if was is None:
                continue
            if was.hit != after.hit or abs(was.coverage - after.coverage) > 1e-9:
                changed.append(
                    {
                        "id": after.id,
                        "category": after.category,
                        "coverage": f"{was.coverage:.2f} -> {after.coverage:.2f}",
                        "missing_before": was.missing,
                        "retrievals": f"{was.retrievals} -> {after.retrievals}",
                    }
                )
        return changed

    def generation_limited(self) -> list[RunRow]:
        """Rows where the evidence arrived and the offline generator still failed."""
        return [r for r in self.agentic if r.bottleneck == "generation"]

    def to_dict(self) -> dict:
        return {
            "tasks": len(self.naive),
            "max_retrievals": self.max_retrievals,
            "overall": {
                "naive": self._overall(self.naive),
                "agentic": self._overall(self.agentic),
            },
            "retrieval": {
                "naive": self._retrieval(self.naive),
                "agentic": self._retrieval(self.agentic),
            },
            "per_category": {
                "naive": self.by_category("naive"),
                "agentic": self.by_category("agentic"),
            },
            "retrieval_fixed": self.retrieval_fixed(),
            "generation_limited": [
                {"id": r.id, "category": r.category, "coverage": r.coverage} for r in self.generation_limited()
            ],
            "per_task": {
                "naive": [r.to_dict() for r in self.naive],
                "agentic": [r.to_dict() for r in self.agentic],
            },
        }

    def render(self) -> str:
        lines: list[str] = []
        header = (
            f"{'strategy':10} {'tasks':>5} {'hit':>6} {'cov':>6} {'mrr':>6} "
            f"{'answer':>7} {'retr':>6} {'tokens':>7}"
        )
        lines.append(f"{'':10} {'':>5} {'--- retrieval ---':>18} {'--- end to end ---':>7} {'--- cost ---':>13}")
        lines.append(header)
        lines.append("-" * len(header))
        for label, rows in (("naive", self.naive), ("agentic", self.agentic)):
            overall = self._overall(rows)
            retrieval = self._retrieval(rows)
            lines.append(
                f"{label:10} {overall['tasks']:>5} {retrieval['hit_rate']:>6.2f} {retrieval['coverage']:>6.2f} "
                f"{retrieval['mrr']:>6.2f} {overall['answer_accuracy']:>7.2f} "
                f"{overall['retrievals']:>6.2f} {overall['prompt_tokens']:>7}"
            )

        lines.append("")
        lines.append("per category (naive -> agentic)")
        lines.append(f"{'category':12} {'hit':>13} {'coverage':>13} {'answer':>13} {'retrievals':>13}")
        naive_cat = self.by_category("naive")
        agentic_cat = self.by_category("agentic")
        for name in sorted(set(naive_cat) | set(agentic_cat)):
            a = naive_cat.get(name, {})
            b = agentic_cat.get(name, {})
            lines.append(
                f"{name:12} "
                f"{a.get('hit_rate', 0):>5.2f} -> {b.get('hit_rate', 0):<5.2f} "
                f"{a.get('coverage', 0):>5.2f} -> {b.get('coverage', 0):<5.2f} "
                f"{a.get('answer_accuracy', 0):>5.2f} -> {b.get('answer_accuracy', 0):<5.2f} "
                f"{a.get('retrievals', 0):>5.2f} -> {b.get('retrievals', 0):<5.2f}"
            )

        fixed = self.retrieval_fixed()
        lines.append("")
        if fixed:
            lines.append("retrieval fixed:")
            for item in fixed:
                missing = f" (was missing {item['missing_before']})" if item["missing_before"] else ""
                lines.append(f"  {item['id']:12} coverage {item['coverage']}, retrievals {item['retrievals']}{missing}")
        else:
            lines.append("retrieval fixed: none")

        limited = self.generation_limited()
        if limited:
            lines.append("")
            lines.append("still failing, but the evidence is now present (offline generator limit):")
            for row in limited:
                lines.append(f"  {row.id:12} coverage {row.coverage:.2f}, bottleneck=generation")
        return "\n".join(lines)


def _naive_rows(
    tasks: Sequence[RagTask],
    index: BM25Index,
    generate: Callable[[str], str],
    *,
    k: int,
) -> list[RunRow]:
    """The baseline, with the same prompt shape the agentic run uses.

    Deliberately not reusing ``run_baseline``: that function builds its own
    prompt through ``naive_rag`` and does not expose the rendered context, so
    comparing against it would mean comparing two prompt constructions rather
    than two retrieval strategies. ``retrieve`` -- the shared retrieval call --
    is reused, which is the part that has to be identical.
    """
    rows: list[RunRow] = []
    for task in tasks:
        hits = retrieve(index, task.question, k=k)
        result = RetrievalResult(
            task_id=task.id,
            category=task.category,
            hits=hits,
            expected_docs=task.expected_docs,
        )
        context = "\n\n".join(f"[{i}] {h.doc_id}\n{h.chunk.text}" for i, h in enumerate(hits, start=1))
        prompt = _render(task, context)
        answer = generate(prompt)
        passed, _ = grade(task, answer)
        rows.append(
            RunRow(
                id=task.id,
                category=task.category,
                answer_ok=passed,
                hit=result.hit,
                coverage=result.coverage,
                expected_rank=result.expected_rank,
                retrievals=1,
                prompt_tokens=max(1, (len(prompt) + 3) // 4),
                sources=result.sources()[:5],
                missing=result.missing,
            )
        )
    return rows


def _agentic_rows(
    tasks: Sequence[RagTask],
    retriever: RetrieverTool,
    generate: Callable[[str], str],
    *,
    k: int,
    max_retrievals: int,
) -> list[RunRow]:
    agent = AgenticRetriever(retriever, max_retrievals=max_retrievals, k=k)
    rows: list[RunRow] = []
    for task in tasks:
        run: AgenticRun = agent.run(task, generate)
        passed, _ = grade(task, run.answer)
        sources = run.sources
        expected = list(task.expected_docs)
        found = [doc for doc in expected if doc in sources]
        rank = None
        for position, doc in enumerate(sources, start=1):
            if doc in expected:
                rank = position
                break
        rows.append(
            RunRow(
                id=task.id,
                category=task.category,
                answer_ok=passed,
                hit=bool(expected) and len(found) == len(expected),
                coverage=(len(found) / len(expected)) if expected else 0.0,
                expected_rank=rank,
                retrievals=run.retrievals,
                prompt_tokens=run.prompt_tokens,
                sources=sources[:5],
                missing=[doc for doc in expected if doc not in sources],
            )
        )
    return rows


def _render(task: RagTask, context: str) -> str:
    """The same prompt template for both strategies.

    Identical on purpose. If the two runs rendered their context differently --
    different headers, a different order -- then part of any measured
    difference would be the template, and the comparison would be of two
    things at once.
    """
    if not context.strip():
        return f"Question: {task.question}\nAnswer:"
    return (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say so plainly.\n\n"
        "Context:\n" + context + f"\n\nQuestion: {task.question}\nAnswer:"
    )


def compare(
    *,
    tasks: Sequence[RagTask] | None = None,
    index: BM25Index | None = None,
    generate: Callable[[str], str] | None = None,
    k: int = 5,
    max_retrievals: int = MAX_RETRIEVALS,
) -> ComparisonReport:
    """Run both strategies and return the comparison."""
    task_list = list(tasks if tasks is not None else load_tasks())
    idx = index if index is not None else build_index_for()
    generator = generate if generate is not None else ExtractiveGenerator()

    return ComparisonReport(
        naive=_naive_rows(task_list, idx, generator, k=k),
        agentic=_agentic_rows(task_list, RetrieverTool(idx), generator, k=k, max_retrievals=max_retrievals),
        max_retrievals=max_retrievals,
    )


__all__ = ["ComparisonReport", "RunRow", "compare"]
