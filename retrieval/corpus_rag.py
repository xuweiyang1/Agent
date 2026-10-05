"""The shared corpus and task set that retrieval is measured against.

Identical inputs to W1, on purpose. The W3.5 baseline and the W6 agentic
retriever must be scored on the same questions over the same documents, or
the comparison between them means nothing. So this module imports W1's
``CORPUS`` rather than copying it: a copy would drift, and a drifted corpus
silently invalidates the very comparison the module exists for.

What is added here is ``EXPECTED_DOCS`` -- for each task, which entry a good
retriever should surface. That mapping is what turns "did it work" into a
number: hit rate needs to know which document counted, and MRR needs to know
its rank. W1 did not need it because its model chose freely between search and
answer; retrieval is scored on the ranking itself.

The tasks come from the same ``eval/tasks.jsonl``. Nothing here is a second
benchmark; it is a second *view* of one benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from agentloop.corpus import CORPUS

from .chunking import Chunk, chunk_corpus
from .index import BM25Index, ScoredChunk, build_index

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
TASKS_PATH = EVAL_DIR / "tasks.jsonl"

# Which entries answer each task. A tuple, because a multi-hop task needs
# more than one: scoring "why does a loop need compaction" as a hit when only
# the loop entry was found would hide exactly the failure the category exists
# to expose. Negative tasks are absent deliberately -- there is no right
# document, and the honest outcome is "nothing relevant", which must not be
# scored as a miss.
EXPECTED_DOCS: dict[str, tuple[str, ...]] = {
    "lookup-01": ("tool-registry",),
    "lookup-02": ("tokenization",),
    "lookup-03": ("prompt-injection",),
    "lookup-04": ("evaluation",),
    # Needs the loop entry *and* the window entry: the question is why the
    # loop needs compaction, which only the combination answers.
    "multi-01": ("agent-loop", "context-window"),
    "multi-02": ("latency-budget",),
    "multi-03": ("backoff",),
    "para-01": ("embedding-retrieval",),
    "para-02": ("observability",),
    "para-03": ("compaction",),
}

# Categories where the naive pipeline is *expected* to fail. Naming them here
# means the baseline report can assert the failure instead of treating it as a
# surprise, which is the difference between a baseline and a bug list.
EXPECTED_WEAK = ("multi_hop", "negative")


@dataclass(frozen=True)
class RagTask:
    """One question plus what an answer must contain.

    Field names match ``agentloop.eval.Task`` so the same grader can be
    reused unchanged; the dataclass is separate only so retrieval can be
    developed before generation exists.
    """

    id: str
    category: str
    question: str
    must_contain: tuple[str, ...] = ()
    must_contain_any: tuple[tuple[str, ...], ...] = ()
    must_not_contain: tuple[str, ...] = ()
    notes: str = ""

    @property
    def expected_docs(self) -> tuple[str, ...]:
        return EXPECTED_DOCS.get(self.id, ())

    @property
    def needs_multiple(self) -> bool:
        return len(self.expected_docs) > 1


def load_tasks(path: Path | None = None) -> list[RagTask]:
    """Read the shared task file into retrieval's own shape."""
    source = path or TASKS_PATH
    tasks: list[RagTask] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        raw = json.loads(line)
        tasks.append(
            RagTask(
                id=raw["id"],
                category=raw["category"],
                question=raw["question"],
                must_contain=tuple(raw.get("must_contain", ())),
                must_contain_any=tuple(tuple(g) for g in raw.get("must_contain_any", ())),
                must_not_contain=tuple(raw.get("must_not_contain", ())),
                notes=raw.get("notes", ""),
            )
        )
    return tasks


def documents() -> dict[str, str]:
    """The corpus being indexed, as a plain mapping."""
    return dict(CORPUS)


def build_index_for(
    docs: dict[str, str] | None = None,
    *,
    size: int = 512,
    overlap: int = 64,
) -> BM25Index:
    """Chunk a corpus and index it, with the parameters kept in one place."""
    return build_index(chunk_corpus(docs or documents(), size=size, overlap=overlap))


@dataclass
class RetrievalResult:
    """What retrieval did for one task, before any answer is written."""

    task_id: str
    category: str
    hits: list[ScoredChunk]
    expected_docs: tuple[str, ...] = ()

    def _rank_of(self, doc_id: str) -> int | None:
        for position, scored in enumerate(self.hits, start=1):
            if scored.doc_id == doc_id:
                return position
        return None

    @property
    def found(self) -> list[str]:
        return [doc for doc in self.expected_docs if self._rank_of(doc) is not None]

    @property
    def missing(self) -> list[str]:
        return [doc for doc in self.expected_docs if self._rank_of(doc) is None]

    @property
    def expected_rank(self) -> int | None:
        """Rank of the first required document, or None if none appeared."""
        for doc in self.expected_docs:
            rank = self._rank_of(doc)
            if rank is not None:
                return rank
        return None

    @property
    def hit(self) -> bool:
        """All required documents present.

        Partial credit would let a multi-hop task count as solved when half
        its evidence was found, and half the evidence is not an answer.
        """
        return bool(self.expected_docs) and not self.missing

    @property
    def coverage(self) -> float:
        """Fraction of required documents retrieved, for partial visibility."""
        if not self.expected_docs:
            return 0.0
        return len(self.found) / len(self.expected_docs)

    @property
    def reciprocal_rank(self) -> float:
        rank = self.expected_rank
        return 0.0 if rank is None else 1.0 / rank

    def sources(self) -> list[str]:
        """Documents that produced the hits, in order, without repeats."""
        seen: list[str] = []
        for scored in self.hits:
            if scored.doc_id not in seen:
                seen.append(scored.doc_id)
        return seen


def retrieve(
    index: BM25Index,
    question: str,
    *,
    k: int = 5,
) -> list[ScoredChunk]:
    """The naive retrieval step: one query, top-k, done.

    No rewriting, no second pass, no judgement about whether the result was
    good enough. That absence is what W3.5 measures and what W6 adds.
    """
    return index.search(question, k=k)


def chunk_documents(docs: dict[str, str] | None = None, **kwargs) -> list[Chunk]:
    return chunk_corpus(docs or documents(), **kwargs)


__all__ = [
    "EVAL_DIR",
    "EXPECTED_DOCS",
    "EXPECTED_WEAK",
    "RagTask",
    "RetrievalResult",
    "TASKS_PATH",
    "build_index_for",
    "chunk_documents",
    "documents",
    "load_tasks",
    "retrieve",
]
