"""The agentic retriever: retrieval as a decision, not as a stage.

``naive.py`` calls retrieval unconditionally, once, and hands whatever comes
back to the generator. Keeping that pipeline is the point -- it is the number
this module has to beat. The four things it cannot do are the four this module
does:

1. **Decide whether to retrieve at all.** The naive pipeline retrieves for
   "how do I bake sourdough" and then relies on a prompt sentence to make the
   model refuse. Checking first is cheaper, and the refusal is then a fact
   about the corpus rather than an instruction.
2. **Judge whether the result is good enough.** Called once and accepted is a
   guess. Judged and found lacking is a reason to act.
3. **Rewrite the query, not repeat it.** Asking the same index the same
   question twice spends two calls to learn one thing.
4. **Retrieve more than once, and combine.** This is the case the W3.5 report
   names as the thing to fix: ``multi-01`` asks why a loop needs compaction,
   and the answer needs the loop entry *and* the window entry.

The fourth one is also the one that cannot be solved by rewording. The
question never says "context window", and the entry that explains the window
scores zero against every phrasing of it -- so no amount of rephrasing the
*question* will find it. What finds it is the first round's own result: the
``compaction`` entry is retrieved, and it *discusses* the context window. So
the next query is built from terms that co-occur with the question's terms
inside what came back -- the classic relevance-feedback move, and the reason
the loop has to be a loop rather than a retry.

The decisions are explicit rules over stemmed terms, not a model. That is a
deliberate trade: the comparison the ROADMAP asks for is naive-vs-agentic
*behaviour*, and behaviour has to be reproducible offline for the comparison
to mean anything. A model decides these things better, and ``SufficiencyJudge``
plus ``AgenticRetriever.next_query`` are the two seams where a model-backed
policy drops in -- at which point the same report becomes a real measurement.

One property worth stating plainly, because it is the honest limitation: the
offline judge cannot tell whether a chunk *answers* the question, only whether
the vocabulary lines up. So it errs toward retrieving again, and the extra
retrievals are visible in the cost column. Making the judge smarter would
reduce that number; making it optimistic would reproduce the baseline and call
it progress.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, Sequence

from agentloop.tools import stem_tokens

from .corpus_rag import RagTask
from .tools import Citation, RetrieverTool

# Retrieval is capped because an agent that can retrieve forever will, on any
# question it cannot answer. The cap is the difference between a bounded cost
# and a surprise on the bill.
MAX_RETRIEVALS = 3

# Words that carry no retrieval signal. Small and fixed: this is a rewrite
# heuristic, not an IR model, and a large stoplist would hide the queries it
# produces from whoever is reading the trace.
_STOPWORDS = frozenset(
    """
    a an and are as at be because before but by for from how i if in into is
    it its me my of on or should so that the their them then there these this
    to too was were what when where which why will with you your
    """.split()
)

_WORD = re.compile(r"[a-z][a-z'-]+")


def _distinctive(question: str) -> set[str]:
    """The question's content terms, stemmed."""
    return {term for term in stem_tokens(question) if term not in _STOPWORDS}


def _stem(word: str) -> str:
    tokens = stem_tokens(word)
    return tokens[0] if tokens else ""


@dataclass
class Vocabulary:
    """Term statistics over the indexed corpus.

    Needed for one decision the citations alone cannot support: telling a
    *topic* word from a *common* one. ``compaction`` is worth adding to a
    query; ``loop`` is not, because it is in half the corpus and would flatten
    the ranking instead of sharpening it.

    Built from the store's chunks when the store can enumerate them. A dense
    store that cannot simply yields ``None``, and expansion falls back to
    plain co-occurrence -- worse, but not absent.
    """

    document_frequency: dict[str, int]
    documents: int
    rare_cutoff: int = 4

    @classmethod
    def from_texts(cls, texts: Iterable[str], *, rare_cutoff: int = 4) -> "Vocabulary":
        frequency: Counter[str] = Counter()
        count = 0
        for text in texts:
            count += 1
            for term in set(stem_tokens(text)):
                frequency[term] += 1
        return cls(document_frequency=dict(frequency), documents=count, rare_cutoff=rare_cutoff)

    @classmethod
    def from_store(cls, store: object) -> "Vocabulary | None":
        """Build from a store that can list its chunks, else ``None``."""
        chunks = getattr(store, "chunks", None)
        if not chunks:
            return None
        return cls.from_texts([getattr(chunk, "text", "") for chunk in chunks])

    def idf(self, term: str) -> float:
        df = self.document_frequency.get(term, 0)
        if self.documents == 0:
            return 0.0
        return math.log((self.documents - df + 0.5) / (df + 0.5) + 1.0)

    def is_rare(self, term: str) -> bool:
        """Rare enough to sharpen a query, and present at all."""
        df = self.document_frequency.get(term, 0)
        return 0 < df <= self.rare_cutoff


class SufficiencyJudge(Protocol):
    """Decides whether what came back is enough to answer with."""

    def __call__(self, question: str, citations: Sequence[Citation]) -> bool: ...


@dataclass
class CoverageJudge:
    """Sufficient when the best single citation carries the question's terms.

    Judged on the *best single* citation rather than the union, and that is
    the decision that matters. A union can look complete while the evidence is
    split across documents that each answer half the question -- which is
    exactly ``multi-01``: the loop entry and the window entry between them
    cover the terms, and neither alone can be answered from.

    Terms absent from the corpus are ignored, because chasing them is chasing
    something no query can find. That keeps the retry decision about the
    retrieval rather than about the question's wording.
    """

    vocabulary: Vocabulary | None = None

    def __call__(self, question: str, citations: Sequence[Citation]) -> bool:
        if not citations:
            return False
        wanted = _distinctive(question)
        if self.vocabulary is not None:
            wanted = {t for t in wanted if self.vocabulary.document_frequency.get(t, 0) > 0}
        if not wanted:
            return True
        best_covered = max(len(wanted & set(stem_tokens(c.text))) for c in citations)
        return best_covered == len(wanted)


@dataclass
class AlwaysSufficient:
    """A policy that never retries -- the baseline's behaviour, kept callable.

    Not a strawman: it is how the harness is validated. If this policy does
    not reproduce the naive numbers, the difference between the two reports is
    a difference in the measuring code, not in retrieval behaviour.
    """

    def __call__(self, question: str, citations: Sequence[Citation]) -> bool:
        return True


def expand_query(
    question: str,
    citations: Sequence[Citation],
    vocabulary: Vocabulary | None,
    *,
    max_terms: int = 4,
) -> str | None:
    """Build the next query from what the first round returned.

    This is relevance feedback: take terms that appear in the retrieved
    documents, skip the ones the question already has, keep the rare ones, and
    weight them by IDF. The rare-term filter is what makes it work -- common
    words already affect the ranking, so adding them changes nothing, while a
    topic word like ``compaction`` pulls in the entries that discuss it.

    Returns ``None`` when nothing useful was found, so the caller can fall
    back rather than re-issuing a query it knows is pointless.
    """
    if not citations:
        return None

    wanted = _distinctive(question)
    surface: dict[str, str] = {}
    scores: dict[str, float] = defaultdict(float)

    for citation in citations[:3]:
        for word in set(_WORD.findall(citation.text.lower())):
            stem = _stem(word) or word
            if stem in wanted or stem in _STOPWORDS:
                continue
            if vocabulary is not None:
                if not vocabulary.is_rare(stem):
                    continue
                scores[stem] += vocabulary.idf(stem)
            else:
                scores[stem] += 1.0
            surface.setdefault(stem, word)

    if not scores:
        return None

    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    terms = [surface[stem] for stem, _ in ranked[:max_terms]]

    kept = [word for word in _WORD.findall(question.lower()) if _stem(word) in wanted]
    return " ".join(kept + terms) or None


def clause_queries(question: str) -> list[str]:
    """Rewrites derived from the question's own structure.

    The fallback when relevance feedback produced nothing, and each strategy
    exists for a named failure:

    - **Split on the question's clauses.** A two-part question is two
      retrievals, because one query matches one half well and the other badly.
    - **Drop the interrogative.** "How much text does one token roughly
      correspond to" ranks worse than "one token text", because the question
      words match everything and sharpen nothing.
    - **Keep only the distinctive terms.** The last resort.

    Derived rather than invented, so the run is reproducible and a trace can
    be explained.
    """
    cleaned = question.strip().rstrip("?").strip()
    candidates: list[str] = []

    for separator in (" and ", ",", " because ", " while "):
        if separator in cleaned:
            parts = [p.strip() for p in cleaned.split(separator) if len(p.strip()) > 8]
            if len(parts) >= 2:
                candidates.extend(parts)

    stripped = _drop_interrogative(cleaned)
    if stripped and stripped.lower() != cleaned.lower():
        candidates.append(stripped)

    distinctive = " ".join(sorted(_distinctive(cleaned)))
    if distinctive:
        candidates.append(distinctive)

    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.lower()
        if key and key != cleaned.lower() and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def _drop_interrogative(text: str) -> str:
    words = text.split()
    while words and words[0].lower() in {"what", "why", "how", "when", "where", "which", "who"}:
        words.pop(0)
    return " ".join(words)


@dataclass
class Step:
    """One retrieval in the loop, kept for the trace."""

    attempt: int
    query: str
    hits: int
    doc_ids: list[str]
    sufficient: bool
    new_docs: list[str] = field(default_factory=list)
    strategy: str = "question"

    def to_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "query": self.query,
            "strategy": self.strategy,
            "hits": self.hits,
            "doc_ids": self.doc_ids,
            "sufficient": self.sufficient,
            "new_docs": self.new_docs,
        }


@dataclass
class AgenticRun:
    """What one agentic run did, in enough detail to argue with."""

    task_id: str
    category: str
    answer: str
    steps: list[Step]
    citations: list[Citation]
    retrieved: bool
    prompt_tokens: int
    generation_calls: int

    @property
    def retrievals(self) -> int:
        return len(self.steps)

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for step in self.steps:
            for doc_id in step.doc_ids:
                if doc_id not in seen:
                    seen.append(doc_id)
        return seen

    def to_dict(self) -> dict:
        return {
            "id": self.task_id,
            "category": self.category,
            "answer": self.answer,
            "retrieved": self.retrieved,
            "retrievals": self.retrievals,
            "sources": self.sources,
            "steps": [s.to_dict() for s in self.steps],
            "prompt_tokens": self.prompt_tokens,
            "generation_calls": self.generation_calls,
        }


class AgenticRetriever:
    """Retrieve, judge, expand, repeat -- then generate once."""

    def __init__(
        self,
        retriever: RetrieverTool,
        *,
        judge: SufficiencyJudge | None = None,
        vocabulary: Vocabulary | None = None,
        max_retrievals: int = MAX_RETRIEVALS,
        k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.vocabulary = vocabulary if vocabulary is not None else Vocabulary.from_store(retriever.store)
        self.judge = judge if judge is not None else CoverageJudge(self.vocabulary)
        self.max_retrievals = max_retrievals
        self.k = k

    # -- the decisions, each callable alone --------------------------------

    def should_retrieve(self, task: RagTask) -> bool:
        """Whether this question needs the corpus at all.

        Always true for a question with content words, because checking is
        cheaper than guessing and the negative questions are exactly the ones
        where "I looked and there is nothing" is the right answer. The
        decision that saves money is the *stopping* one, not this.
        """
        return bool(_distinctive(task.question))

    def next_query(
        self,
        task: RagTask,
        attempt: int,
        previous: Sequence[str],
        citations: Sequence[Citation],
    ) -> tuple[str, str] | None:
        """The next query and its strategy, or ``None`` if there is no better one.

        Returning ``None`` rather than repeating the last query is what makes
        the cap meaningful: an attempt that re-asks the same question is not an
        attempt, and counting it would make the loop look busy while learning
        nothing.
        """
        asked = {q.lower() for q in previous}

        if attempt <= 0:
            return task.question, "question"

        expanded = expand_query(task.question, citations, self.vocabulary)
        if expanded and expanded.lower() not in asked:
            return expanded, "relevance_feedback"

        for candidate in clause_queries(task.question):
            if candidate.lower() not in asked:
                return candidate, "clause"

        return None

    # -- the loop ----------------------------------------------------------

    def run(self, task: RagTask, generate: Callable[[str], str]) -> AgenticRun:
        """Retrieve with a decision at each step, then answer once.

        One generation call, at the end, over the accumulated evidence. Not
        one per retrieval: the comparison against the baseline is about
        retrieval, and multiplying generation calls would let a token
        difference be explained by something other than retrieval.
        """
        self.retriever.reset()
        steps: list[Step] = []
        citations: list[Citation] = []
        asked: list[str] = []

        if self.should_retrieve(task):
            for attempt in range(self.max_retrievals):
                chosen = self.next_query(task, attempt, asked, citations)
                if chosen is None:
                    break
                query, strategy = chosen
                asked.append(query)

                result = self.retriever.retrieve(query, k=self.k)
                known = {c.doc_id for c in citations}
                new_docs = [c.doc_id for c in result.citations if c.doc_id not in known]
                citations.extend(c for c in result.citations if c.doc_id not in known)

                sufficient = self.judge(task.question, citations)
                steps.append(
                    Step(
                        attempt=attempt + 1,
                        query=query,
                        strategy=strategy,
                        hits=len(result.citations),
                        doc_ids=[c.doc_id for c in result.citations],
                        sufficient=sufficient,
                        new_docs=new_docs,
                    )
                )
                # Stop on sufficiency, or on a retrieval that added nothing --
                # a further call after two with no new documents spends tokens
                # confirming a dead end.
                if sufficient or not new_docs:
                    break

        prompt = self._prompt(task, citations)
        answer = generate(prompt)
        return AgenticRun(
            task_id=task.id,
            category=task.category,
            answer=answer,
            steps=steps,
            citations=citations,
            retrieved=bool(steps),
            prompt_tokens=_estimate(prompt),
            generation_calls=1,
        )

    def _prompt(self, task: RagTask, citations: Sequence[Citation]) -> str:
        if not citations:
            return f"Question: {task.question}\nAnswer:"
        blocks = [f"[{i}] {c.doc_id}\n{c.text}" for i, c in enumerate(citations, start=1)]
        return (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say so plainly.\n\n"
            "Context:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {task.question}\nAnswer:"
        )


def _estimate(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


__all__ = [
    "MAX_RETRIEVALS",
    "AgenticRetriever",
    "AgenticRun",
    "AlwaysSufficient",
    "CoverageJudge",
    "Step",
    "SufficiencyJudge",
    "Vocabulary",
    "clause_queries",
    "expand_query",
]
