"""BM25 over chunks, behind an interface that a vector store can also satisfy.

Two things are being decided here, and the second matters more than the first.

**Why BM25 first.** A dense embedding model was the obvious choice and the
wrong first move for this project: it needs weights downloaded, which needs
network access this environment does not reliably have, and it makes the
index non-deterministic across model versions. BM25 is offline, has no
weights, and is genuinely good at the failure that dominates a technical
corpus -- rare exact terms. It is also one half of the hybrid retrieval that
W6 wants anyway, so it is not scaffolding to be thrown away.

**Why an interface.** ``VectorStore`` is the seam. W6 swaps in a dense store
and the retriever keeps calling ``search``. Writing that seam on day one
costs nothing and makes the W6 comparison honest: one component changes, so
a difference in the numbers is attributable to that component. Bolting on an
interface later means the comparison is against a refactor, which is not a
comparison.

Two implementation details are load-bearing:

- **Tokenization reuses W1's stemmer** via ``stem_tokens``. ``retry`` has to
  match ``retries``; that bug was fixed once in the corpus search, and a
  second copy of the stemmer would be a second place for it to reappear.
- **BM25's IDF floor is why the k values match intution.** A term in more
  than half the chunks gets a negative IDF by the textbook formula, which
  makes a common word a *penalty*. The floor at a small positive value keeps
  matching monotonic.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from agentloop.tools import stem_tokens

from .chunking import Chunk

BM25_K1 = 1.5  # term-frequency saturation
BM25_B = 0.75  # length normalisation


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with its score, ready to be cited or re-ranked."""

    chunk: Chunk
    score: float

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

    def to_dict(self) -> dict:
        row = self.chunk.to_dict()
        row["score"] = round(self.score, 6)
        return row


class VectorStore(Protocol):
    """What a retriever needs from a store. Deliberately just a search.

    No ``add``, no ``embed``, no distance metric: a caller that can only
    search is the whole contract, which keeps a dense implementation free to
    look nothing like this one on the inside.
    """

    def search(self, query: str, *, k: int = 5) -> list[ScoredChunk]:
        ...

    def __len__(self) -> int:
        ...


class BM25Index:
    """A sparse, deterministic index over a fixed set of chunks.

    Built once from a list of chunks and not mutated afterwards. Immutability
    is what makes a retrieval score reproducible: an index that grows between
    queries changes the document frequencies, so the same query returns
    different rankings and two runs cannot be compared.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.k1 = k1
        self.b = b

        self._terms: list[Counter[str]] = []
        self._lengths: list[int] = []
        for chunk in self.chunks:
            tokens = stem_tokens(chunk.text)
            self._terms.append(Counter(tokens))
            self._lengths.append(len(tokens))

        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._df: Counter[str] = Counter()
        for counts in self._terms:
            self._df.update(counts.keys())

    def __len__(self) -> int:
        return len(self.chunks)

    def _idf(self, term: str) -> float:
        """Inverse document frequency with a positive floor.

        The textbook formula goes negative for a term present in more than
        half the documents, which turns a common word into a penalty and makes
        ranking non-monotonic -- a rare word would outrank a perfect match.
        The floor keeps a very common term worth approximately nothing
        instead of less than nothing.
        """
        total = len(self.chunks)
        if total == 0:
            return 0.0
        seen = self._df.get(term, 0)
        return max(1e-6, math.log(1 + (total - seen + 0.5) / (seen + 0.5)))

    def score(self, query: str, chunk_index: int) -> float:
        tokens = stem_tokens(query)
        if not tokens:
            return 0.0
        counts = self._terms[chunk_index]
        length = self._lengths[chunk_index] or 1
        total = 0.0
        for term in tokens:
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * length / (self._avg_length or 1)
            )
            total += self._idf(term) * (frequency * (self.k1 + 1)) / denominator
        return total

    def search(self, query: str, *, k: int = 5) -> list[ScoredChunk]:
        """Return the best ``k`` chunks, highest score first.

        Ties break on ``(doc_id, index)`` so equal scores come back in a
        stable order. Without that, two chunks with the same score swap places
        between runs and a diff of two benchmark reports shows phantom changes.
        """
        if k <= 0 or not self.chunks:
            return []
        scored = [
            ScoredChunk(chunk, self.score(query, position))
            for position, chunk in enumerate(self.chunks)
        ]
        hits = [item for item in scored if item.score > 0]
        hits.sort(key=lambda item: (-item.score, item.chunk.doc_id, item.chunk.index))
        return hits[:k]


def build_index(chunks: Sequence[Chunk]) -> BM25Index:
    """Convenience constructor, so callers do not import the class directly."""
    return BM25Index(chunks)


__all__ = [
    "BM25_B",
    "BM25_K1",
    "BM25Index",
    "ScoredChunk",
    "VectorStore",
    "build_index",
]
