"""A dense store behind the same ``VectorStore`` seam as BM25.

The ROADMAP names ChromaDB as the vector store this project is meant to grow
into. The honest way to get there is to first show *why* dense retrieval is
worth a dependency, and to do it under the same offline rule the rest of the
suite lives under. That is what this module is: a dense index with no
download, no network, and a reproducible score.

**Why not a real embedding model.** The obvious implementation is
``sentence-transformers``. It is also the one that breaks two constraints at
once. It downloads weights, and this environment is offline. Worse for a
benchmark, its vectors change between model versions, so a stored number is
not reproducible -- and reproducibility is the whole reason W3.5's reports
can be diffed at all.

**So: hashing.** Character n-grams hashed into a fixed-width vector is the
classic offline stand-in. It needs no weights, gives the same vector for the
same text on every machine, and captures the sub-word overlap a word-level
BM25 misses: ``retry`` and ``retries`` share trigrams without a stemmer. It
is weaker than a trained encoder on paraphrase, and that tradeoff is stated
rather than hidden -- this store measures the *shape* of the dense-vs-sparse
difference here, it does not pretend to be a production embedder.

Three details are load-bearing:

- **Hashes come from ``hashlib``, not ``hash()``.** The builtin is salted per
  process, so the same corpus would embed differently in a fresh run and the
  reproducibility test would fail intermittently. ``blake2b`` is stable.
- **Signed weights, then L2.** Each n-gram is added with a sign from a second
  hash, which lets collisions cancel instead of always inflating similarity,
  and every vector is normalised so cosine similarity is a plain dot product
  and a longer chunk is not rewarded for being longer.
- **Ties break on ``(doc_id, index)``**, the same rule BM25 uses, so the two
  stores produce a diffable order rather than one that reshuffles per run.
"""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, Protocol, Sequence

from .chunking import Chunk
from .index import ScoredChunk

DEFAULT_DIM = 512
DEFAULT_NGRAMS = (2, 3, 4)
DEFAULT_RRF_K = 60  # the standard reciprocal-rank constant from Cormack et al.


class Embedder(Protocol):
    """Text in, a fixed-width vector out. Small on purpose.

    The injectable seam that lets a real model replace the hashing stand-in
    without touching the index. Kept to ``__call__`` and ``dim`` because
    anything more would couple the index to one library's API.
    """

    dim: int

    def __call__(self, text: str) -> list[float]:
        ...


class HashingEmbedder:
    """Deterministic character n-gram hashing, offline and version-stable.

    Not a language model and labelled as such. Its value is that it is the
    same function on every machine forever, which is what a benchmark needs,
    and that it fails on paraphrase in a *predictable* way rather than a
    random one -- so a dense-vs-sparse gap measured with it is real signal
    about the retrieval method, not noise about a model checkpoint.
    """

    def __init__(
        self,
        *,
        dim: int = DEFAULT_DIM,
        ngrams: Sequence[int] = DEFAULT_NGRAMS,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.ngrams = tuple(ngrams)

    def _grams(self, text: str) -> Iterable[str]:
        """Lowercased, whitespace-collapsed character n-grams.

        Whitespace is collapsed first so the same sentence does not embed
        differently depending on indentation, which is the kind of accidental
        corpus dependency that makes two reports incomparable.
        """
        normalised = " ".join(text.lower().split())
        for n in self.ngrams:
            if n <= 0 or len(normalised) < n:
                continue
            for start in range(len(normalised) - n + 1):
                gram = normalised[start : start + n]
                if gram.strip():
                    yield gram

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for gram in self._grams(text):
            encoded = gram.encode("utf-8")
            bucket = int.from_bytes(
                hashlib.blake2b(encoded, digest_size=8).digest(), "big"
            ) % self.dim
            # A second, independent hash decides the sign. Deriving it from
            # the bucket hash instead would correlate sign with position and
            # weaken the collision cancellation.
            sign = 1.0 if hashlib.blake2b(b"sign:" + encoded, digest_size=1).digest()[0] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class DenseIndex:
    """Cosine search over embedded chunks, behind ``VectorStore``.

    Built once and not mutated, for the same reason ``BM25Index`` is: a store
    that changes between queries cannot be compared to itself.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        embedder: Embedder | None = None,
    ) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.embedder: Embedder = embedder or HashingEmbedder()
        self._vectors: list[list[float]] = [self.embedder(c.text) for c in self.chunks]

    def __len__(self) -> int:
        return len(self.chunks)

    def _dot(self, query_vector: Sequence[float], position: int) -> float:
        return sum(a * b for a, b in zip(query_vector, self._vectors[position]))

    def score(self, query: str, chunk_index: int) -> float:
        """Similarity of one chunk to the query, the same shape BM25 exposes."""
        return self._dot(self.embedder(query), chunk_index)

    def search(self, query: str, *, k: int = 5) -> list[ScoredChunk]:
        """Best ``k`` by cosine, highest first, ties on ``(doc_id, index)``.

        A dense store returns *something* for almost any query, which is a
        real behavioural difference from BM25 and the reason the comparison
        report watches the negative tasks: an always-answerable retriever is
        exactly what an abstention test is for.
        """
        if k <= 0 or not self.chunks:
            return []
        query_vector = self.embedder(query)
        if not any(query_vector):
            return []
        scored = [
            ScoredChunk(chunk, self._dot(query_vector, position))
            for position, chunk in enumerate(self.chunks)
        ]
        hits = [item for item in scored if item.score > 0]
        hits.sort(key=lambda item: (-item.score, item.chunk.doc_id, item.chunk.index))
        return hits[:k]


class HybridIndex:
    """Reciprocal-rank fusion of several stores, also a ``VectorStore``.

    Fusion happens on *ranks*, not scores: BM25 and cosine live on different
    scales, and normalising them to compare is a source of tuning that would
    make the comparison depend on a constant someone picked. Ranks are already
    comparable, which is the argument for RRF and the reason it is the
    baseline hybrid here rather than a weighted sum.

    A chunk's fused score is ``sum(1 / (k + rank))`` over the stores that
    returned it, so agreement between stores is what wins. Ties break on
    ``(doc_id, index)`` like every other store in this module.
    """

    def __init__(
        self,
        stores: Sequence[VectorStore],
        *,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        if not stores:
            raise ValueError("HybridIndex needs at least one store")
        self.stores = list(stores)
        self.rrf_k = rrf_k

    def __len__(self) -> int:
        # The union is bounded by the widest store; enough for a caller that
        # only wants a scale, and cheap to compute without touching a query.
        return max(len(store) for store in self.stores)

    def search(self, query: str, *, k: int = 5) -> list[ScoredChunk]:
        if k <= 0:
            return []
        fused: dict[tuple[str, int], float] = {}
        by_key: dict[tuple[str, int], Chunk] = {}
        for store in self.stores:
            for rank, hit in enumerate(store.search(query, k=max(k, 20)), start=1):
                key = (hit.chunk.doc_id, hit.chunk.index)
                by_key[key] = hit.chunk
                fused[key] = fused.get(key, 0.0) + 1.0 / (self.rrf_k + rank)
        scored = [ScoredChunk(by_key[key], score) for key, score in fused.items()]
        scored.sort(key=lambda item: (-item.score, item.chunk.doc_id, item.chunk.index))
        return scored[:k]


def build_vector_index(
    chunks: Sequence[Chunk],
    *,
    embedder: Embedder | None = None,
) -> DenseIndex:
    """Convenience constructor, mirroring ``build_index`` for BM25."""
    return DenseIndex(chunks, embedder=embedder)


__all__ = [
    "DEFAULT_DIM",
    "DEFAULT_NGRAMS",
    "DEFAULT_RRF_K",
    "DenseIndex",
    "Embedder",
    "HashingEmbedder",
    "HybridIndex",
    "build_vector_index",
]