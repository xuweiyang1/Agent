"""ChromaDB as a ``VectorStore``, loaded only when it is installed.

The ROADMAP's "ChromaDB start -> Milvus" line is a promise that the store is
a substitution. This module is the evidence: it implements the same two
methods ``BM25Index`` and ``DenseIndex`` implement, and nothing above it --
``RetrieverTool``, ``run_baseline``, ``compare`` -- changes.

**Why the import is inside the functions.** Chroma pulls in a sizeable
dependency tree, and the suite is offline by rule. Importing it at module
scope would make ``retrieval/__init__.py`` fail to import on a machine
without it, taking the whole package down for a store nobody asked for. The
lazy import keeps the default path dependency-free and turns a missing
install into one clear sentence instead of an ImportError at collection time.

**What is passed in, not guessed.** The embedder is injected, exactly as in
``DenseIndex``, so the vectors Chroma indexes are the vectors this project
controls. Letting Chroma embed by default would silently swap the embedding
model and make its numbers incomparable with the local dense row. The store
computes the query vector here and hands Chroma ``query_embeddings``.

Distance is requested as cosine and converted back to a similarity
(``1 - distance``), so the score means the same thing it means in
``DenseIndex`` and the two dense rows can sit next to each other.
"""

from __future__ import annotations

from typing import Sequence

from .chunking import Chunk
from .index import ScoredChunk
from .vector_store import Embedder, HashingEmbedder

_MISSING = (
    "chromadb is not installed. It is an optional dependency: the offline "
    "default is DenseIndex, and chromadb is only needed to run the store this "
    "module wraps. Install it with `pip install chromadb` where network is "
    "available."
)


def chroma_available() -> bool:
    """Whether ChromaDB is importable in this environment."""
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


def _require_chroma():
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover - exercised via skipUnless
        raise ImportError(_MISSING) from exc
    return chromadb


class ChromaIndex:
    """A ``VectorStore`` backed by ChromaDB, in-memory by default.

    ``EphemeralClient`` is the default because a retrieval index in this
    project is built from a corpus and discarded at process exit -- persisting
    it would leave a file that grows silently and is never read again. A
    caller that wants a durable store passes ``persist_path``.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        embedder: Embedder | None = None,
        collection_name: str = "retrieval",
        persist_path: str | None = None,
    ) -> None:
        chromadb = _require_chroma()
        self.chunks: list[Chunk] = list(chunks)
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.collection_name = collection_name

        if persist_path:
            client = chromadb.PersistentClient(path=persist_path)
        else:
            client = chromadb.EphemeralClient()

        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        if self.chunks:
            self._collection.add(
                ids=[f"{chunk.doc_id}:{chunk.index}" for chunk in self.chunks],
                documents=[chunk.text for chunk in self.chunks],
                metadatas=[
                    {"doc_id": chunk.doc_id, "index": chunk.index} for chunk in self.chunks
                ],
                embeddings=[self.embedder(chunk.text) for chunk in self.chunks],
            )

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, *, k: int = 5) -> list[ScoredChunk]:
        if k <= 0 or not self.chunks:
            return []
        query_vector = self.embedder(query)
        if not any(query_vector):
            return []
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=min(k, len(self.chunks)),
            include=["distances", "metadatas"],
        )
        by_id = {(chunk.doc_id, chunk.index): chunk for chunk in self.chunks}
        hits: list[ScoredChunk] = []
        for metadata, distance in zip(result["metadatas"][0], result["distances"][0]):
            key = (metadata["doc_id"], metadata["index"])
            chunk = by_id.get(key)
            if chunk is None:  # defensive: a store should never invent a chunk
                continue
            score = 1.0 - float(distance)
            if score > 0:
                hits.append(ScoredChunk(chunk, score))
        return hits[:k]


__all__ = ["ChromaIndex", "chroma_available"]