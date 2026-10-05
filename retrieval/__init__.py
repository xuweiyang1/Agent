"""Retrieval: chunking, an index, and a generator.

Built to be reused rather than replaced. The naive pipeline in W3.5 and the
agentic retriever in W6 share this exact code; only the caller changes. That
is what makes the W6 comparison meaningful -- one variable moves, not two.
"""

from .chunking import (
    DEFAULT_OVERLAP,
    DEFAULT_SIZE,
    Chunk,
    chunk_corpus,
    chunk_document,
    chunk_text,
)
from .index import BM25Index, ScoredChunk, VectorStore, build_index
from .chroma_store import ChromaIndex, chroma_available
from .vector_store import (
    DEFAULT_DIM,
    DEFAULT_NGRAMS,
    DEFAULT_RRF_K,
    DenseIndex,
    Embedder,
    HashingEmbedder,
    HybridIndex,
    build_vector_index,
)

__all__ = [
    "BM25Index",
    "ChromaIndex",
    "DEFAULT_DIM",
    "DEFAULT_NGRAMS",
    "DEFAULT_RRF_K",
    "DenseIndex",
    "Embedder",
    "HashingEmbedder",
    "HybridIndex",
    "DEFAULT_OVERLAP",
    "DEFAULT_SIZE",
    "Chunk",
    "ScoredChunk",
    "VectorStore",
    "build_index",
    "build_vector_index",
    "chunk_corpus",
    "chunk_document",
    "chunk_text",
    "chroma_available",
]
