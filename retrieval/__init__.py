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

__all__ = [
    "DEFAULT_OVERLAP",
    "DEFAULT_SIZE",
    "Chunk",
    "chunk_corpus",
    "chunk_document",
    "chunk_text",
]
