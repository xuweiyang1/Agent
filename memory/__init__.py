"""W5: three layers of memory, and the reason there are three.

``WorkingMemory`` is what is happening now, ``SessionMemory`` is what has
been said, ``LongTermMemory`` is what should outlive the conversation. Each
exists because it loses a different thing if you try to do its job with one
of the others -- a single store cannot be bounded and complete at once.

``build_memory`` is the one entry point, and it wires the long-term semantic
store so that W6's retriever can take it directly: the memory layer and the
retrieval layer share ``retrieval.index.VectorStore`` rather than defining
two interfaces that would drift apart.
"""

from __future__ import annotations

from pathlib import Path

from .longterm import Clock, LongTermMemory, system_clock
from .session import SessionMemory, Summarizer, session_from_checkpoint, truncating_summarizer
from .store import (
    InMemoryStore,
    JsonFileStore,
    MemoryRecord,
    SemanticStore,
    StructuredStore,
)
from .working import ScratchEntry, WorkingMemory

__all__ = [
    "Clock",
    "InMemoryStore",
    "JsonFileStore",
    "LongTermMemory",
    "MemoryRecord",
    "ScratchEntry",
    "SemanticStore",
    "SessionMemory",
    "StructuredStore",
    "session_from_checkpoint",
    "Summarizer",
    "WorkingMemory",
    "build_memory",
    "system_clock",
    "truncating_summarizer",
]


def build_memory(
    *,
    path: str | Path | None = None,
    clock: Clock | None = None,
    structured: StructuredStore | None = None,
    semantic: SemanticStore | None = None,
    window: int = 6,
    summarize: Summarizer | None = None,
) -> tuple[LongTermMemory, SessionMemory, WorkingMemory]:
    """Assemble the three layers.

    Returning a tuple instead of a container object is deliberate: the three
    are peers with different lifetimes, and a wrapper would invite callers to
    treat them as one thing -- which is the mistake the module docstring
    describes.

    With ``path``, long-term memory is a file and therefore actually outlives
    the process. Without it, in-memory, which is what the tests want.
    """
    structured_store = structured or (JsonFileStore(path) if path else InMemoryStore())
    semantic_store = semantic or SemanticStore()
    if semantic is None:
        # Seed semantic recall with everything already in the structured
        # store. Without this a file loaded from disk is findable by key but
        # invisible to ``recall`` -- the two stores would hold different
        # memories, which is worse than holding fewer.
        for record in structured_store.all():
            semantic_store.add(record)

    longterm = LongTermMemory(
        structured=structured_store,
        semantic=semantic_store,
        clock=clock or system_clock,
    )
    session = SessionMemory(window=window, summarize=summarize)
    working = WorkingMemory()
    return longterm, session, working
