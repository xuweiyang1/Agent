"""Where long-term memory lives, and the two seams that keep it swappable.

Long-term memory is deliberately two stores, because it holds two kinds of
thing and they need opposite treatment:

- **Preferences and past decisions are structured.** "Prefers aisle seats",
  "rejected the Lisbon plan on 2026-05-04". These are read by exact key, they
  must not be re-derived by similarity, and a wrong or missing one silently
  changes behaviour. A similarity search that returns a *nearly* matching
  preference is worse than no answer at all.
- **Everything else is semantic.** "The trip the user liked was the one with
  the food market." This is only findable by meaning, and losing precision is
  acceptable because it is a hint, not a setting.

Putting both in one vector store is the common shortcut and the one that
fails: it makes a lookup that must be exact into a lookup that is approximate.

Two seams are exported as protocols. ``StructuredStore`` is what
``LongTermMemory`` reads, and ``VectorStore`` is deliberately *not* redefined
here -- it is imported from ``retrieval.index``, because W6 has to use the
same one. A second, memory-flavoured protocol would let the two drift until
the W6 comparison is between two different interfaces rather than two
retrievers.

Persistence is a real requirement rather than a nicety: "remembers across
sessions" cannot be demonstrated by an object that dies with the process. So
the JSON store is the default in the demo and the in-memory one is the test
double, which is the reverse of the usual arrangement and for a stated
reason.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from retrieval.chunking import Chunk, chunk_text
from retrieval.index import BM25Index, ScoredChunk, build_index

# The kinds a record can be. Closed on purpose: a free-form ``kind`` string is
# how a preference ends up stored as a decision and then not found.
KIND_PREFERENCE = "preference"
KIND_DECISION = "decision"
KIND_FACT = "fact"
KINDS = (KIND_PREFERENCE, KIND_DECISION, KIND_FACT)


@dataclass(frozen=True)
class MemoryRecord:
    """One remembered thing.

    ``key`` is the exact handle for structured lookup, and ``text`` is the
    searchable sentence. Both are present because the two uses are different:
    ``key="seat"`` is how a preference is found reliably, and ``text="prefers
    an aisle seat"`` is how it is found by meaning when the key is unknown.

    ``session`` and ``created_at`` exist so a recall can say where it came
    from. A memory with no provenance cannot be revoked or corrected, and
    "why did it do that" becomes unanswerable.
    """

    id: str
    kind: str
    text: str
    key: str = ""
    value: Any = None
    session: str = ""
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown memory kind {self.kind!r}; expected one of {KINDS}")
        if not self.text.strip():
            raise ValueError("a memory record needs text; an empty memory is not a memory")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "key": self.key,
            "value": self.value,
            "session": self.session,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryRecord":
        """Rebuild from stored JSON.

        Tolerant of unknown fields being *added* by a newer version and strict
        about the required ones, so an old file keeps loading and a corrupt
        one fails loudly instead of producing a half-record.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@runtime_checkable
class StructuredStore(Protocol):
    """Exact-key storage for preferences and decisions."""

    def put(self, record: MemoryRecord) -> None: ...

    def all(self) -> list[MemoryRecord]: ...

    def find(self, *, kind: str | None = None, key: str | None = None) -> list[MemoryRecord]: ...

    def delete(self, record_id: str) -> bool: ...

    def __len__(self) -> int: ...


class InMemoryStore:
    """A structured store that forgets when the process does.

    The test double, and named that way so nobody mistakes it for the
    production path. It exists because most tests are about *logic* -- does a
    preference override a default -- and writing a file to ask that question
    would make them slow and order-dependent.
    """

    def __init__(self, records: Iterable[MemoryRecord] = ()) -> None:
        # Insertion order is preserved and duplicate ids replace in place, so
        # ``all()`` is stable across runs. A dict keyed by id gives both.
        self._records: dict[str, MemoryRecord] = {r.id: r for r in records}

    def put(self, record: MemoryRecord) -> None:
        self._records[record.id] = record

    def all(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def delete(self, record_id: str) -> bool:
        return self._records.pop(record_id, None) is not None

    def find(self, *, kind: str | None = None, key: str | None = None) -> list[MemoryRecord]:
        found = self._records.values()
        if kind is not None:
            found = [r for r in found if r.kind == kind]
        if key is not None:
            found = [r for r in found if r.key == key]
        return list(found)

    def __len__(self) -> int:
        return len(self._records)


class JsonFileStore:
    """A structured store that survives the process, which is the point.

    Written whole on every change. That is O(n) per write and entirely
    adequate for the tens of records a personal assistant accumulates; an
    append-only log or a real database would be the move at thousands, and
    saying so is cheaper than guessing.

    The write is temp-file-then-replace so an interrupted write cannot leave a
    truncated file. That matters more here than for a cache: a corrupt memory
    file loses preferences the user cannot easily restate.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: dict[str, MemoryRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            # Loud, not silent: a corrupt memory file is a bug to fix, and
            # starting fresh would hide it behind plausible amnesia.
            raise ValueError(f"corrupt memory file {self.path}: {exc}") from exc
        self._records = {item["id"]: MemoryRecord.from_dict(item) for item in payload}

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.to_dict() for r in self._records.values()]
        handle, temporary = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def put(self, record: MemoryRecord) -> None:
        self._records[record.id] = record
        self._flush()

    def all(self) -> list[MemoryRecord]:
        return list(self._records.values())

    def delete(self, record_id: str) -> bool:
        if self._records.pop(record_id, None) is None:
            return False
        self._flush()
        return True

    def find(self, *, kind: str | None = None, key: str | None = None) -> list[MemoryRecord]:
        found = list(self._records.values())
        if kind is not None:
            found = [r for r in found if r.kind == kind]
        if key is not None:
            found = [r for r in found if r.key == key]
        return found

    def __len__(self) -> int:
        return len(self._records)


class SemanticStore:
    """Search over remembered text, satisfying W3.5's ``VectorStore``.

    BM25 for the same reason the retrieval baseline chose it: offline, no
    weights, deterministic. The point of implementing the *existing* protocol
    is that W6 can swap this for Chroma or Milvus without the retriever
    noticing, and the memory layer does not need a second retrieval path.

    Rebuilt on write rather than updated in place. An incremental BM25 would
    have to recompute document frequencies anyway, and a store holding tens of
    memories does not need the optimisation. What it does need is that a
    search after a write reflects the write, which rebuilding guarantees.
    """

    def __init__(self, records: Sequence[MemoryRecord] = (), *, size: int = 400, overlap: int = 60) -> None:
        self.size = size
        self.overlap = overlap
        self._records: dict[str, MemoryRecord] = {}
        self._index: BM25Index = build_index([])
        for record in records:
            self._records[record.id] = record
        self._rebuild()

    def add(self, record: MemoryRecord) -> None:
        self._records[record.id] = record
        self._rebuild()

    def delete(self, record_id: str) -> bool:
        if self._records.pop(record_id, None) is None:
            return False
        self._rebuild()
        return True

    def _rebuild(self) -> None:
        chunks: list[Chunk] = []
        for record in self._records.values():
            # ``doc_id`` is the record id, so a hit is traceable back to the
            # memory it came from without keeping a parallel mapping.
            chunks.extend(
                chunk_text(record.text, doc_id=record.id, size=self.size, overlap=self.overlap)
            )
        self._index = build_index(chunks)

    def search(self, query: str, *, k: int = 5) -> list[ScoredChunk]:
        return self._index.search(query, k=k)

    def record_of(self, doc_id: str) -> MemoryRecord | None:
        return self._records.get(doc_id)

    def __len__(self) -> int:
        return len(self._records)


def with_created_at(record: MemoryRecord, timestamp: str) -> MemoryRecord:
    """Return a copy stamped with ``timestamp``.

    A free function rather than a ``MemoryRecord`` default, because the value
    has to come from the caller's clock or an injected one -- a dataclass
    default would freeze the import time and make every record look created at
    startup.
    """
    return replace(record, created_at=timestamp)


__all__ = [
    "KINDS",
    "KIND_DECISION",
    "KIND_FACT",
    "KIND_PREFERENCE",
    "InMemoryStore",
    "JsonFileStore",
    "MemoryRecord",
    "SemanticStore",
    "StructuredStore",
    "with_created_at",
]
