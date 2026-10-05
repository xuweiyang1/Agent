"""Long-term memory: what outlives the session, and how it changes behaviour.

The requirement is "remember the user's preferences and past decisions", and
the interesting part is the second half of that sentence: **change
behaviour**. A note that is stored but never consulted is a diary, not a
memory. So this module is built around the two ways a memory gets used, and
each has its own store:

- ``preference(key)`` -- exact lookup. ``preference("seat")`` returns "aisle"
  or nothing, and never a close match. A preference that is 90% right is a
  bug that will be blamed on the model.
- ``recall(query)`` -- semantic search, for the things with no key. "Which
  plan did I reject" is not a field, it is a sentence.

Use is recorded. ``MemoryRecord.metadata["uses"]`` counts how often a memory
was actually consulted, because "the long-term memory is populated" and "the
long-term memory is used" are different claims, and only the second one is
worth making.

Two rules that exist because of specific failure modes:

- **A preference can be revised, not accumulated.** Setting ``seat`` twice
  replaces it. Appending would leave two answers to one question and the
  lookup order would silently become the policy.
- **A decision records what was rejected and why.** "Did not go to Lisbon,
  over budget" is what stops the same suggestion coming back -- the same
  mechanism W4's ``avoid`` list implements within a run, here across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from retrieval.index import ScoredChunk

from .store import (
    KIND_DECISION,
    KIND_FACT,
    KIND_PREFERENCE,
    MemoryRecord,
    SemanticStore,
    StructuredStore,
)

Clock = Callable[[], str]


@dataclass
class LongTermMemory:
    """Structured facts plus semantic recall, over one write path.

    Both stores are injected. The structured one is what makes persistence
    testable (in-memory in tests, a file in the demo), and the semantic one is
    what makes W6 able to reuse it: ``SemanticStore`` satisfies the same
    ``VectorStore`` protocol that a dense store will, so the retriever in W6
    takes this as its backend and the comparison is honest.
    """

    structured: StructuredStore
    semantic: SemanticStore
    clock: Clock
    _counter: int = 0

    def _next_id(self, kind: str) -> str:
        self._counter += 1
        return f"{kind}-{self._counter:04d}"

    def _store(self, record: MemoryRecord) -> MemoryRecord:
        self.structured.put(record)
        self.semantic.add(record)
        return record

    # -- writing -----------------------------------------------------------

    def remember_preference(self, key: str, value: Any, *, text: str = "", session: str = "") -> MemoryRecord:
        """Set a preference, replacing any earlier value for the same key.

        Replacement rather than addition is the whole correctness argument:
        two live answers to "what seat does the user want" means the answer
        depends on iteration order, which is the kind of bug that shows up as
        an agent that changes its mind for no reason.
        """
        for existing in self.structured.find(kind=KIND_PREFERENCE, key=key):
            self._forget(existing.id)
        sentence = text or f"The user prefers {key} = {value}."
        return self._store(
            MemoryRecord(
                id=self._next_id("pref"),
                kind=KIND_PREFERENCE,
                text=sentence,
                key=key,
                value=value,
                session=session,
                created_at=self.clock(),
            )
        )

    def remember_decision(
        self,
        subject: str,
        outcome: str,
        *,
        reason: str = "",
        session: str = "",
    ) -> MemoryRecord:
        """Record what was decided, and why.

        The reason is not optional in spirit even though the argument is:
        "rejected Lisbon" without "over budget" cannot be acted on later,
        because there is no way to tell whether the objection still stands.
        """
        sentence = f"Decision on {subject}: {outcome}"
        if reason:
            sentence += f" (because {reason})"
        return self._store(
            MemoryRecord(
                id=self._next_id("decision"),
                kind=KIND_DECISION,
                text=sentence + ".",
                key=subject,
                value={"outcome": outcome, "reason": reason},
                session=session,
                created_at=self.clock(),
            )
        )

    def remember_fact(self, text: str, *, session: str = "", metadata: dict[str, Any] | None = None) -> MemoryRecord:
        """Remember something with no key and no decision attached."""
        return self._store(
            MemoryRecord(
                id=self._next_id("fact"),
                kind=KIND_FACT,
                text=text,
                session=session,
                created_at=self.clock(),
                metadata=dict(metadata or {}),
            )
        )

    def _forget(self, record_id: str) -> None:
        """Drop a record from both stores, so they cannot disagree.

        Through the public ``delete`` rather than by reaching into internals:
        a preference that is removed from one store and left in the other is
        found by ``recall`` and invisible to ``preference``, which is exactly
        the kind of split-brain the two-store design has to be careful about.
        """
        self.structured.delete(record_id)
        self.semantic.delete(record_id)

    # -- reading -----------------------------------------------------------

    def preference(self, key: str, default: Any = None) -> Any:
        """Exact lookup, most recent wins.

        Falls back through the newest record for the key, so a caller does
        not have to know that replacement is implemented as forget-then-add.
        """
        matches = self.structured.find(kind=KIND_PREFERENCE, key=key)
        if not matches:
            return default
        return matches[-1].value

    def preferences(self) -> dict[str, Any]:
        """Every current preference, as a plain mapping."""
        return {r.key: r.value for r in self.structured.find(kind=KIND_PREFERENCE)}

    def decisions(self, subject: str | None = None) -> list[MemoryRecord]:
        return self.structured.find(kind=KIND_DECISION, key=subject)

    def recall(self, query: str, *, k: int = 3, mark_used: bool = True) -> list[ScoredChunk]:
        """Semantic search across everything remembered.

        ``mark_used`` exists because a *read* that mutates storage is
        surprising; a caller doing a speculative search should be able to opt
        out. The default is on, because the count is only useful if ordinary
        use is what increments it.
        """
        hits = self.semantic.search(query, k=k)
        if mark_used:
            for hit in hits:
                self._mark_used(hit.doc_id)
        return hits

    def _mark_used(self, record_id: str) -> None:
        record = next((r for r in self.structured.all() if r.id == record_id), None)
        if record is None:
            return
        uses = int(record.metadata.get("uses", 0)) + 1
        # ``metadata`` is mutable by design here: the record is frozen so its
        # identity cannot change, but the usage counter is not identity.
        record.metadata["uses"] = uses
        self.structured.put(record)

    def context_for(self, query: str, *, k: int = 3) -> str:
        """A prompt-ready block: preferences as facts, then recall as hints.

        The two are labelled differently on purpose. A preference is a
        constraint the answer must respect; a recalled memory is evidence that
        may be stale. Presenting them identically invites the model to treat a
        three-session-old recollection as a rule.
        """
        lines: list[str] = []
        prefs = self.preferences()
        if prefs:
            lines.append("Known preferences:")
            lines.extend(f"- {key}: {value}" for key, value in sorted(prefs.items()))
        # Preferences are already listed as constraints above, so they are
        # filtered out of the semantic half. Printing the same fact twice, once
        # as a rule and once as a hint, invites the model to weigh it twice.
        hints = [
            hit
            for hit in self.recall(query, k=k)
            if (record := self.semantic.record_of(hit.doc_id)) is not None
            and record.kind != KIND_PREFERENCE
        ]
        if hints:
            lines.append("Maybe relevant from earlier:")
            for hit in hints:
                record = self.semantic.record_of(hit.doc_id)
                lines.append(f"- {record.text}")

        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.structured)


def system_clock() -> str:
    """The default clock: local time, ISO-8601, seconds precision."""
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


__all__ = ["Clock", "LongTermMemory", "system_clock"]
