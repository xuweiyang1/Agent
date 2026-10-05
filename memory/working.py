"""Working memory: the scratchpad for what is happening right now.

This layer exists to answer one question: *what should not be in the prompt?*
A tool returns a 4 KB document and the model needs one line from it; a search
returns eight hits and the plan needs one. Putting all of it into the
transcript is what makes the next turn expensive, and it is the most common
way an agent gets slower and dumber at the same time.

So the raw results land here, bounded, and only what a node chooses to
surface reaches the transcript. Three design points:

- **The bound is on tokens, not entries.** An entry cap does not do the job
  the layer exists for: eight one-line results and eight documents are the
  same count and wildly different costs.
- **Eviction is oldest-first and recorded.** A silent eviction is how a bug
  looks like the memory layer forgetting something on purpose. The count of
  evicted entries is kept so a caller can notice.
- **It is cleared per turn, and that is deliberate.** Working memory that
  survives between turns is session memory with a misleading name, and the
  two need different lifetime rules.

The layer is intentionally dumb: a list with a token budget. The interesting
decisions live in ``session`` and ``longterm``; making this one clever would
just make it harder to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentloop.context import estimate_tokens

DEFAULT_TOKEN_BUDGET = 1200


@dataclass(frozen=True)
class ScratchEntry:
    """One intermediate result, with where it came from."""

    source: str
    text: str
    tokens: int


@dataclass
class WorkingMemory:
    """A bounded, per-turn scratchpad.

    ``note`` is the only way in, and ``recent`` the only way out, so the
    budget is enforced in one place and cannot be bypassed by a caller that
    appends to a list.
    """

    token_budget: int = DEFAULT_TOKEN_BUDGET
    turn: int = 0
    _entries: list[ScratchEntry] = field(default_factory=list, init=False)
    _evicted: int = field(default=0, init=False)

    def note(self, source: str, text: str) -> ScratchEntry:
        """Record an intermediate result, evicting the oldest to fit.

        A single entry larger than the whole budget is kept rather than
        dropped: refusing it would leave the caller with nothing and no
        explanation, whereas keeping one oversized entry is visible in
        ``tokens`` and lets the caller decide to summarise it.
        """
        entry = ScratchEntry(source=source, text=text, tokens=estimate_tokens(text))
        self._entries.append(entry)
        self._enforce_budget()
        return entry

    def _enforce_budget(self) -> None:
        while self.tokens() > self.token_budget and len(self._entries) > 1:
            self._entries.pop(0)
            self._evicted += 1

    def recent(self, n: int | None = None) -> list[ScratchEntry]:
        """The most recent entries, newest last, in original order."""
        if n is None or n >= len(self._entries):
            return list(self._entries)
        return self._entries[-n:]

    def find(self, source: str) -> list[ScratchEntry]:
        return [e for e in self._entries if e.source == source]

    def tokens(self) -> int:
        return sum(entry.tokens for entry in self._entries)

    @property
    def evicted(self) -> int:
        """How many entries the budget pushed out. Non-zero is a signal."""
        return self._evicted

    def clear(self) -> None:
        """End the turn.

        Reset rather than accumulate: the count of evictions goes with it,
        because a new turn starting with the previous turn's eviction count
        would report a problem that has already been handled.
        """
        self._entries.clear()
        self._evicted = 0
        self.turn += 1

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["DEFAULT_TOKEN_BUDGET", "ScratchEntry", "WorkingMemory"]
