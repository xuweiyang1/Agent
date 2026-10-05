"""Session memory: the answer to "the conversation got long, now what?".

This is the most-asked question about agents and the answer is not "use a
bigger model". A conversation that grows without bound fails three ways --
cost, latency, and the model getting *worse* as the context fills with
irrelevant older turns. The fix is not compression, it is **layering**, and
this module is three layers over the same transcript:

1. **A recent window, verbatim.** The last few turns stay untouched, because
   they are what the next reply is actually about. Nothing is gained by
   summarising the thing the user just said.
2. **A rolling summary of everything older.** Cheaper than the turns it
   replaces and, crucially, *bounded*: summarising an old summary is how the
   size stays flat while the conversation grows. The compression rate is
   measurable, which is why ``summarized_tokens`` is reported rather than
   hidden.
3. **Retrieval over the turns that were dropped.** This is the part naive
   compaction misses. A summary is lossy on purpose; the fix for the loss is
   not a better summary, it is being able to get the original back. "What
   exactly did I say about the hotel" searches the evicted turns and returns
   them, instead of hoping the gist survived summarisation.

So the invariant is: **the window keeps what is happening, the summary keeps
the shape of what happened, and retrieval keeps the details recoverable.**
Three mechanisms, because they lose different things.

The session turns are read from W4's checkpoint rather than stored again.
That is the whole reason the checkpoint was built, and duplicating the
transcript here would create two histories that disagree after a resume --
the exact class of bug the ROADMAP warns about by saying to reuse it.

What this module deliberately does *not* do: call a model. Summarisation is
an injected callable, so the tests are offline and the demo can use a real
one. A session layer that hard-codes an API call cannot be tested, and a
memory bug is precisely the bug you want a test for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from agentloop.context import estimate_tokens
from retrieval.chunking import Chunk, chunk_text
from retrieval.index import BM25Index, ScoredChunk, build_index

Summarizer = Callable[[list[str], str], str]
"""``(older_turns, previous_summary) -> new_summary``.

The previous summary is an argument rather than being appended afterwards
because a summariser that cannot see its own last output will re-derive it,
and re-derivation is where the meaning drifts.
"""

DEFAULT_WINDOW = 6


@dataclass
class Turn:
    """One exchange, as the session layer sees it."""

    index: int
    role: str
    text: str

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class SessionMemory:
    """Turn history with a window, a rolling summary, and recall.

    The three are exposed separately on purpose. A caller assembling a prompt
    decides what to include; the layer does not silently decide for it, since
    "what goes in the prompt" is the cost decision this whole project is
    about.
    """

    window: int = DEFAULT_WINDOW
    summarize: Summarizer | None = None
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    summarized_tokens: int = 0
    index: BM25Index = field(default_factory=lambda: build_index([]))
    _evicted: list[Turn] = field(default_factory=list, init=False)
    # Highest turn index already folded into the summary. A private field
    # rather than a local because folding happens incrementally: each new
    # turn can push one more turn out of the window, and re-summarising the
    # whole history on every turn would grow the cost without bound.
    _summarized_upto: int = field(default=0, init=False)

    def add(self, role: str, text: str) -> Turn:
        turn = Turn(index=len(self.turns), role=role, text=text)
        self.turns.append(turn)
        self._reindex()
        self._fold()
        return turn

    def extend(self, exchanges: Iterable[tuple[str, str]]) -> list[Turn]:
        return [self.add(role, text) for role, text in exchanges]

    # -- the three layers --------------------------------------------------

    def recent(self) -> list[Turn]:
        """The verbatim window: the last ``window`` turns, oldest first."""
        return list(self.turns[-self.window :]) if self.window else []

    def prompt(self) -> str:
        """The assembled form: summary first, then the live window.

        Summary before window, and that order is not cosmetic -- it is the
        prompt-caching rule from the ROADMAP. The stable prefix (summary)
        comes first and the volatile part (the newest turns) comes last, so a
        provider caching on a prefix keeps hitting.
        """
        parts: list[str] = []
        if self.summary:
            parts.append(f"Earlier in this conversation:\n{self.summary}")
        for turn in self.recent():
            parts.append(f"{turn.role}: {turn.text}")
        return "\n\n".join(parts)

    def recall(self, query: str, *, k: int = 3) -> list[ScoredChunk]:
        """Search the turns that the window no longer holds.

        Only evicted turns are indexed. Searching the live window as well
        would return what the caller already has, which wastes budget on the
        one thing that is certainly not missing.
        """
        if not self._evicted:
            return []
        return self.index.search(query, k=k)

    def turn_for(self, doc_id: str) -> Turn | None:
        """Resolve a recall hit back to the turn it came from."""
        prefix = "session-turn-"
        if not doc_id.startswith(prefix):
            return None
        try:
            position = int(doc_id[len(prefix) :].split("-")[0])
        except ValueError:
            return None
        return next((t for t in self._evicted if t.index == position), None)

    # -- internals ---------------------------------------------------------

    def _fold(self) -> None:
        """Move turns out of the window, into the summary and the recall index.

        Two things happen here and they are deliberately independent:

        - **Eviction is unconditional.** A turn older than the window leaves
          the verbatim layer either way, and it becomes recall material either
          way. Tying this to the presence of a summariser would mean the
          retrieval layer silently disappears in the no-summariser
          configuration -- the one place where recall is the *only* way to get
          an old detail back.
        - **Summarisation is optional.** Without a summariser the window is
          all the prompt holds, and ``prompt()`` says so by containing no
          summary. That is a stated configuration rather than a silent
          failure: nothing claims to have compressed what it did not.

        Only the turns not yet folded are summarised, because re-summarising
        the whole history on every turn would make the cost grow with the
        conversation -- the exact growth this layer exists to prevent.
        """
        if len(self.turns) <= self.window:
            return

        older = self.turns[: -self.window] if self.window else list(self.turns)
        self._evicted = list(older)
        self._reindex()

        if self.summarize is None:
            return
        fresh = [t for t in older if t.index >= self._summarized_upto]
        if not fresh:
            return

        self.summary = self.summarize([f"{t.role}: {t.text}" for t in fresh], self.summary)
        self._summarized_upto = older[-1].index + 1
        self.summarized_tokens += sum(t.tokens for t in fresh)

    def _reindex(self) -> None:
        chunks: list[Chunk] = []
        for turn in self._evicted:
            # The id encodes the turn index so a hit can be traced back to the
            # exact exchange, the same way a chunk traces to its document.
            chunks.extend(chunk_text(turn.text, doc_id=f"session-turn-{turn.index}", size=400, overlap=60))
        self.index = build_index(chunks)

    @property
    def compression(self) -> float:
        """How much smaller the summary is than what it replaced.

        Reported because it is the number this whole layer is judged on. A
        summariser that returns something the size of its input has not
        compressed anything, and that should be visible rather than felt.
        """
        if not self.summarized_tokens:
            return 1.0
        return estimate_tokens(self.summary) / self.summarized_tokens

    def tokens(self) -> int:
        """The cost of the three layers as they would enter a prompt."""
        return estimate_tokens(self.prompt())

    def __len__(self) -> int:
        return len(self.turns)


def turns_from_checkpoint(app: Any, config: dict[str, Any], *, window: int = 20) -> list[tuple[str, str]]:
    """Read a completed W4 run out of its checkpoint, as ``(role, text)`` pairs.

    This is the "session memory reuses the W4 checkpoint" requirement, made
    concrete. The alternative -- keeping a second list of turns alongside the
    graph state -- creates two histories that disagree the moment a run is
    resumed, because the checkpoint is what actually survives a pause.

    ``get_state_history`` returns newest-first, so it is reversed here to get
    the run in the order it happened. Each snapshot contributes the notes the
    node *just* added, which is the checkpoint's own record of what the run
    learned at that step; nothing is re-derived from the final state, because
    reconstructing history from a summary of history is how the details are
    lost.

    Only ``window`` snapshots are taken. A long run has one checkpoint per
    node, and hydrating all of them would rebuild the unbounded transcript
    this layer exists to avoid.
    """
    history = list(app.get_state_history(config))
    if not history:
        return []

    # Oldest first, and bounded to the tail: the recent part of a run is what
    # a follow-up question is about.
    ordered = list(reversed(history))[-window:]

    turns: list[tuple[str, str]] = []
    seen = 0
    for snapshot in ordered:
        values = getattr(snapshot, "values", None) or {}
        notes = values.get("notes") or []
        # ``notes`` is a reducer channel, so it accumulates. Only the new tail
        # belongs to this step.
        fresh = notes[seen:]
        seen = len(notes)
        for note in fresh:
            turns.append(("graph", str(note)))
    return turns


def session_from_checkpoint(
    app: Any,
    config: dict[str, Any],
    *,
    window: int = DEFAULT_WINDOW,
    summarize: Summarizer | None = None,
    history_window: int = 20,
) -> SessionMemory:
    """Hydrate a ``SessionMemory`` from a W4 run, without a second store."""
    session = SessionMemory(window=window, summarize=summarize)
    session.extend(turns_from_checkpoint(app, config, window=history_window))
    return session


def truncating_summarizer(limit: int = 240) -> Summarizer:
    """An offline summariser: first and last line of the folded block.

    Deliberately crude and deliberately honest about it. It compresses by
    construction, which is what makes it useful in tests -- it lets the
    session layer's *behaviour* (does folding happen, does recall work) be
    asserted without paying for a model or pretending a heuristic is a
    summary. Real summarisation is the injected callable's job.
    """

    def summarize(older: Sequence[str], previous: str) -> str:
        if not older:
            return previous
        head = older[0].split(":", 1)[-1].strip()
        tail = older[-1].split(":", 1)[-1].strip()
        body = head if head == tail else f"{head} ... {tail}"
        combined = f"{previous} {body}".strip()
        return combined[-limit:]

    return summarize


__all__ = [
    "DEFAULT_WINDOW",
    "SessionMemory",
    "Summarizer",
    "Turn",
    "session_from_checkpoint",
    "truncating_summarizer",
    "turns_from_checkpoint",
]
