"""Splitting documents into retrievable pieces.

Chunking is where a retrieval system is decided, and it is usually described
as an implementation detail. It is not. A chunk has to be small enough that
the answer inside it is not diluted by unrelated text, and large enough that
the answer is not cut in half. Those pull in opposite directions, so the two
parameters here -- ``size`` and ``overlap`` -- are the real knobs.

Two decisions are deliberate:

- **Overlap exists to protect sentences that straddle a boundary.** Without
  it, a fact split across two chunks matches neither. The cost is duplicate
  text in the index, which is cheap; the alternative is a retrieval failure
  that looks random.

- **Boundaries prefer whitespace, and the fallback is not silent.** Cutting
  mid-word inflates the token count and produces chunks that read as typos.
  When a single run exceeds the window there is nothing to prefer, so the cut
  is hard -- but the chunk records ``hard_cut`` so the cause is visible in a
  result instead of being reconstructed from confusing behaviour.

Provenance is carried on every chunk (``doc_id``, ``start``, ``end``,
``index``) because an answer without a citation is not checkable. W6 will
need to cite; making that possible is cheapest here, at creation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_SIZE = 512
DEFAULT_OVERLAP = 64

# A paragraph break is a better cut than a space: two paragraphs are usually
# two topics, so a boundary between them costs the least meaning.
_BREAKS = ("\n\n", "\n", ". ", "。", "! ", "? ", "; ", " ")


@dataclass(frozen=True)
class Chunk:
    """One retrievable span of text, with the provenance needed to cite it."""

    doc_id: str
    index: int
    text: str
    start: int
    end: int
    hard_cut: bool = False

    @property
    def char_len(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "index": self.index,
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "hard_cut": self.hard_cut,
        }


def _split_point(text: str, limit: int, floor: int) -> tuple[int, bool]:
    """Choose where to cut a window that starts at 0 and must not pass ``limit``.

    Returns the cut offset and whether it was a hard cut. Searches backwards
    for the best break, so the chunk stays as full as possible while ending on
    something that reads as a boundary.
    """
    best = -1
    for break_token in _BREAKS:
        position = text.rfind(break_token, 0, limit)
        if position > floor and position > best:
            best = position + len(break_token)
    if best > 0:
        return best, False
    return limit, True


def chunk_text(
    text: str,
    *,
    doc_id: str = "",
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks of at most ``size`` characters.

    ``overlap`` must be smaller than ``size``. If it is not, each chunk would
    start before the previous one ended and the loop would never advance --
    an infinite loop is a bad way to discover a misconfiguration, so this is
    rejected up front as a plain ``ValueError``.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    body = text.strip()
    if not body:
        return []

    chunks: list[Chunk] = []
    cursor = 0
    length = len(body)

    while cursor < length:
        window_end = min(cursor + size, length)
        remaining = window_end - cursor

        if remaining < size:
            # The tail fits entirely: take it, no boundary search needed.
            cut, hard = remaining, False
        else:
            # A break must leave a useful minimum, or a break near the very
            # start would produce a tiny chunk and defeat the point of a
            # boundary search.
            cut, hard = _split_point(body[cursor:], size, floor=int(size * 0.5))

        piece = body[cursor : cursor + cut]
        if piece.strip():
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    index=len(chunks),
                    text=piece.strip(),
                    start=cursor,
                    end=cursor + cut,
                    hard_cut=hard,
                )
            )

        if cursor + cut >= length:
            break
        # Step forward by the chunk minus the overlap, but never stand still.
        step = max(1, cut - overlap)
        cursor += step

    return chunks


def chunk_document(
    doc_id: str,
    text: str,
    *,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk one document, tagging every piece with its id."""
    return chunk_text(text, doc_id=doc_id, size=size, overlap=overlap)


def chunk_corpus(
    documents: dict[str, str],
    *,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Chunk every document, keeping document order for reproducibility.

    Sorted rather than insertion-ordered: a dict built in a different order
    would otherwise produce a different index, and an index that changes with
    construction order makes a retrieval score impossible to compare across
    runs.
    """
    chunks: list[Chunk] = []
    for doc_id in sorted(documents):
        chunks.extend(chunk_document(doc_id, documents[doc_id], size=size, overlap=overlap))
    return chunks


__all__ = [
    "DEFAULT_OVERLAP",
    "DEFAULT_SIZE",
    "Chunk",
    "chunk_corpus",
    "chunk_document",
    "chunk_text",
]
