"""Tests for chunking.

The properties that matter are structural, not textual: no chunk exceeds the
window, consecutive chunks actually overlap, and the provenance needed to
cite a source survives the split. Each of those is a way retrieval fails
later while looking fine here.
"""

from __future__ import annotations

import unittest

from retrieval.chunking import (
    DEFAULT_OVERLAP,
    DEFAULT_SIZE,
    Chunk,
    chunk_corpus,
    chunk_document,
    chunk_text,
)

PARAGRAPHS = (
    "The context window is a hard budget of tokens. Everything shares it.\n\n"
    "Compaction keeps the system prompt and the most recent turns.\n\n"
    "A tool result left without its request is an invalid transcript.\n\n"
    "Retries are for transient failures only.\n\n"
) * 8


class ShapeTests(unittest.TestCase):
    def test_empty_input_yields_no_chunks(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\t "), [])

    def test_short_input_is_a_single_chunk(self):
        chunks = chunk_text("one short sentence", doc_id="d")
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "one short sentence")
        self.assertEqual(chunks[0].index, 0)

    def test_no_chunk_exceeds_the_window(self):
        chunks = chunk_text(PARAGRAPHS, size=200, overlap=40)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk.char_len, 200, f"chunk {chunk.index} is too long")

    def test_consecutive_chunks_overlap(self):
        """Without overlap, a fact split across a boundary matches neither."""
        chunks = chunk_text(PARAGRAPHS, size=200, overlap=40)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertLess(current.start, previous.end, "no overlap between consecutive chunks")

    def test_chunks_without_overlap_are_contiguous(self):
        chunks = chunk_text(PARAGRAPHS, size=200, overlap=0)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(current.start, previous.end)

    def test_indices_are_sequential_from_zero(self):
        chunks = chunk_text(PARAGRAPHS, size=150, overlap=30)
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))

    def test_covers_the_document_without_losing_the_tail(self):
        chunks = chunk_text(PARAGRAPHS, size=180, overlap=20)
        self.assertEqual(chunks[-1].end, len(PARAGRAPHS.strip()))
        self.assertTrue(PARAGRAPHS.strip().endswith(chunks[-1].text[-40:].strip()[-20:]))


class BoundaryTests(unittest.TestCase):
    def test_prefers_a_paragraph_break_over_a_hard_cut(self):
        chunks = chunk_text(PARAGRAPHS, size=200, overlap=20)
        self.assertFalse(any(c.hard_cut for c in chunks), "a text with breaks should never hard-cut")

    def test_splits_a_continuous_run_with_a_hard_cut(self):
        """No whitespace means there is no better boundary to choose."""
        chunks = chunk_text("x" * 500, size=100, overlap=10)
        self.assertTrue(any(c.hard_cut for c in chunks))

    def test_a_break_too_close_to_the_start_is_not_used(self):
        """A boundary near the start would produce a tiny, useless chunk."""
        text = "ab " + ("y" * 400)
        chunks = chunk_text(text, size=100, overlap=20)
        self.assertGreater(chunks[0].char_len, 50)

    def test_strips_surrounding_whitespace_from_chunk_text(self):
        # overlap is passed explicitly: the default (64) is larger than this
        # window, and that combination is rejected rather than silently
        # clamped -- see ValidationTests.
        chunks = chunk_text("   hello world   ", size=50, overlap=0)
        self.assertEqual(chunks[0].text, "hello world")


class ProvenanceTests(unittest.TestCase):
    def test_chunks_carry_their_document_id(self):
        chunks = chunk_document("notes/todo.md", PARAGRAPHS, size=200, overlap=20)
        self.assertTrue(chunks)
        self.assertTrue(all(c.doc_id == "notes/todo.md" for c in chunks))

    def test_start_and_end_point_into_the_original_text(self):
        chunks = chunk_text(PARAGRAPHS, size=200, overlap=20)
        for chunk in chunks:
            self.assertEqual(PARAGRAPHS.strip()[chunk.start:chunk.end].strip(), chunk.text)

    def test_to_dict_is_json_ready(self):
        import json

        chunk = Chunk("d", 0, "text", 0, 4)
        json.dumps(chunk.to_dict())

    def test_corpus_ordering_is_stable_regardless_of_input_order(self):
        """An index that changes with construction order cannot be compared."""
        forward = chunk_corpus({"a": "aaa", "b": "bbb"})
        reverse = chunk_corpus({"b": "bbb", "a": "aaa"})
        self.assertEqual([c.doc_id for c in forward], [c.doc_id for c in reverse])
        self.assertEqual([c.doc_id for c in forward], ["a", "b"])

    def test_corpus_concatenates_every_document(self):
        chunks = chunk_corpus({"a": "short a", "b": "short b"})
        self.assertEqual(len(chunks), 2)
        self.assertEqual({c.doc_id for c in chunks}, {"a", "b"})


class ValidationTests(unittest.TestCase):
    def test_overlap_not_smaller_than_size_is_rejected(self):
        """Otherwise each chunk starts before the last ended and never advances."""
        with self.assertRaises(ValueError):
            chunk_text("some text", size=100, overlap=100)
        with self.assertRaises(ValueError):
            chunk_text("some text", size=100, overlap=150)

    def test_negative_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("some text", overlap=-1)

    def test_non_positive_size_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_text("some text", size=0)

    def test_terminates_on_pathological_input(self):
        """A one-character overlap on repetitive text must still finish."""
        chunks = chunk_text("ab" * 500, size=50, overlap=1)
        self.assertGreater(len(chunks), 1)

    def test_a_window_smaller_than_the_default_overlap_needs_it_overridden(self):
        """The foot-gun: size alone is not enough below the default overlap.

        Silently clamping would hide a misconfiguration, so the combination is
        rejected and the message names both values.
        """
        with self.assertRaises(ValueError) as caught:
            chunk_text("hello world", size=50)
        self.assertIn("64", str(caught.exception))
        self.assertIn("50", str(caught.exception))

    def test_defaults_are_usable(self):
        chunks = chunk_text(PARAGRAPHS)
        self.assertTrue(chunks)
        self.assertLessEqual(max(c.char_len for c in chunks), DEFAULT_SIZE)
        self.assertGreater(DEFAULT_OVERLAP, 0)


if __name__ == "__main__":
    unittest.main()
