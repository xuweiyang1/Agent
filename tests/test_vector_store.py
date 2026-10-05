"""Tests for the dense store and rank fusion.

Four properties carry the weight, and each is a claim the module makes in its
docstring: the embedder is deterministic across processes, the index honours
the ``VectorStore`` protocol, ties break the way BM25 breaks them, and fusion
rewards agreement between stores rather than one store's raw score scale.

The first is the one worth a dedicated test. ``hash()`` is salted per process,
so an implementation that reached for the builtin would pass every test inside
one run and fail the moment the suite was re-run with a different seed. The
cross-process test is what actually guards that choice.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from retrieval.chunking import Chunk
from retrieval.index import VectorStore, build_index
from retrieval.vector_store import (
    DenseIndex,
    HashingEmbedder,
    HybridIndex,
    build_vector_index,
)
from retrieval.chroma_store import chroma_available

ROOT = Path(__file__).resolve().parent.parent


def chunks(*pairs: tuple[str, str]) -> list[Chunk]:
    return [Chunk(doc_id=doc_id, index=0, text=text, start=0, end=len(text)) for doc_id, text in pairs]


CORPUS = chunks(
    ("compaction", "Compaction keeps the system prompt and the most recent turns, and summarises what it drops."),
    ("retries", "A retryable error is transient: a rate limit or a timeout. Backoff then retries the request."),
    ("passport", "The passport expires in March. Renewal needs a photo, the old passport, and a fee."),
    ("laundry", "The laundry room is on level B1 and takes contactless payment."),
)


class EmbedderTests(unittest.TestCase):
    def test_vector_has_the_declared_width_and_unit_norm(self):
        vector = HashingEmbedder(dim=64)("passport renewal")
        self.assertEqual(len(vector), 64)
        norm = sum(value * value for value in vector) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=9)

    def test_same_text_embeds_identically(self):
        embedder = HashingEmbedder()
        self.assertEqual(embedder("retryable error"), embedder("retryable error"))

    def test_case_and_whitespace_do_not_change_the_vector(self):
        """Indentation is not information; a corpus edit must not move a score."""
        embedder = HashingEmbedder()
        self.assertEqual(embedder("Passport   Renewal\n"), embedder("passport renewal"))

    def test_empty_text_is_a_zero_vector_not_a_crash(self):
        vector = HashingEmbedder(dim=8)("")
        self.assertEqual(vector, [0.0] * 8)

    def test_identical_text_in_two_processes_embeds_the_same(self):
        """The regression guard for hashlib over the salted builtin ``hash``."""
        script = (
            "import sys; sys.path.insert(0, r'%s');"
            "from retrieval.vector_store import HashingEmbedder;"
            "print(sum(HashingEmbedder(dim=32)('passport renewal')));"
        ) % str(ROOT)
        runs = [
            subprocess.run(
                [sys.executable, "-X", "utf8", "-c", script],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])
        self.assertNotEqual(runs[0], "0")


class DenseIndexTests(unittest.TestCase):
    def setUp(self):
        self.index = DenseIndex(CORPUS)

    def test_returns_the_relevant_document_first(self):
        self.assertEqual(self.index.search("passport renewal")[0].doc_id, "passport")
        self.assertEqual(self.index.search("retryable error")[0].doc_id, "retries")

    def test_similarity_descends(self):
        scores = [hit.score for hit in self.index.search("passport renewal")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_low_similarity_hits_are_filtered(self):
        """A dense store should not return the whole corpus for a miss.

        A flat top-k of zero-score neighbours is a hit rate that always reads
        1.0 and a report nobody can trust, so hits must clear zero.
        """
        hits = self.index.search("passport renewal", k=50)
        self.assertLess(len(hits), len(CORPUS))
        self.assertTrue(all(hit.score > 0 for hit in hits))

    def test_k_limits_the_result_count(self):
        self.assertEqual(len(self.index.search("passport renewal", k=2)), 2)
        self.assertEqual(len(self.index.search("passport renewal", k=1)), 1)

    def test_k_of_zero_and_empty_index_return_nothing(self):
        self.assertEqual(self.index.search("passport", k=0), [])
        self.assertEqual(DenseIndex([]).search("passport"), [])
        self.assertEqual(len(DenseIndex([])), 0)

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.index.search(""), [])
        self.assertEqual(self.index.search("   "), [])

    def test_ties_break_deterministically(self):
        tied = chunks(("b", "shared term here"), ("a", "shared term here"))
        first = [h.doc_id for h in DenseIndex(tied).search("shared")]
        second = [h.doc_id for h in DenseIndex(tied).search("shared")]
        self.assertEqual(first, second)
        self.assertEqual(first, ["a", "b"])

    def test_the_same_query_scores_the_same_across_builds(self):
        first = [(h.doc_id, h.score) for h in DenseIndex(CORPUS).search("passport renewal")]
        second = [(h.doc_id, h.score) for h in DenseIndex(CORPUS).search("passport renewal")]
        self.assertEqual(first, second)

    def test_scored_chunk_exposes_provenance(self):
        hit = self.index.search("passport renewal")[0]
        self.assertEqual(hit.doc_id, "passport")
        self.assertIn("score", hit.to_dict())

    def test_an_injected_embedder_is_actually_used(self):
        """The seam has to be real, or the Chroma swap cannot reuse the index."""

        class Fixed:
            dim = 4

            def __call__(self, text: str) -> list[float]:
                return [1.0, 0.0, 0.0, 0.0]

        index = DenseIndex(CORPUS, embedder=Fixed())
        self.assertEqual(index.embedder.dim, 4)
        self.assertEqual(len(index.search("anything", k=2)), 2)


class HybridIndexTests(unittest.TestCase):
    def setUp(self):
        self.bm25 = build_index(CORPUS)
        self.dense = DenseIndex(CORPUS)
        self.hybrid = HybridIndex([self.bm25, self.dense])

    def test_fusion_keeps_the_combined_ranking(self):
        self.assertEqual(self.hybrid.search("passport renewal")[0].doc_id, "passport")

    def test_a_document_both_stores_agree_on_merges_to_the_top(self):
        """RRF rewards agreement, which is the only reason to fuse at all."""
        hits = self.hybrid.search("retryable error backoff")
        self.assertEqual(hits[0].doc_id, "retries")

    def test_fusion_is_order_stable(self):
        first = [(h.doc_id, h.score) for h in self.hybrid.search("passport renewal")]
        second = [(h.doc_id, h.score) for h in self.hybrid.search("passport renewal")]
        self.assertEqual(first, second)

    def test_k_limits_the_result_count(self):
        self.assertEqual(len(self.hybrid.search("passport renewal", k=2)), 2)
        self.assertEqual(self.hybrid.search("passport renewal", k=0), [])

    def test_an_empty_store_list_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            HybridIndex([])

    def test_length_is_the_widest_store(self):
        self.assertEqual(len(self.hybrid), len(CORPUS))


class InterfaceTests(unittest.TestCase):
    def test_dense_index_satisfies_the_vector_store_protocol(self):
        """Same assertion ``test_index`` makes for BM25: the seam is shared."""
        from retrieval.index import VectorStore as Protocol_

        candidate = DenseIndex(CORPUS)
        for name in ("search", "__len__"):
            self.assertTrue(hasattr(candidate, name), f"{name} missing")
        self.assertIs(Protocol_, VectorStore)

    def test_hybrid_index_satisfies_the_vector_store_protocol(self):
        candidate = HybridIndex([build_index(CORPUS), DenseIndex(CORPUS)])
        for name in ("search", "__len__"):
            self.assertTrue(hasattr(candidate, name), f"{name} missing")

    def test_build_vector_index_wraps_the_same_configuration(self):
        self.assertEqual(len(build_vector_index(CORPUS)), len(CORPUS))

    def test_bm25_and_dense_are_both_usable_as_the_retriever_tool_store(self):
        """The retriever takes any ``VectorStore``; dense must slot in unchanged."""
        from retrieval.tools import RetrieverTool

        for store in (build_index(CORPUS), DenseIndex(CORPUS)):
            tool = RetrieverTool(store, default_k=3)
            result = tool.retrieve("passport renewal")
            self.assertEqual(result.citations[0].doc_id, "passport")


if __name__ == "__main__":
    unittest.main()
