"""Tests for the BM25 index.

Three properties carry the weight: ranking is relevance-ordered, scoring is
reproducible across runs, and the stemmer shared with W1 is actually in use.
The last one is a regression guard -- the plain substring bug was fixed once
already and this index is in a position to reintroduce it.
"""

from __future__ import annotations

import unittest

from retrieval.chunking import Chunk
from retrieval.index import BM25Index, build_index


def chunks(*pairs: tuple[str, str]) -> list[Chunk]:
    return [Chunk(doc_id=doc_id, index=0, text=text, start=0, end=len(text)) for doc_id, text in pairs]


CORPUS = chunks(
    ("compaction", "Compaction keeps the system prompt and the most recent turns, and summarises what it drops."),
    ("retries", "A retryable error is transient: a rate limit or a timeout. Backoff then retries the request."),
    ("passport", "The passport expires in March. Renewal needs a photo, the old passport, and a fee."),
    ("laundry", "The laundry room is on level B1 and takes contactless payment."),
)


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index(CORPUS)

    def test_returns_the_relevant_document_first(self):
        self.assertEqual(self.index.search("retryable error")[0].doc_id, "retries")
        self.assertEqual(self.index.search("passport renewal")[0].doc_id, "passport")
        self.assertEqual(self.index.search("laundry payment")[0].doc_id, "laundry")

    def test_scores_descend(self):
        scores = [hit.score for hit in self.index.search("retryable timeout backoff")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_unmatched_query_returns_nothing_not_everything(self):
        """A zero-hit query must be empty; returning the corpus is worse than useless."""
        self.assertEqual(self.index.search("zzzqqqxyz"), [])

    def test_single_char_terms_are_ignored(self):
        self.assertEqual(self.index.search("a"), [])

    def test_k_limits_the_result_count(self):
        # Two terms matching two different documents, so the result set is
        # genuinely larger than one. Zero-score documents are filtered, so a
        # single-term query would not exercise the limit.
        self.assertEqual(len(self.index.search("passport laundry", k=2)), 2)
        self.assertEqual(len(self.index.search("passport laundry", k=1)), 1)
        self.assertEqual(len(self.index.search("passport", k=0)), 0)

    def test_k_larger_than_the_corpus_is_safe(self):
        self.assertLessEqual(len(self.index.search("payment", k=999)), len(CORPUS))

    def test_multiple_query_terms_accumulate(self):
        """Two matching terms must outrank one, or BM25 is not serving relevance."""
        both = self.index.search("passport fee")[0].score
        one = self.index.search("passport")[0].score
        self.assertGreater(both, one)


class StemmingTests(unittest.TestCase):
    def test_singular_query_matches_plural_document(self):
        """Regression: the corpus says 'retries' and a user asks for 'retry'."""
        self.assertEqual(
            [hit.doc_id for hit in build_index(CORPUS).search("retry")],
            ["retries"],
        )

    def test_singular_and_plural_score_identically(self):
        index = build_index(CORPUS)
        self.assertEqual(index.search("retry")[0].score, index.search("retries")[0].score)

    def test_no_substring_false_positive(self):
        """'err' must not match 'error' by accident, which is the W1 lesson."""
        self.assertEqual(build_index(CORPUS).search("err"), [])


class ReproducibilityTests(unittest.TestCase):
    def test_the_same_query_scores_the_same_across_index_builds(self):
        first = build_index(CORPUS).search("passport renewal")
        second = build_index(CORPUS).search("passport renewal")
        self.assertEqual(
            [(h.doc_id, h.score) for h in first], [(h.doc_id, h.score) for h in second]
        )

    def test_ties_break_deterministically(self):
        """Equal scores must not swap places between runs, or diffs show noise."""
        tied = chunks(("b", "shared term here"), ("a", "shared term here"))
        first = [h.doc_id for h in build_index(tied).search("shared")]
        second = [h.doc_id for h in build_index(tied).search("shared")]
        self.assertEqual(first, second)
        self.assertEqual(first, ["a", "b"])

    def test_empty_index_answers_without_crashing(self):
        index = build_index([])
        self.assertEqual(len(index), 0)
        self.assertEqual(index.search("anything"), [])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(build_index(CORPUS).search(""), [])
        self.assertEqual(build_index(CORPUS).search("   "), [])


class ScoringTests(unittest.TestCase):
    def test_common_terms_do_not_go_negative(self):
        """Textbook BM25 goes negative past half the corpus; a common word must
        be worth nothing, not less than nothing."""
        every_doc = chunks(("a", "shared"), ("b", "shared"), ("c", "shared"), ("d", "shared rare"))
        index = build_index(every_doc)
        hits = index.search("shared")
        self.assertTrue(hits)
        self.assertTrue(all(hit.score > 0 for hit in hits))

    def test_rare_terms_outrank_common_ones(self):
        docs = chunks(
            ("a", "the the the the unique"),
            ("b", "the the the the the the the the the the the the common"),
        )
        hits = build_index(docs).search("unique common")
        self.assertEqual(hits[0].doc_id, "a")

    def test_scored_chunk_exposes_provenance(self):
        hit = build_index(CORPUS).search("passport")[0]
        self.assertEqual(hit.doc_id, "passport")
        self.assertIn("score", hit.to_dict())
        self.assertIn("doc_id", hit.to_dict())


class InterfaceTests(unittest.TestCase):
    def test_bm25_satisfies_the_vector_store_protocol(self):
        """The W6 swap depends on this protocol being genuinely small."""
        from retrieval.index import VectorStore

        self.assert_is_protocol_satisfied(build_index(CORPUS), VectorStore)

    def assert_is_protocol_satisfied(self, candidate, protocol):
        for name in ("search", "__len__"):
            self.assertTrue(hasattr(candidate, name), f"{name} missing")

    def test_index_reports_its_size(self):
        self.assertEqual(len(build_index(CORPUS)), 4)


if __name__ == "__main__":
    unittest.main()
