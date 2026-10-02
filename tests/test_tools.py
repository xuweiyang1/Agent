"""Tests for the built-in tools and the retrieval ranking.

Two of these cover bugs that a real run against DeepSeek exposed, so they
are regression tests rather than hypothetical ones.
"""

from __future__ import annotations

import unittest

from agentloop import ToolCall, ToolError, build_default_registry
from agentloop.tools import rank_entries, tokenize


class TokenizeTests(unittest.TestCase):
    def test_stopwords_and_single_chars_are_dropped(self):
        self.assertEqual(tokenize("What is an agent loop?"), ["agent", "loop"])

    def test_case_is_normalized(self):
        self.assertEqual(tokenize("Agent LOOP"), ["agent", "loop"])


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.corpus = {
            "backoff": "Exponential backoff retries a transient failure.",
            "context-window": "The context window is a hard budget of tokens.",
        }

    def test_title_hits_outrank_body_hits(self):
        ranked = rank_entries("context window", self.corpus)
        self.assertEqual(ranked[0][0], "context-window")

    def test_unrelated_query_returns_nothing(self):
        self.assertEqual(rank_entries("zzz qqq", self.corpus), [])

    def test_singular_and_plural_agree(self):
        """Regression: the corpus says "retries" and the model asked "retry"."""
        singular = rank_entries("retry", self.corpus)
        plural = rank_entries("retries", self.corpus)
        self.assertEqual([k for k, _ in singular], [k for k, _ in plural])
        self.assertTrue(singular, "the singular form must match the plural in the corpus")


class SearchToolTests(unittest.TestCase):
    def setUp(self):
        self.registry = build_default_registry()

    def _search(self, query):
        import json

        return json.loads(self.registry.call(ToolCall("c", "search", {"query": query})))

    def test_search_returns_a_snippet_not_only_an_id(self):
        """Regression: returning ids only forced a second `read` call."""
        payload = self._search("context window")
        hit = payload["hits"][0]
        self.assertEqual(hit["id"], "context-window")
        self.assertIn("snippet", hit)
        self.assertGreater(len(hit["snippet"]), 20, "snippet must carry real content")

    def test_search_answers_the_question_the_model_actually_asked(self):
        payload = self._search("retryable error")
        self.assertTrue(payload["hits"], "the query that failed live must now match")
        self.assertEqual(payload["hits"][0]["id"], "backoff")

    def test_hits_are_ranked_by_score_descending(self):
        payload = self._search("tool registry")
        scores = [hit["score"] for hit in payload["hits"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], scores[-1])

    def test_natural_language_question_still_matches(self):
        payload = self._search("What is an agent loop?")
        self.assertEqual(payload["hits"][0]["id"], "agent-loop")

    def test_no_match_is_an_empty_list_not_an_error(self):
        self.assertEqual(self._search("zzz qqq")["hits"], [])


class ReadToolTests(unittest.TestCase):
    def test_unknown_id_is_reported(self):
        with self.assertRaises(ToolError):
            build_default_registry().call(ToolCall("c", "read", {"id": "missing"}))

    def test_known_id_returns_full_body(self):
        body = build_default_registry().call(ToolCall("c", "read", {"id": "backoff"}))
        self.assertIn("retryable", body)


if __name__ == "__main__":
    unittest.main()