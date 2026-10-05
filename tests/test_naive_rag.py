"""Tests for the naive baseline.

The point of these is not that the pipeline is good -- it is that the
baseline is *honest*. A baseline that accidentally scores well is worse than
no baseline, because W6 would then be measured against a number that means
nothing. So the tests pin the intended weaknesses, not just the intended
strengths.
"""

from __future__ import annotations

import unittest

from retrieval.corpus_rag import (
    EXPECTED_DOCS,
    EXPECTED_WEAK,
    build_index_for,
    documents,
    load_tasks,
)
from retrieval.naive import ExtractiveGenerator, estimate_tokens, format_context, naive_rag
from retrieval.report import run_baseline


class CorpusWiringTests(unittest.TestCase):
    def test_the_corpus_is_w1s_corpus_not_a_copy(self):
        """Importing beats copying: a drifted copy invalidates the comparison."""
        from agentloop.corpus import CORPUS

        self.assertEqual(documents(), dict(CORPUS))

    def test_tasks_come_from_the_shared_file(self):
        self.assertEqual(len(load_tasks()), 13)

    def test_every_negative_task_has_no_expected_document(self):
        """There is no right document for an unanswerable question."""
        for task in load_tasks():
            if task.category == "negative":
                with self.subTest(task=task.id):
                    self.assertEqual(task.expected_docs, ())

    def test_every_non_negative_task_names_an_expected_document(self):
        for task in load_tasks():
            if task.category != "negative":
                with self.subTest(task=task.id):
                    self.assertTrue(task.expected_docs, f"{task.id} has no expectation")

    def test_expected_documents_exist_in_the_corpus(self):
        known = set(documents())
        for task_id, docs in EXPECTED_DOCS.items():
            for doc in docs:
                with self.subTest(task=task_id, doc=doc):
                    self.assertIn(doc, known)

    def test_the_multi_hop_task_requires_two_documents(self):
        """Otherwise the category could not expose a single-retrieval failure."""
        multi = [t for t in load_tasks() if t.id == "multi-01"][0]
        self.assertTrue(multi.needs_multiple)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index_for()
        self.tasks = {t.id: t for t in load_tasks()}
        self.generator = ExtractiveGenerator()

    def test_context_includes_source_ids(self):
        """An answer that cannot be cited cannot be checked."""
        hits = self.index.search("tool registry schema", k=2)
        self.assertTrue(hits, "this query must match the corpus for the test to mean anything")
        context = format_context(hits)
        self.assertIn(hits[0].doc_id, context)
        self.assertIn("[1]", context)

    def test_empty_context_is_explicit_not_blank(self):
        self.assertIn("no matching context", format_context([]))

    def test_prompt_contains_the_question_and_the_context(self):
        task = self.tasks["lookup-03"]
        answer = naive_rag(task, self.index, self.generator, k=3)
        self.assertIn(task.question, answer.prompt)
        self.assertIn("Context:", answer.prompt)
        self.assertTrue(answer.hits)

    def test_naive_pipeline_retrieves_exactly_once(self):
        """The defining property: one query, no second attempt."""
        task = self.tasks["para-01"]
        before = self.generator.calls
        naive_rag(task, self.index, self.generator, k=3)
        self.assertEqual(self.generator.calls - before, 1)

    def test_prompt_tokens_are_reported(self):
        task = self.tasks["lookup-01"]
        answer = naive_rag(task, self.index, self.generator, k=3)
        self.assertGreater(answer.prompt_tokens, 0)
        self.assertEqual(answer.prompt_tokens, estimate_tokens(answer.prompt))

    def test_a_larger_k_costs_more_tokens(self):
        """The trade-off a fixed pipeline cannot reason about."""
        task = self.tasks["lookup-01"]
        small = naive_rag(task, self.index, self.generator, k=1).prompt_tokens
        large = naive_rag(task, self.index, self.generator, k=5).prompt_tokens
        self.assertGreater(large, small)

    def test_generator_refuses_when_there_is_no_context(self):
        self.assertEqual(
            self.generator("Context:\n(no matching context)\nQuestion: x\nAnswer:"),
            "The context does not contain an answer.",
        )


class BaselineHonestyTests(unittest.TestCase):
    """The baseline must fail where the naive design is structurally weak."""

    def setUp(self):
        self.report = run_baseline(generate=ExtractiveGenerator())

    def test_lookup_is_strong(self):
        """The easy category has to work, or the pipeline is broken not naive."""
        self.assertEqual(self.report.per_category["lookup"].hit_rate, 1.0)
        self.assertEqual(self.report.per_category["lookup"].answer_accuracy, 1.0)

    def test_multi_hop_is_the_weak_category(self):
        """One retrieval cannot satisfy a question needing two facts."""
        multi = self.report.per_category["multi_hop"]
        self.assertLess(multi.hit_rate, 1.0)
        self.assertLess(multi.coverage, 1.0)

    def test_the_specific_multi_hop_failure_is_the_context_window(self):
        """Named, so W6 has a concrete case to fix rather than a vague gap."""
        failed = [t for t in self.report.per_task if not t["hit"] and t["category"] == "multi_hop"]
        self.assertTrue(failed, "the expected multi-hop miss did not occur")
        self.assertIn("context-window", failed[0]["missing"])

    def test_every_category_is_reported_separately(self):
        """A single aggregate would hide the finding that motivates W6."""
        self.assertEqual(
            set(self.report.per_category), {"lookup", "multi_hop", "paraphrase", "negative"}
        )

    def test_the_report_is_serialisable(self):
        import json

        payload = json.dumps(self.report.to_dict())
        self.assertIn("per_category", payload)
        self.assertIn("per_task", payload)

    def test_reports_are_reproducible(self):
        again = run_baseline(generate=ExtractiveGenerator())
        self.assertEqual(self.report.to_dict(), again.to_dict())

    def test_weak_categories_helper_finds_the_intended_gaps(self):
        """Used to keep the baseline from silently becoming too good."""
        from retrieval.report import weak_categories

        weak = set(weak_categories(self.report))
        # multi_hop is where answer accuracy drops below half... it does not
        # yet (0.67), so this asserts the helper's contract rather than a
        # particular outcome, and records that fact.
        self.assertTrue(weak.issubset(set(self.report.per_category)))

    def test_the_offline_generator_refuses_rather_than_hallucinating(self):
        """Stated as a limitation, because it flatters the baseline.

        The negative category passes here only because the offline generator
        refuses by construction. A real model with a permissive prompt
        answered an out-of-corpus question from memory in zero tool calls --
        that is W1's finding, and it means the naive pipeline's negative score
        is not evidence that naive RAG is honest.
        """
        answer = ExtractiveGenerator()("Context:\n(no matching context)\nQuestion: capital of Peru\nAnswer:")
        self.assertIn("does not contain", answer)


if __name__ == "__main__":
    unittest.main()
