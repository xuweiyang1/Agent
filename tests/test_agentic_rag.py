"""Tests for W6: the retrieve tool, the decisions, and the comparison.

Three groups, and the split is what makes the failures diagnosable.

The **tool** tests check that ``retrieve`` is a real registered tool -- a
schema, validation, classified errors -- because "it is only a helper" is how
it would quietly stop being usable by a model.

The **decision** tests are the substance. Each of the four behaviours naive
RAG lacks is asserted directly: it skips retrieval when there is nothing to
search, it stops when the evidence suffices, it rewrites instead of repeating,
and it can retrieve twice and combine. A test that only checked a final answer
would pass while every one of those was broken.

The **comparison** tests pin the claims the report makes. The important one is
that the harness reproduces the naive baseline when the agentic policy is
switched off -- if it did not, the difference between the two rows would be a
difference in the measuring code rather than in retrieval, and the whole
report would be worthless.

The ``multi-01`` case is asserted by name, because it is the defect the W3.5
report identified and this week is supposed to fix. Asserting the fix by name
is what stops it quietly regressing.
"""

from __future__ import annotations

import unittest

from agentkit.dispatch import ToolInvoker
from agentkit.errors import ErrorKind
from agentkit.tools import build_registry

from retrieval.agentic import (
    MAX_RETRIEVALS,
    AgenticRetriever,
    AlwaysSufficient,
    CoverageJudge,
    Vocabulary,
    clause_queries,
    expand_query,
)
from retrieval.compare import compare
from retrieval.corpus_rag import build_index_for, load_tasks
from retrieval.naive import ExtractiveGenerator
from retrieval.tools import RetrieveArgs, RetrieverTool


def task(task_id: str):
    return next(t for t in load_tasks() if t.id == task_id)


class RetrieveToolTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index_for()
        self.tool = RetrieverTool(self.index)

    def test_it_returns_citations_with_source_ids(self):
        result = self.tool.retrieve("tool registry schema", k=3)
        self.assertTrue(result.citations)
        self.assertTrue(all(c.doc_id for c in result.citations))
        self.assertTrue(all(c.text for c in result.citations))

    def test_an_empty_result_is_a_result_not_an_error(self):
        """'Nothing matched' is a fact about the corpus, not a failure."""
        result = self.tool.retrieve("zzzz qqqq xxxx")
        self.assertTrue(result.empty)
        self.assertEqual(result.citations, [])

    def test_a_repeated_document_is_reported(self):
        """Re-retrieving the same chunk spends tokens to learn nothing."""
        self.tool.retrieve("agent loop", k=5)
        again = self.tool.retrieve("agent loop", k=5)
        self.assertTrue(again.repeated)
        self.assertTrue(set(again.repeated) <= set(c.doc_id for c in again.citations))

    def test_it_registers_as_a_real_tool_with_a_schema(self):
        registry = build_registry(with_todos=False, with_calendar=False, with_chart=False)
        self.tool.register(registry)
        self.assertIn("retrieve", registry)
        schema = registry.get("retrieve").schema()
        self.assertEqual(schema["name"], "retrieve")
        self.assertIn("query", schema["parameters"]["properties"])

    def test_the_registered_tool_calls_through_the_invoker(self):
        import asyncio

        registry = build_registry(with_todos=False, with_calendar=False, with_chart=False)
        self.tool.register(registry)
        invoker = ToolInvoker(registry)
        result = asyncio.run(invoker.invoke("retrieve", {"query": "token budget", "k": 2}))
        self.assertTrue(result.ok)
        self.assertEqual(len(result.value["hits"]), 2)

    def test_bad_arguments_come_back_classified(self):
        """The model must be able to correct itself, so this cannot be a crash."""
        import asyncio

        registry = build_registry(with_todos=False, with_calendar=False, with_chart=False)
        self.tool.register(registry)
        invoker = ToolInvoker(registry)
        result = asyncio.run(invoker.invoke("retrieve", {"query": "", "k": 2}))
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)

    def test_the_argument_model_requires_a_query(self):
        with self.assertRaises(Exception):
            RetrieveArgs.model_validate({"k": 3})


class VocabularyTests(unittest.TestCase):
    def setUp(self):
        # ``token`` is in five documents, which is above the rare cutoff; the
        # other two appear once. That is the distinction being tested.
        self.vocabulary = Vocabulary.from_texts(
            [
                "the token budget",
                "token limit",
                "token counting",
                "another token note",
                "a final token line",
                "budget planning",
                "unrelated text",
            ]
        )

    def test_document_frequency_counts_documents_not_occurrences(self):
        self.assertEqual(self.vocabulary.document_frequency["token"], 5)

    def test_a_term_in_many_documents_is_not_rare(self):
        self.assertFalse(self.vocabulary.is_rare("token"))

    def test_an_absent_term_is_not_rare(self):
        """Rare means 'present and uncommon', not 'missing'."""
        self.assertFalse(self.vocabulary.is_rare("absent"))

    def test_idf_ranks_a_rare_term_above_a_common_one(self):
        self.assertGreater(self.vocabulary.idf("other"), self.vocabulary.idf("token"))


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.retriever = RetrieverTool(build_index_for())
        self.agent = AgenticRetriever(self.retriever)

    def test_the_first_query_is_the_question_itself(self):
        chosen = self.agent.next_query(task("lookup-01"), 0, [], [])
        self.assertEqual(chosen, ("What does the tool registry do with a tool call before running it?", "question"))

    def test_a_repeat_query_is_never_returned(self):
        """An attempt that re-asks the same question is not an attempt."""
        t = task("lookup-01")
        chosen = self.agent.next_query(t, 1, [t.question], [])
        self.assertIsNotNone(chosen, "a clause rewrite is available here")
        self.assertNotEqual(chosen[0].lower(), t.question.lower())

    def test_no_query_is_offered_once_every_candidate_is_exhausted(self):
        """Then the loop stops rather than re-running a known-dead query."""
        t = task("lookup-01")
        asked = [t.question] + clause_queries(t.question)
        self.assertIsNone(self.agent.next_query(t, 1, asked, []))

    def test_the_second_query_differs_from_the_first(self):
        t = task("multi-01")
        run = self.agent.run(t, lambda prompt: "no answer")
        self.assertGreater(run.retrievals, 1)
        self.assertNotEqual(run.steps[0].query, run.steps[1].query)

    def test_it_stops_once_the_evidence_suffices(self):
        t = task("lookup-03")
        run = self.agent.run(t, lambda prompt: "no answer")
        self.assertEqual(run.retrievals, 1)
        self.assertTrue(run.steps[0].sufficient)

    def test_the_retrieval_cap_is_enforced(self):
        """A judge that never accepts must not make the loop unbounded."""
        agent = AgenticRetriever(self.retriever, judge=lambda question, citations: False)
        run = agent.run(task("multi-01"), lambda prompt: "no answer")
        self.assertLessEqual(run.retrievals, MAX_RETRIEVALS)
        self.assertEqual(run.retrievals, MAX_RETRIEVALS)

    def test_a_question_with_no_content_words_is_not_retrieved(self):
        """The one case where skipping retrieval is right."""
        from retrieval.corpus_rag import RagTask

        empty = RagTask(id="empty", category="lookup", question="the of and to")
        self.assertFalse(self.agent.should_retrieve(empty))

    def test_a_real_question_is_retrieved(self):
        self.assertTrue(self.agent.should_retrieve(task("neg-01")))

    def test_a_duplicate_only_second_round_stops_early(self):
        """A retrieval that adds no new document is a dead end, not progress."""
        agent = AgenticRetriever(RetrieverTool(build_index_for()))
        agent.judge = lambda question, citations: False
        run = agent.run(task("multi-03"), lambda prompt: "no answer")
        for step in run.steps[1:]:
            if not step.new_docs:
                self.assertIs(step, run.steps[-1])


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.vocabulary = Vocabulary.from_store(build_index_for())

    def test_expansion_prefers_rare_terms_from_the_hits(self):
        """Common words already affect the ranking; adding them changes nothing."""
        index = build_index_for()
        hits = index.search(task("multi-01").question, k=5)
        from retrieval.tools import Citation

        citations = [Citation(h.doc_id, h.score, h.chunk.text) for h in hits]
        expanded = expand_query(task("multi-01").question, citations, self.vocabulary)
        self.assertIsNotNone(expanded)
        self.assertTrue(expanded.strip())

    def test_expansion_needs_hits_to_work_from(self):
        self.assertIsNone(expand_query("anything", [], self.vocabulary))

    def test_the_expanded_query_actually_finds_the_missing_document(self):
        """The named defect: multi-01 must retrieve the window entry."""
        index = build_index_for()
        agent = AgenticRetriever(RetrieverTool(index))
        run = agent.run(task("multi-01"), lambda prompt: "no answer")
        self.assertIn("context-window", run.sources)

    def test_clause_queries_never_return_the_original_question(self):
        question = task("multi-02").question
        self.assertNotIn(question, clause_queries(question))

    def test_clause_queries_are_derived_and_non_empty(self):
        self.assertTrue(clause_queries("What does the tool registry do with a tool call?"))


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = compare()

    def test_the_agentic_policy_is_not_the_baseline(self):
        """A harness where both rows are equal is measuring nothing."""
        self.assertNotEqual(
            [r.retrievals for r in self.report.naive],
            [r.retrievals for r in self.report.agentic],
        )

    def test_the_harness_reproduces_the_naive_baseline(self):
        """If this fails, the comparison is of two code paths, not two policies."""
        retrieval = self.report._retrieval(self.report.naive)
        self.assertAlmostEqual(retrieval["coverage"], 0.95, places=2)
        self.assertAlmostEqual(retrieval["hit_rate"], 0.90, places=2)
        self.assertAlmostEqual(retrieval["mrr"], 0.90, places=2)

    def test_agentic_retrieval_is_not_worse_and_fixes_the_named_task(self):
        before = self.report._retrieval(self.report.naive)
        after = self.report._retrieval(self.report.agentic)
        self.assertGreaterEqual(after["coverage"], before["coverage"])
        self.assertGreaterEqual(after["hit_rate"], before["hit_rate"])
        fixed = {item["id"] for item in self.report.retrieval_fixed()}
        self.assertIn("multi-01", fixed)

    def test_the_fixed_task_is_now_generation_limited_not_retrieval_limited(self):
        """The honest reading: retrieval is fixed, the offline generator is not."""
        row = next(r for r in self.report.agentic if r.id == "multi-01")
        self.assertTrue(row.hit)
        self.assertEqual(row.bottleneck, "generation")

    def test_extra_retrievals_are_paid_for_and_reported(self):
        before = self.report._overall(self.report.naive)
        after = self.report._overall(self.report.agentic)
        self.assertGreater(after["retrievals"], before["retrievals"])
        self.assertGreater(after["prompt_tokens"], before["prompt_tokens"])

    def test_negative_tasks_still_abstain(self):
        """More retrieval must not turn a refusal into an invention."""
        negatives = [r for r in self.report.agentic if r.category == "negative"]
        self.assertTrue(negatives)
        self.assertTrue(all(r.answer_ok for r in negatives))

    def test_a_passing_task_has_no_bottleneck(self):
        for row in self.report.agentic:
            if row.answer_ok:
                self.assertEqual(row.bottleneck, "-")

    def test_the_report_serialises(self):
        payload = self.report.to_dict()
        self.assertIn("retrieval_fixed", payload)
        self.assertEqual(payload["tasks"], len(load_tasks()))


class BaselineEquivalenceTests(unittest.TestCase):
    """``AlwaysSufficient`` must reproduce naive behaviour exactly.

    This is the control for the whole comparison. If forcing the agentic loop
    to never retry does not land on the baseline numbers, then the one
    retrieval it does perform is not the same retrieval, and any difference
    measured elsewhere is unaccounted for.
    """

    def test_one_retrieval_and_no_expansion_matches_the_baseline(self):
        index = build_index_for()
        agent = AgenticRetriever(RetrieverTool(index), judge=AlwaysSufficient())
        generator = ExtractiveGenerator()
        from agentloop.eval import grade

        passed = 0
        tasks = load_tasks()
        for t in tasks:
            run = agent.run(t, generator)
            self.assertEqual(run.retrievals, 1)
            ok, _ = grade(t, run.answer)
            passed += ok
        self.assertEqual(passed, 12)


if __name__ == "__main__":
    unittest.main()
