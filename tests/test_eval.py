"""Tests for the evaluation harness itself.

A benchmark that always passes is worse than no benchmark, so these check
that grading rejects the answers it is supposed to reject.
"""

from __future__ import annotations

import unittest

from agentloop.eval import Task, grade, load_tasks


class LoadTests(unittest.TestCase):
    def test_tasks_load_and_are_well_formed(self):
        tasks = load_tasks()
        self.assertGreaterEqual(len(tasks), 10)
        ids = [t.id for t in tasks]
        self.assertEqual(len(ids), len(set(ids)), "task ids must be unique")
        for task in tasks:
            self.assertTrue(task.question.strip(), f"{task.id} has no question")
            self.assertGreater(task.checks, 0, f"{task.id} has no assertions")

    def test_every_category_is_represented(self):
        categories = {t.category for t in load_tasks()}
        self.assertEqual(categories, {"lookup", "multi_hop", "paraphrase", "negative"})


class GradeTests(unittest.TestCase):
    def test_missing_required_text_fails(self):
        task = Task(id="t", category="lookup", question="q", must_contain=["budget"])
        passed, failures = grade(task, "the window is large")
        self.assertFalse(passed)
        self.assertIn("budget", failures[0])

    def test_matching_is_case_insensitive(self):
        task = Task(id="t", category="lookup", question="q", must_contain=["Budget"])
        self.assertTrue(grade(task, "the BUDGET is fixed")[0])

    def test_alternatives_satisfy_a_group(self):
        task = Task(id="t", category="lookup", question="q", must_contain_any=[["vector", "embedding"]])
        self.assertTrue(grade(task, "use embeddings instead")[0])
        self.assertFalse(grade(task, "use grep instead")[0])

    def test_forbidden_text_fails_even_when_required_text_is_present(self):
        task = Task(
            id="t",
            category="negative",
            question="q",
            must_contain=["knowledge base"],
            must_not_contain=["lima"],
        )
        self.assertFalse(grade(task, "Peru's capital, Lima, is not in the knowledge base")[0])

    def test_a_blank_answer_fails_a_negative_task(self):
        """Refusing must be stated, not implied by silence."""
        task = Task(id="t", category="negative", question="q", must_contain_any=[["knowledge base", "no entry"]])
        self.assertFalse(grade(task, "")[0])

    def test_every_negative_task_rejects_the_obvious_hallucination(self):
        cases = {
            "neg-01": "The capital of Peru is Lima.",
            "neg-02": "It was sunny and 18 degrees in Shanghai that day.",
            "neg-03": "Mix flour and water, then let it ferment for 12 hours.",
        }
        tasks = {t.id: t for t in load_tasks()}
        for task_id, hallucination in cases.items():
            with self.subTest(task=task_id):
                self.assertFalse(
                    grade(tasks[task_id], hallucination)[0],
                    f"{task_id} accepted a bare hallucination",
                )


class PromptTests(unittest.TestCase):
    def test_both_variants_exist_and_differ(self):
        from agentloop.prompts import PROMPTS

        self.assertEqual(set(PROMPTS), {"default", "retrieval"})
        self.assertNotEqual(PROMPTS["default"], PROMPTS["retrieval"])

    def test_retrieval_prompt_actually_asks_for_retrieval(self):
        from agentloop.prompts import RETRIEVAL_FIRST

        lowered = RETRIEVAL_FIRST.lower()
        self.assertIn("search the knowledge base", lowered)
        self.assertIn("does not cover", lowered, "must instruct the model to flag a gap")

    def test_unknown_prompt_name_fails_loudly(self):
        from agentloop.prompts import get

        with self.assertRaises(KeyError):
            get("nope")

    def test_a_negative_task_answer_must_label_its_source(self):
        """The retrieval prompt allows a general answer, so grading must still
        require the model to say the knowledge base does not cover it."""
        task = Task(
            id="t",
            category="negative",
            question="What is the capital of Peru?",
            must_contain_any=[["knowledge base", "no entry"]],
        )
        labelled = "The knowledge base does not cover this. From my own knowledge, Lima."
        self.assertTrue(grade(task, labelled)[0])
        unlabelled = "The capital of Peru is Lima."
        self.assertFalse(grade(task, unlabelled)[0])


if __name__ == "__main__":
    unittest.main()