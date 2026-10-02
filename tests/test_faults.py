"""Tests for the fault-injecting transport.

These are offline: the injected transport wraps a stub, so the retry and
backoff paths run in the test suite without a network or a key.
"""

from __future__ import annotations

import unittest

from agentloop import Agent, build_default_registry
from agentloop.llm import LLMError
from agentloop.providers import OpenAICompatibleModel, ReplayTransport
from agentloop.providers.faults import FaultInjectingTransport

FIXTURE = "tests/fixtures/deepseek_search.json"


def reply(content: str = "done") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class FaultTests(unittest.TestCase):
    def test_no_faults_means_no_injection(self):
        transport = FaultInjectingTransport(lambda u, h, p: reply(), fail_first=0)
        model = OpenAICompatibleModel(api_key="k", transport=transport)
        self.assertEqual(model.complete([], []).content, "done")
        self.assertEqual(transport.injected, 0)

    def test_failures_are_injected_then_recovered(self):
        transport = FaultInjectingTransport(lambda u, h, p: reply(), fail_first=2)
        model = OpenAICompatibleModel(api_key="k", transport=transport)
        agent = Agent(model, build_default_registry(), sleep=lambda _: None)
        result = agent.run("q")
        self.assertEqual(result.answer, "done")
        self.assertEqual(transport.injected, 2)
        self.assertEqual(result.retries, 2)

    def test_a_per_call_rate_keeps_failing_throughout_the_run(self):
        """A front-loaded fault only tests the first call; this tests the loop."""
        transport = FaultInjectingTransport(lambda u, h, p: reply(), fail_first=0, per_call=2)
        model = OpenAICompatibleModel(api_key="k", transport=transport)
        agent = Agent(model, build_default_registry(), sleep=lambda _: None, max_retries=5)
        for _ in range(3):
            agent.run("q")
        self.assertGreaterEqual(transport.injected, 2, "failures must recur, not only at the start")

    def test_a_non_retryable_fault_is_not_retried(self):
        transport = FaultInjectingTransport(lambda u, h, p: reply(), fail_first=99, retryable=False)
        model = OpenAICompatibleModel(api_key="k", transport=transport)
        agent = Agent(model, build_default_registry(), sleep=lambda _: None)
        with self.assertRaises(LLMError):
            agent.run("q")
        self.assertEqual(transport.calls, 1, "a non-retryable failure must stop immediately")

    def test_recovered_counter_tracks_success_after_a_fault(self):
        transport = FaultInjectingTransport(lambda u, h, p: reply(), fail_first=1)
        model = OpenAICompatibleModel(api_key="k", transport=transport)
        Agent(model, build_default_registry(), sleep=lambda _: None).run("q")
        self.assertEqual(transport.recovered, 1)

    def test_the_retry_path_runs_against_a_recorded_fixture(self):
        """The whole point: exercise retry with real response shapes, offline."""
        inner = ReplayTransport(FIXTURE)
        transport = FaultInjectingTransport(inner, fail_first=1)
        model = OpenAICompatibleModel(api_key="k", transport=transport)
        agent = Agent(model, build_default_registry(), sleep=lambda _: None)
        result = agent.run("How does the context window relate to an agent loop?")
        self.assertIn("context window", result.answer)
        self.assertEqual(transport.injected, 1)


if __name__ == "__main__":
    unittest.main()