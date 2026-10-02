"""Tests for the agent loop.

These are the assertions worth keeping: they pin the behaviours that are
easy to regress and hard to notice, namely retry semantics, error
containment, and context compaction validity.
"""

from __future__ import annotations

import unittest

from agentloop import (
    Agent,
    FlakyModel,
    LLMError,
    Message,
    FakeModel,
    ToolCall,
    ToolError,
    ToolRegistry,
    build_default_registry,
    compact,
    message_tokens,
)


def call_then_answer(tool_name: str, arguments: dict, answer: str = "done"):
    """Script helper: ask for one tool, then answer once it has run."""
    return [
        Message(role="assistant", tool_calls=[ToolCall("c1", tool_name, arguments)]),
        Message(role="assistant", content=answer),
    ]


class LoopTests(unittest.TestCase):
    def test_tool_round_trip(self):
        model = FakeModel(call_then_answer("search", {"query": "agent"}))
        agent = Agent(model, build_default_registry())
        result = agent.run("what is an agent loop?")
        self.assertEqual(result.answer, "done")
        self.assertEqual(result.turns, 2)
        self.assertEqual(len(result.results), 1)
        self.assertTrue(result.results[0].ok)
        self.assertIn("agent-loop", result.results[0].output)

    def test_tool_message_is_linked_to_its_request(self):
        model = FakeModel(call_then_answer("read", {"id": "backoff"}))
        agent = Agent(model, build_default_registry())
        result = agent.run("explain backoff")
        tool_messages = [m for m in result.messages if m.role == "tool"]
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].tool_call_id, "c1")

    def test_unknown_tool_becomes_an_error_message_not_a_crash(self):
        model = FakeModel(call_then_answer("nope", {}))
        agent = Agent(model, build_default_registry())
        result = agent.run("try a bad tool")
        self.assertFalse(result.results[0].ok)
        self.assertIn("unknown tool", result.results[0].output)
        self.assertEqual(result.answer, "done")

    def test_missing_argument_is_reported(self):
        registry = build_default_registry()
        with self.assertRaises(ToolError):
            registry.call(ToolCall("c1", "search", {}))

    def test_wrong_argument_type_is_reported(self):
        registry = build_default_registry()
        with self.assertRaises(ToolError):
            registry.call(ToolCall("c1", "search", {"query": 42}))

    def test_duplicate_registration_is_rejected(self):
        registry = build_default_registry()
        with self.assertRaises(ValueError):
            registry.register(registry._tools["search"])

    def test_max_turns_stops_a_tool_loop(self):
        endless = [Message(role="assistant", tool_calls=[ToolCall(f"c{i}", "search", {"query": "x"})]) for i in range(20)]
        agent = Agent(FakeModel(endless), build_default_registry(), max_turns=3)
        result = agent.run("loop forever")
        self.assertEqual(result.turns, 3)
        self.assertEqual(result.answer, "")


class RetryTests(unittest.TestCase):
    def test_retryable_error_is_retried_with_backoff(self):
        sleeps: list[float] = []
        inner = FakeModel([Message(role="assistant", content="recovered")])
        model = FlakyModel(inner, fail_times=2)
        agent = Agent(model, build_default_registry(), sleep=sleeps.append, base_delay=0.5, seed=1)
        result = agent.run("hello")
        self.assertEqual(result.answer, "recovered")
        self.assertEqual(result.retries, 2)
        self.assertEqual(len(sleeps), 2)
        self.assertLess(sleeps[0], sleeps[1], "backoff must grow")

    def test_non_retryable_error_is_raised_immediately(self):
        inner = FakeModel([Message(role="assistant", content="unused")])
        model = FlakyModel(inner, fail_times=1, retryable=False)
        agent = Agent(model, build_default_registry(), sleep=lambda _: None)
        with self.assertRaises(LLMError):
            agent.run("hello")
        self.assertEqual(model.attempts, 1)

    def test_retries_are_bounded(self):
        inner = FakeModel([Message(role="assistant", content="unused")])
        model = FlakyModel(inner, fail_times=99)
        agent = Agent(model, build_default_registry(), sleep=lambda _: None, max_retries=2)
        with self.assertRaises(LLMError):
            agent.run("hello")
        self.assertEqual(model.attempts, 3)


class CompactionTests(unittest.TestCase):
    def _transcript(self, n: int):
        messages = [Message(role="system", content="system prompt")]
        for i in range(n):
            messages.append(Message(role="user", content=f"question {i} " + "x" * 200))
            messages.append(Message(role="assistant", content=f"answer {i} " + "y" * 200))
        return messages

    def test_small_transcript_is_left_alone(self):
        messages = self._transcript(1)
        outcome = compact(messages, max_tokens=10000)
        self.assertEqual(outcome.dropped, 0)
        self.assertEqual(len(outcome.messages), len(messages))

    def test_large_transcript_is_compacted_and_saves_tokens(self):
        messages = self._transcript(50)
        outcome = compact(messages, max_tokens=300)
        self.assertGreater(outcome.dropped, 0)
        self.assertGreater(outcome.saved, 0)
        self.assertLess(outcome.tokens_after, outcome.tokens_before)

    def test_system_prompt_survives_compaction(self):
        messages = self._transcript(50)
        outcome = compact(messages, max_tokens=300)
        self.assertEqual(outcome.messages[0].content, "system prompt")

    def test_compaction_never_orphans_a_tool_result(self):
        messages = [Message(role="system", content="s")]
        for i in range(30):
            messages.append(Message(role="assistant", tool_calls=[ToolCall(f"c{i}", "search", {"query": "z"})]))
            messages.append(Message(role="tool", content="z" * 400, tool_call_id=f"c{i}"))
        outcome = compact(messages, max_tokens=200, keep_tail=3)
        first_body = next(m for m in outcome.messages if m.role != "system")
        self.assertNotEqual(first_body.role, "tool", "compaction must not start the tail with an orphan tool result")

    def test_agent_compacts_during_a_long_run(self):
        script = [Message(role="assistant", tool_calls=[ToolCall(f"c{i}", "read", {"id": "backoff"})]) for i in range(10)]
        script.append(Message(role="assistant", content="finally"))
        agent = Agent(FakeModel(script), build_default_registry(), max_tokens=250, max_turns=12)
        result = agent.run("long task")
        self.assertGreater(result.compacted, 0)
        self.assertEqual(result.answer, "finally")


class TruncationTests(unittest.TestCase):
    def test_large_tool_output_is_truncated(self):
        registry = ToolRegistry()

        def loud() -> str:
            return "A" * 5000

        from agentloop import Tool

        registry.register(Tool("loud", "returns a lot", {"type": "object", "properties": {}}, loud))
        model = FakeModel([Message(role="assistant", tool_calls=[ToolCall("c1", "loud", {})]), Message(role="assistant", content="ok")])
        agent = Agent(model, registry, max_tool_output_chars=100)
        result = agent.run("be loud")
        output = result.results[0].output
        self.assertLess(len(output), 5000)
        self.assertIn("truncated", output)


if __name__ == "__main__":
    unittest.main()