"""Tests for the provider adapter, all offline.

The point of these is that the adapter's wire format is pinned. If a
refactor changes how messages or tool calls are encoded, these fail before
a live call would.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentloop import Agent, LLMError, Message, ToolCall, build_default_registry
from agentloop.providers import OpenAICompatibleModel, ReplayTransport
from agentloop.providers.openai_compat import _decode_arguments, _encode_message

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_search.json"


def replay_model(**kwargs) -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        model="deepseek-chat",
        api_key="test-key",
        transport=ReplayTransport(FIXTURE),
        **kwargs,
    )


class EncodingTests(unittest.TestCase):
    def test_plain_message_encodes_without_tool_fields(self):
        encoded = _encode_message(Message(role="user", content="hi"))
        self.assertEqual(encoded, {"role": "user", "content": "hi"})

    def test_tool_message_carries_its_call_id(self):
        encoded = _encode_message(Message(role="tool", content="42", tool_call_id="call_0"))
        self.assertEqual(encoded["role"], "tool")
        self.assertEqual(encoded["tool_call_id"], "call_0")

    def test_assistant_tool_call_is_nested_under_function(self):
        message = Message(role="assistant", tool_calls=[ToolCall("c1", "search", {"query": "x"})])
        encoded = _encode_message(message)
        self.assertIsNone(encoded["content"])
        self.assertEqual(encoded["tool_calls"][0]["type"], "function")
        self.assertEqual(encoded["tool_calls"][0]["function"]["name"], "search")
        self.assertEqual(json.loads(encoded["tool_calls"][0]["function"]["arguments"]), {"query": "x"})

    def test_malformed_tool_arguments_degrade_to_empty_dict(self):
        self.assertEqual(_decode_arguments('{"query": "x"'), {})
        self.assertEqual(_decode_arguments(None), {})
        self.assertEqual(_decode_arguments("[1, 2]"), {})
        self.assertEqual(_decode_arguments('{"a": 1}'), {"a": 1})

    def test_arguments_already_decoded_are_passed_through(self):
        self.assertEqual(_decode_arguments({"a": 1}), {"a": 1})


class ParsingTests(unittest.TestCase):
    def test_response_parses_into_a_tool_call(self):
        model = replay_model()
        reply = model.complete([Message(role="user", content="x")], [])
        self.assertEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0].name, "search")
        self.assertEqual(reply.tool_calls[0].arguments, {"query": "context"})

    def test_usage_is_accumulated_across_calls(self):
        model = replay_model()
        model.complete([Message(role="user", content="x")], [])
        model.complete([Message(role="user", content="x")], [])
        self.assertEqual(model.usage.calls, 2)
        self.assertEqual(model.usage.prompt_tokens, 142 + 268)
        self.assertEqual(model.usage.total, 163 + 342)

    def test_empty_assistant_message_is_a_hard_error(self):
        model = OpenAICompatibleModel(
            api_key="k",
            transport=lambda url, headers, payload: {
                "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}]
            },
        )
        with self.assertRaises(LLMError) as ctx:
            model.complete([Message(role="user", content="x")], [])
        self.assertFalse(ctx.exception.retryable)
        self.assertIn("length", str(ctx.exception))

    def test_response_without_choices_is_a_hard_error(self):
        model = OpenAICompatibleModel(api_key="k", transport=lambda url, headers, payload: {"error": "nope"})
        with self.assertRaises(LLMError) as ctx:
            model.complete([Message(role="user", content="x")], [])
        self.assertFalse(ctx.exception.retryable)

    def test_replay_exhaustion_is_not_retried(self):
        model = replay_model()
        model.complete([Message(role="user", content="x")], [])
        model.complete([Message(role="user", content="x")], [])
        with self.assertRaises(LLMError) as ctx:
            model.complete([Message(role="user", content="x")], [])
        self.assertFalse(ctx.exception.retryable)


class IntegrationTests(unittest.TestCase):
    def test_full_loop_runs_against_replayed_responses(self):
        model = replay_model()
        agent = Agent(model, build_default_registry())
        result = agent.run("How does the context window relate to an agent loop?")

        self.assertEqual(result.turns, 2)
        self.assertEqual(len(result.results), 1)
        self.assertTrue(result.results[0].ok)
        self.assertIn("context window", result.answer)
        self.assertEqual(model.usage.prompt_tokens, 410)

    def test_transport_receives_the_expected_request_shape(self):
        transport = ReplayTransport(FIXTURE)
        model = OpenAICompatibleModel(api_key="k", transport=transport)
        agent = Agent(model, build_default_registry())
        agent.run("How does the context window relate to an agent loop?")

        first = transport.requests[0]
        self.assertEqual(first["model"], "deepseek-chat")
        self.assertEqual([t["function"]["name"] for t in first["tools"]], ["read", "search"])
        self.assertEqual(first["messages"][0]["role"], "system")
        second = transport.requests[1]
        self.assertEqual([m["role"] for m in second["messages"]], ["system", "user", "assistant", "tool"])
        self.assertEqual(second["messages"][-1]["tool_call_id"], "call_0")

    def test_no_authorization_header_leaks_into_the_fixture(self):
        raw = FIXTURE.read_text(encoding="utf-8")
        self.assertNotIn("Authorization", raw)
        self.assertNotIn("Bearer", raw)


class FromEnvTests(unittest.TestCase):
    def test_missing_env_var_is_a_hard_error(self):
        import os

        os.environ.pop("AGENTLOOP_TEST_KEY", None)
        with self.assertRaises(LLMError) as ctx:
            OpenAICompatibleModel.from_env(env_var="AGENTLOOP_TEST_KEY")
        self.assertFalse(ctx.exception.retryable)
        self.assertIn("AGENTLOOP_TEST_KEY", str(ctx.exception))

    def test_env_var_is_picked_up(self):
        import os

        os.environ["AGENTLOOP_TEST_KEY"] = "sk-test"
        try:
            model = OpenAICompatibleModel.from_env(
                env_var="AGENTLOOP_TEST_KEY",
                transport=lambda url, headers, payload: {"choices": [{"message": {"content": "ok"}}]},
            )
            self.assertEqual(model.complete([Message(role="user", content="x")], []).content, "ok")
        finally:
            os.environ.pop("AGENTLOOP_TEST_KEY", None)


if __name__ == "__main__":
    unittest.main()