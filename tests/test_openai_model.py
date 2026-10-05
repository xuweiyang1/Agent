"""Tests for the async model adapter.

These run entirely offline: the transport is a stub that captures the request
and returns a canned response. The point is not to test HTTP, it is to pin the
two things the adapter exists for -- images arriving as content blocks on the
wire, and tool schemas being published byte-for-byte as the registry rendered
them.
"""

from __future__ import annotations

import asyncio
import json
import unittest

from agentkit import ChatMessage, ImageBlock, TextBlock, ToolInvoker, build_registry
from agentkit.openai_model import AsyncOpenAICompatibleModel


def _canned_response(content: str = "ok", tool_calls=None):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


class CapturingTransport:
    """Records the payload and returns a canned body."""

    def __init__(self, body=None):
        self.payloads = []
        self.body = body or _canned_response()

    def __call__(self, url, headers, payload):
        self.payloads.append(payload)
        return self.body


def run(coro):
    return asyncio.run(coro)


class AdapterTests(unittest.TestCase):
    def _model(self, transport):
        return AsyncOpenAICompatibleModel("test-key", transport=transport)

    def test_text_only_message_is_sent_as_a_string(self):
        transport = CapturingTransport()
        model = self._model(transport)
        run(model.acomplete([ChatMessage(role="user", content="hello")], []))
        sent = transport.payloads[0]["messages"][0]["content"]
        self.assertEqual(sent, "hello")

    def test_image_message_is_sent_as_blocks(self):
        """The regression this adapter was written to fix."""
        transport = CapturingTransport()
        model = self._model(transport)
        message = ChatMessage(
            role="user",
            content=[TextBlock("what is this?"), ImageBlock("data:image/png;base64,AAAA")],
        )
        run(model.acomplete([message], []))
        sent = transport.payloads[0]["messages"][0]["content"]
        self.assertIsInstance(sent, list)
        self.assertEqual(sent[0], {"type": "text", "text": "what is this?"})
        self.assertEqual(sent[1]["type"], "image_url")
        self.assertEqual(sent[1]["image_url"]["url"], "data:image/png;base64,AAAA")

    def test_tool_schemas_are_published_as_the_registry_renders_them(self):
        transport = CapturingTransport()
        model = self._model(transport)
        registry = build_registry()
        run(model.acomplete([ChatMessage(role="user", content="hi")], registry.schemas()))
        sent = transport.payloads[0]["tools"]
        self.assertEqual(len(sent), 5)
        names = {t["function"]["name"] for t in sent}
        self.assertEqual(names, set(registry.names()))
        for entry in sent:
            self.assertEqual(entry["type"], "function")
            self.assertFalse(entry["function"]["parameters"]["additionalProperties"])

    def test_tool_calls_are_parsed_into_our_type(self):
        body = _canned_response(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        )
        model = self._model(CapturingTransport(body))
        reply = run(model.acomplete([ChatMessage(role="user", content="weather?")], build_registry().schemas()))
        self.assertEqual(len(reply.tool_calls), 1)
        self.assertEqual(reply.tool_calls[0].name, "weather")
        self.assertEqual(reply.tool_calls[0].arguments, {"city": "Paris"})

    def test_malformed_arguments_do_not_crash_the_adapter(self):
        """A model emitting bad JSON is normal; dispatch reports the bad args."""
        body = _canned_response(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": "{not json"},
                }
            ],
        )
        model = self._model(CapturingTransport(body))
        reply = run(model.acomplete([ChatMessage(role="user", content="x")], []))
        self.assertEqual(reply.tool_calls[0].arguments, {})
        result = run(ToolInvoker(build_registry()).invoke("weather", reply.tool_calls[0].arguments))
        self.assertEqual(result.failure.kind.value, "bad_arguments")

    def test_usage_is_accumulated(self):
        transport = CapturingTransport()
        model = self._model(transport)
        run(model.acomplete([ChatMessage(role="user", content="a")], []))
        run(model.acomplete([ChatMessage(role="user", content="b")], []))
        self.assertEqual(model.usage.calls, 2)
        self.assertEqual(model.usage.total, 30)

    def test_from_env_fails_loudly_without_a_key(self):
        import os

        from agentloop.llm import LLMError

        saved = os.environ.pop("AGENTKIT_TEST_KEY", None)
        try:
            with self.assertRaises(LLMError):
                AsyncOpenAICompatibleModel.from_env(env_var="AGENTKIT_TEST_KEY")
        finally:
            if saved is not None:
                os.environ["AGENTKIT_TEST_KEY"] = saved


if __name__ == "__main__":
    unittest.main()
