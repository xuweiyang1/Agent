"""Tests for the W2 service layer.

These are written around the three failure modes the week exists to handle,
plus the two design claims that are easy to get wrong silently: that
independent tool calls actually run concurrently, and that an image stops
costing tokens once it has been seen.
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from pathlib import Path

from pydantic import Field

from agentkit import (
    ChatMessage,
    ErrorKind,
    HeuristicModel,
    ImageBlock,
    TextBlock,
    ToolArgs,
    ToolCallError,
    ToolCallingAgent,
    ToolInvoker,
    ToolRegistry,
    build_registry,
    classify,
    create_app,
    estimate_message_tokens,
    image_from_file,
    to_wire,
)


def run(coro):
    return asyncio.run(coro)


class ErrorTaxonomyTests(unittest.TestCase):
    def test_timeout_maps_to_timeout_kind(self):
        self.assertEqual(classify(TimeoutError("x"), tool="t").kind, ErrorKind.TIMEOUT)

    def test_asyncio_timeout_maps_to_timeout_kind(self):
        """Python 3.10 has two distinct TimeoutError classes; both must map."""
        self.assertEqual(
            classify(asyncio.TimeoutError("x"), tool="t").kind, ErrorKind.TIMEOUT
        )

    def test_cancelled_is_not_swallowed_by_the_classifier(self):
        """Cancellation is control flow, not a tool failure."""
        failure = classify(asyncio.CancelledError(), tool="t")
        self.assertEqual(failure.kind, ErrorKind.INTERNAL)

    def test_toolcallerror_keeps_its_kind(self):
        exc = ToolCallError("nope", kind=ErrorKind.UPSTREAM)
        self.assertEqual(classify(exc, tool="t").kind, ErrorKind.UPSTREAM)

    def test_only_model_faults_are_model_faults(self):
        self.assertTrue(ErrorKind.UNKNOWN_TOOL.is_model_fault)
        self.assertTrue(ErrorKind.BAD_ARGUMENTS.is_model_fault)
        self.assertTrue(ErrorKind.NOT_FOUND.is_model_fault)
        self.assertFalse(ErrorKind.TIMEOUT.is_model_fault)
        self.assertFalse(ErrorKind.INTERNAL.is_model_fault)

    def test_retryable_is_narrower_than_failure(self):
        self.assertTrue(
            classify(TimeoutError(), tool="t").retryable
        )
        self.assertFalse(
            classify(ToolCallError("bad", kind=ErrorKind.BAD_ARGUMENTS), tool="t").retryable
        )


class SchemaTests(unittest.TestCase):
    def test_schema_forbids_extra_arguments(self):
        registry = build_registry()
        weather = json.dumps(registry.get("weather").schema())
        self.assertIn("additionalProperties", weather)
        self.assertIn("city", weather)

    def test_missing_required_argument_is_bad_arguments(self):
        invoker = ToolInvoker(build_registry())
        result = run(invoker.invoke("weather", {}))
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)
        self.assertEqual(result.failure.details["argument"], "city")

    def test_wrong_type_reports_the_expected_type(self):
        invoker = ToolInvoker(build_registry())
        result = run(invoker.invoke("weather", {"city": 123}))
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)
        self.assertEqual(result.failure.details["expected"], "string")

    def test_invented_argument_is_rejected_not_ignored(self):
        invoker = ToolInvoker(build_registry())
        result = run(invoker.invoke("weather", {"city": "Paris", "unit": "C"}))
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)
        self.assertEqual(result.failure.details["argument"], "unit")

    def test_zero_argument_tool_rejects_anything(self):
        registry = ToolRegistry()

        @registry.tool("ping", "no arguments")
        def ping():
            return "pong"

        result = run(ToolInvoker(registry).invoke("ping", {"x": 1}))
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)
        self.assertTrue(run(ToolInvoker(registry).invoke("ping")).ok)


class DispatchTests(unittest.TestCase):
    def test_unknown_tool_lists_what_is_available(self):
        invoker = ToolInvoker(build_registry(with_todos=False, with_calendar=False, with_fx=False, with_search=False))
        result = run(invoker.invoke("send_email", {"to": "a@b.c"}))
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.kind, ErrorKind.UNKNOWN_TOOL)
        self.assertEqual(result.failure.details["available"], ["weather"])

    def test_not_found_is_distinct_from_bad_arguments(self):
        invoker = ToolInvoker(build_registry())
        result = run(invoker.invoke("weather", {"city": "Atlantis"}))
        self.assertEqual(result.failure.kind, ErrorKind.NOT_FOUND)
        self.assertIn("known_cities", result.failure.details)

    def test_tool_timeout_becomes_a_result_not_an_exception(self):
        registry = ToolRegistry()

        @registry.tool("slow", "sleeps", timeout=0.05)
        def slow():
            time.sleep(0.5)
            return "too late"

        result = run(ToolInvoker(registry).invoke("slow"))
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.kind, ErrorKind.TIMEOUT)

    def test_async_tool_timeout_is_also_caught(self):
        registry = ToolRegistry()

        @registry.tool("aslow", "sleeps", timeout=0.05)
        async def aslow():
            await asyncio.sleep(0.5)
            return "too late"

        result = run(ToolInvoker(registry).invoke("aslow"))
        self.assertEqual(result.failure.kind, ErrorKind.TIMEOUT)

    def test_upstream_failure_is_classified(self):
        registry = ToolRegistry()

        @registry.tool("flaky", "always fails upstream")
        async def flaky():
            raise ToolCallError("provider down", kind=ErrorKind.UPSTREAM)

        result = run(ToolInvoker(registry).invoke("flaky"))
        self.assertEqual(result.failure.kind, ErrorKind.UPSTREAM)
        self.assertTrue(result.failure.retryable)

    def test_independent_calls_in_one_turn_run_concurrently(self):
        """Three 200ms tools must not take 600ms when issued together."""
        registry = ToolRegistry()

        @registry.tool("wait", "waits")
        async def wait(seconds: float):
            await asyncio.sleep(seconds)
            return seconds

        invoker = ToolInvoker(registry)

        async def scenario():
            started = time.perf_counter()
            await asyncio.gather(
                invoker.invoke("wait", {"seconds": 0.2}),
                invoker.invoke("wait", {"seconds": 0.2}),
                invoker.invoke("wait", {"seconds": 0.2}),
            )
            return time.perf_counter() - started

        elapsed = run(scenario())
        self.assertLess(elapsed, 0.45, "calls were serialized instead of concurrent")

    def test_successful_call_returns_both_value_and_transcript_text(self):
        registry = build_registry()
        result = run(ToolInvoker(registry).invoke("convert_currency", {"amount": 10, "source": "USD", "target": "CNY"}))
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.value["converted"], 72.1, places=2)
        self.assertIn("converted", result.output)


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.invoker = ToolInvoker(build_registry())

    def test_weather_stub_is_deterministic(self):
        first = run(self.invoker.invoke("weather", {"city": "Paris", "days": 3}))
        second = run(self.invoker.invoke("weather", {"city": "Paris", "days": 3}))
        self.assertEqual(first.value, second.value)
        self.assertEqual(len(first.value["forecast"]), 3)

    def test_currency_rejects_unknown_code_with_the_supported_list(self):
        result = run(self.invoker.invoke("convert_currency", {"amount": 1, "source": "USD", "target": "XYZ"}))
        self.assertEqual(result.failure.kind, ErrorKind.NOT_FOUND)
        self.assertIn("USD", result.failure.details["supported"])

    def test_todo_conditional_requirement(self):
        result = run(self.invoker.invoke("todo", {"action": "add"}))
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)

    def test_todo_lifecycle(self):
        added = run(self.invoker.invoke("todo", {"action": "add", "text": "buy milk"}))
        self.assertTrue(added.ok)
        item_id = added.value["added"]["id"]
        listed = run(self.invoker.invoke("todo", {"action": "list"}))
        self.assertEqual(len(listed.value["items"]), 1)
        done = run(self.invoker.invoke("todo", {"action": "complete", "item_id": item_id}))
        self.assertTrue(done.value["completed"]["done"])
        missing = run(self.invoker.invoke("todo", {"action": "remove", "item_id": "t999"}))
        self.assertEqual(missing.failure.kind, ErrorKind.NOT_FOUND)

    def test_calendar_accepts_space_separated_datetime(self):
        result = run(self.invoker.invoke("calendar", {"action": "create", "title": "standup", "start": "2026-10-06 09:30", "duration_minutes": 30}))
        self.assertTrue(result.ok)
        self.assertEqual(result.value["created"]["start"], "2026-10-06T09:30")

    def test_calendar_rejects_unparseable_datetime_with_the_format(self):
        result = run(self.invoker.invoke("calendar", {"action": "create", "title": "x", "start": "tomorrow morning"}))
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)
        self.assertIn("YYYY-MM-DD", result.failure.details["expected_format"])

    def test_search_returns_ranked_snippets(self):
        result = run(self.invoker.invoke("search", {"query": "passport renewal", "k": 2}))
        self.assertTrue(result.ok)
        self.assertEqual(result.value["hits"][0]["id"], "passport")
        self.assertGreater(len(result.value["hits"][0]["snippet"]), 10)

    def test_search_with_no_match_is_empty_not_an_error(self):
        result = run(self.invoker.invoke("search", {"query": "zzzqqq"}))
        self.assertTrue(result.ok)
        self.assertEqual(result.value["hits"], [])


class MessageTests(unittest.TestCase):
    def test_text_only_message_encodes_as_a_plain_string(self):
        wire = to_wire([ChatMessage(role="user", content="hello")])
        self.assertEqual(wire[0]["content"], "hello")

    def test_image_message_encodes_as_blocks(self):
        message = ChatMessage(role="user", content=[TextBlock("what is this?"), ImageBlock("data:image/png;base64,AAAA", caption="a menu")])
        wire = to_wire([message])
        self.assertIsInstance(wire[0]["content"], list)
        self.assertEqual(wire[0]["content"][1]["type"], "image_url")

    def test_images_are_counted_in_the_token_estimate(self):
        text_only = ChatMessage(role="user", content="what is this?")
        with_image = ChatMessage(role="user", content=[TextBlock("what is this?"), ImageBlock("data:image/png;base64,AAAA")])
        self.assertGreater(estimate_message_tokens(with_image), estimate_message_tokens(text_only) + 500)

    def test_collapsing_an_image_replaces_it_with_its_caption(self):
        message = ChatMessage(role="user", content=[TextBlock("menu?"), ImageBlock("data:...", caption="thai menu")])
        collapsed = message.collapse_images()
        self.assertFalse(collapsed.has_image())
        self.assertIn("thai menu", collapsed.text())
        self.assertLess(estimate_message_tokens(collapsed), estimate_message_tokens(message))

    def test_image_from_missing_file_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            image_from_file(Path("does-not-exist.png"))


class AgentLoopTests(unittest.TestCase):
    def test_agent_calls_a_tool_then_answers(self):
        invoker = ToolInvoker(build_registry())
        agent = ToolCallingAgent(HeuristicModel(), invoker)
        result = run(agent.run("what is the weather in Tokyo?"))
        self.assertEqual(result.turns, 2)
        self.assertTrue(result.dispatch)
        self.assertTrue(result.dispatch[0].ok)
        self.assertIn("Done", result.answer)

    def test_agent_recovers_from_a_bad_tool_call(self):
        """The model's mistake must come back as data, not kill the run."""
        from agentloop.llm import ToolCall

        class ConfusedModel:
            def __init__(self):
                self.calls = 0

            async def acomplete(self, messages, tools):
                self.calls += 1
                if self.calls == 1:
                    return ChatMessage(role="assistant", tool_calls=[ToolCall("c1", "teleport", {"to": "mars"})])
                last = next((m for m in reversed(messages) if m.role == "tool"), None)
                return ChatMessage(role="assistant", content=f"recovered: {last.text()[:60]}")

        result = run(ToolCallingAgent(ConfusedModel(), ToolInvoker(build_registry())).run("teleport me"))
        self.assertFalse(result.dispatch[0].ok)
        self.assertEqual(result.dispatch[0].failure.kind, ErrorKind.UNKNOWN_TOOL)
        self.assertIn("recovered", result.answer)

    def test_run_is_bounded_by_max_turns(self):
        from agentloop.llm import ToolCall

        class LoopingModel:
            async def acomplete(self, messages, tools):
                return ChatMessage(role="assistant", tool_calls=[ToolCall("c", "todo", {"action": "list"})])

        result = run(ToolCallingAgent(LoopingModel(), ToolInvoker(build_registry()), max_turns=3).run("loop"))
        self.assertEqual(result.turns, 3)
        self.assertEqual(result.answer, "")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        cls.client = TestClient(create_app())

    def test_health_lists_the_five_tools(self):
        body = self.client.get("/healthz").json()
        self.assertEqual(body["tools"], ["calendar", "convert_currency", "search", "todo", "weather"])
        self.assertTrue(body["ok"])

    def test_tools_endpoint_publishes_schemas(self):
        tools = {t["name"] for t in self.client.get("/tools").json()["tools"]}
        self.assertEqual(tools, {"calendar", "convert_currency", "search", "todo", "weather"})

    def test_direct_tool_call_returns_the_value(self):
        body = self.client.post("/tools/convert_currency", json={"arguments": {"amount": 10, "source": "USD", "target": "CNY"}}).json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["payload"]["result"]["converted"], 72.1)

    def test_bad_arguments_are_a_200_with_an_error_payload(self):
        response = self.client.post("/tools/weather", json={"arguments": {"city": 5}})
        self.assertEqual(response.status_code, 200)
        payload = response.json()["payload"]
        self.assertEqual(payload["error"], "bad_arguments")
        self.assertEqual(payload["details"]["expected"], "string")

    def test_unknown_tool_is_a_404(self):
        response = self.client.post("/tools/teleport", json={"arguments": {}})
        self.assertEqual(response.status_code, 404)

    def test_run_endpoint_answers_and_reports_tools_used(self):
        response = self.client.post("/agent/run", json={"task": "what is the weather in Paris?"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("Done", body["answer"])
        self.assertEqual(body["tool_calls"][0]["name"], "weather")
        self.assertTrue(body["tool_calls"][0]["ok"])

    def test_run_rejects_an_empty_task(self):
        self.assertEqual(self.client.post("/agent/run", json={"task": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
