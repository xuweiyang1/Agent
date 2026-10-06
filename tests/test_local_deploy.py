"""Tests for the local deployment: persistent stores, chat memory, and the web layer.

These are deliberately separate from ``test_chain``. That file checks the
*chain's* joints; this one checks what the local deployment added around it --
that a todo survives a restart, that a supplied read skips perception without
pretending to have read an image, and that the HTTP surface turns a form post
into a run with its results on disk.

The chat tests drive a scripted model, so they exercise the persistence and
memory wiring offline -- no key, no network. They exist because the first
version of this layer had three real defects: the transcript lived only in a
browser variable, long-term memory had no write path, and a todo created in
chat was never persisted. Each has a test here now.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from agentkit.dispatch import ToolInvoker
from agentkit.errors import ErrorKind
from agentkit.messages import ChatMessage
from agentkit.tools import build_registry
from agentkit.tools.calendar import CalendarService
from agentkit.tools.notes import NoteService
from agentkit.tools.todo import TodoService
from agentkit.tools.weather import WeatherService, network_with_stub_fallback
from agentkit.errors import ToolCallError
from agentloop.llm import ToolCall
from assistant import run_chain
from assistant.vision import _clean, _extract_json
from assistant.webapp import create_app
from memory import build_memory

TODAY = date(2026, 10, 5)


class PersistentTodoTests(unittest.TestCase):
    def test_a_todo_survives_a_new_service_over_the_same_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "todos.json"
            first = TodoService(path=path)
            item = first.add("pack a bag")
            second = TodoService(path=path)
            ids = [entry["id"] for entry in second.list_items()]
        self.assertEqual(ids, [item["id"]])
        self.assertEqual(second.list_items()[0]["text"], "pack a bag")

    def test_ids_do_not_collide_after_a_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "todos.json"
            TodoService(path=path).add("one")
            reopened = TodoService(path=path)
            second = reopened.add("two")
        self.assertEqual(second["id"], "t2")

    def test_an_in_memory_service_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = TodoService()
            svc.add("ephemeral")
            leftover = list(Path(tmp).iterdir())
        self.assertEqual(leftover, [])


class PersistentCalendarTests(unittest.TestCase):
    def test_an_event_survives_a_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calendar.json"
            CalendarService(path=path).create("Trip", "2026-10-10T09:00", 60)
            reopened = CalendarService(path=path)
            events = reopened.list_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["title"], "Trip")

    def test_a_corrupt_file_does_not_brick_startup(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calendar.json"
            path.write_text("{not json", encoding="utf-8")
            svc = CalendarService(path=path)
            event = svc.create("Still works", "2026-10-10T09:00", 30)
        self.assertEqual(event["title"], "Still works")


class OverridePathTests(unittest.TestCase):
    """A supplied read must skip perception, and say so."""

    def test_fields_supplied_directly_are_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = asyncio.run(
                run_chain(
                    today=TODAY,
                    artifacts_dir=tmp,
                    state_dir=tmp,
                    perceived_override={
                        "destination": "Lisbon",
                        "budget": "9000 CNY",
                        "note": "museum afternoon",
                    },
                )
            )
        self.assertTrue(run.ok)
        self.assertEqual(run.perceived["destination"], "Lisbon")
        perceive = run.step("perceive")
        self.assertEqual(perceive.inputs.get("source"), "override")
        self.assertIn("no image", perceive.detail)

    def test_an_empty_override_stops_without_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = asyncio.run(
                run_chain(today=TODAY, artifacts_dir=tmp, state_dir=tmp, perceived_override={})
            )
        self.assertFalse(run.ok)
        self.assertIn("could not be read", run.answer)


class WebLayerTests(unittest.TestCase):
    def test_the_form_runs_the_chain_and_returns_a_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(state_dir=tmp, model=None))
            response = client.post(
                "/plan",
                data={
                    "request": "plan a weekend trip",
                    "destination": "Lisbon",
                    "budget": "9000 CNY",
                    "note": "museum afternoon",
                    "today": "2026-10-05",
                    "nights": "2",
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Result", response.text)

    def test_state_endpoint_reads_the_persisted_stores(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(state_dir=tmp, model=None))
            client.post(
                "/plan",
                data={
                    "request": "plan a weekend trip",
                    "destination": "Lisbon",
                    "budget": "9000 CNY",
                    "today": "2026-10-05",
                    "nights": "2",
                },
            )
            body = client.get("/state").json()
        self.assertGreaterEqual(len(body["todos"]), 1)
        self.assertGreaterEqual(len(body["events"]), 1)

    def test_a_photo_without_a_model_is_a_400_with_a_sentence(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(state_dir=tmp, model=None))
            response = client.post(
                "/plan",
                data={"request": "plan a weekend trip", "today": "2026-10-05", "nights": "2"},
                files={"photo": ("x.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("no vision model", response.json()["detail"])

    def test_healthz_reports_that_no_model_is_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(state_dir=tmp, model=None))
            body = client.get("/healthz").json()
        self.assertTrue(body["ok"])
        self.assertIsNone(body["vision"])


class VisionParsingTests(unittest.TestCase):
    """Parsing is defended because a model wraps JSON in prose routinely."""

    def test_json_is_extracted_from_surrounding_prose(self):
        fields = _extract_json('Sure! Here you go:\n{"destination": "Lisbon", "budget": "9000 CNY"}')
        self.assertEqual(fields["destination"], "Lisbon")

    def test_a_non_object_reply_yields_nothing(self):
        self.assertEqual(_extract_json("I cannot read this image."), {})

    def test_unknown_keys_and_empties_are_dropped(self):
        cleaned = _clean({"destination": "Lisbon", "budget": "", "note": "none", "junk": "x"})
        self.assertEqual(cleaned, {"destination": "Lisbon"})


class RunLogTests(unittest.TestCase):
    def test_each_run_writes_a_log_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(state_dir=tmp, model=None))
            client.post(
                "/plan",
                data={
                    "request": "plan a weekend trip",
                    "destination": "Lisbon",
                    "budget": "9000 CNY",
                    "today": "2026-10-05",
                    "nights": "2",
                },
            )
            logs = list((Path(tmp) / "runs").glob("*.json"))
            payload = json.loads(logs[0].read_text(encoding="utf-8"))
        self.assertEqual(len(logs), 1)
        self.assertIn("steps", payload)


# ---------------------------------------------------------------------------
# The chat layer: a scripted model so the persistence path is exercised offline
# ---------------------------------------------------------------------------


class _EchoModel:
    """A stand-in that answers immediately and records the prompts it saw."""

    def __init__(self, answer: str = "好的，我记住了。") -> None:
        self.answer = answer
        self.seen: list[list[ChatMessage]] = []

    async def acomplete(self, messages, tools):
        self.seen.append(list(messages))
        return ChatMessage(role="assistant", content=self.answer)


class _TodoCallingModel:
    """Calls the todo tool on its first turn, then answers."""

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(self, messages, tools):
        self.calls += 1
        if self.calls == 1:
            return ChatMessage(
                role="assistant",
                tool_calls=[ToolCall("c1", "todo", {"action": "add", "text": "买牛奶"})],
            )
        return ChatMessage(role="assistant", content="已记下这条待办。")


class ChatPersistenceTests(unittest.TestCase):
    def test_a_turn_is_written_to_the_conversation_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(state_dir=tmp, model=_EchoModel()))
            client.get("/")  # creates the first conversation
            client.post("/chat", json={"message": "你好"})
            path = Path(tmp) / "conversations" / "c1.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
        roles = [turn["role"] for turn in payload["turns"]]
        self.assertEqual(roles, ["user", "assistant"])
        self.assertEqual(payload["turns"][0]["text"], "你好")
        self.assertEqual(payload["title"], "你好")

    def test_the_page_replays_the_stored_transcript_after_a_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = TestClient(create_app(state_dir=tmp, model=_EchoModel()))
            first.get("/")
            first.post("/chat", json={"message": "我上周去了杭州"})
            # A brand-new app over the same directory stands in for a restart.
            page = TestClient(create_app(state_dir=tmp, model=_EchoModel())).get("/").text
        self.assertIn("我上周去了杭州", page)

    def test_long_term_memory_is_injected_into_the_prompt(self):
        model = _EchoModel()
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, model=model)
            app.state.longterm.remember_preference("seat", "aisle")
            client = TestClient(app)
            client.get("/")
            client.post("/chat", json={"message": "帮我选个座位"})
        system = next(m for m in model.seen[0] if m.role == "system")
        self.assertIn("aisle", system.text())

    def test_a_todo_created_in_chat_lands_in_state_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = TestClient(create_app(state_dir=tmp, model=_TodoCallingModel()))
            client.post("/chat", json={"message": "记一条待办：买牛奶"})
            todos = client.get("/state").json()["todos"]
        self.assertEqual([item["text"] for item in todos], ["买牛奶"])

    def test_clear_empties_the_transcript_but_keeps_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(state_dir=tmp, model=_EchoModel())
            app.state.longterm.remember_preference("hotel", "quiet")
            client = TestClient(app)
            client.get("/")
            client.post("/chat", json={"message": "你好"})
            client.post("/chat/clear", json={"conversation": "c1"})
            turns = json.loads(
                (Path(tmp) / "conversations" / "c1.json").read_text(encoding="utf-8")
            )["turns"]
        self.assertEqual(turns, [])
        self.assertEqual(app.state.longterm.preference("hotel"), "quiet")


class ChatFrontendTests(unittest.TestCase):
    """Guards for the page's own bugs.

    The first version shipped a composer that never sent anything: ``bubble()``
    returned the wrong element, ``firstChild`` was ``null``, and the TypeError
    fired before ``fetch``. Nothing server-side could catch it, so the symptom
    was a spinner forever. These assert the two things that broke -- the
    composer sends, and the tool list is a usable menu rather than decoration.
    """

    def _page(self):
        with tempfile.TemporaryDirectory() as tmp:
            return TestClient(create_app(state_dir=tmp, model=_EchoModel())).get("/").text

    def test_no_leftover_template_placeholders(self):
        page = self._page()
        for placeholder in ("__TOOL_LIST__", "__MEMORY_PILLS__", "__MESSAGES__", "__TOKENS__", "__TURNS__"):
            self.assertNotIn(placeholder, page)

    def test_the_page_has_a_usable_send_path(self):
        page = self._page()
        # The bug was assigning to this null node, which aborted before fetch.
        self.assertNotIn("lastChild.firstChild", page)
        self.assertIn("fetch('/chat'", page)

    def test_tools_are_clickable_and_carry_an_example(self):
        page = self._page()
        self.assertIn('class="tool"', page)
        self.assertIn("data-prompt=", page)
        self.assertIn("dataset.prompt", page.split("<script>")[-1])



def _fallback_over(fetcher):
    """The same rule ``network_with_stub_fallback`` applies, over any fetcher.

    Extracted so the fallback decision -- known city degrades, unknown city
    stays NOT_FOUND -- is testable without reaching the network.
    """
    from agentkit.tools.weather import _BUILTIN, _stub

    async def fallback(city, days):
        try:
            return await fetcher(city, days)
        except ToolCallError as exc:
            base = _BUILTIN.get(city.strip().lower())
            if base is not None and exc.kind in (ErrorKind.UPSTREAM, ErrorKind.TIMEOUT):
                return _stub(city, days, base)
            raise

    return fallback


class ConversationTests(unittest.TestCase):
    """Multiple conversations, which the single ``chat.json`` made impossible."""

    def _client(self, tmp):
        return TestClient(create_app(state_dir=tmp, model=_EchoModel()))

    def test_a_new_conversation_does_not_touch_the_old_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            client.get("/")  # creates c1
            client.post("/chat", json={"message": "第一个话题"})
            second = client.post("/conversations").json()["id"]
            client.post("/chat", json={"message": "第二个话题", "conversation": second})

            first_body = client.get("/?c=c1").text
            second_body = client.get(f"/?c={second}").text
        self.assertIn("第一个话题", first_body)
        self.assertNotIn("第二个话题", first_body.split('id="messages"')[1].split("</main>")[0])
        self.assertIn("第二个话题", second_body)

    def test_a_conversation_is_titled_by_its_first_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            client.get("/")
            client.post("/chat", json={"message": "帮我规划周末"})
            rows = client.get("/conversations").json()["conversations"]
        self.assertEqual(rows[0]["title"], "帮我规划周末")
        self.assertEqual(rows[0]["turns"], 2)

    def test_clearing_one_conversation_leaves_the_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            client.get("/")
            client.post("/chat", json={"message": "甲"})
            second = client.post("/conversations").json()["id"]
            client.post("/chat", json={"message": "乙", "conversation": second})

            client.post("/chat/clear", json={"conversation": "c1"})
            first_body = client.get("/?c=c1").text
            second_body = client.get(f"/?c={second}").text
        self.assertNotIn("甲", first_body.split('id="messages"')[1].split("</main>")[0])
        self.assertIn("乙", second_body)

    def test_deleting_a_conversation_removes_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            client.get("/")
            client.post("/chat", json={"message": "甲"})
            second = client.post("/conversations").json()["id"]
            client.delete("/conversations/c1")
            rows = client.get("/conversations").json()["conversations"]
        self.assertEqual([row["id"] for row in rows], [second])

    def test_a_traversing_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            self.assertEqual(client.get("/?c=..%2F..%2Fetc%2Fpasswd").status_code, 400)

    def test_a_legacy_chat_json_is_imported_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "chat.json"
            legacy.write_text(
                json.dumps({"turns": [{"role": "user", "text": "旧的对话"}]}),
                encoding="utf-8",
            )
            client = self._client(tmp)
            page = client.get("/").text
            imported = (Path(tmp) / "chat.json.imported").is_file()
        self.assertIn("旧的对话", page)
        self.assertTrue(imported)


class WeatherToolTests(unittest.TestCase):
    """The weather tool must answer a Chinese city name, not refuse it.

    The first version matched six hardcoded English names, so 南充 and 成都
    both came back NOT_FOUND and the model padded the refusal into an
    apology. Geocoding is the fix; these pin the seam with an injected
    fetcher so the suite stays offline.
    """

    def _invoke(self, fetcher, city="南充"):
        registry = build_registry(
            with_todos=False, with_calendar=False, with_search=False,
            with_chart=False, with_fx=False, with_notes=False,
            with_weather=True, weather_service=WeatherService(fetcher=fetcher),
        )
        return asyncio.run(ToolInvoker(registry).invoke("weather", {"city": city, "days": 1}))

    def test_a_fetcher_supplies_the_forecast(self):
        async def fetcher(city, days):
            return {"city": city, "days": days, "source": "test",
                    "forecast": [{"day": 1, "condition": "晴", "high": 24, "low": 15}]}

        result = self._invoke(fetcher)
        self.assertTrue(result.ok)
        self.assertEqual(result.value["forecast"][0]["condition"], "晴")

    def test_without_a_fetcher_a_known_city_still_answers_offline(self):
        result = self._invoke(None, city="Shanghai")
        self.assertTrue(result.ok)
        self.assertEqual(result.value["source"], "stub")

    def test_without_a_fetcher_an_unknown_city_is_not_found(self):
        result = self._invoke(None, city="Atlantis")
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.kind, ErrorKind.NOT_FOUND)

    def test_a_network_failure_falls_back_only_for_known_cities(self):
        async def broken(city, days):
            raise ToolCallError("down", kind=ErrorKind.UPSTREAM)

        # ``network_with_stub_fallback`` wraps the live fetcher; composing the
        # same rule over an always-failing fetcher tests the rule without the
        # network.
        fallback = _fallback_over(broken)
        self.assertTrue(self._invoke(fallback, city="Shanghai").ok)
        self.assertFalse(self._invoke(fallback, city="Atlantis").ok)


class MemoryToolTests(unittest.TestCase):
    """W5's store, reachable from the tool layer so the chat can write to it."""

    def _registry(self, tmp):
        longterm, _, _ = build_memory(path=Path(tmp) / "memory.json")
        registry = build_registry(
            with_weather=False,
            with_fx=False,
            with_search=False,
            with_chart=False,
            with_todos=False,
            with_calendar=False,
            with_memory=True,
            longterm=longterm,
        )
        return registry, longterm

    def test_a_preference_is_stored_through_the_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry, longterm = self._registry(tmp)
            result = asyncio.run(
                ToolInvoker(registry).invoke(
                    "memory",
                    {"action": "remember_preference", "key": "seat", "value": "aisle"},
                )
            )
        self.assertTrue(result.ok)
        self.assertEqual(longterm.preference("seat"), "aisle")

    def test_the_store_survives_a_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            first = build_memory(path=path)[0]
            first.remember_fact("The user's dog is named Momo.")
            second = build_memory(path=path)[0]
            hits = second.recall("dog", k=3)
        self.assertTrue(hits)

    def test_storing_a_preference_without_a_value_is_a_bad_arguments_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry, _ = self._registry(tmp)
            result = asyncio.run(
                ToolInvoker(registry).invoke(
                    "memory", {"action": "remember_preference", "key": "seat"}
                )
            )
        self.assertFalse(result.ok)
        self.assertEqual(result.failure.kind, ErrorKind.BAD_ARGUMENTS)


class NoteToolTests(unittest.TestCase):
    def test_a_note_survives_a_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.json"
            NoteService(path=path).add("买牛奶", title="购物")
            reopened = NoteService(path=path)
            notes = reopened.list_notes()
        self.assertEqual(notes[0]["text"], "买牛奶")
        self.assertEqual(notes[0]["title"], "购物")

    def test_the_note_tool_adds_then_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_registry(
                with_weather=False,
                with_fx=False,
                with_search=False,
                with_chart=False,
                with_todos=False,
                with_calendar=False,
                with_notes=True,
                note_service=NoteService(path=Path(tmp) / "notes.json"),
            )
            invoker = ToolInvoker(registry)
            asyncio.run(invoker.invoke("note", {"action": "add", "text": "周五开会"}))
            listed = asyncio.run(invoker.invoke("note", {"action": "list"}))
        self.assertTrue(listed.ok)
        self.assertEqual(listed.value["notes"][0]["text"], "周五开会")


if __name__ == "__main__":
    unittest.main()