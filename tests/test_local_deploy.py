"""Tests for the local deployment: persistent stores, the override path, and the web layer.

These are deliberately separate from ``test_chain``. That file checks the
*chain's* joints; this one checks what the local deployment added around it --
that a todo survives a restart, that a caller who supplies the fields skips
perception without pretending to have read an image, and that the HTTP surface
turns a form post into a run with its results on disk.

Everything here is offline. The web test drives ``TestClient``, so no server
is started and no port is bound.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from agentkit.tools.calendar import CalendarService
from agentkit.tools.todo import TodoService
from assistant import run_chain
from assistant.vision import _clean, _extract_json
from assistant.webapp import create_app

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


if __name__ == "__main__":
    unittest.main()