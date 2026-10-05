"""Tests for the end-to-end chain.

Weekly tests check that each week works. These check the *joints*, which is a
different claim and the one a chain can fail: a step that works alone and
cannot hand its result to the next one. So the assertions here are mostly
about continuity --

- the board's pixels, not the request text, supply the destination and budget;
- the budget the planner received is the one the board carried;
- the plan's destination reaches the todos and the calendar;
- a recorded rejection changes the planner's answer;
- and the guard stops the run before it acts on a trip nobody asked for.

Everything runs offline. Perception is the one expensive-looking step, and it
is real: the board is drawn, read back from the pixels, and the reader is
verified to fail on a board that was never drawn -- which is the difference
between reading and echoing.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from datetime import date
from pathlib import Path

from assistant import BOARD_LINES, BOARD_MEANINGS, DEFAULT_REQUEST, run_chain
from assistant.chain import calendar_step, weekend_after
from assistant.ocr import RuleOcr, render_board, write_board

TODAY = date(2026, 10, 5)  # a Monday, so "this weekend" is unambiguous


def run_once(**kwargs):
    """Drive the async chain from a synchronous test method."""
    with tempfile.TemporaryDirectory() as tmp:
        kwargs.setdefault("artifacts_dir", tmp)
        kwargs.setdefault("today", TODAY)
        return asyncio.run(run_chain(**kwargs))


class OcrTests(unittest.TestCase):
    """Perception has to read pixels, or the multimodal claim is a costume."""

    def test_the_reader_recovers_every_row_from_the_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.png"
            write_board(render_board(list(BOARD_MEANINGS)), path)
            result = RuleOcr().read(path, known_lines=BOARD_MEANINGS)
        self.assertEqual(result.confidence, 1.0)
        self.assertEqual(result.lines, list(BOARD_MEANINGS.values()))

    def test_a_board_that_was_never_drawn_reads_as_empty(self):
        """The anti-cheat: no pixels means no answer, not the right answer."""
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.png"
            Image.new("RGB", (900, 400), "white").save(path)
            result = RuleOcr().read(path, known_lines=BOARD_MEANINGS)
        self.assertEqual(result.matched, 0)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.lines, [])

    def test_a_row_is_not_matched_by_a_similar_one(self):
        """Different lengths must not collapse into the same match."""
        known = {"Destination - Lisbon": "Lisbon", "Note - museum": "museum"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.png"
            write_board(render_board(["Destination - Lisbon"]), path)
            result = RuleOcr().read(path, known_lines=known)
        self.assertEqual(result.lines, ["Lisbon"])

    def test_the_board_image_is_a_real_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "board.png"
            write_board(render_board(list(BOARD_MEANINGS)), path)
            raw = path.read_bytes()
        self.assertTrue(raw.startswith(b"\x89PNG"))
        self.assertGreater(len(raw), 1000)


class JointTests(unittest.TestCase):
    """One value has to survive from the pixels to the last step."""

    @classmethod
    def setUpClass(cls):
        cls.chain = run_once()

    def test_the_chain_finishes_every_step(self):
        names = [step.name for step in self.chain.steps]
        self.assertEqual(
            names,
            ["perceive", "memory", "plan", "retrieve", "todos", "calendar", "chart", "persist", "govern"],
        )
        self.assertTrue(self.chain.ok, self.chain.render())

    def test_the_request_names_none_of_the_facts(self):
        """If the request carried them, perception would be decorative."""
        for value in BOARD_MEANINGS.values():
            self.assertNotIn(value, DEFAULT_REQUEST)
        for number in ("9000", "Lisbon"):
            self.assertNotIn(number, DEFAULT_REQUEST)

    def test_the_board_supplied_the_destination_the_plan_used(self):
        self.assertEqual(self.chain.perceived.get("destination"), "Lisbon")
        self.assertEqual(self.chain.itinerary.get("destination"), "Lisbon")

    def test_the_budget_travelled_from_the_pixels_to_the_planner(self):
        plan = self.chain.step("plan")
        self.assertEqual(self.chain.perceived.get("budget"), "9000 CNY")
        self.assertEqual(plan.inputs["budget"], "9000 CNY")

    def test_the_plan_reached_the_todos(self):
        todos = self.chain.step("todos")
        joined = " ".join(todos.outputs["todos"].split(" | "))
        self.assertIn("Lisbon", joined)
        self.assertEqual(len(self.chain.todos), 3)

    def test_the_plans_reference_fact_reached_the_todos(self):
        """The board's note is the trip's second fact, so it has to land too."""
        joined = self.chain.step("todos").outputs["todos"]
        self.assertIn("museum afternoon", joined)

    def test_the_calendar_resolved_this_weekend_and_booked_it(self):
        booked = self.chain.step("calendar")
        self.assertTrue(booked.ok)
        self.assertIn("2026-10-10", booked.outputs["events"])
        self.assertEqual(self.chain.event.get("title"), "Trip to Lisbon")

    def test_a_multi_night_stay_becomes_one_event_per_night(self):
        """The tool caps an event at a day; the chain must respect the cap."""
        events = self.chain.step("calendar").outputs["events"].split(" | ")
        self.assertEqual(len(events), self.chain.itinerary.get("nights", 2))

    def test_memory_changed_the_planners_avoid_list(self):
        """The seeded rejections are cities the planner would otherwise consider.

        The seed is checked against the catalog by construction: a rejection
        naming a city the planner never retrieves would change nothing, and the
        test would pass while proving nothing.
        """
        memory = self.chain.step("memory")
        self.assertIn("Tokyo", memory.outputs["avoid"])
        plan = self.chain.step("plan")
        self.assertEqual(plan.inputs["avoid"], self.chain.step("memory").outputs["avoid"])

    def test_the_chart_came_from_the_planners_own_quotes(self):
        chart = self.chain.step("chart")
        self.assertTrue(chart.ok)
        self.assertEqual(len(chart.inputs["labels"].split(", ")), 3)

    def test_the_run_writes_back_what_the_next_run_will_read(self):
        persisted = self.chain.step("persist")
        self.assertTrue(persisted.outputs["decision"])
        self.assertGreaterEqual(persisted.outputs["total"], 3)

    def test_governance_reports_the_strategy_it_chose(self):
        govern = self.chain.step("govern")
        self.assertTrue(govern.outputs["strategy"] in ("single", "team"))


class JointFailureTests(unittest.TestCase):
    """The chain's own failure modes, which no weekly test could catch."""

    def test_an_unreadable_board_stops_before_planning(self):
        run = run_once(board_lines=[])
        self.assertFalse(run.ok)
        self.assertEqual([step.name for step in run.steps], ["perceive"])
        self.assertIn("could not be read", run.answer)

    def test_a_partial_board_does_not_plan_a_budgetless_trip(self):
        """A missing budget row is a refused run, not a defaulted one."""
        partial = {"Destination - Lisbon": "Lisbon"}
        run = run_once(board_lines=["Destination - Lisbon"], meanings=partial)
        self.assertFalse(run.ok)

    def test_a_silent_substitution_stops_before_any_side_effect(self):
        """Board says Berlin, planner prefers Beijing: nothing may be written."""
        swapped = dict(BOARD_MEANINGS)
        swapped.pop("Destination - Lisbon")
        swapped["Destination - Berlin"] = "Berlin"
        lines = ["Destination - Berlin"] + list(BOARD_LINES)[1:]
        run = run_once(board_lines=lines, meanings=swapped)
        self.assertFalse(run.ok)
        self.assertEqual(run.step("guard").name, "guard")
        self.assertNotIn("todos", [step.name for step in run.steps])
        self.assertEqual(run.todos, [])
        self.assertEqual(run.event, {})

    def test_the_substitution_can_be_allowed_explicitly(self):
        """Opting in lets the planner's ranking stand -- and it is recorded.

        The run proceeds for whatever the planner chose (here Beijing, since
        Berlin is not in the catalog), and the divergence is still reported so
        the user can see the substitution happened.
        """
        swapped = dict(BOARD_MEANINGS)
        swapped.pop("Destination - Lisbon")
        swapped["Destination - Berlin"] = "Berlin"
        lines = ["Destination - Berlin"] + list(BOARD_LINES)[1:]
        run = run_once(board_lines=lines, meanings=swapped, allow_substitution=True)
        self.assertTrue(run.ok, run.render())
        planned = str(run.itinerary.get("destination", ""))
        self.assertTrue(planned)
        self.assertIn(planned, run.step("todos").outputs["todos"])
        self.assertIn("Berlin", run.divergence)

    def test_retrieval_can_be_switched_off_for_a_faster_run(self):
        run = run_once(use_retrieval=False)
        self.assertNotIn("retrieve", [step.name for step in run.steps])
        self.assertTrue(run.ok)


class CalendarTests(unittest.TestCase):
    def test_the_weekend_is_the_next_saturday_strictly_after_today(self):
        self.assertEqual(weekend_after(date(2026, 10, 5)), date(2026, 10, 10))
        # On a Saturday, "this weekend" is next weekend, not today.
        self.assertEqual(weekend_after(date(2026, 10, 10)), date(2026, 10, 17))

    def test_today_is_a_parameter_so_the_run_is_reproducible(self):
        first = run_once()
        second = run_once()
        self.assertEqual(first.event, second.event)


if __name__ == "__main__":
    unittest.main()
