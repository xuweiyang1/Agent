"""Tests for the three memory layers.

The three are tested separately because they fail differently and a single
end-to-end test would hide which one broke. The assertions are about the
properties the layers exist for:

- working memory: the budget holds, and eviction is visible rather than silent
- session memory: the window stays small, recall can still reach what left it,
  and the no-summariser configuration degrades honestly
- long-term memory: a preference read back in a *new object over the same
  file* is the acceptance criterion, because that is what "across sessions"
  means and an in-process dictionary cannot demonstrate it

The last one is why the persistence tests use a real temporary file rather
than a double. Testing persistence with an in-memory store tests nothing.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from graph.build import build_graph
from graph.checkpoint import thread_config
from graph.state import initial_state
from memory import (
    InMemoryStore,
    JsonFileStore,
    session_from_checkpoint,
    LongTermMemory,
    MemoryRecord,
    SemanticStore,
    SessionMemory,
    WorkingMemory,
    build_memory,
    truncating_summarizer,
)
from memory.store import KIND_DECISION, KIND_PREFERENCE


class FixedClock:
    def __init__(self) -> None:
        self.now = "2026-05-04T09:00:00"

    def __call__(self) -> str:
        return self.now


class RecordTests(unittest.TestCase):
    def test_an_unknown_kind_is_rejected(self):
        """A free-form kind is how a preference gets stored as a decision."""
        with self.assertRaises(ValueError):
            MemoryRecord(id="x", kind="vibe", text="something")

    def test_empty_text_is_rejected(self):
        with self.assertRaises(ValueError):
            MemoryRecord(id="x", kind=KIND_PREFERENCE, text="   ")

    def test_round_trips_through_a_dict(self):
        record = MemoryRecord(id="x", kind=KIND_DECISION, text="t", key="k", value={"a": 1})
        self.assertEqual(MemoryRecord.from_dict(record.to_dict()), record)

    def test_unknown_fields_are_ignored_not_fatal(self):
        """A newer version's extra field must not break an older reader."""
        payload = MemoryRecord(id="x", kind=KIND_DECISION, text="t").to_dict()
        payload["future_field"] = "whatever"
        self.assertEqual(MemoryRecord.from_dict(payload).id, "x")


class WorkingMemoryTests(unittest.TestCase):
    def test_token_budget_is_enforced_by_evicting_the_oldest(self):
        memory = WorkingMemory(token_budget=50)
        memory.note("a", "x" * 200)
        memory.note("b", "y" * 200)
        memory.note("c", "short")
        self.assertLessEqual(memory.tokens(), 100)
        self.assertGreater(memory.evicted, 0)
        self.assertEqual(memory.recent()[-1].source, "c")

    def test_one_oversized_entry_is_kept_rather_than_dropped(self):
        """Losing it silently would leave the caller with nothing to explain."""
        memory = WorkingMemory(token_budget=10)
        memory.note("big", "x" * 400)
        self.assertEqual(len(memory), 1)

    def test_clear_resets_the_eviction_count_with_the_entries(self):
        memory = WorkingMemory(token_budget=20)
        memory.note("a", "x" * 200)
        memory.note("b", "y" * 200)
        self.assertGreater(memory.evicted, 0)
        memory.clear()
        self.assertEqual((len(memory), memory.evicted, memory.turn), (0, 0, 1))

    def test_find_filters_by_source(self):
        memory = WorkingMemory()
        memory.note("weather", "clear")
        memory.note("search", "hits")
        self.assertEqual([e.text for e in memory.find("weather")], ["clear"])


class SessionMemoryTests(unittest.TestCase):
    def _filled(self, *, window: int = 3, summarize=None) -> SessionMemory:
        session = SessionMemory(window=window, summarize=summarize)
        for i in range(8):
            session.add("user", f"turn {i}: thinking about option {i}")
            session.add("assistant", f"turn {i}: noted")
        return session

    def test_the_window_keeps_only_the_newest_turns_verbatim(self):
        session = self._filled()
        self.assertEqual([t.index for t in session.recent()], [13, 14, 15])

    def test_recall_reaches_a_turn_the_window_dropped(self):
        """The window is lossy; recall is what makes the loss recoverable."""
        session = self._filled()
        hits = session.recall("option 2")
        self.assertTrue(hits)
        found = session.turn_for(hits[0].doc_id)
        self.assertIsNotNone(found)
        self.assertLess(found.index, 13)

    def test_recall_works_even_without_a_summariser(self):
        """Eviction must not depend on summarisation, or recall disappears."""
        session = self._filled(summarize=None)
        self.assertTrue(session.recall("option 2"))
        self.assertEqual(session.summary, "")

    def test_the_summary_is_bounded_and_reported(self):
        session = self._filled(summarize=truncating_summarizer())
        self.assertTrue(session.summary)
        self.assertLess(session.compression, 1.0)

    def test_the_prompt_puts_the_stable_summary_before_the_live_window(self):
        """The prompt-caching rule: stable prefix first, volatile last."""
        session = self._filled(summarize=truncating_summarizer())
        prompt = session.prompt()
        newest = session.recent()[-1].text
        self.assertLess(prompt.index("Earlier in this conversation"), prompt.index(newest))

    def test_older_turns_are_not_summarised_twice(self):
        """Re-summarising the history each turn is the growth being prevented."""
        session = SessionMemory(window=2, summarize=truncating_summarizer())
        for i in range(6):
            session.add("user", f"turn {i}")
        first = len(session.recall("turn 0"))
        session.add("user", "turn 6")
        session.add("user", "turn 7")
        self.assertEqual(len(session.recall("turn 0")), first)


class CheckpointBridgeTests(unittest.IsolatedAsyncioTestCase):
    """Session memory has to be the W4 run, not a second copy of it.

    This is the acceptance criterion "session memory reuses the W4
    checkpoint". A test that only hydrated a hand-built list would pass while
    the actual wiring was wrong, so these run the real graph first.
    """

    async def _finished_run(self, thread: str):
        app = build_graph()
        config = thread_config(thread)
        await app.ainvoke(initial_state("plan a sunny weekend trip", nights=2, budget=9000.0), config)
        app.update_state(config, {"approved": True})
        await app.ainvoke(None, config)
        return app, config

    async def test_a_finished_run_hydrates_into_session_memory(self):
        app, config = await self._finished_run("bridge-basic")
        session = session_from_checkpoint(app, config, window=3)
        self.assertGreater(len(session), 0)
        self.assertLessEqual(len(session.recent()), 3)

    async def test_the_checkpoint_record_is_the_graphs_own_notes(self):
        """Nothing is re-derived: the notes are what the run actually learned."""
        app, config = await self._finished_run("bridge-notes")
        session = session_from_checkpoint(app, config, window=3)
        text = " ".join(t.text for t in session.turns)
        self.assertIn("chose", text)

    async def test_recall_reaches_a_note_the_window_dropped(self):
        app, config = await self._finished_run("bridge-recall")
        session = session_from_checkpoint(app, config, window=2, summarize=truncating_summarizer())
        hits = session.recall("forecast weather")
        self.assertTrue(hits)
        found = session.turn_for(hits[0].doc_id)
        self.assertIsNotNone(found)

    async def test_an_empty_thread_hydrates_to_an_empty_session(self):
        app = build_graph()
        session = session_from_checkpoint(app, thread_config("bridge-never-run"))
        self.assertEqual(len(session), 0)


class LongTermMemoryTests(unittest.TestCase):
    def _memory(self, path: Path) -> LongTermMemory:
        longterm, _, _ = build_memory(path=path, clock=FixedClock())
        return longterm

    def test_a_preference_survives_a_new_object_over_the_same_file(self):
        """The acceptance criterion: this is what "across sessions" means."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            self._memory(path).remember_preference("seat", "aisle")

            reopened = self._memory(path)
            self.assertEqual(reopened.preference("seat"), "aisle")

    def test_a_decision_survives_with_its_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            self._memory(path).remember_decision("Lisbon", "rejected", reason="over budget")

            reopened = self._memory(path)
            self.assertEqual(reopened.decisions("Lisbon")[0].value["reason"], "over budget")

    def test_setting_a_preference_twice_replaces_it(self):
        """Two live answers to one question would make order the policy."""
        with tempfile.TemporaryDirectory() as tmp:
            longterm = self._memory(Path(tmp) / "memory.json")
            longterm.remember_preference("seat", "aisle")
            longterm.remember_preference("seat", "window")
            self.assertEqual(longterm.preference("seat"), "window")
            self.assertEqual(len(longterm.structured.find(kind=KIND_PREFERENCE, key="seat")), 1)

    def test_a_replaced_preference_leaves_no_semantic_orphan(self):
        """Otherwise recall returns a value that lookup has already retired."""
        with tempfile.TemporaryDirectory() as tmp:
            longterm = self._memory(Path(tmp) / "memory.json")
            longterm.remember_preference("seat", "aisle")
            longterm.remember_preference("seat", "window")
            self.assertEqual(longterm.recall("aisle", mark_used=False), [])

    def test_a_missing_preference_returns_the_default_not_a_near_match(self):
        """``seating`` must not surface the value stored under ``seat``."""
        with tempfile.TemporaryDirectory() as tmp:
            longterm = self._memory(Path(tmp) / "memory.json")
            longterm.remember_preference("seat", "aisle")
            self.assertEqual(longterm.preference("seating", "fallback"), "fallback")
            self.assertEqual(longterm.preference("seat"), "aisle")

    def test_recall_count_is_recorded_and_can_be_opted_out_of(self):
        with tempfile.TemporaryDirectory() as tmp:
            longterm = self._memory(Path(tmp) / "memory.json")
            record = longterm.remember_fact("Liked the Time Out Market food hall.")
            longterm.recall("food hall", mark_used=False)
            self.assertIsNone(record.metadata.get("uses"))

            longterm.recall("food hall")
            used = next(r for r in longterm.structured.all() if r.id == record.id)
            self.assertEqual(used.metadata["uses"], 1)

    def test_semantic_recall_satisfies_the_retrieval_protocol(self):
        """W6 reuses this as a retriever, so the seam has to be the same one."""
        with tempfile.TemporaryDirectory() as tmp:
            longterm = self._memory(Path(tmp) / "memory.json")
            longterm.remember_fact("The quiet hotel was away from the nightlife.")
            hits = longterm.recall("quiet hotel", mark_used=False)
            self.assertTrue(hits)
            self.assertTrue(hasattr(hits[0], "doc_id"))

    def test_context_for_labels_constraints_and_hints_differently(self):
        with tempfile.TemporaryDirectory() as tmp:
            longterm = self._memory(Path(tmp) / "memory.json")
            longterm.remember_preference("seat", "aisle")
            longterm.remember_fact("An old note about a food market.")
            block = longterm.context_for("food market")
            self.assertIn("Known preferences:", block)
            self.assertIn("Maybe relevant from earlier:", block)

    def test_a_corrupt_file_fails_loudly(self):
        """Silent amnesia is worse than a visible error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(ValueError):
                JsonFileStore(path)

    def test_the_file_is_written_atomically_and_readable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.json"
            longterm = self._memory(path)
            longterm.remember_preference("seat", "aisle")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([item["key"] for item in payload], ["seat"])


class StoreTests(unittest.TestCase):
    def test_the_semantic_store_finds_a_record_added_after_construction(self):
        """A write must be visible to the next search, not the next rebuild."""
        store = SemanticStore()
        store.add(MemoryRecord(id="r1", kind="fact", text="The passport expires in March."))
        self.assertTrue(store.search("passport"))
        self.assertEqual(store.record_of(store.search("passport")[0].doc_id).id, "r1")

    def test_delete_removes_from_both_sides(self):
        structured = InMemoryStore()
        semantic = SemanticStore()
        record = MemoryRecord(id="r1", kind="fact", text="something borrowed")
        structured.put(record)
        semantic.add(record)
        self.assertTrue(structured.delete("r1"))
        self.assertTrue(semantic.delete("r1"))
        self.assertEqual(len(structured), 0)
        self.assertEqual(semantic.search("borrowed"), [])


if __name__ == "__main__":
    unittest.main()
