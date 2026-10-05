"""Tests for the W4 planner.

Split deliberately in two, because the two halves fail for different reasons.

The first half calls the pure functions in ``plan`` directly, with no graph
and no runtime. Those functions hold every decision the planner makes, so
this is where "does a wet forecast change the plan" and "does the budget
actually exclude anything" are answered -- quickly, and with a failure that
points at the function rather than at LangGraph.

The second half drives the compiled graph with the real W2 tools, offline.
It is asserting plumbing, not policy: that the run pauses where it should,
that a rejected plan re-enters the loop with the destination excluded, that a
dead tool degrades into a recorded note instead of an exception, and that
three bad attempts end in an explicit give-up. Those are the failure modes a
graph can have while every pure function above still passes.
"""

from __future__ import annotations

import unittest

from agentkit.dispatch import DispatchResult, ToolInvoker
from agentkit.errors import ErrorKind, ToolFailure
from agentkit.tools import build_registry

from graph import plan
from graph.build import build_context, build_graph
from graph.checkpoint import thread_config
from graph.destinations import CATALOG, as_corpus
from graph.edges import after_approval, after_decide, after_search
from graph.state import initial_state


def hit(name: str, score: float = 1.0) -> dict:
    return {"id": name.lower(), "score": score, "snippet": f"A weekend in {name} is easy to fill."}


class CandidateTests(unittest.TestCase):
    def test_hits_become_candidates(self):
        found = plan.candidates_from_hits([hit("Lisbon"), hit("Tokyo")])
        self.assertEqual([c["name"] for c in found], ["Lisbon", "Tokyo"])

    def test_unknown_destinations_are_dropped(self):
        """A name the catalog cannot price is not a candidate, it is noise."""
        found = plan.candidates_from_hits([hit("Atlantis"), hit("Lisbon")])
        self.assertEqual([c["name"] for c in found], ["Lisbon"])

    def test_the_candidate_cap_is_enforced(self):
        hits = [hit(d.name) for d in CATALOG.values()]
        self.assertEqual(len(plan.candidates_from_hits(hits, limit=2)), 2)


class WeatherTests(unittest.TestCase):
    def test_rain_moves_the_plan_indoors(self):
        verdict = plan.weather_verdict({"forecast": [{"condition": "rain"}]}, days=2)
        self.assertFalse(verdict["outdoor_ok"])

    def test_clear_weather_keeps_the_outdoor_plan(self):
        verdict = plan.weather_verdict({"forecast": [{"condition": "clear"}]}, days=2)
        self.assertTrue(verdict["outdoor_ok"])

    def test_missing_forecast_is_not_bad_weather(self):
        """Fail open: an upstream gap must not silently forbid outdoor plans."""
        verdict = plan.weather_verdict({}, days=2)
        self.assertTrue(verdict["outdoor_ok"])
        self.assertIn("assuming", verdict["summary"])


class PricingTests(unittest.TestCase):
    def test_quotes_are_converted_through_the_fx_callable(self):
        lisbon = CATALOG["lisbon"]
        seen: list[tuple] = []

        def convert(amount: float, source: str, target: str) -> dict:
            seen.append((amount, source, target))
            return {"converted": amount * 7.0, "rate": 7.0}

        quote = plan.quote_destination(lisbon, nights=2, budget_currency="CNY", convert=convert)
        self.assertEqual(seen, [(lisbon.nightly_cost * 2, "EUR", "CNY")])
        self.assertAlmostEqual(quote.budget_cost, lisbon.nightly_cost * 2 * 7.0, places=2)

    def test_build_itinerary_follows_the_forecast(self):
        chosen = {
            "name": "Tokyo",
            "country": "Japan",
            "indoor": ["museum A", "market B"],
            "outdoor": ["shrine C", "park D"],
            "outdoor_ok": False,
            "weather_summary": "rain",
            "cost": 1.0,
            "cost_currency": "CNY",
        }
        built = plan.build_itinerary(chosen, nights=2, request="weekend")
        self.assertTrue(all(day["mode"] == "indoor" for day in built["days"]))

    def test_longer_plans_repeat_activities_instead_of_crashing(self):
        chosen = {
            "name": "Lisbon",
            "country": "Portugal",
            "indoor": ["museum A"],
            "outdoor": ["miradouro B"],
            "outdoor_ok": True,
            "weather_summary": "sunny",
            "cost": 1.0,
            "cost_currency": "CNY",
        }
        built = plan.build_itinerary(chosen, nights=5, request="long weekend")
        self.assertEqual(len(built["days"]), 5)
        self.assertTrue(all(day["morning"] for day in built["days"]))


class ChooseTests(unittest.TestCase):
    def _candidates(self, *names: str) -> list[dict]:
        return [{"name": n, "score": 1.0} for n in names]

    def _weather(self, **by_name: bool) -> dict:
        return {
            name: {"outdoor_ok": ok, "summary": "ok" if ok else "rain"}
            for name, ok in by_name.items()
        }

    def _quote(self, name: str, cost: float) -> plan.Quote:
        return plan.Quote(name, 2, cost, "CNY", cost, "CNY", 1.0)

    def test_a_wet_city_loses_to_a_dry_one(self):
        """The reason to check weather at all is that it changes the answer."""
        chosen = plan.choose(
            self._candidates("Tokyo", "Lisbon"),
            weather=self._weather(tokyo=False, lisbon=True),
            quotes=[self._quote("Tokyo", 100.0), self._quote("Lisbon", 900.0)],
            budget=1000.0,
        )
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen["name"], "Lisbon")

    def test_over_budget_candidates_are_excluded(self):
        chosen = plan.choose(
            self._candidates("Paris"),
            weather=self._weather(paris=True),
            quotes=[self._quote("Paris", 5000.0)],
            budget=100.0,
        )
        self.assertIsNone(chosen)

    def test_excluded_candidates_are_not_reconsidered(self):
        chosen = plan.choose(
            self._candidates("Lisbon", "Beijing"),
            weather=self._weather(lisbon=True, beijing=True),
            quotes=[self._quote("Lisbon", 100.0), self._quote("Beijing", 200.0)],
            budget=1000.0,
            avoid=["Lisbon"],
        )
        self.assertEqual(chosen["name"], "Beijing")

    def test_nothing_viable_returns_none_rather_than_raising(self):
        """"Look again" is a normal outcome, so it routes instead of throwing."""
        self.assertIsNone(plan.choose([], weather={}, quotes=[], budget=1.0))


class EdgeTests(unittest.TestCase):
    def test_search_without_candidates_rewrites_the_query(self):
        state = initial_state("trip")
        state["attempt"] = 1
        self.assertEqual(after_search(state), "broaden")

    def test_the_attempt_cap_becomes_a_give_up(self):
        state = initial_state("trip")
        state["attempt"] = plan.MAX_ATTEMPTS
        self.assertEqual(after_search(state), "give_up")
        self.assertEqual(after_decide(state), "give_up")

    def test_approval_routes_to_the_end(self):
        state = initial_state("trip")
        state["approved"] = True
        self.assertEqual(after_approval(state), "done")

    def test_rejection_routes_back_to_search(self):
        self.assertEqual(after_approval(initial_state("trip")), "broaden")

    def test_queries_change_between_attempts(self):
        self.assertNotEqual(plan.broaden_query(0), plan.broaden_query(1))


class FailingInvoker:
    """An invoker where every tool call fails, to exercise the degrade path."""

    def __init__(self, kind: ErrorKind = ErrorKind.UPSTREAM, *, fail: set[str] | None = None) -> None:
        self.kind = kind
        self.fail = fail
        self.calls: list[str] = []

    async def invoke(self, name, arguments=None, **kwargs) -> DispatchResult:
        self.calls.append(name)
        if self.fail is not None and name not in self.fail:
            return await _real(name, arguments)
        return DispatchResult(
            name=name,
            call_id="",
            ok=False,
            output="",
            failure=ToolFailure(self.kind, "simulated failure", tool=name),
        )


_REAL_INVOKER = ToolInvoker(
    build_registry(search_corpus=as_corpus(), with_todos=False, with_calendar=False, with_chart=False)
)


async def _real(name, arguments) -> DispatchResult:
    return await _REAL_INVOKER.invoke(name, arguments)


class PlannerGraphTests(unittest.IsolatedAsyncioTestCase):
    """The graph itself: pause, resume, degrade, and give up."""

    async def _run(self, *, thread: str, budget: float = 9000.0, context=None, approve=True):
        """Drive one run to its pause, then resume it.

        The pause is only guaranteed on the happy path: a run that exhausts
        its attempts reaches ``give_up`` and ends without ever drafting a
        plan, so asserting a pause here would turn a correct give-up into a
        test failure. The assertion is therefore conditional, and callers that
        specifically care about the gate assert it themselves.
        """
        app = build_graph(context)
        config = thread_config(thread)
        paused = await app.ainvoke(initial_state("plan a sunny weekend", nights=2, budget=budget), config)
        if app.get_state(config).next == ("approve",):
            app.update_state(config, {"approved": approve})
            resumed = await app.ainvoke(None, config)
        else:
            resumed = paused
        return app, config, paused, resumed

    async def test_an_approved_run_finishes_and_cites_its_tools(self):
        _, _, _, state = await self._run(thread="approved")
        self.assertTrue(state["approved"])
        self.assertTrue(state["done"])
        steps = {a["step"] for a in state["attempts"]}
        # search, weather, price and decide all ran: the plan is a function of
        # the tool results, not of a single guess.
        self.assertEqual(steps, {"search", "weather", "price", "decide"})
        self.assertIn("weather", state)
        self.assertIn("quotes", state)

    async def test_the_chosen_destination_is_inside_the_budget(self):
        _, _, _, state = await self._run(thread="budget")
        self.assertLessEqual(state["chosen"]["cost"], 9000.0)

    async def test_the_checkpoint_returns_the_paused_state(self):
        """Resume is the point of checkpointing, so it is asserted directly."""
        _, _, paused, resumed = await self._run(thread="resume")
        self.assertEqual(paused["itinerary"], resumed["itinerary"])
        self.assertEqual(paused["notes"], resumed["notes"][: len(paused["notes"])])

    async def test_state_history_is_recorded(self):
        app, config, _, _ = await self._run(thread="history")
        history = list(app.get_state_history(config))
        self.assertGreater(len(history), 2)
        self.assertIn("search", {task for snapshot in history for task in snapshot.next})

    async def test_a_rejected_plan_re_searches_a_different_city(self):
        """Rejection has to change the next offer, or the gate is decoration."""
        _, _, paused, resumed = await self._run(thread="reject", approve=False)
        first = paused["itinerary"]["destination"]
        self.assertIn(first, resumed.get("avoid", []))
        self.assertFalse(resumed["done"])
        if resumed.get("itinerary"):
            self.assertNotEqual(resumed["itinerary"]["destination"], first)

    async def test_a_budget_nothing_can_meet_ends_in_an_explicit_give_up(self):
        _, _, _, state = await self._run(thread="poor", budget=10.0)
        self.assertTrue(state["done"])
        self.assertIn("gave up", " ".join(state["notes"]))
        self.assertEqual(state["attempt"], plan.MAX_ATTEMPTS)

    async def test_a_dead_search_tool_degrades_instead_of_raising(self):
        context = build_context()
        context.invoker = FailingInvoker()
        _, _, _, state = await self._run(thread="dead", context=context)
        self.assertTrue(state["done"])
        self.assertIn("gave up", " ".join(state["notes"]))
        self.assertTrue(any("search failed" in note for note in state["notes"]))

    async def test_a_dead_weather_tool_drops_that_candidate_only(self):
        """One city without a forecast must not take the whole run down."""
        context = build_context()
        context.invoker = FailingInvoker(fail={"weather"})
        _, _, _, state = await self._run(thread="no-weather", context=context)
        self.assertTrue(any("weather unavailable" in note for note in state["notes"]))
        self.assertEqual(state.get("weather"), {})


if __name__ == "__main__":
    unittest.main()
