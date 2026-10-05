"""The graph nodes: each one reads state, calls one decision, returns a delta.

Every method here is deliberately a few lines. The check that keeps it honest
is: if a node contains an ``if`` that decides something about the *trip*, that
``if`` belongs in ``plan.py``. What a node is allowed to contain is the
plumbing -- calling a tool, raising a failure into a note, shaping a return
value -- because that is the part that genuinely depends on the runtime.

Two consequences worth naming, because they are the interview answers:

- **Tools are called through ``ToolInvoker``, not directly.** Timeout, retry
  classification and the "unknown tool" case were solved in W2; a node that
  called ``weather()`` itself would re-solve them worse. The graph is a new
  scheduler over old capabilities, not a new stack.
- **A failed tool call is a note, not an exception.** A planner that dies
  when one city has no forecast cannot plan around the outage, and planning
  around outages is most of what a planner does. Every failure becomes a
  recorded note and a missing piece of data, and ``choose`` simply does not
  consider that candidate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentkit.dispatch import DispatchResult, ToolInvoker
from langgraph.types import interrupt

from . import plan
from .destinations import CATALOG
from .state import TravelState


@dataclass
class PlannerContext:
    """What the nodes are allowed to reach for.

    Kept as one object so a test can build the graph against a fake invoker
    without patching module globals, and so the node methods stay methods of
    something a reader can name.
    """

    invoker: ToolInvoker
    hits_per_query: int = 3
    notes: list[str] = field(default_factory=list)

    async def call(self, tool: str, arguments: dict[str, Any]) -> DispatchResult:
        return await self.invoker.invoke(tool, arguments)


class Planner:
    """The node bodies. ``build.py`` wires these into the StateGraph."""

    def __init__(self, context: PlannerContext) -> None:
        self.ctx = context

    # -- search ------------------------------------------------------------

    async def search(self, state: TravelState) -> dict[str, Any]:
        """Query for destinations, and remember what this attempt asked.

        The query is carried in the state rather than recomputed, because on
        the broaden branch the next attempt needs to know which query already
        failed. Recomputing it from the request would silently re-run the same
        search, which is the bug the branch exists to avoid.
        """
        query = state.get("query") or state["request"]
        result = await self.ctx.call("search", {"query": query, "k": self.ctx.hits_per_query})

        if not result.ok:
            note = f"search failed ({result.failure.kind.value if result.failure else 'error'}); trying a broader query"
            return {
                "candidates": [],
                "notes": [note],
                "attempts": [{"step": "search", "query": query, "n": 0}],
                "attempt": state.get("attempt", 0) + 1,
            }

        hits = (result.value or {}).get("hits", []) if isinstance(result.value, dict) else []
        candidates = plan.candidates_from_hits(hits)
        note = f"search {query!r} -> {len(candidates)} usable candidate(s) from {len(hits)} hit(s)"
        return {
            "candidates": candidates,
            "notes": [note],
            "attempts": [{"step": "search", "query": query, "n": len(candidates)}],
            "attempt": state.get("attempt", 0) + 1,
        }

    # -- weather -----------------------------------------------------------

    async def weather(self, state: TravelState) -> dict[str, Any]:
        """Check the forecast for the most promising candidates.

        Only the top ``MAX_WEATHER_CALLS`` are checked. Fetching all of them
        would be more thorough and is the wrong trade: the cap is what bounds
        a planning run's cost, and a candidate that loses on the first three
        forecasts was not going to win on the fourth.
        """
        candidates = state.get("candidates", [])[: plan.MAX_WEATHER_CALLS]
        forecasts: dict[str, Any] = {}
        notes: list[str] = []

        for candidate in candidates:
            name = str(candidate.get("name", ""))
            result = await self.ctx.call("weather", {"city": name, "days": int(state.get("nights", 2))})
            if not result.ok:
                notes.append(f"weather unavailable for {name}; that candidate cannot be judged")
                continue
            verdict = plan.weather_verdict(result.value or {}, days=state.get("nights", 2))
            forecasts[name.lower()] = verdict
            notes.append(f"{name}: {verdict['summary']}")

        outdoor_ok = all(v.get("outdoor_ok", True) for v in forecasts.values()) if forecasts else True
        return {
            "weather": forecasts,
            "outdoor_ok": outdoor_ok,
            "notes": notes,
            "attempts": [{"step": "weather", "checked": sorted(forecasts), "outdoor_ok": outdoor_ok}],
        }

    # -- price -------------------------------------------------------------

    async def price(self, state: TravelState) -> dict[str, Any]:
        """Convert each candidate's cost into the budget currency.

        A destination whose conversion fails is dropped from ``quotes`` but
        kept in the notes: the difference between "too expensive" and "the
        rate table does not know this currency" matters to whoever debugs the
        plan, and collapsing both into "skipped" throws that away.
        """
        nights = state.get("nights", 2)
        budget_currency = state.get("currency", "CNY")
        forecast_names = set(state.get("weather", {}))
        quotes: list[plan.Quote] = []
        notes: list[str] = []

        for candidate in state.get("candidates", []):
            name = str(candidate.get("name", ""))
            found = CATALOG.get(name.lower())
            if found is None or name.lower() not in forecast_names:
                continue

            result = await self.ctx.call(
                "convert_currency",
                {"amount": found.nightly_cost * nights, "source": found.currency, "target": budget_currency},
            )
            if not result.ok:
                notes.append(f"{found.name}: price unavailable in {budget_currency}")
                continue
            payload = result.value or {}
            quotes.append(
                plan.Quote(
                    destination=found.name,
                    nights=nights,
                    local_cost=found.nightly_cost * nights,
                    local_currency=found.currency,
                    budget_cost=round(float(payload.get("converted", 0.0)), 2),
                    budget_currency=budget_currency,
                    rate=float(payload.get("rate", 1.0)),
                )
            )

        budget = float(state.get("budget", 0.0))
        within = [q.destination for q in quotes if q.budget_cost <= budget]
        notes.append(f"{len(within)} of {len(quotes)} priced candidate(s) inside budget")
        return {
            "quotes": [q.__dict__ for q in quotes],
            "notes": notes,
            "attempts": [{"step": "price", "priced": len(quotes), "within_budget": len(within)}],
        }

    # -- decide ------------------------------------------------------------

    async def decide(self, state: TravelState) -> dict[str, Any]:
        """Pick a destination, or record that nothing survived."""
        quotes = [plan.Quote(**q) for q in state.get("quotes", [])]
        chosen = plan.choose(
            state.get("candidates", []),
            weather=state.get("weather", {}),
            quotes=quotes,
            budget=float(state.get("budget", 0.0)),
            avoid=state.get("avoid", []),
        )
        if chosen is None:
            return {
                "chosen": {},
                "notes": ["no candidate survived the weather and budget checks; looking for another"],
            }
        return {
            "chosen": chosen,
            "notes": [f"chose {chosen['name']} -- {chosen['why']}"],
            "attempts": [{"step": "decide", "choice": chosen["name"], "cost": chosen["cost"]}],
        }

    # -- itinerary and approval -------------------------------------------

    async def itinerary(self, state: TravelState) -> dict[str, Any]:
        chosen = state.get("chosen") or {}
        if not chosen:
            return {}
        built = plan.build_itinerary(chosen, nights=state.get("nights", 2), request=state["request"])
        return {"itinerary": built, "notes": [f"drafted a {len(built['days'])}-day plan for {built['destination']}"]}

    async def approve(self, state: TravelState) -> dict[str, Any]:
        """Read the human decision the run was paused for, and act on it.

        The pause is a *static breakpoint* -- the graph is compiled with
        ``interrupt_before=["approve"]`` -- so by the time this node runs, a
        person has either set ``approved`` or left it unset. That is why this
        node only branches on state and never blocks: nothing here waits for a
        person, because the waiting already happened between two invocations.

        ``interrupt()`` would express the same gate in one call and is the
        documented way to do it. It is unusable here, and the reason is worth
        recording because it is a Python-version trap, not a style choice:
        ``interrupt`` needs the runnable config, which LangGraph propagates
        through a ``contextvars``-based task context. On Python 3.10 that
        context does not survive ``asyncio.create_task``, so every async run
        raised "Called get_config outside of a runnable context". A static
        breakpoint needs no config lookup and works on 3.10. Upgrading the
        interpreter is what would allow the tidier form back.
        """
        itinerary = state.get("itinerary") or {}
        if not itinerary:
            return {}

        if state.get("approved"):
            return {"done": True, "notes": ["plan approved"]}

        return {
            "approved": False,
            "avoid": [itinerary["destination"]],
            "notes": [f"plan for {itinerary['destination']} rejected; it will not be proposed again"],
        }

    # -- recovery ----------------------------------------------------------

    async def broaden(self, state: TravelState) -> dict[str, Any]:
        """Rewrite the query after a bad result, instead of re-running it."""
        attempt = state.get("attempt", 0)
        query = plan.broaden_query(attempt)
        return {
            "query": query,
            "notes": [f"rewriting the query for attempt {attempt + 1}: {query!r}"],
        }

    async def give_up(self, state: TravelState) -> dict[str, Any]:
        """Stop, with the reason recorded. The only honest failure mode.

        An empty plan with no note is the worst outcome in the whole graph:
        indistinguishable from a bug. This node exists so the run ends with a
        sentence a person can read.
        """
        return {
            "done": True,
            "notes": [
                f"gave up after {state.get('attempt', 0)} attempt(s): no destination satisfied "
                f"the {state.get('budget', 0):.0f} {state.get('currency', '')} budget and the forecast"
            ],
        }


__all__ = ["Planner", "PlannerContext"]
