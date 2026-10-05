"""The planning decisions, as pure functions, deliberately outside the graph.

LangGraph is a scheduler, not a place to think. Every decision this week makes
-- which destinations survived, whether the weather rules out the outdoor
half, whether the price fits -- lives here as a function over plain dicts, and
``nodes.py`` does nothing but call one of these and hand the result back.

That split is not tidiness. It is the difference between a graph whose
behaviour can be tested and one whose behaviour can only be observed. A node
that reaches into the runtime is untestable without the runtime; these
functions are called directly, with no graph, no checkpointer and no LLM, and
the tests do exactly that. The ROADMAP warns that node and edge bodies must
stay thin because the framework's API moves; this module is what "thin" means
in practice.

The second theme is that a decision must be able to say *why*. ``chosen`` is
never a bare destination: it carries the forecast it was judged against and
the converted price it was judged by. When a run picks the safe indoor city
over the sunny one, the state contains the reason, which is what makes the
resumed run auditable and the trace worth reading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .destinations import CATALOG, OUTDOOR_BLOCKERS, Destination

# How many search hits to consider, and how many of those to spend a weather
# call on. Both are caps rather than loops: a planner that will call tools
# until it is satisfied is a planner that can be made to call tools forever.
MAX_CANDIDATES = 4
MAX_WEATHER_CALLS = 3
MAX_ATTEMPTS = 3

# Query rewrites for the "the results were not good enough, look elsewhere"
# branch. Ordered, and shorter than MAX_ATTEMPTS is not the same as equal to
# it: the last one is the fallback and must exist whenever the cap allows
# another attempt.
BROADEN_QUERIES = (
    "indoor museum weekend city",
    "cheap weekend city break",
    "sunny walkable weekend",
)


@dataclass(frozen=True)
class Quote:
    """One destination priced in the budget currency."""

    destination: str
    nights: int
    local_cost: float
    local_currency: str
    budget_cost: float
    budget_currency: str
    rate: float


def candidates_from_hits(
    hits: Sequence[Mapping[str, Any]],
    *,
    limit: int = MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Turn search hits into candidates.

    A hit naming a destination the catalog does not know is dropped rather
    than carried as an opaque string: every later step needs its currency and
    its activities, and a candidate that cannot be priced is not a candidate.
    """
    candidates: list[dict[str, Any]] = []
    for hit in hits:
        name = _hit_name(hit)
        if name is None:
            continue
        found = CATALOG.get(name.lower())
        if found is None:
            continue
        candidates.append(
            {
                "name": found.name,
                "score": float(hit.get("score", 0) or 0),
                "why": "matched the request",
            }
        )
    return candidates[:limit]


def _hit_name(hit: Mapping[str, Any]) -> str | None:
    """Read a destination name out of a search hit, whatever shape it took.

    The search tool returns ``{"id": ..., "snippet": ...}``, but the snippet
    is a clipped note and the id is not guaranteed to be the display name. So
    the name is matched against the catalog rather than parsed: the first
    catalog key appearing in the hit's text wins. Written this way because a
    tool result is data from outside the graph, and trusting its shape is how
    a pipeline breaks the day the tool changes.
    """
    for key in ("id", "snippet", "text"):
        value = hit.get(key)
        if not isinstance(value, str):
            continue
        for name in CATALOG.values():
            if name.name.lower() in value.lower():
                return name.name
    return None


def weather_verdict(payload: Mapping[str, Any], *, days: int) -> dict[str, Any]:
    """Read a weather tool result into a go / no-go for outdoor plans.

    Blocking conditions anywhere in the horizon mean the outdoor half of the
    plan has to move indoors. This is the step that makes the trip plan depend
    on a tool result instead of on the model's mood: the branch is a function
    of ``condition``, not a judgement call.
    """
    forecast = payload.get("forecast") or []
    conditions = [str(day.get("condition", "")).lower() for day in forecast if isinstance(day, dict)]
    if not conditions:
        # No forecast is not the same as bad weather. Treat it as "unknown"
        # and keep the outdoor plan, because refusing to plan on a missing
        # result would make the graph fail closed on an upstream hiccup.
        return {"outdoor_ok": True, "conditions": [], "summary": "no forecast available; assuming outdoor plans hold"}
    wet = [c for c in conditions if c in OUTDOOR_BLOCKERS]
    if wet:
        return {
            "outdoor_ok": False,
            "conditions": conditions,
            "summary": f"{wet[0]} in the forecast; move the outdoor half indoors",
        }
    return {
        "outdoor_ok": True,
        "conditions": conditions,
        "summary": f"{conditions[0]} and dry across {min(days, len(conditions))} day(s); outdoor plans hold",
    }



def quote_destination(
    destination: Destination,
    *,
    nights: int,
    budget_currency: str,
    convert: Callable[[float, str, str], Mapping[str, Any]],
) -> Quote:
    """Price one destination through the fx tool.

    The conversion is a real tool call, not multiplication by a rate table,
    because "the plan consulted the currency tool" is the behaviour being
    demonstrated. ``convert`` is injected so this stays a pure function of its
    arguments and the tests can pass a stub instead of wiring a registry.
    """
    local_cost = destination.nightly_cost * nights
    result = convert(local_cost, destination.currency, budget_currency)
    budget_cost = float(result.get("converted", local_cost))
    rate = float(result.get("rate", 1.0))
    return Quote(
        destination=destination.name,
        nights=nights,
        local_cost=local_cost,
        local_currency=destination.currency,
        budget_cost=round(budget_cost, 2),
        budget_currency=budget_currency,
        rate=rate,
    )


def choose(
    candidates: Sequence[Mapping[str, Any]],
    *,
    weather: Mapping[str, Mapping[str, Any]],
    quotes: Sequence[Quote],
    budget: float,
    avoid: Iterable[str] = (),
) -> dict[str, Any] | None:
    """Pick the best surviving candidate, or ``None`` meaning "look again".

    A candidate survives only if it is not excluded, its weather permits the
    plan, and it is inside the budget. When several survive, the one with the
    driest forecast wins, and search score breaks the tie -- weather first
    because the whole point of checking it is to let it change the answer.

    Returning ``None`` instead of raising is what makes the conditional edge
    work: "nothing viable" is a normal outcome that routes back to search, not
    an exception to be caught somewhere else.
    """
    avoided = {name.lower() for name in avoid}
    priced = {q.destination.lower(): q for q in quotes}
    viable: list[dict[str, Any]] = []

    for candidate in candidates:
        name = str(candidate.get("name", ""))
        found = CATALOG.get(name.lower())
        if found is None or name.lower() in avoided:
            continue
        verdict = weather.get(name.lower())
        if verdict is None:
            continue
        quote = priced.get(name.lower())
        if quote is None or quote.budget_cost > budget:
            continue
        viable.append(
            {
                "name": found.name,
                "country": found.country,
                "currency": found.currency,
                "indoor": list(found.indoor),
                "outdoor": list(found.outdoor),
                "outdoor_ok": bool(verdict.get("outdoor_ok", True)),
                "weather_summary": verdict.get("summary", ""),
                "cost": quote.budget_cost,
                "cost_currency": quote.budget_currency,
                "local_cost": quote.local_cost,
                "local_currency": quote.local_currency,
                "score": float(candidate.get("score", 0) or 0),
            }
        )

    if not viable:
        return None

    viable.sort(key=lambda item: (not item["outdoor_ok"], -item["score"], item["cost"]))
    best = viable[0]
    best["alternatives"] = [item["name"] for item in viable[1:]]
    best["why"] = _rationale(best, budget)
    return best


def _rationale(chosen: Mapping[str, Any], budget: float) -> str:
    """One sentence explaining the pick, carried in the state for the trace."""
    weather_part = "forecast suits outdoor plans" if chosen["outdoor_ok"] else "forecast favours the indoor plan"
    return (
        f"{chosen['name']}: {weather_part}; "
        f"{chosen['cost']:.0f} {chosen['cost_currency']} of a {budget:.0f} budget"
    )


def build_itinerary(
    chosen: Mapping[str, Any],
    *,
    nights: int,
    request: str,
) -> dict[str, Any]:
    """Turn a chosen destination into a day-by-day plan.

    The outdoor/indoor split follows ``outdoor_ok``, which the weather step
    set. That is the mechanism by which a forecast actually changes the plan
    rather than being printed next to it: a wet forecast swaps the morning
    slot, it does not merely append a warning.

    Activities are consumed from copies, so a plan longer than the activity
    list repeats the last item rather than running out -- a two-night trip to
    a city with three ideas should not crash the planner.
    """
    outdoor = list(chosen.get("outdoor") or [])
    indoor = list(chosen.get("indoor") or [])
    outdoor_ok = bool(chosen.get("outdoor_ok", True))

    # The morning is the outdoor slot when the forecast allows it; the
    # afternoon is indoors either way, which is what makes a wet forecast a
    # change of plan rather than a warning printed next to one.
    primary = outdoor if outdoor_ok else indoor
    secondary = indoor
    days: list[dict[str, Any]] = []

    for index in range(max(1, nights)):
        days.append(
            {
                "day": index + 1,
                "morning": _pick(primary, index),
                "afternoon": _pick(secondary, index),
                "mode": "outdoor" if outdoor_ok else "indoor",
            }
        )

    return {
        "destination": chosen["name"],
        "country": chosen["country"],
        "nights": nights,
        "request": request,
        "outdoor_ok": outdoor_ok,
        "weather": chosen.get("weather_summary", ""),
        "cost": chosen.get("cost"),
        "cost_currency": chosen.get("cost_currency"),
        "days": days,
    }


def _pick(items: Sequence[str], index: int) -> str:
    """The activity for slot ``index``, repeating the last one if it runs out."""
    if not items:
        return ""
    return items[min(index, len(items) - 1)]


def broaden_query(attempt: int) -> str:
    """The next query to try when the results were not good enough.

    Rewriting the query rather than re-running it is the whole point of the
    branch: asking the same index the same question twice cannot help, and a
    planner that only retries is a planner that has not understood why the
    first attempt failed.
    """
    if attempt < 1:
        return BROADEN_QUERIES[0]
    return BROADEN_QUERIES[min(attempt, len(BROADEN_QUERIES) - 1)]


def render(itinerary: Mapping[str, Any]) -> str:
    """Render a plan as the text the model and the user actually see."""
    lines = [
        f"{itinerary['destination']}, {itinerary['country']} -- {itinerary['nights']} nights",
        f"weather: {itinerary['weather']}",
        f"cost: {itinerary['cost']} {itinerary['cost_currency']}",
    ]
    for day in itinerary.get("days", []):
        lines.append(f"day {day['day']} ({day['mode']}): {day['morning']} / {day['afternoon']}")
    return "\n".join(lines)


__all__ = [
    "BROADEN_QUERIES",
    "MAX_ATTEMPTS",
    "MAX_CANDIDATES",
    "MAX_WEATHER_CALLS",
    "Quote",
    "broaden_query",
    "build_itinerary",
    "candidates_from_hits",
    "choose",
    "quote_destination",
    "render",
    "weather_verdict",
]
