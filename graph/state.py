"""The shape of a planning run, and why each field earns its place.

A LangGraph state is the contract between nodes, and a state that grows by
accident is how a graph becomes a place where logic hides. So every field
here is here for a named reason:

- the request fields are what the user stated, frozen at entry;
- ``candidates`` / ``quotes`` / ``weather`` are what the *tools* reported, so
  a node never reaches into another node's internals to get them;
- ``chosen`` / ``itinerary`` / ``approved`` are what the *agent* decided;
- ``notes`` and ``attempts`` are provenance, and they are first-class because
  the interesting case is not "found a destination" -- it is "the first one
  was rejected and the plan had to change". A resumed run that cannot explain
  why it is where it is cannot be trusted to continue.

``total=False`` is deliberate and load-bearing. It is what lets a node return
only the keys it owns, which LangGraph merges into the state. Requiring every
key would force each node to restate fields it never touched -- exactly the
coupling this file exists to avoid.

``notes`` and ``attempts`` are the one exception to replacement-merge: they
carry an ``add`` reducer, so a node appends an increment and cannot silently
erase what earlier nodes recorded. A reducer needs a starting value, which is
why ``initial_state`` seeds them instead of leaving them absent.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

Notes = Annotated[list[str], add]
Attempts = Annotated[list[dict[str, Any]], add]


class TravelState(TypedDict, total=False):
    """One planning run. Field groups are commented in the module docstring."""

    # The request, as the user stated it.
    request: str
    nights: int
    budget: float
    currency: str

    # What search surfaced, and what has been ruled out.
    candidates: list[dict[str, Any]]
    avoid: list[str]
    # The query currently in flight, and how many search attempts have run.
    # Both are state rather than locals because the broaden branch is a later
    # node reading an earlier node's decision.
    query: str
    attempt: int

    # What the tools reported.
    weather: dict[str, Any]
    outdoor_ok: bool
    quotes: list[dict[str, Any]]

    # What the agent decided.
    chosen: dict[str, Any]
    itinerary: dict[str, Any]
    approved: bool
    done: bool

    # Provenance.
    notes: Notes
    attempts: Attempts


def initial_state(
    request: str,
    *,
    nights: int = 2,
    budget: float = 6000.0,
    currency: str = "CNY",
    **overrides: Any,
) -> TravelState:
    """Seed a run.

    The reducer channels are seeded here rather than at each call site,
    because a reducer with no initial value is a subtle bug: the first node to
    append works, and the failure shows up later as a missing earlier note.
    """
    state: TravelState = {
        "request": request,
        "nights": nights,
        "budget": budget,
        "currency": currency,
        "avoid": [],
        "attempt": 0,
        "notes": [],
        "attempts": [],
        "approved": False,
        "done": False,
    }
    state.update(overrides)
    return state


__all__ = ["Attempts", "Notes", "TravelState", "initial_state"]
