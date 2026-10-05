"""The routing decisions, as plain functions of the state.

Separated from ``build.py`` so the tests can call a routing function directly
and assert where a run goes, without compiling a graph to find out. "Which way
did this state route" is the question that actually breaks during a refactor,
and it is cheap to answer when the router is a function.
"""

from __future__ import annotations

from typing import Literal

from .plan import MAX_ATTEMPTS
from .state import TravelState

Route = Literal["weather", "broaden", "give_up"]
DecideRoute = Literal["itinerary", "broaden", "give_up"]
ApprovalRoute = Literal["done", "broaden"]


def after_search(state: TravelState) -> Route:
    """Candidates found -> check weather; otherwise rewrite, then give up.

    The attempt cap is checked here rather than inside ``broaden`` because a
    router that can loop forever is a graph that can hang, and the cap has to
    be visible at the point the loop is created.
    """
    if state.get("candidates"):
        return "weather"
    if state.get("attempt", 0) < MAX_ATTEMPTS:
        return "broaden"
    return "give_up"


def after_decide(state: TravelState) -> DecideRoute:
    """A chosen destination -> draft it; nothing viable -> try again."""
    if state.get("chosen"):
        return "itinerary"
    if state.get("attempt", 0) < MAX_ATTEMPTS:
        return "broaden"
    return "give_up"


def after_approval(state: TravelState) -> ApprovalRoute:
    """Approved -> finish; rejected -> search again without that destination."""
    if state.get("approved"):
        return "done"
    return "broaden"


__all__ = ["ApprovalRoute", "DecideRoute", "Route", "after_approval", "after_decide", "after_search"]
