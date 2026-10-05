"""Assembling the StateGraph: nodes, edges, and the conditional branches.

The graph is a scheduler and this file is the wiring diagram. Reading it
should tell you the shape of a planning run without opening any node:

    START -> search -> (weather | broaden | give up)
    weather -> price -> decide -> (itinerary | broaden | give up)
    itinerary -> approve -> (done | broaden)
    broaden -> search

Three properties of that shape are deliberate:

- **The recovery loop goes through search, never around it.** ``broaden``
  rewrites the query and hands control back to search. A retry that jumped
  straight to ``weather`` would be re-judging the same candidates, which is
  the thing the branch exists to avoid.
- **``give_up`` is reachable from both decision points.** The attempt cap is
  enforced in the routers, so there is no path from START that loops without
  a bound.
- **Rejection re-enters the loop.** A refused plan puts its destination into
  ``avoid`` and re-searches, so a second approval request offers something
  new rather than the same plan again.
- **The approval gate is a static breakpoint.** ``compile`` passes
  ``interrupt_before=["approve"]``, so the graph stops before the gate node
  and the caller resumes it once a person has recorded a decision. This is
  the Python-3.10-compatible form of human-in-the-loop; ``nodes.Planner.approve``
  explains why the ``interrupt()`` call is not usable here.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agentkit.dispatch import ToolInvoker
from agentkit.registry import ToolRegistry
from agentkit.tools import build_registry
from . import checkpoint as checkpoint_module
from .destinations import CATALOG, as_corpus
from .edges import after_approval, after_decide, after_search
from .nodes import Planner, PlannerContext
from .state import TravelState


def build_context(
    *,
    registry: ToolRegistry | None = None,
    timeout: float = 5.0,
) -> PlannerContext:
    """Default context: the W2 tools, with the destination notes indexed.

    The registry is built with the destination notes as the search corpus, so
    the search step retrieves destinations instead of the personal-notes
    corpus. Same tool, same schema, different data -- which is the point of
    having put the corpus behind an argument in W2.
    """
    tools = registry or build_registry(
        search_corpus=as_corpus(),
        with_todos=False,
        with_calendar=False,
        with_chart=False,
    )
    return PlannerContext(invoker=ToolInvoker(tools, default_timeout=timeout), hits_per_query=4)


def build_graph(
    context: PlannerContext | None = None,
    *,
    saver: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the planner.

    Returns the compiled graph, ready for ``invoke`` / ``stream`` with a
    ``thread_config``. The saver defaults to a fresh in-memory one so that
    compiling twice never shares history by accident.
    """
    ctx = context or build_context()
    planner = Planner(ctx)

    graph: StateGraph = StateGraph(TravelState)
    graph.add_node("search", planner.search)
    graph.add_node("weather", planner.weather)
    graph.add_node("price", planner.price)
    graph.add_node("decide", planner.decide)
    graph.add_node("itinerary", planner.itinerary)
    graph.add_node("approve", planner.approve)
    graph.add_node("broaden", planner.broaden)
    graph.add_node("give_up", planner.give_up)

    graph.add_edge(START, "search")
    graph.add_conditional_edges(
        "search",
        after_search,
        {"weather": "weather", "broaden": "broaden", "give_up": "give_up"},
    )
    graph.add_edge("weather", "price")
    graph.add_edge("price", "decide")
    graph.add_conditional_edges(
        "decide",
        after_decide,
        {"itinerary": "itinerary", "broaden": "broaden", "give_up": "give_up"},
    )
    graph.add_edge("itinerary", "approve")
    graph.add_conditional_edges(
        "approve",
        after_approval,
        {"done": END, "broaden": "broaden"},
    )
    graph.add_edge("broaden", "search")
    graph.add_edge("give_up", END)

    return graph.compile(
        checkpointer=saver or checkpoint_module.default_saver(),
        # Pause just before the approval gate. Compiled here rather than in a
        # node so the breakpoint is part of the graph's shape: a reader of
        # build.py should see where the run can stop without opening nodes.
        interrupt_before=["approve"],
    )


__all__ = ["build_context", "build_graph"]
