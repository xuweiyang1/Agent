"""W4: a planner that schedules search, weather and price into one trip plan.

``build_graph`` is the entry point; ``graph.nodes`` and ``graph.edges`` hold
the thin layer over LangGraph, and ``graph.plan`` holds the decisions.
"""

from __future__ import annotations

from .build import build_graph

__all__ = ["build_graph"]
