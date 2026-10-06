"""The five tools a personal assistant needs, and the registry that holds them.

``build_registry`` is the one entry point. Each ``register`` returns the
service it wired up, so a test can seed state (a todo, a calendar entry)
without reaching into private attributes.
"""

from __future__ import annotations

from pathlib import Path

from ..registry import Tool, ToolRegistry
from . import calendar, chart, files, fx, search, todo, weather

__all__ = [
    "build_registry",
    "calendar",
    "chart",
    "files",
    "fx",
    "search",
    "todo",
    "weather",
]


def build_registry(
    *,
    search_corpus: dict[str, str] | None = None,
    with_todos: bool = True,
    with_calendar: bool = True,
    with_search: bool = True,
    with_weather: bool = True,
    with_fx: bool = True,
    with_chart: bool = True,
    with_files: bool = False,
    chart_output_dir: str | None = None,
    workspace: str | None = None,
    todo_service: "todo.TodoService | None" = None,
    calendar_service: "calendar.CalendarService | None" = None,
) -> ToolRegistry:
    """Assemble a registry, optionally with a subset of the tools.

    The flags exist so a test can build the one-tool registry it needs
    instead of re-deriving it by mutation, which would make the test depend
    on registration order.
    """
    registry = ToolRegistry()
    if with_weather:
        weather.register(registry)
    if with_fx:
        fx.register(registry)
    if with_todos:
        todo.register(registry, todo_service)
    if with_calendar:
        calendar.register(registry, calendar_service)
    if with_search:
        search.register(registry, search.SearchService(corpus=dict(search_corpus or search.DEFAULT_CORPUS)))
    if with_chart:
        output_dir = Path(chart_output_dir) if chart_output_dir else None
        chart.register(registry, chart.ChartService(output_dir=output_dir))
    if with_files:
        files.register(registry, workspace)
    return registry
