"""The tools a personal assistant needs, and the registry that holds them.

``build_registry`` is the one entry point. Each ``register`` returns the
service it wired up, so a test can seed state (a todo, a calendar entry)
without reaching into private attributes.

Two tools are opt-in because they need a store the caller owns: ``memory``
takes a ``LongTermMemory`` and ``note`` writes into the deployment's
``state_dir``. Defaulting them off is what keeps ``build_registry()``
constructible with no arguments and no filesystem, which every week's tests
rely on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..registry import Tool, ToolRegistry
from . import calendar, chart, files, fx, memory, notes, search, todo, weather

__all__ = [
    "build_registry",
    "calendar",
    "chart",
    "files",
    "fx",
    "memory",
    "notes",
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
    with_memory: bool = False,
    with_notes: bool = False,
    chart_output_dir: str | None = None,
    workspace: str | None = None,
    todo_service: "todo.TodoService | None" = None,
    calendar_service: "calendar.CalendarService | None" = None,
    longterm: Any | None = None,
    note_service: "notes.NoteService | None" = None,
    weather_service: "weather.WeatherService | None" = None,
) -> ToolRegistry:
    """Assemble a registry, optionally with a subset of the tools.

    The flags exist so a test can build the one-tool registry it needs
    instead of re-deriving it by mutation, which would make the test depend
    on registration order.
    """
    registry = ToolRegistry()
    if with_weather:
        weather.register(registry, weather_service)
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
    if with_memory:
        if longterm is None:
            raise ValueError("with_memory=True needs a longterm store; pass longterm=...")
        memory.register(registry, longterm)
    if with_notes:
        notes.register(registry, note_service)
    return registry