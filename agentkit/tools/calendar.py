"""Calendar events with real timestamp parsing.

The parse is the lesson here. A model asked for "next Tuesday at 3" will send
something like ``2026-10-06T15:00`` -- or ``2026-10-06 15:00``, or a bare
date. Accepting the two common shapes and rejecting the rest with a message
that names the expected format turns a confusing failure into a one-line fix,
which is exactly what ``BAD_ARGUMENTS`` is for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from pydantic import Field

from ..errors import ErrorKind, ToolCallError
from ..registry import ToolRegistry
from ..schema import ToolArgs

_ACCEPTED = "YYYY-MM-DDTHH:MM (a space instead of 'T' is also accepted)"


class CalendarArgs(ToolArgs):
    action: str = Field(..., pattern="^(create|list)$", description="Either 'create' or 'list'.")
    title: str | None = Field(None, description="Event title; required when action is 'create'.")
    start: str | None = Field(None, description=f"Event start, {_ACCEPTED}; required when action is 'create'.")
    duration_minutes: int = Field(60, gt=0, le=24 * 60, description="Event length in minutes.")
    day: str | None = Field(None, description="Filter for 'list', as YYYY-MM-DD. Omit to list everything.")


@dataclass
class CalendarService:
    _events: list[dict[str, Any]] = field(default_factory=list)
    _next: int = field(default=1, init=False)

    def create(self, title: str, start: str, duration_minutes: int) -> dict[str, Any]:
        begins = _parse(start)
        event = {
            "id": f"e{self._next}",
            "title": title.strip(),
            "start": begins.isoformat(timespec="minutes"),
            "end": (begins + timedelta(minutes=duration_minutes)).isoformat(timespec="minutes"),
        }
        self._next += 1
        self._events.append(event)
        return event

    def list_events(self, day: str | None = None) -> list[dict[str, Any]]:
        if day is None:
            return sorted(self._events, key=lambda e: e["start"])
        wanted = _parse(day).date().isoformat()
        return sorted(
            (e for e in self._events if e["start"].startswith(wanted)),
            key=lambda e: e["start"],
        )


def _parse(raw: str) -> datetime:
    """Parse the shapes a model actually sends, or fail with the format."""
    candidate = raw.strip().replace(" ", "T")
    try:
        return datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise ToolCallError(
            f"could not parse datetime {raw!r}",
            kind=ErrorKind.BAD_ARGUMENTS,
            details={"expected_format": _ACCEPTED, "got": raw},
        ) from exc


def register(registry: ToolRegistry, service: CalendarService | None = None) -> CalendarService:
    """Attach the ``calendar`` tool."""
    svc = service or CalendarService()

    @registry.tool(
        "calendar",
        "Create a calendar event, or list events for a day. Use it after "
        "resolving a relative date such as 'next Saturday' into a real date.",
        args_model=CalendarArgs,
        timeout=2.0,
    )
    def calendar(
        action: str,
        title: str | None = None,
        start: str | None = None,
        duration_minutes: int = 60,
        day: str | None = None,
    ) -> dict[str, Any]:
        if action == "create":
            if not title or not start:
                raise ToolCallError(
                    "calendar: create needs both 'title' and 'start'",
                    kind=ErrorKind.BAD_ARGUMENTS,
                    details={"missing": [k for k, v in (("title", title), ("start", start)) if not v]},
                )
            return {"created": svc.create(title, start, duration_minutes)}
        return {"events": svc.list_events(day)}


__all__ = ["CalendarArgs", "CalendarService", "register"]
