"""Calendar events with real timestamp parsing, optionally stored on disk.

The parse is the lesson here. A model asked for "next Tuesday at 3" will send
something like ``2026-10-06T15:00`` -- or ``2026-10-06 15:00``, or a bare
date. Accepting the two common shapes and rejecting the rest with a message
that names the expected format turns a confusing failure into a one-line fix,
which is exactly what ``BAD_ARGUMENTS`` is for.

Like ``todo``, storage is opt-in: tests want a fresh, in-memory calendar, while
a local deployment wants the events to survive a restart. The file is a plain
JSON list of event dicts so a human can fix a bad entry without a tool.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
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
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.path is not None:
            self._load()

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
        self._save()
        return event

    def list_events(self, day: str | None = None) -> list[dict[str, Any]]:
        if day is None:
            return sorted(self._events, key=lambda e: e["start"])
        wanted = _parse(day).date().isoformat()
        return sorted(
            (e for e in self._events if e["start"].startswith(wanted)),
            key=lambda e: e["start"],
        )

    # -- persistence -------------------------------------------------------

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"events": self._events, "next": self._next}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        raw_events = payload.get("events", payload if isinstance(payload, list) else [])
        if not isinstance(raw_events, list):
            return
        for event in raw_events:
            if isinstance(event, dict) and event.get("id") and event.get("start"):
                self._events.append(event)
        self._next = int(payload.get("next") or self._next_from_events())

    def _next_from_events(self) -> int:
        highest = 0
        for event in self._events:
            digits = "".join(ch for ch in str(event.get("id", "")) if ch.isdigit())
            highest = max(highest, int(digits) if digits else 0)
        return highest + 1


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

    return svc


__all__ = ["CalendarArgs", "CalendarService", "register"]