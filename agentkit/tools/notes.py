"""Notes: the assistant's own scratch space, distinct from the user's files.

``files.py`` reads and writes the *workspace* -- the user's documents, under a
sandbox. This is the other half: things the assistant jots down for the user,
kept in the deployment's own ``state_dir`` so they survive a restart. Merging
the two would mean an assistant note and a real document share a namespace,
and cleaning up one would risk the other.

Appending is the only write. A note tool that overwrote would be a text
editor with one slot; an append-only list is what "keep a note for me" means,
and it makes the file trivially auditable by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ..errors import ErrorKind, ToolCallError
from ..registry import ToolRegistry
from ..schema import ToolArgs


class NoteArgs(ToolArgs):
    action: str = Field(..., pattern="^(add|list)$", description="Either 'add' or 'list'.")
    text: str | None = Field(None, description="Note body; required when action is 'add'.")
    title: str | None = Field(None, description="Short title; optional, defaults to a trimmed body.")

    @model_validator(mode="after")
    def _required_by_action(self) -> "NoteArgs":
        if self.action == "add" and not (self.text or "").strip():
            raise ValueError("text is required when action is 'add'")
        return self


@dataclass
class NoteService:
    """A JSON-backed list of notes, written whole on each add."""

    path: Path | None = None
    _notes: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._load()

    def add(self, text: str, title: str = "") -> dict[str, Any]:
        note = {
            "id": f"n{len(self._notes) + 1}",
            "title": (title or text).strip().splitlines()[0][:60],
            "text": text.strip(),
        }
        self._notes.append(note)
        self._save()
        return note

    def list_notes(self) -> list[dict[str, Any]]:
        return list(self._notes)

    # -- persistence -------------------------------------------------------
    #
    # Same reasoning as ``todo.py``: a corrupt file must not brick startup, and
    # a note typed by a user is exactly what they would miss after a crash.

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"notes": self._notes}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        raw = payload.get("notes", payload if isinstance(payload, list) else [])
        if isinstance(raw, list):
            self._notes = [item for item in raw if isinstance(item, dict) and item.get("text")]


def register(registry: ToolRegistry, service: NoteService | None = None) -> NoteService:
    """Attach the ``note`` tool (one tool, an action enum -- see ``todo.py``)."""
    svc = service or NoteService()

    @registry.tool(
        "note",
        "Keep a note for the user, or list the notes already kept. Use it when "
        "asked to remember something for later without a todo or a calendar event.",
        args_model=NoteArgs,
        timeout=2.0,
    )
    def note(action: str, text: str | None = None, title: str | None = None) -> dict[str, Any]:
        if action == "add":
            return {"added": svc.add(text or "", title or "")}
        return {"notes": svc.list_notes()}

    return svc


__all__ = ["NoteArgs", "NoteService", "register"]