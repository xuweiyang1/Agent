"""A tiny todo list, mutated through one tool, optionally backed by a file.

One tool with an ``action`` field rather than four tools is a deliberate
schema choice. Four near-identical tools inflate the tool list the model reads
every turn and make "add" versus "create" a coin flip; one tool with a small
enum keeps the prompt short and the intent unambiguous.

Persistence is opt-in for one reason: the weekly tests want a fresh service
per test, and a service that always wrote to disk would make those tests order
dependent. Passing ``path`` turns it into a small JSON store, which is what a
single-user local deployment needs -- a todo list that vanishes on restart is
a demo, not an assistant. The file format is the plain item dicts, so it can
be read and repaired by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ..errors import ErrorKind, ToolCallError
from ..registry import ToolRegistry
from ..schema import ToolArgs


class TodoAction(str, Enum):
    ADD = "add"
    LIST = "list"
    COMPLETE = "complete"
    REMOVE = "remove"


class TodoArgs(ToolArgs):
    action: TodoAction = Field(..., description="One of: add, list, complete, remove.")
    text: str | None = Field(None, description="Item text; required when action is 'add'.")
    item_id: str | None = Field(None, description="Item id, e.g. 't1'; required for complete/remove.")

    @model_validator(mode="after")
    def _check_required_by_action(self) -> "TodoArgs":
        """Conditional requirements, enforced by the model, not the tool body.

        ``expected``/``got`` already covers type errors; this covers "the right
        type but the wrong shape for this action", which is the same class of
        model mistake and deserves the same actionable error.
        """
        if self.action is TodoAction.ADD and not (self.text or "").strip():
            raise ValueError("text is required when action is 'add'")
        if self.action in (TodoAction.COMPLETE, TodoAction.REMOVE) and not self.item_id:
            raise ValueError("item_id is required when action is complete or remove")
        return self


@dataclass
class TodoService:
    _items: dict[str, dict[str, Any]] = field(default_factory=dict)
    _next: int = field(default=1, init=False)
    path: Path | None = None

    def __post_init__(self) -> None:
        if self.path is not None:
            self._load()

    def add(self, text: str) -> dict[str, Any]:
        item_id = f"t{self._next}"
        self._next += 1
        item = {"id": item_id, "text": text.strip(), "done": False}
        self._items[item_id] = item
        self._save()
        return item

    def list_items(self) -> list[dict[str, Any]]:
        return list(self._items.values())

    def complete(self, item_id: str) -> dict[str, Any]:
        return self._get(item_id, "complete")

    def remove(self, item_id: str) -> dict[str, Any]:
        return self._get(item_id, "remove")

    def _get(self, item_id: str, why: str) -> dict[str, Any]:
        item = self._items.get(item_id)
        if item is None:
            raise ToolCallError(
                f"no todo with id {item_id!r}",
                kind=ErrorKind.NOT_FOUND,
                details={"known_ids": sorted(self._items)},
            )
        if why == "complete":
            item["done"] = True
        else:
            del self._items[item_id]
        self._save()
        return item

    # -- persistence -------------------------------------------------------
    #
    # Saving on every mutation rather than on shutdown is the choice that makes
    # a crash survivable: a killed process never runs a shutdown hook, and the
    # state a user typed in is exactly what they would miss.

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"items": list(self._items.values()), "next": self._next}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt file should not brick startup; it is reported by being
            # ignored, and the next write replaces it with known-good state.
            return
        raw_items = payload.get("items", payload if isinstance(payload, list) else [])
        if not isinstance(raw_items, list):
            return
        for item in raw_items:
            if isinstance(item, dict) and item.get("id"):
                self._items[str(item["id"])] = item
        self._next = int(payload.get("next") or self._next_from_items())

    def _next_from_items(self) -> int:
        highest = 0
        for item_id in self._items:
            digits = "".join(ch for ch in str(item_id) if ch.isdigit())
            highest = max(highest, int(digits) if digits else 0)
        return highest + 1


def register(registry: ToolRegistry, service: TodoService | None = None) -> TodoService:
    """Attach the ``todo`` tool."""
    svc = service or TodoService()

    @registry.tool(
        "todo",
        "Manage the user's todo list: add an item, list items, complete or "
        "remove one by id.",
        args_model=TodoArgs,
        timeout=2.0,
    )
    def todo(action: TodoAction, text: str | None = None, item_id: str | None = None) -> dict[str, Any]:
        if action is TodoAction.ADD:
            return {"added": svc.add(text or "")}
        if action is TodoAction.LIST:
            return {"items": svc.list_items()}
        if action is TodoAction.COMPLETE:
            return {"completed": svc.complete(item_id or "")}
        return {"removed": svc.remove(item_id or "")}

    return svc


__all__ = ["TodoAction", "TodoArgs", "TodoService", "register"]