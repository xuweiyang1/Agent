"""A tiny in-memory todo list, mutated through one tool.

One tool with an ``action`` field rather than four tools is a deliberate
schema choice. Four near-identical tools inflate the tool list the model reads
every turn and make "add" versus "create" a coin flip; one tool with a small
enum keeps the prompt short and the intent unambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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

    def add(self, text: str) -> dict[str, Any]:
        item_id = f"t{self._next}"
        self._next += 1
        item = {"id": item_id, "text": text.strip(), "done": False}
        self._items[item_id] = item
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
        return item


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
