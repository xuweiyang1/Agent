"""The write path into long-term memory, exposed as one tool.

Long-term memory that nothing ever writes to is a decorated empty file. W5
built the two stores and the read paths, but the chat agent had no way to put
a preference *in* -- the only writer was the seeded chain. This module closes
that loop, and it is deliberately a single ``memory`` tool with an action
enum, the same argument ``todo.py`` makes, rather than three near-identical
tools that inflate the schema the model rereads every turn.

Three actions map onto W5's three kinds, and the mapping is not cosmetic:

- ``remember_preference`` replaces by key. Two live answers to "what seat
  does the user want" would make behaviour depend on iteration order, which
  shows up as an assistant that changes its mind for no reason.
- ``remember_decision`` records the *reason*, because "rejected Lisbon"
  without "over budget" cannot be acted on later.
- ``remember_fact`` is the unkeyed remainder, findable only by recall.

``list`` and ``recall`` exist so the model can check what it already believes
before storing a duplicate, which is the cheap way to keep the store honest.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from memory import LongTermMemory

from ..registry import ToolRegistry
from ..schema import ToolArgs


class MemoryArgs(ToolArgs):
    action: str = Field(
        ...,
        pattern="^(remember_preference|remember_decision|remember_fact|list|recall)$",
        description="One of: remember_preference, remember_decision, remember_fact, list, recall.",
    )
    key: str | None = Field(None, description="Preference key, e.g. 'seat'.")
    value: str | None = Field(None, description="Preference value, e.g. 'aisle'.")
    subject: str | None = Field(None, description="What a decision is about, e.g. 'Lisbon trip'.")
    outcome: str | None = Field(None, description="The decision itself, e.g. 'rejected'.")
    reason: str | None = Field(None, description="Why, e.g. 'over budget'.")
    text: str | None = Field(None, description="A fact to remember, as one sentence.")
    query: str | None = Field(None, description="What to search for, for 'recall'.")
    k: int = Field(3, ge=1, le=10, description="Maximum memories to return.")

    @model_validator(mode="after")
    def _required_by_action(self) -> "MemoryArgs":
        """Conditional requirements, enforced here rather than in the body.

        Same reasoning as ``TodoArgs``: "right type, wrong shape for this
        action" is a model mistake the schema should convert into one
        actionable sentence, not an exception from three frames down.
        """
        needs = {
            "remember_preference": ("key", "value"),
            "remember_decision": ("subject", "outcome"),
            "remember_fact": ("text",),
            "recall": ("query",),
        }
        for field in needs.get(self.action, ()):
            if not (getattr(self, field) or "").strip():
                raise ValueError(f"{field} is required when action is {self.action!r}")
        return self


def register(registry: ToolRegistry, longterm: LongTermMemory) -> LongTermMemory:
    """Attach the ``memory`` tool over an injected long-term store."""

    @registry.tool(
        "memory",
        "Long-term memory that survives across sessions: store a user "
        "preference, a decision and its reason, or a fact; list or search what "
        "is already remembered.",
        args_model=MemoryArgs,
        timeout=2.0,
    )
    def memory(
        action: str,
        key: str | None = None,
        value: str | None = None,
        subject: str | None = None,
        outcome: str | None = None,
        reason: str | None = None,
        text: str | None = None,
        query: str | None = None,
        k: int = 3,
    ) -> dict[str, Any]:
        if action == "remember_preference":
            record = longterm.remember_preference(key or "", value)
            return {"remembered": record.to_dict()}
        if action == "remember_decision":
            record = longterm.remember_decision(subject or "", outcome or "", reason=reason or "")
            return {"remembered": record.to_dict()}
        if action == "remember_fact":
            record = longterm.remember_fact(text or "")
            return {"remembered": record.to_dict()}
        if action == "recall":
            memories = []
            for hit in longterm.recall(query or "", k=k):
                record = longterm.semantic.record_of(hit.doc_id)
                if record is not None:
                    memories.append({"id": record.id, "kind": record.kind, "text": record.text})
            return {"query": query, "memories": memories}
        return {
            "preferences": longterm.preferences(),
            "decisions": [record.text for record in longterm.decisions()],
            "count": len(longterm),
        }

    return longterm


__all__ = ["MemoryArgs", "register"]