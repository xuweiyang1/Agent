"""Tool registry and a couple of built-in tools.

A tool is a plain function plus a JSON-schema-ish description. Keeping the
schema hand-written rather than generated is deliberate: the schema is the
contract shown to the model, and it is the part most worth reading.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .llm import ToolCall


class ToolError(RuntimeError):
    """Raised inside a tool. The runtime turns this into a tool message."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Holds tools and validates arguments before calling them.

    Validation is intentionally shallow, just required-key and type checks.
    Deep schema validation is a solved problem and would bury the point of
    this module, which is the dispatch path and its error handling.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def tool(
        self, name: str, description: str, parameters: dict[str, Any]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(Tool(name, description, parameters, fn))
            return fn

        return decorator

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def call(self, call: ToolCall) -> str:
        """Run one tool call and always return a string for the transcript."""
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names()) or "none"
            raise ToolError(f"unknown tool {call.name!r}; available: {available}")

        properties = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])
        missing = [key for key in required if key not in call.arguments]
        if missing:
            raise ToolError(f"{call.name} missing required argument(s): {missing}")
        for key, value in call.arguments.items():
            expected = properties.get(key, {}).get("type")
            if expected and not _matches_type(value, expected):
                raise ToolError(
                    f"{call.name} argument {key!r} expected {expected}, "
                    f"got {type(value).__name__}"
                )

        result = tool.fn(**call.arguments)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=False, sort_keys=True)


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return True


def build_default_registry(corpus: dict[str, str] | None = None) -> ToolRegistry:
    """A registry with two offline tools, enough to exercise the loop."""
    data = corpus or {
        "agent-loop": "An agent loop alternates model calls with tool calls "
        "until the model stops requesting tools.",
        "context-window": "The context window is a hard budget of tokens "
        "shared by system prompt, history, and tool output.",
        "backoff": "Exponential backoff retries a transient failure with "
        "growing delays, and is a safety requirement once writes are involved.",
    }

    registry = ToolRegistry()

    @registry.tool(
        "search",
        "Search the local knowledge base for a keyword and return matching ids.",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    def search(query: str) -> dict[str, Any]:
        needle = query.lower()
        hits = [key for key, body in data.items() if needle in key.lower() or needle in body.lower()]
        return {"query": query, "hits": hits}

    @registry.tool(
        "read",
        "Read the full text of one knowledge base entry by id.",
        {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    )
    def read(id: str) -> str:
        if id not in data:
            raise ToolError(f"no entry with id {id!r}")
        return data[id]

    return registry