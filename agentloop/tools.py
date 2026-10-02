"""Tool registry and a couple of built-in tools.

A tool is a plain function plus a JSON-schema-ish description. Keeping the
schema hand-written rather than generated is deliberate: the schema is the
contract shown to the model, and it is the part most worth reading.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .corpus import CORPUS
from .llm import ToolCall

_WORD = re.compile(r"[a-z0-9]+")

# Words this short carry no retrieval signal and only create false hits.
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "for",
    "from", "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "what", "when", "where", "which", "why", "with",
}


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


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stopwords and single characters removed."""
    return [w for w in _WORD.findall(text.lower()) if len(w) > 1 and w not in _STOPWORDS]


def _stem(word: str) -> str:
    """Strip the few suffixes that actually show up in this corpus.

    Not a real stemmer. It exists because substring matching failed on a
    real query: the corpus says "retries" and the model asked for "retry".
    Teaching the index those two are the same word is the whole fix, and a
    full Porter stemmer would be more machinery than the problem needs.
    """
    for suffix in ("ies", "es", "s", "ed", "ing", "y"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def rank_entries(query: str, corpus: dict[str, str]) -> list[tuple[str, int]]:
    """Score each entry by how many distinct query terms it contains.

    Matching is on stemmed whole tokens rather than substrings, so "retry"
    and "retries" collide while "err" no longer matches "error" by accident.
    Title hits score double: a term in the key is a stronger signal than the
    same term buried in the body.
    """
    terms = {_stem(t) for t in tokenize(query)}
    if not terms:
        return []

    scored: list[tuple[str, int]] = []
    for key, body in corpus.items():
        key_terms = {_stem(t) for t in tokenize(key)}
        body_terms = {_stem(t) for t in tokenize(body)}
        score = 2 * len(terms & key_terms) + len(terms & body_terms)
        if score:
            scored.append((key, score))

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def _snippet(body: str, query: str, width: int = 140) -> str:
    """Return the part of the body most likely to answer the query.

    Sending a snippet instead of only an id is what removes the second tool
    call for most questions: the model can often answer from the search
    result and never needs `read`.
    """
    if len(body) <= width:
        return body
    terms = {_stem(t) for t in tokenize(query)}
    lowered = tokenize(body)
    for index, word in enumerate(lowered):
        if _stem(word) in terms:
            start = max(0, body.lower().find(word) - width // 3)
            return body[start : start + width].strip()
    return body[:width].strip()


def build_default_registry(corpus: dict[str, str] | None = None) -> ToolRegistry:
    """A registry with two offline tools, enough to exercise the loop."""
    data = CORPUS if corpus is None else corpus

    registry = ToolRegistry()

    @registry.tool(
        "search",
        "Search the local knowledge base. Returns matching entry ids with a "
        "short snippet of each, ranked by relevance.",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
    def search(query: str) -> dict[str, Any]:
        ranked = rank_entries(query, data)
        return {
            "query": query,
            "hits": [
                {"id": key, "score": score, "snippet": _snippet(data[key], query)}
                for key, score in ranked
            ],
        }

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