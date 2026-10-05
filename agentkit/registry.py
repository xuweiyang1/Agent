"""Tool registry: one place that maps a name to its schema and its callable.

The registry is deliberately agnostic about sync versus async. Whether a
tool is ``async def`` is a property of its implementation, discovered with
``inspect``, not a separate registration path. That keeps the schema the
model sees identical for both, which matters because the model has no way to
know or care which one it is calling.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .schema import ToolArgs, schema_from_signature, schema_for

ArgsModel = type[ToolArgs]
ToolFn = Callable[..., Any]


@dataclass(frozen=True)
class Tool:
    """A registered tool: its published contract plus its implementation.

    ``args_model`` is ``None`` only for a genuine zero-argument tool. The
    ``timeout`` is per-tool because the right budget is a property of the
    dependency: a local search is milliseconds, a remote weather call is
    seconds, and forcing both to share one number guarantees one of them is
    wrong.
    """

    name: str
    description: str
    args_model: ArgsModel | None
    fn: ToolFn
    timeout: float | None = None

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.fn)

    def parameters(self) -> dict[str, Any]:
        if self.args_model is None:
            return schema_from_signature(self.fn)
        return schema_for(self.args_model)

    def schema(self) -> dict[str, Any]:
        """The object handed to the model as one entry of ``tools``."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters(),
        }


class ToolRegistry:
    """Holds tools and publishes their schemas.

    Registration order is not preserved on purpose: ``names()`` sorts, so the
    tool list in a prompt is stable across runs. An unstable tool list would
    defeat prompt caching, which is the cheapest saving available.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def tool(
        self,
        name: str,
        description: str,
        *,
        args_model: ArgsModel | None = None,
        timeout: float | None = None,
    ) -> Callable[[ToolFn], ToolFn]:
        """Decorator form, mirroring ``agentloop``'s registry so the two read alike."""

        def decorator(fn: ToolFn) -> ToolFn:
            self.register(Tool(name, description, args_model, fn, timeout))
            return fn

        return decorator

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in self.names()]

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


__all__ = ["Tool", "ToolRegistry"]
