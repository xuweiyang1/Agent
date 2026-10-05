"""Turning one model tool call into one transcript-ready result.

This is the module the week is really about. Three failures have to be
handled identically whether they arrive as an exception, a timeout, or a
name that does not exist, and the caller must never see a raw exception:

1. The model invents a tool name (``UNKNOWN_TOOL``).
2. The model sends arguments that do not fit the schema (``BAD_ARGUMENTS``).
3. The tool itself fails or hangs (``TIMEOUT`` / ``UPSTREAM`` / ``INTERNAL``).

The third case is why dispatch is async. A blocking tool call inside an
async service would stall every other request, so a sync tool is pushed to
a worker thread and the whole invocation is wrapped in a time budget. The
event loop stays free, and a hung tool becomes a classified result instead
of a hung server.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import ErrorKind, ToolCallError, ToolFailure, classify
from .registry import Tool, ToolRegistry
from .schema import ToolArgs, validate_args

DEFAULT_TIMEOUT = 10.0

Observer = Callable[[str, dict[str, Any]], None]


@dataclass
class DispatchResult:
    """One completed tool call, successful or not.

    ``value`` is the raw return value for a caller that wants the object, and
    ``output`` is its serialized form for the transcript. Keeping both means
    the HTTP layer can return structured JSON while the model still sees the
    string it expects.
    """

    name: str
    call_id: str
    ok: bool
    output: str
    value: Any = None
    failure: ToolFailure | None = None
    latency_ms: float = 0.0

    def payload(self) -> dict[str, Any]:
        """The object form: an error envelope, or the successful value."""
        if self.failure is not None:
            return self.failure.to_payload()
        return {"ok": True, "result": self.value}


class ToolInvoker:
    """Validates, runs, times out, and classifies a tool call.

    It owns no policy beyond the timeout: retrying an ``UPSTREAM`` failure is
    left to the caller, because the right retry depends on whether the caller
    is a live agent turn (retry) or a cached batch job (fail fast).
    """

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        default_timeout: float = DEFAULT_TIMEOUT,
        observer: Observer | None = None,
    ) -> None:
        self.registry = registry
        self.default_timeout = default_timeout
        self.observer = observer

    async def invoke(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        call_id: str = "",
        timeout: float | None = None,
    ) -> DispatchResult:
        """Run one call and always return a result, never raise.

        Cancellation is the one exception that is re-raised: a ``CancelledError``
        means the surrounding request is going away, and swallowing it would
        make shutdown hang.
        """
        start = time.perf_counter()
        try:
            value = await self._run(name, arguments or {}, timeout=timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failure = classify(exc, tool=name)
            result = DispatchResult(
                name=name,
                call_id=call_id,
                ok=False,
                output=_dump(failure.to_payload()),
                failure=failure,
            )
        else:
            result = DispatchResult(
                name=name,
                call_id=call_id,
                ok=True,
                output=_dump(value) if not isinstance(value, str) else value,
                value=value,
            )

        result.latency_ms = (time.perf_counter() - start) * 1000
        self._emit(result)
        return result

    async def _run(self, name: str, arguments: dict[str, Any], *, timeout: float | None) -> Any:
        if name not in self.registry:
            available = self.registry.names()
            raise ToolCallError(
                f"unknown tool {name!r}; available: {', '.join(available) or 'none'}",
                kind=ErrorKind.UNKNOWN_TOOL,
                details={"available": available},
            )

        tool = self.registry.get(name)
        kwargs = self._prepare_arguments(tool, arguments)

        budget = timeout if timeout is not None else (
            tool.timeout if tool.timeout is not None else self.default_timeout
        )

        if tool.is_async:
            awaitable = tool.fn(**kwargs)
        else:
            awaitable = asyncio.to_thread(tool.fn, **kwargs)

        if budget is None or budget <= 0:
            return await awaitable
        return await asyncio.wait_for(awaitable, budget)

    @staticmethod
    def _prepare_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate before the tool body runs, so no tool sees bad input."""
        if not isinstance(arguments, dict):
            raise ToolCallError(
                f"{tool.name}: arguments must be an object, got {type(arguments).__name__}",
                kind=ErrorKind.BAD_ARGUMENTS,
                details={"expected": "object"},
            )
        if tool.args_model is None:
            if arguments:
                raise ToolCallError(
                    f"{tool.name}: takes no arguments",
                    kind=ErrorKind.BAD_ARGUMENTS,
                    details={"unexpected": sorted(arguments)},
                )
            return {}
        validated: ToolArgs = validate_args(tool.args_model, arguments, tool=tool.name)
        return validated.model_dump()

    def _emit(self, result: DispatchResult) -> None:
        if self.observer is None:
            return
        self.observer(
            "tool_call",
            {
                "tool": result.name,
                "ok": result.ok,
                "error": result.failure.kind.value if result.failure else None,
                "latency_ms": round(result.latency_ms, 3),
            },
        )


def _dump(value: Any) -> str:
    """Serialize a tool return value, never raising on an exotic type.

    A tool that returns a ``datetime`` should not turn a successful call into
    an ``INTERNAL`` error, so anything json cannot handle falls back to its
    ``str``. The transcript stays stringly, which is what the model reads.
    """
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return str(value)


__all__ = ["DEFAULT_TIMEOUT", "DispatchResult", "ToolInvoker"]
