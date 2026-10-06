"""The async turn loop, and a deterministic model to drive it offline.

Same shape as W1's runtime -- call the model, dispatch tools, append results,
repeat -- with two changes that a service forces:

- Every tool call in a turn is awaited concurrently, not in sequence. Three
  independent lookups should take one round trip, not three.
- A whole turn has a deadline. W1 bounded each tool; here the run itself is
  bounded, because the caller is an HTTP request and has a client waiting.

``HeuristicModel`` is not a mock. It is a real, deterministic policy that
picks a tool from the task text, so the service and its tests run with no key
and no network, and a real model can replace it without touching the loop.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from agentloop.llm import ToolCall

from .dispatch import DispatchResult, ToolInvoker
from .messages import ChatMessage, estimate_message_tokens

Observer = Callable[[str, dict[str, Any]], None]


class AsyncModel(Protocol):
    """The async contract the loop depends on. One method, like W1's."""

    async def acomplete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]
    ) -> ChatMessage:
        ...


@dataclass
class TurnResult:
    """Everything a caller or a test needs to judge one run."""

    messages: list[ChatMessage]
    answer: str
    dispatch: list[DispatchResult] = field(default_factory=list)
    turns: int = 0
    truncated: bool = False
    elapsed_ms: float = 0.0

    @property
    def tokens(self) -> int:
        return sum(estimate_message_tokens(m) for m in self.messages)

    @property
    def failed_calls(self) -> list[DispatchResult]:
        return [d for d in self.dispatch if not d.ok]


class ToolCallingAgent:
    """Runs turns until the model answers without asking for a tool."""

    def __init__(
        self,
        model: AsyncModel,
        invoker: ToolInvoker,
        *,
        system_prompt: str = "You are a helpful assistant. Use tools when they help.",
        max_turns: int = 8,
        run_timeout: float = 60.0,
        observer: Observer | None = None,
    ) -> None:
        self.model = model
        self.invoker = invoker
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.run_timeout = run_timeout
        self.observer = observer

    async def run(self, task: str, *, history: Sequence[ChatMessage] | None = None) -> TurnResult:
        started = time.perf_counter()
        messages: list[ChatMessage] = []
        if self.system_prompt:
            messages.append(ChatMessage(role="system", content=self.system_prompt))
        messages.extend(history or [])
        messages.append(ChatMessage(role="user", content=task))

        dispatch: list[DispatchResult] = []
        turns = 0
        truncated = False

        while turns < self.max_turns:
            turns += 1
            if time.perf_counter() - started > self.run_timeout:
                truncated = True
                self._emit("run_timeout", turns=turns)
                break

            remaining = self.run_timeout - (time.perf_counter() - started)
            if remaining <= 0:
                truncated = True
                self._emit("run_timeout", turns=turns)
                break
            try:
                # ``run_timeout`` is a real deadline, not only a check between
                # turns.  A provider can hang inside one await, and the HTTP
                # caller should still receive a bounded response.  Keep both
                # timeout classes for Python 3.10, where they are distinct.
                reply = await asyncio.wait_for(
                    self.model.acomplete(
                        list(messages), self.invoker.registry.schemas()
                    ),
                    timeout=remaining,
                )
            except (asyncio.TimeoutError, TimeoutError):
                truncated = True
                self._emit("run_timeout", turns=turns)
                break
            messages.append(reply)

            if not reply.tool_calls:
                messages = [m.collapse_images() for m in messages]
                elapsed = (time.perf_counter() - started) * 1000
                self._emit("answer", turns=turns, elapsed_ms=round(elapsed, 3))
                return TurnResult(messages, reply.text(), dispatch, turns, False, elapsed)

            # Independent calls in one turn are concurrent on purpose.  The
            # same deadline covers the tool round too; otherwise a model that
            # answers quickly could still leave an HTTP request waiting on a
            # slow provider tool after the advertised run budget elapsed.
            remaining = self.run_timeout - (time.perf_counter() - started)
            if remaining <= 0:
                truncated = True
                self._emit("run_timeout", turns=turns)
                break
            try:
                results = await asyncio.wait_for(
                    self._invoke_all(reply.tool_calls), timeout=remaining
                )
            except (asyncio.TimeoutError, TimeoutError):
                truncated = True
                self._emit("run_timeout", turns=turns)
                break
            for result in results:
                dispatch.append(result)
                messages.append(
                    ChatMessage(role="tool", content=result.output, tool_call_id=result.call_id)
                )

        if turns >= self.max_turns:
            self._emit("max_turns", turns=turns)
        elapsed = (time.perf_counter() - started) * 1000
        return TurnResult(messages, "", dispatch, turns, truncated, elapsed)

    async def _invoke_all(self, calls: Sequence[ToolCall]) -> list[DispatchResult]:
        import asyncio

        return list(
            await asyncio.gather(
                *(self.invoker.invoke(c.name, c.arguments, call_id=c.id) for c in calls)
            )
        )

    def _emit(self, event: str, **fields: Any) -> None:
        if self.observer is not None:
            self.observer(event, fields)


class HeuristicModel:
    """A deterministic stand-in that actually calls the tools.

    Written so the service is exercisable without a key. It is a keyword
    router, not a language model, and it is labelled that way: its job is to
    prove the *plumbing* (schema, dispatch, errors, timeouts) end to end, and
    a real model swaps in behind ``AsyncModel`` unchanged.
    """

    _RULES: tuple[tuple[str, str, dict[str, Any]], ...] = (
        (r"天气|weather|forecast|气温", "weather", {}),
        (r"汇率|换算|convert|currency|美元|欧元", "convert_currency", {}),
        (r"待办|todo|任务清单|记一下|提醒我", "todo", {}),
        (r"日历|日程|安排|calendar|会议", "calendar", {}),
        (r"搜|查一下|notes|检索|search", "search", {}),
    )

    def __init__(self) -> None:
        self.calls = 0

    async def acomplete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]
    ) -> ChatMessage:
        self.calls += 1
        if self.calls > 1:
            # Second pass: the tool has run, so summarize instead of looping.
            last = next((m for m in reversed(messages) if m.role == "tool"), None)
            body = last.text() if last else ""
            return ChatMessage(role="assistant", content=f"Done. Tool said: {body[:300]}")

        task = next((m.text() for m in reversed(messages) if m.role == "user"), "")
        for pattern, name, defaults in self._RULES:
            if re.search(pattern, task, re.IGNORECASE):
                return ChatMessage(
                    role="assistant",
                    tool_calls=[ToolCall("c1", name, self._args(name, task, defaults))],
                )
        return ChatMessage(role="assistant", content=f"Nothing to look up for: {task}")

    @staticmethod
    def _args(name: str, task: str, defaults: dict[str, Any]) -> dict[str, Any]:
        """Best-effort arguments so the router produces *valid* calls."""
        if name == "weather":
            # Match a known city before falling back, so a question's first
            # word ("what") is never mistaken for a place name.
            for city in ("Beijing", "Shanghai", "Tokyo", "Paris", "Lisbon", "Sydney"):
                if city.lower() in task.lower():
                    return {"city": city, "days": 1}
            return {"city": "Shanghai", "days": 1}
        if name == "convert_currency":
            amount = re.search(r"\d+(?:\.\d+)?", task)
            return {
                "amount": float(amount.group(0)) if amount else 100.0,
                "source": "USD",
                "target": "CNY",
            }
        if name == "todo":
            return {"action": "add", "text": task.strip()[:80] or "new task"}
        if name == "calendar":
            return {"action": "list"}
        return {"query": task.strip() or "help", "k": 3}


__all__ = ["AsyncModel", "HeuristicModel", "ToolCallingAgent", "TurnResult"]
