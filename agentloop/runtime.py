"""The agent loop: model call, tool dispatch, retry, repeat.

This module is the reason the project exists. Everything else is support.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from .context import CompactFn, compact as default_compact, message_tokens
from .llm import LLMError, Message, Model, ToolCall
from .tools import ToolError, ToolRegistry


@dataclass
class ToolResult:
    call: ToolCall
    ok: bool
    output: str
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    messages: list[Message]
    answer: str
    results: list[ToolResult] = field(default_factory=list)
    turns: int = 0
    retries: int = 0
    compacted: int = 0

    @property
    def tokens(self) -> int:
        return sum(message_tokens(m) for m in self.messages)


class Agent:
    """Runs a bounded loop until the model answers without tool calls."""

    def __init__(
        self,
        model: Model,
        tools: ToolRegistry,
        *,
        system_prompt: str = "You are a helpful assistant. Use tools when they help.",
        max_turns: int = 8,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_tokens: Optional[int] = None,
        max_tool_output_chars: Optional[int] = None,
        compact: CompactFn = default_compact,
        sleep: Callable[[float], None] = time.sleep,
        seed: int = 0,
        observer: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_tokens = max_tokens
        self.max_tool_output_chars = max_tool_output_chars
        self.compact = compact
        self.sleep = sleep
        self.random = random.Random(seed)
        self.observer = observer

    def run(self, task: str) -> AgentResult:
        messages: list[Message] = []
        if self.system_prompt:
            messages.append(Message(role="system", content=self.system_prompt))
        messages.append(Message(role="user", content=task))

        results: list[ToolResult] = []
        retries = 0
        compacted = 0
        turns = 0

        while turns < self.max_turns:
            turns += 1
            if self.max_tokens is not None:
                outcome = self.compact(messages, max_tokens=self.max_tokens)
                if outcome.dropped:
                    messages = outcome.messages
                    compacted += outcome.dropped
                    self._emit("compact", saved=outcome.saved, dropped=outcome.dropped)

            reply = self._complete(messages)
            retries += self._last_retries
            messages.append(reply)

            if not reply.tool_calls:
                return AgentResult(messages, reply.content, results, turns, retries, compacted)

            for call in reply.tool_calls:
                result = self._dispatch(call)
                results.append(result)
                messages.append(
                    Message(role="tool", content=result.output, tool_call_id=call.id)
                )

        self._emit("max_turns", turns=turns)
        return AgentResult(messages, "", results, turns, retries, compacted)

    def _complete(self, messages: Sequence[Message]) -> Message:
        """Call the model, retrying only transient failures, with backoff."""
        self._last_retries = 0
        attempt = 0
        while True:
            try:
                return self.model.complete(list(messages), self.tools.schemas())
            except LLMError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                delay = self.base_delay * (2**attempt) * (1 + 0.3 * self.random.random())
                attempt += 1
                self._last_retries = attempt
                self._emit("retry", attempt=attempt, delay=round(delay, 4), error=str(exc))
                self.sleep(delay)

    def _dispatch(self, call: ToolCall) -> ToolResult:
        start = time.perf_counter()
        try:
            output = self.tools.call(call)
            ok = True
        except ToolError as exc:
            output = f"error: {exc}"
            ok = False
        elapsed = (time.perf_counter() - start) * 1000
        output = self._truncate(output)
        self._emit("tool", name=call.name, ok=ok, latency_ms=round(elapsed, 3))
        return ToolResult(call, ok, output, elapsed)

    def _truncate(self, text: str) -> str:
        """Bound one tool output so a single result cannot flood the window."""
        limit = self.max_tool_output_chars
        if limit is None or len(text) <= limit:
            return text
        head = text[: limit // 2]
        tail = text[-(limit // 2) :]
        dropped = len(text) - len(head) - len(tail)
        return f"{head}\n... [{dropped} chars truncated] ...\n{tail}"

    def _emit(self, event: str, **fields: Any) -> None:
        if self.observer is not None:
            self.observer(event, fields)