"""Model boundary: one tiny protocol plus a scripted fake.

The runtime never talks to a vendor SDK directly. It only needs something
that maps a message list plus a tool schema to the next assistant message,
which is what `Model` describes. `FakeModel` makes the whole loop testable
offline, which is what lets us assert on retries and compaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence, runtime_checkable


class LLMError(RuntimeError):
    """Raised when a model call fails.

    A `retryable` error is a transient failure: rate limit, timeout, flaky
    socket. The runtime is allowed to retry those with backoff. A
    non-retryable error, such as a malformed request, must not be retried,
    because the next attempt would fail identically.
    """

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class ToolCall:
    """A model's request to run one tool with already-parsed arguments."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """One entry in the conversation.

    `role` is one of: system, user, assistant, tool. Assistant messages may
    carry `tool_calls`; tool messages must carry `tool_call_id` so the model
    can match a result to the request that produced it.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            payload["tool_calls"] = [
                {"id": call.id, "name": call.name, "arguments": call.arguments}
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@runtime_checkable
class Model(Protocol):
    """The only contract the runtime depends on."""

    def complete(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]
    ) -> Message:
        ...


class FakeModel:
    """A deterministic model driven by a script of responses.

    Each script entry is either a `Message`, which is returned as-is, or a
    callable that receives the current message list. Callables let a test
    react to the transcript, for example to echo back a tool's output.
    """

    def __init__(self, script: Iterable[Message | Callable[[Sequence[Message]], Message]]):
        self._script: list[Any] = list(script)
        self.calls: list[list[Message]] = []

    @property
    def remaining(self) -> int:
        return len(self._script)

    def complete(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]
    ) -> Message:
        self.calls.append(list(messages))
        if not self._script:
            raise LLMError("script exhausted", retryable=False)
        step = self._script.pop(0)
        if callable(step):
            return step(messages)
        return step

    def push(self, step: Message | Callable[[Sequence[Message]], Message]) -> None:
        """Queue another response, e.g. after a retry was consumed."""
        self._script.append(step)


class FlakyModel:
    """Wraps a model and fails the first `fail_times` calls it sees.

    Used to prove that backoff actually recovers instead of only existing
    in a docstring.
    """

    def __init__(self, inner: Model, fail_times: int, *, retryable: bool = True) -> None:
        self.inner = inner
        self.fail_times = fail_times
        self.retryable = retryable
        self.attempts = 0

    def complete(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]
    ) -> Message:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise LLMError(f"transient failure #{self.attempts}", retryable=self.retryable)
        return self.inner.complete(messages, tools)