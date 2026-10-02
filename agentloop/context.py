"""Context compaction.

The interesting failure mode of an agent loop is not a wrong answer, it is
a transcript that grows without bound until the request no longer fits.
`compact` keeps the system prompt and the most recent exchanges, and turns
everything older into a short summary. That is deliberately lossy, and the
loss is measurable: `estimate_tokens` before and after is the number you
would put on a resume slide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .llm import Message


CompactFn = Callable[..., "CompactionResult"]


def estimate_tokens(text: str) -> int:
    """A cheap token estimate: about four characters per token.

    Not a real tokenizer on purpose. The runtime only needs a monotonic
    budget, and a real tokenizer would add a dependency without changing
    any decision this code makes.
    """
    return max(1, (len(text) + 3) // 4)


def message_tokens(message: Message) -> int:
    total = estimate_tokens(message.content)
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments))
    return total


@dataclass
class CompactionResult:
    messages: list[Message]
    dropped: int
    tokens_before: int
    tokens_after: int

    @property
    def saved(self) -> int:
        return self.tokens_before - self.tokens_after


def compact(
    messages: Sequence[Message],
    *,
    max_tokens: int,
    keep_tail: int = 4,
    summarize: bool = True,
) -> CompactionResult:
    """Shrink `messages` to fit `max_tokens`.

    System prompts are never dropped. Tool call and tool result pairs at the
    tail are kept whole, because a tool message without its matching request
    is not a valid transcript for most providers.
    """
    before = sum(message_tokens(m) for m in messages)
    if before <= max_tokens:
        return CompactionResult(list(messages), 0, before, before)

    system = [m for m in messages if m.role == "system"]
    body = [m for m in messages if m.role != "system"]

    split = _safe_split(body, keep_tail)
    head, tail = body[:split], body[split:]

    rebuilt: list[Message] = list(system)
    if summarize and head:
        rebuilt.append(_summarize(head))
    rebuilt.extend(tail)

    after = sum(message_tokens(m) for m in rebuilt)
    return CompactionResult(rebuilt, len(head), before, after)


def _safe_split(body: list[Message], keep_tail: int) -> int:
    """Find a cut point that does not orphan a tool result."""
    cut = max(0, len(body) - max(0, keep_tail))
    while cut > 0 and body[cut].role == "tool":
        cut -= 1
    return cut


def _summarize(head: Iterable[Message]) -> Message:
    """Replace a run of old messages with one note describing what happened."""
    lines: list[str] = []
    for message in head:
        if message.role == "tool":
            lines.append(f"- tool output ({len(message.content)} chars)")
        elif message.tool_calls:
            names = ", ".join(call.name for call in message.tool_calls)
            lines.append(f"- assistant called {names}")
        elif message.content:
            snippet = message.content.strip().splitlines()[0][:80]
            lines.append(f"- {message.role}: {snippet}")
    body = "\n".join(lines) if lines else "(nothing)"
    return Message(
        role="system",
        content="Summary of earlier turns:\n" + body,
    )