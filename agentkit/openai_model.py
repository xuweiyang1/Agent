"""An async OpenAI-compatible model that keeps W1's record/replay.

The whole point of reusing ``agentloop.providers`` here is that a real
function-calling run can be recorded once and then replayed offline forever.
Tool schemas are published exactly as the registry renders them, so what the
model was allowed to call is the same object the dispatcher validates against.

The one thing the W1 adapter cannot do is encode an image, because its
payload builder assumes string content. Rather than reaching into its
privates, ``_BlockAwareModel`` subclasses it and overrides only the payload
builder, inheriting the transport handling, retry classification, usage
accounting, and response parsing unchanged. A subclass overriding one seam is
the difference between an adapter and a pile of ``getattr`` calls.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Sequence

from agentloop.llm import LLMError, Message
from agentloop.providers import OpenAICompatibleModel

from .messages import ChatMessage, to_wire


class _BlockAwareModel(OpenAICompatibleModel):
    """W1's adapter, with payload building taught about content blocks."""

    enable_search: bool = False

    def _build_payload(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": to_wire(messages),
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = "auto"
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        if self.enable_search:
            payload["enable_search"] = True
        return payload


class AsyncOpenAICompatibleModel:
    """``AsyncModel`` over ``/chat/completions``, with a pluggable transport.

    Pass a ``ReplayTransport`` and no network is touched; pass nothing and the
    live HTTP transport is built lazily, so importing this module never needs
    a key.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        temperature: float | None = None,
        max_tokens: int | None = None,
        transport: Any | None = None,
        timeout: float = 60.0,
        enable_search: bool = False,
    ) -> None:
        self._sync = _BlockAwareModel(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            transport=transport,
            timeout=timeout,
        )
        self._sync.enable_search = enable_search
        self.model = self._sync.model

    @property
    def usage(self) -> Any:
        """Token accounting, since the adapter is the only thing that sees it."""
        return self._sync.usage

    @classmethod
    def from_env(
        cls, *, env_var: str = "DEEPSEEK_API_KEY", **kwargs: Any
    ) -> "AsyncOpenAICompatibleModel":
        key = os.environ.get(env_var)
        if not key:
            raise LLMError(f"{env_var} is not set", retryable=False)
        return cls(key, **kwargs)

    async def acomplete(
        self, messages: Sequence[ChatMessage], tools: Sequence[dict[str, Any]]
    ) -> ChatMessage:
        """One model call, off the event loop, with the transcript intact.

        ``http_transport`` blocks, so the whole call runs in a worker thread.
        That keeps the event loop free for other requests and lets a
        cancelled request propagate instead of pinning a thread until the
        socket times out.
        """
        reply: Message = await asyncio.to_thread(
            self._sync.complete, list(messages), list(tools)
        )
        return ChatMessage(
            role="assistant",
            content=reply.content or "",
            tool_calls=list(reply.tool_calls),
        )


__all__ = ["AsyncOpenAICompatibleModel"]
