"""An OpenAI-compatible chat-completions adapter, with record and replay.

DeepSeek, Moonshot, Qwen, and most self-hosted gateways all speak the same
`/chat/completions` shape, so one adapter covers them by changing `base_url`
and `model`. Two transports are provided:

- `http_transport`  talks to the network.
- `RecordTransport` wraps another transport and writes every exchange to a
  JSON file, so a real run can be replayed later.
- `ReplayTransport` reads those files, which is how the test suite stays
  offline and how a benchmark run stays reproducible after the model behind
  an endpoint has changed.
"""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from ..llm import LLMError, Message, ToolCall

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class Transport(Protocol):
    """Sends one request payload and returns the decoded response body."""

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass
class Usage:
    """Token accounting, summed across every call a model makes."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, other: dict[str, Any]) -> None:
        self.calls += 1
        self.prompt_tokens += int(other.get("prompt_tokens") or 0)
        self.completion_tokens += int(other.get("completion_tokens") or 0)


def http_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    *,
    timeout: float = 60.0,
) -> dict[str, Any]:
    """POST JSON over the network and classify failures for the retry loop."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = _error_detail(exc)
        raise LLMError(
            f"HTTP {exc.code}: {detail}",
            retryable=exc.code in RETRYABLE_STATUS,
        ) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        raise LLMError(f"network failure: {exc}", retryable=True) from exc
    except json.JSONDecodeError as exc:
        raise LLMError(f"malformed response body: {exc}", retryable=False) from exc


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return exc.reason or "no body"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw[:500]
    error = parsed.get("error")
    if isinstance(error, dict):
        return str(error.get("message") or error)
    return str(error or parsed)[:500]


class RecordTransport:
    """Wraps a transport and appends each exchange to a JSON file.

    The file is a list of `{"request": ..., "response": ...}` objects, which
    is exactly what `ReplayTransport` expects. Requests are recorded too,
    because when a replay stops matching, the diff between the recorded and
    the new request is the whole diagnostic.
    """

    def __init__(self, inner: Transport, path: str | Path) -> None:
        self.inner = inner
        self.path = Path(path)
        self.exchanges: list[dict[str, Any]] = []
        if self.path.exists():
            self.exchanges = json.loads(self.path.read_text(encoding="utf-8"))

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        response = self.inner(url, headers, payload)
        self.exchanges.append({"request": payload, "response": response})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.exchanges, indent=2), encoding="utf-8")
        return response


class ReplayTransport:
    """Serves recorded responses in order, with no network access.

    Replaying by position rather than by request hash is deliberate: a hash
    would silently serve a stale response that happens to have the same
    request, which is the failure mode you least want in a benchmark.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.exchanges: list[dict[str, Any]] = json.loads(self.path.read_text(encoding="utf-8"))
        self.cursor = 0
        self.requests: list[dict[str, Any]] = []

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        if self.cursor >= len(self.exchanges):
            raise LLMError(
                f"replay exhausted after {self.cursor} exchange(s): {self.path.name}",
                retryable=False,
            )
        entry = self.exchanges[self.cursor]
        self.cursor += 1
        return entry["response"]


class OpenAICompatibleModel:
    """Adapts a `/chat/completions` endpoint to the `Model` protocol."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        transport: Transport | None = None,
        timeout: float = 60.0,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers or {}
        self.usage = Usage()
        self.durations: list[float] = []

        if transport is not None:
            self._transport = transport
            self._api_key = api_key or ""
        else:
            # Read the key at call time, not import time, so tests and
            # notebooks can set it after the module is loaded.
            self._api_key = api_key
            self._transport = lambda url, headers, payload: http_transport(
                url, headers, payload, timeout=self.timeout
            )

    @classmethod
    def from_env(
        cls,
        *,
        env_var: str = "DEEPSEEK_API_KEY",
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        **kwargs: Any,
    ) -> "OpenAICompatibleModel":
        """Build a live model from an environment variable.

        The key is never stored in a file that could be committed; the repo's
        `.gitignore` covers `.env` for the same reason.
        """
        key = os.environ.get(env_var)
        if not key:
            raise LLMError(f"{env_var} is not set", retryable=False)
        return cls(api_key=key, model=model, base_url=base_url, **kwargs)

    def complete(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]
    ) -> Message:
        payload = self._build_payload(messages, tools)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            **self.extra_headers,
        }

        started = time.perf_counter()
        body = self._transport(url, headers, payload)
        self.durations.append(time.perf_counter() - started)

        if "usage" in body and isinstance(body["usage"], dict):
            self.usage.add(body["usage"])

        return self._parse(body)

    def _build_payload(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]]
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_encode_message(m) for m in messages],
        }
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
            payload["tool_choice"] = "auto"
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _parse(self, body: dict[str, Any]) -> Message:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(f"response has no choices: {str(body)[:300]}", retryable=False)

        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content") or ""

        calls: list[ToolCall] = []
        for index, raw in enumerate(message.get("tool_calls") or []):
            function = raw.get("function") or {}
            calls.append(
                ToolCall(
                    id=str(raw.get("id") or f"call_{index}"),
                    name=str(function.get("name") or ""),
                    arguments=_decode_arguments(function.get("arguments")),
                )
            )

        if not content and not calls:
            finish = choice.get("finish_reason")
            raise LLMError(
                f"assistant message is empty (finish_reason={finish!r})",
                retryable=False,
            )

        return Message(role="assistant", content=content, tool_calls=calls)


def _encode_message(message: Message) -> dict[str, Any]:
    """Translate our Message into the OpenAI wire format."""
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id or "",
        }
    if message.tool_calls:
        return {
            "role": message.role,
            "content": message.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ],
        }
    return {"role": message.role, "content": message.content}


def _decode_arguments(raw: Any) -> dict[str, Any]:
    """Parse tool arguments, which arrive as a JSON string, sometimes invalid.

    A model that emits malformed JSON is a normal occurrence, not a bug in
    this code, so it is surfaced as an empty argument set and the registry
    reports the missing required key. That keeps the error inside the
    transcript where the model can see and correct it.
    """
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}