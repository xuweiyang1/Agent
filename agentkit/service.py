"""The HTTP surface: one endpoint that runs the agent, one that inspects it.

``create_app`` is a factory rather than a module-level ``app`` so a test can
build a service around a chosen registry or model and get a fresh state each
time. Import-time construction would make the tests share mutable todo lists,
which is the classic way a green suite hides a real bug.

Note the split on errors: a *tool* failure is data inside the 200 response
(the model is expected to read and recover), while a *request* failure is a
4xx. Conflating the two is what turns a recoverable model mistake into a 500.
"""

from __future__ import annotations

from typing import Any, Iterable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .chat import AsyncModel, HeuristicModel, ToolCallingAgent
from .dispatch import ToolInvoker
from .messages import ChatMessage, ImageBlock, TextBlock, to_wire
from .registry import ToolRegistry
from .tools import build_registry


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, description="What the user asked for.")
    max_turns: int = Field(8, ge=1, le=20)
    history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Prior turns as {role, content}; content may be a string or block list.",
    )


class RunResponse(BaseModel):
    answer: str
    turns: int
    tokens: int
    elapsed_ms: float
    truncated: bool
    tool_calls: list[dict[str, Any]]
    trace: list[dict[str, Any]]


class ToolCallRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)


def create_app(
    *,
    registry: ToolRegistry | None = None,
    model: AsyncModel | None = None,
    system_prompt: str = "You are a helpful assistant. Use tools when they help.",
) -> FastAPI:
    """Build the FastAPI app around a registry and a model."""
    reg = registry or build_registry()
    invoker = ToolInvoker(reg)
    events: list[dict[str, Any]] = []

    def observe(event: str, fields: dict[str, Any]) -> None:
        events.append({"event": event, **fields})

    agent = ToolCallingAgent(
        model or HeuristicModel(),
        invoker,
        system_prompt=system_prompt,
        observer=observe,
    )

    app = FastAPI(title="agentkit", version="0.2.0")
    app.state.registry = reg
    app.state.agent = agent
    app.state.invoker = invoker

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"ok": True, "tools": reg.names(), "model": type(agent.model).__name__}

    @app.get("/tools")
    async def tools() -> dict[str, Any]:
        return {"tools": reg.schemas()}

    @app.post("/tools/{name}")
    async def call_tool(name: str, request: ToolCallRequest) -> dict[str, Any]:
        """Call one tool directly. Useful for debugging and for tests."""
        result = await invoker.invoke(name, request.arguments, call_id="direct")
        if not result.ok and result.failure is not None and result.failure.kind.value == "unknown_tool":
            raise HTTPException(status_code=404, detail=result.failure.to_payload())
        return {"ok": result.ok, "latency_ms": result.latency_ms, "payload": result.payload()}

    @app.post("/agent/run", response_model=RunResponse)
    async def run(request: RunRequest) -> RunResponse:
        history = [_decode_message(item) for item in request.history]
        agent.max_turns = request.max_turns
        events.clear()
        result = await agent.run(request.task, history=history)
        return RunResponse(
            answer=result.answer,
            turns=result.turns,
            tokens=result.tokens,
            elapsed_ms=round(result.elapsed_ms, 3),
            truncated=result.truncated,
            tool_calls=[
                {
                    "name": d.name,
                    "ok": d.ok,
                    "latency_ms": round(d.latency_ms, 3),
                    "error": d.failure.kind.value if d.failure else None,
                }
                for d in result.dispatch
            ],
            trace=list(events),
        )

    return app


def _decode_message(item: dict[str, Any]) -> ChatMessage:
    """Rebuild a history entry, including images, from JSON."""
    role = str(item.get("role", "user"))
    raw = item.get("content", "")
    if isinstance(raw, list):
        blocks: list[Any] = []
        for block in raw:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                url = (block.get("image_url") or {}).get("url", "")
                blocks.append(ImageBlock(url=url, caption=block.get("caption", "")))
            else:
                blocks.append(TextBlock(str(block.get("text", ""))))
        return ChatMessage(role=role, content=blocks)
    return ChatMessage(role=role, content=str(raw))


__all__ = ["RunRequest", "RunResponse", "ToolCallRequest", "create_app"]
