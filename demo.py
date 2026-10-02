"""Run the agent loop offline and print a trace.

    py demo.py

No API key is needed. The model is scripted, so the trace shows the loop's
mechanics rather than a vendor's output quality.
"""

from __future__ import annotations

from agentloop import Agent, FakeModel, Message, ToolCall, build_default_registry


def main() -> None:
    events: list[str] = []

    def observe(event: str, fields: dict) -> None:
        detail = " ".join(f"{k}={v}" for k, v in sorted(fields.items()))
        events.append(f"  [{event}] {detail}")

    script = [
        Message(role="assistant", tool_calls=[ToolCall("c1", "search", {"query": "context"})]),
        lambda _: Message(role="assistant", tool_calls=[ToolCall("c2", "read", {"id": "context-window"})]),
        lambda messages: Message(role="assistant", content=f"Answer: {messages[-1].content}"),
    ]

    agent = Agent(
        FakeModel(script),
        build_default_registry(),
        max_tokens=200,
        max_tool_output_chars=120,
        observer=observe,
        sleep=lambda _: None,
    )

    result = agent.run("How does the context window relate to an agent loop?")

    print("trace:")
    print("\n".join(events) if events else "  (none)")
    print()
    print(f"turns      : {result.turns}")
    print(f"retries    : {result.retries}")
    print(f"compacted  : {result.compacted} messages")
    print(f"tool calls : {len(result.results)}")
    print(f"tokens     : ~{result.tokens}")
    print()
    print(f"answer     : {result.answer}")


if __name__ == "__main__":
    main()