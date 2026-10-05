"""Run the W2 service offline and print a trace of all three failure modes.

    python scripts/demo_w2.py

No key, no network. The model is the deterministic router, so what you see
is the plumbing: tool selection, argument validation, error classification,
timeout containment, and the token cost of an image that has been seen.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Same convention as the W1 scripts: runnable from the repo root without an
# install step.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from agentkit import (
    ChatMessage,
    ErrorKind,
    ImageBlock,
    TextBlock,
    ToolCallingAgent,
    ToolInvoker,
    ToolRegistry,
    build_registry,
    create_app,
    estimate_message_tokens,
)
from agentkit.chat import HeuristicModel


def header(text: str) -> None:
    print(f"\n=== {text} ===")


async def main() -> None:
    client = TestClient(create_app())

    header("tool catalog the model sees")
    for tool in client.get("/tools").json()["tools"]:
        required = tool["parameters"].get("required", [])
        print(f"  {tool['name']:<17} required={required}")

    header("1. model chooses a tool and supplies arguments")
    body = client.post("/agent/run", json={"task": "what is the weather in Paris?"}).json()
    print("  answer     :", body["answer"][:64])
    print("  tool calls :", body["tool_calls"])
    print("  tokens     :", body["tokens"], f"({body['elapsed_ms']} ms)")

    header("2. the model invents a tool name")
    response = client.post("/tools/send_email", json={"arguments": {"to": "a@b.c"}})
    print("  http       :", response.status_code)
    print("  kind       :", response.json()["detail"]["error"])
    print("  tells model:", response.json()["detail"]["message"])

    header("3. the model sends an argument of the wrong type")
    response = client.post("/tools/weather", json={"arguments": {"city": 123}})
    payload = response.json()["payload"]
    print("  http       :", response.status_code, "(the run is fine; the call is not)")
    print("  detail     :", json.dumps(payload["details"], ensure_ascii=False))

    header("4. a tool hangs")
    registry = ToolRegistry()

    @registry.tool("slow_lookup", "pretends to hang", timeout=0.05)
    def slow_lookup():
        import time

        time.sleep(0.4)
        return "too late"

    result = await ToolInvoker(registry).invoke("slow_lookup")
    print(f"  kind       : {result.failure.kind.value} (retryable={result.failure.retryable})")
    print(f"  bounded at : {result.latency_ms:.0f} ms instead of 400 ms")

    header("5. what an image costs, and what collapsing it saves")
    fresh = ChatMessage(
        role="user",
        content=[TextBlock("what is on this receipt?"), ImageBlock("data:image/png;base64,AAAA", caption="receipt: 42.50 CNY, coffee")],
    )
    seen = fresh.collapse_images()
    print(f"  with the image    : ~{estimate_message_tokens(fresh)} tokens")
    print(f"  after it is seen  : ~{estimate_message_tokens(seen)} tokens")
    print(f"  later turns read  : {seen.text()}")

    header("6. the agent still answers after a bad call")
    invoker = ToolInvoker(build_registry())
    agent = ToolCallingAgent(HeuristicModel(), invoker)
    result = await agent.run("add buy milk to my todo")
    print("  answer     :", result.answer[:64])
    print("  ok calls   :", [d.name for d in result.dispatch if d.ok])
    print("  failed     :", [(d.name, d.failure.kind.value) for d in result.failed_calls])


if __name__ == "__main__":
    asyncio.run(main())
