"""Talk to a real endpoint and print what the runtime measured.

    set DEEPSEEK_API_KEY=sk-...
    py scripts\smoke_live.py

    # or point it at any OpenAI-compatible gateway
    py scripts\smoke_live.py --base-url https://api.moonshot.cn/v1 --model kimi-k2

Nothing here writes to the transcript files; run with --record to save the
exchanges so the same task can be replayed offline later.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentloop import Agent, build_default_registry
from agentloop.providers import OpenAICompatibleModel, RecordTransport, http_transport

QUESTIONS = [
    "What is an agent loop, and why does it need a tool registry?",
    "Why can a long agent run exceed the context window?",
    "What is the difference between a retryable and a non-retryable error?",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--record", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    if not os.environ.get(args.env_var):
        print(f"error: {args.env_var} is not set", file=sys.stderr)
        return 2

    transport = None
    if args.record is not None:
        transport = RecordTransport(
            lambda url, headers, payload: http_transport(url, headers, payload, timeout=90.0),
            args.record,
        )

    model = OpenAICompatibleModel.from_env(
        env_var=args.env_var,
        model=args.model,
        base_url=args.base_url,
        transport=transport,
        max_tokens=args.max_tokens,
    )

    agent = Agent(model, build_default_registry(), max_tokens=4000, max_tool_output_chars=2000)

    for index, question in enumerate(QUESTIONS, start=1):
        print(f"\n{'=' * 72}\n[{index}/{len(QUESTIONS)}] {question}\n{'=' * 72}")
        result = agent.run(question)
        print(f"answer     : {result.answer[:400]}")
        print(f"turns      : {result.turns}")
        print(f"retries    : {result.retries}")
        print(f"tool calls : {len(result.results)} ({sum(1 for r in result.results if r.ok)} ok)")
        print(f"compacted  : {result.compacted} messages")

    print(f"\n{'=' * 72}\ntotals\n{'=' * 72}")
    print(f"api calls       : {model.usage.calls}")
    print(f"prompt tokens   : {model.usage.prompt_tokens}")
    print(f"completion      : {model.usage.completion_tokens}")
    print(f"total tokens    : {model.usage.total}")
    if model.durations:
        print(f"mean latency    : {sum(model.durations) / len(model.durations):.2f}s")
        print(f"max latency     : {max(model.durations):.2f}s")
    if args.record:
        print(f"recorded to     : {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())