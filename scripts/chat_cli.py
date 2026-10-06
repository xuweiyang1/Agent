"""Interactive command-line assistant with real model + tools.

    $env:DASHSCOPE_API_KEY = "sk-..."
    python scripts/chat_cli.py

    # or use a different provider:
    python scripts/chat_cli.py --base-url https://api.deepseek.com --model deepseek-chat

Type your message and press Enter. Type /exit to quit, /tools to see available tools,
/tokens to see usage so far.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentkit.chat import ToolCallingAgent
from agentkit.dispatch import ToolInvoker
from agentkit.messages import ChatMessage
from agentkit.openai_model import AsyncOpenAICompatibleModel
from agentkit.tools import build_registry

SYSTEM_PROMPT = """You are a helpful personal assistant. You have access to tools:

- weather: check the weather for a city
- convert_currency: convert between currencies
- search: search the knowledge base for information
- todo: manage your todo list (add / list / complete)
- calendar: manage calendar events (create / list)
- render_chart: render a bar chart to a PNG file

Guidelines:
- Use tools when they help answer the question.
- Be concise and friendly.
- If you create a calendar event or todo, confirm it clearly.
- When showing numbers, keep them reasonable.
"""


def print_banner() -> None:
    print("\n" + "=" * 60)
    print("  Personal Assistant (type /exit to quit, /tools for help)")
    print("=" * 60 + "\n")


def print_tools(agent: ToolCallingAgent) -> None:
    names = agent.invoker.registry.names()
    print(f"\n  Available tools: {', '.join(names)}\n")


def print_tokens(model: AsyncOpenAICompatibleModel) -> None:
    u = model.usage
    print(f"\n  Tokens so far: {u.calls} calls, "
          f"{u.prompt_tokens} prompt, {u.completion_tokens} completion, "
          f"{u.total} total\n")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive assistant CLI")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--env-var", default="DASHSCOPE_API_KEY")
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    try:
        model = AsyncOpenAICompatibleModel.from_env(
            env_var=args.env_var,
            model=args.model,
            base_url=args.base_url,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(f"hint: set {args.env_var} first, e.g.:", file=sys.stderr)
        print(f'  $env:{args.env_var} = "sk-..."', file=sys.stderr)
        return 2

    registry = build_registry()
    invoker = ToolInvoker(registry, default_timeout=30.0)
    agent = ToolCallingAgent(
        model,
        invoker,
        system_prompt=SYSTEM_PROMPT,
        max_turns=args.max_turns,
    )

    print_banner()
    print(f"  Model: {args.model}")
    print(f"  Base URL: {args.base_url}\n")

    messages: list[ChatMessage] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  bye!\n")
            break

        if not user_input:
            continue

        if user_input == "/exit":
            print("\n  bye!\n")
            break

        if user_input == "/tools":
            print_tools(agent)
            continue

        if user_input == "/tokens":
            print_tokens(model)
            continue

        if user_input == "/clear":
            messages = []
            print("  conversation cleared.\n")
            continue

        try:
            result = await agent.run(user_input, history=messages)
            messages = result.messages
            print(f"\nAssistant: {result.answer}\n")
        except Exception as exc:
            print(f"\n  error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
