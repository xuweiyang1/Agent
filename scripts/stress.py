"""Force the two paths the normal runs never touch: retry and compaction.

    py scripts\stress.py --live          # needs a key
    py scripts\stress.py                 # re-uses the last recorded run

Three experiments, each answering a question the code currently cannot:

1. `retry`      inject failures into a real transport and confirm the run
                still completes, and how much it costs.
2. `compaction` run the same long task with a generous window and with a
                window tight enough to force compaction, then compare the
                answers. A compaction that saves tokens by changing the answer
                is a regression, not an optimisation.
3. `ceiling`    give a task a turn budget too small to finish and confirm the
                loop stops instead of running away.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentloop import Agent, build_default_registry
from agentloop.providers import (
    OpenAICompatibleModel,
    RecordTransport,
    ReplayTransport,
    http_transport,
)
from agentloop.providers.faults import FaultInjectingTransport
from agentloop.prompts import get as get_prompt

# Long enough that a tight window actually has to compact several times.
LONG_TASK = (
    "Explain, step by step and in detail, how an agent loop works, what the "
    "context window is, why compaction is needed, how retries should be "
    "classified, and how embedding retrieval differs from keyword search. "
    "Cover each topic in a separate paragraph."
)


def transport_for(args, name: str, timeout: float):
    """Replay a recording when one exists, otherwise record a live run.

    Re-running the stress experiments must not cost money, but the very first
    run has nothing to replay, so the same call site serves both cases.
    """
    path = Path(args.record_dir) / name
    if not args.live and path.exists():
        return ReplayTransport(path)
    return RecordTransport(
        lambda u, h, p: http_transport(u, h, p, timeout=timeout),
        path,
    )


def make_model(args, transport):
    if args.live:
        return OpenAICompatibleModel.from_env(
            env_var=args.env_var, model=args.model, base_url=args.base_url, transport=transport
        )
    return OpenAICompatibleModel(model=args.model, api_key="live", transport=transport)


def experiment_retry(args) -> dict:
    print("\n[1/3] retry under injected failures")
    print("-" * 72)
    outcomes = []

    for fail_first in (1, 2, 3):
        inner = transport_for(args, f"retry-{fail_first}.json", 90.0)
        fault = FaultInjectingTransport(inner, fail_first=fail_first)
        model = make_model(args, fault)
        agent = Agent(model, build_default_registry(), max_retries=5)

        result = agent.run("What is an agent loop?")
        ok = bool(result.answer)
        outcomes.append({"fail_first": fail_first, "recovered": ok, "retries": result.retries,
                         "injected": fault.injected, "tokens": result.tokens})
        print(f"  fail_first={fail_first}  recovered={ok}  retries={result.retries}"
              f"  injected={fault.injected}  tokens={result.tokens}")

    return {"experiment": "retry", "outcomes": outcomes}


def experiment_compaction(args) -> dict:
    print("\n[2/3] compaction: does a tight window change the answer?")
    print("-" * 72)
    outcomes = []

    for label, window in (("generous", None), ("tight", args.window)):
        model = make_model(args, transport_for(args, f"compact-{label}.json", 120.0))
        agent = Agent(
            model,
            build_default_registry(),
            system_prompt=get_prompt(args.prompt),
            max_tokens=window,
            max_turns=args.max_turns,
        )
        result = agent.run(LONG_TASK)
        outcomes.append({
            "variant": label,
            "window": window,
            "compacted": result.compacted,
            "turns": result.turns,
            "tokens": result.tokens,
            "answer_chars": len(result.answer),
            "answer": result.answer,
        })
        print(f"  {label:9} window={str(window):>7}  compacted={result.compacted:3}"
              f"  turns={result.turns}  final_tokens={result.tokens:5}  chars={len(result.answer)}")

    generous, tight = outcomes[0]["answer"], outcomes[1]["answer"]
    kept = _overlap(generous, tight)
    print(f"\n  keyword overlap between the two answers: {kept:.0%}")
    return {"experiment": "compaction", "overlap": kept, "outcomes": outcomes}


def experiment_ceiling(args) -> dict:
    print("\n[3/3] turn ceiling")
    print("-" * 72)
    model = make_model(args, transport_for(args, "ceiling.json", 90.0))
    agent = Agent(model, build_default_registry(), max_turns=1)
    result = agent.run(LONG_TASK)
    print(f"  max_turns=1  turns={result.turns}  answer_empty={not result.answer}"
          f"  tool_calls={len(result.results)}")
    return {
        "experiment": "ceiling",
        "turns": result.turns,
        "answer_empty": not result.answer,
        "tool_calls": len(result.results),
    }


def _overlap(a: str, b: str) -> float:
    """Share of significant words in `a` that also appear in `b`."""
    import re

    words = lambda s: {w for w in re.findall(r"[a-z]{4,}", s.lower())}
    left, right = words(a), words(b)
    if not left:
        return 0.0
    return len(left & right) / len(left)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--env-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--prompt", default="retrieval")
    parser.add_argument("--window", type=int, default=600, help="token window that forces compaction")
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--record-dir", default="eval/stress")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    Path(args.record_dir).mkdir(parents=True, exist_ok=True)

    if not args.live:
        import os

        has_recordings = any(Path(args.record_dir).glob("*.json"))
        if not has_recordings and not os.environ.get(args.env_var):
            print(f"error: no recordings in {args.record_dir} and {args.env_var} is not set", file=sys.stderr)
            print("       run once with --live to record, then re-run offline", file=sys.stderr)
            return 2

    report = {
        "retry": experiment_retry(args),
        "compaction": experiment_compaction(args),
        "ceiling": experiment_ceiling(args),
    }

    print("\n" + "=" * 72)
    retry = report["retry"]["outcomes"]
    print(f"retry      : {sum(1 for o in retry if o['recovered'])}/{len(retry)} runs recovered"
          f" after {sum(o['injected'] for o in retry)} injected failures")
    print(f"compaction : {report['compaction']['outcomes'][1]['compacted']} messages dropped,"
          f" {report['compaction']['overlap']:.0%} answer overlap with the uncompacted run")
    print(f"ceiling    : stopped at {report['ceiling']['turns']} turn,"
          f" empty answer = {report['ceiling']['answer_empty']}")
    print("=" * 72)

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())