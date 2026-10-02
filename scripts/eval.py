"""Run the evaluation set and report a pass rate.

    py scripts\eval.py                      # replay against the fixture, no key
    py scripts\eval.py --live               # against a real endpoint
    py scripts\eval.py --record out.json    # live, and save for later replay
    py scripts\eval.py --compare a.json b.json

Replay mode is the default because a benchmark has to be repeatable. The
fixture is produced by --record, so the usual loop is: run --record once
against the live endpoint, then grade against the recorded run as often as
needed while changing prompts or retrieval.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentloop import Agent, build_default_registry
from agentloop.llm import LLMError
from agentloop.eval import TaskResult, grade, load_tasks
from agentloop.providers import OpenAICompatibleModel, RecordTransport, ReplayTransport, http_transport

DEFAULT_FIXTURE = Path(__file__).resolve().parent.parent / "eval" / "baseline.json"


def build_agent(args) -> tuple[Agent, OpenAICompatibleModel]:
    transport = None
    if args.record:
        transport = RecordTransport(
            lambda url, headers, payload: http_transport(url, headers, payload, timeout=90.0),
            args.record,
        )
    elif not args.live:
        transport = ReplayTransport(args.fixture)

    if args.live or args.record:
        model = OpenAICompatibleModel.from_env(
            env_var=args.env_var, model=args.model, base_url=args.base_url, transport=transport
        )
    else:
        model = OpenAICompatibleModel(
            model=args.model, api_key="replay", transport=transport
        )

    agent = Agent(model, build_default_registry(), max_turns=args.max_turns)
    return agent, model


def run(args) -> dict:
    tasks = load_tasks()
    agent, model = build_agent(args)
    results: list[TaskResult] = []

    for task in tasks:
        started = time.perf_counter()
        try:
            run_result = agent.run(task.question)
            answer, turns = run_result.answer, run_result.turns
            tool_calls = len(run_result.results)
            failed = sum(1 for r in run_result.results if not r.ok)
            retries = run_result.retries
            tokens = run_result.tokens
            elapsed = time.perf_counter() - started
        except Exception as exc:  # a transport failure is a task failure, not a crash
            results.append(
                TaskResult(task, "", 0, 0, 0, 0, 0, False, [f"exception: {type(exc).__name__}: {exc}"])
            )
            print(f"  {task.id:10} ERROR  {type(exc).__name__}: {exc}")
            continue

        passed, failures = grade(task, answer)
        results.append(
            TaskResult(task, answer, turns, tool_calls, failed, retries, tokens, passed, failures)
        )
        mark = "PASS" if passed else "FAIL"
        detail = "" if passed else "  <- " + "; ".join(failures)
        print(f"  {task.id:10} {mark}  turns={turns} calls={tool_calls} tokens={tokens}{detail}")

    return {
        "model": args.model,
        "mode": "live" if (args.live or args.record) else "replay",
        "results": [
            {
                "id": r.task.id,
                "category": r.task.category,
                "passed": r.passed,
                "failures": r.failures,
                "turns": r.turns,
                "tool_calls": r.tool_calls,
                "failed_calls": r.failed_calls,
                "retries": r.retries,
                "tokens": r.tokens,
                "answer": r.answer,
            }
            for r in results
        ],
        "usage": {
            "calls": model.usage.calls,
            "prompt_tokens": model.usage.prompt_tokens,
            "completion_tokens": model.usage.completion_tokens,
            "total_tokens": model.usage.total,
        },
    }


def summarize(report: dict) -> None:
    rows = report["results"]
    passed = sum(1 for r in rows if r["passed"])

    print()
    print("=" * 72)
    print(f"pass rate      : {passed}/{len(rows)}  ({100 * passed / max(1, len(rows)):.1f}%)")
    print(f"total tokens   : {report['usage']['total_tokens']}")
    print(f"tool calls     : {sum(r['tool_calls'] for r in rows)}"
          f"  (failed: {sum(r['failed_calls'] for r in rows)})")
    print(f"retries        : {sum(r['retries'] for r in rows)}")
    print()

    by_category: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row["passed"])
    print("by category")
    for category in sorted(by_category):
        outcomes = by_category[category]
        rate = 100 * sum(outcomes) / len(outcomes)
        print(f"  {category:12} {sum(outcomes)}/{len(outcomes)}  ({rate:5.1f}%)")

    failed = [r for r in rows if not r["passed"]]
    if failed:
        print()
        print("failures")
        for row in failed:
            print(f"  {row['id']:10} {'; '.join(row['failures'])}")
    print("=" * 72)


def compare(paths: list[str]) -> None:
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    ids = [r["id"] for r in reports[0]["results"]]

    print(f"{'task':12}" + "".join(f"{Path(p).name:>18}" for p in paths) + "   change")
    print("-" * 72)
    for task_id in ids:
        marks = []
        for report in reports:
            row = next(r for r in report["results"] if r["id"] == task_id)
            marks.append("PASS" if row["passed"] else "FAIL")
        change = ""
        if marks[0] != marks[-1]:
            change = "  REGRESSED" if marks[0] == "PASS" else "  FIXED"
        print(f"{task_id:12}" + "".join(f"{m:>18}" for m in marks) + change)
    print("-" * 72)
    for report, path in zip(reports, paths):
        rows = report["results"]
        passed = sum(1 for r in rows if r["passed"])
        print(f"{Path(path).name:12} {passed}/{len(rows)} passed, "
              f"{report['usage']['total_tokens']} tokens")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="call the real endpoint")
    parser.add_argument("--record", type=Path, default=None, help="live, and save exchanges here")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="fixture to replay")
    parser.add_argument("--report", type=Path, default=None, help="write the report as JSON")
    parser.add_argument("--compare", nargs="+", default=None, help="compare saved reports")
    parser.add_argument("--env-var", default="DEEPSEEK_API_KEY")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return 0

    if not args.live and not args.record and not args.fixture.exists():
        print(f"error: fixture not found: {args.fixture}", file=sys.stderr)
        print("       run with --record first, or use --live", file=sys.stderr)
        return 2

    try:
        report = run(args)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.live or args.record:
            print(f"       set the key first, e.g. `$env:{args.env_var} = 'sk-...'", file=sys.stderr)
        return 2

    summarize(report)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written to {args.report}")

    passed = sum(1 for r in report["results"] if r["passed"])
    return 0 if passed == len(report["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())