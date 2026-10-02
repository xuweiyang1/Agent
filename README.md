# agentloop

A minimal agent runtime in pure Python, no dependencies. Built as a reading
exercise in how an agent loop actually works: model call, tool dispatch,
retry with backoff, context compaction.

## Why it exists

Most agent projects wrap a framework and never touch the loop. This one is
the loop. It is deliberately small enough to read end to end, and every
non-obvious decision is explained in a docstring at the point it matters.

## Layout

| File | Responsibility |
| --- | --- |
| `agentloop/llm.py` | Message and ToolCall types, the `Model` protocol, a scripted `FakeModel`, a `FlakyModel` for retry tests |
| `agentloop/tools.py` | Tool registry, argument validation, error containment, two offline tools |
| `agentloop/context.py` | Token estimation and compaction that never orphans a tool result |
| `agentloop/runtime.py` | The loop itself: turns, dispatch, backoff, truncation, observer hooks |
| `agentloop/providers/` | OpenAI-compatible adapter (DeepSeek, Moonshot, Qwen), plus record and replay transports |

## Run it

```powershell
py demo.py
py -m unittest discover -s tests -t . -v
```

`demo.py` needs no API key. It runs a scripted model and prints a trace of
tool calls, turns, retry count, compaction, and estimated tokens.

With a key set, the same loop runs against a real endpoint instead of the
scripted one:

## What is worth looking at

- **Retry semantics.** `LLMError` carries `retryable`. Transient failures
  are retried with exponential backoff plus jitter; a non-retryable error
  is raised on the first attempt, because retrying it would fail the same
  way. Both paths are pinned by tests.
- **Error containment.** An unknown tool or a bad argument becomes an error
  message in the transcript, not an exception that kills the run. The model
  gets a chance to correct itself.
- **Compaction validity.** Naively cutting history can leave a tool result
  with no matching request, which several providers reject. `compact` walks
  the cut point backwards until the tail starts on a non-tool message, and
  the system prompt is never dropped.
- **Bounded tool output.** One verbose tool result can consume the window,
  so outputs are truncated from the middle with a visible marker.

## Numbers this produces

The runtime reports `turns`, `retries`, `compacted`, tool latency, and an
estimated token count per run. Those are the measurements a resume bullet
should be built from, and `max_tokens` makes the cost of a long run visible.

## Not done yet

These are the natural next steps, in the order they add value:

1. ~~A real provider adapter~~ and offline replay: done, see `agentloop/providers/`.
2. An evaluation set with a pass rate, so prompt and compaction changes can
   be compared instead of guessed at.
3. Persistent trace logs for after-the-fact replay.
4. Sub-agent spawning with an explicit budget.


## Talking to a real model

The loop only depends on the `Model` protocol, so a provider is an adapter
and nothing in `runtime.py` changes. One adapter covers DeepSeek, Moonshot,
Qwen, and most self-hosted gateways, because they all speak the same
`/chat/completions` shape.

```powershell
$env:DEEPSEEK_API_KEY = "sk-..."
py scripts\smoke_live.py
```

`smoke_live.py` runs three questions and prints what was actually measured:
turns, retries, tool call success, tokens, and latency. Those numbers, not
impressions, are what belongs on a resume line.

Point it at another endpoint without touching code:

```powershell
py scripts\smoke_live.py --base-url https://api.moonshot.cn/v1 --model kimi-k2
```

### Record and replay

A benchmark is worthless if the model behind the endpoint changes underneath
it. `--record` saves every request and response, and `ReplayTransport` serves
those back with no network access, which is why the test suite runs offline:

```powershell
py scripts\smoke_live.py --record tests\fixtures\my_run.json
```

Replay matches by position rather than by request hash. A hash would quietly
serve a stale response whenever two requests happen to look alike, which is
the one failure mode a benchmark must not have.

### Keys

Never put a key in a file git can see. Use an environment variable, or a
`.env` file, which `.gitignore` already covers. The adapter reads the key at
call time, and a test fails if the string `Bearer` ever appears in a fixture.
## Evaluation

`eval/tasks.jsonl` holds 13 tasks in four categories, chosen to separate the
failure modes rather than to look impressive:

| Category | What it isolates |
| --- | --- |
| `lookup` | one entry, one search should find it |
| `multi_hop` | two entries must be combined |
| `paraphrase` | a query worded unlike the corpus, testing retrieval rather than phrasing |
| `negative` | the corpus has no answer, so the honest reply is to say so |

Grading is deterministic string checking, not a second model as a judge. A
judge would drift, and a benchmark whose score moves when the repository did
not is worthless. `tests/test_eval.py` grades known hallucinations against the
negative tasks and requires every one of them to fail: a benchmark that always
passes is worse than no benchmark.

```powershell
# one live pass that also saves the exchanges for later replay
py scripts\eval.py --record eval\baseline.json --report eval\report-v1.json

# re-grade offline, no key and no cost, as often as you like
py scripts\eval.py --report eval\report-v2.json

# what changed between two runs
py scripts\eval.py --compare eval\report-v1.json eval\report-v2.json
```

### Two token numbers, both real

The summary reports *billed tokens* and *final transcript length* separately,
because they differ by the amplification factor: every turn resends the entire
history, so a four-turn answer bills roughly four times its final size. On the
first baseline run that factor was `2.92x`, which is the number that matters
when predicting cost.

### Prompt variants

`agentloop/prompts.py` holds a permissive prompt and a retrieval-first one, so
a prompt change becomes a measurable result. The retrieval-first variant exists
because of a measured failure: with the permissive prompt, the model answered
an out-of-corpus question from its own memory in zero tool calls.

Change the variant with `--prompt retrieval`, then compare reports.
## Getting the code onto an offline server

If the machine that runs this code cannot reach GitHub, then GitHub cannot
deliver the code to it: `git clone` there will fail. GitHub is the public
archive; a `git push` over SSH is the transfer channel.

On the server, once:

```bash
bash scripts/setup_server.sh
```

It checks `git` and a Python >= 3.10, creates a bare repo at `~/agent.git`,
clones a working tree, and prints the exact commands for the local side.

If the server is reachable over SSH directly:

```bash
git remote add server ssh://you@server/home/you/agent.git
```

If it is not, forward its sshd to a local port and push through the tunnel:

```bash
ssh -N -L 2222:localhost:22 you@gateway
git remote add server ssh://you@127.0.0.1:2222/home/you/agent.git
```

Both destinations at once, so one `git push origin main` updates both:

```powershell
.\scripts\push.ps1 -GitHubUrl git@github.com:you/agent.git `
                   -ServerUrl ssh://you@127.0.0.1:2222/home/you/agent.git
```

Then verify on the server:

```bash
cd ~/agent && python3 -m unittest discover -s tests -t . -v
```

Commit on whichever side you are working, push, and pull on the other. Do
not mount a shared filesystem and edit from both places, and do not move
files by hand after the first import.

## Requirements

Python 3.10+. Standard library only. `from __future__ import annotations`
is used throughout, so the type hints are inert at runtime.