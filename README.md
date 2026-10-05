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
## Results

Two prompt variants over the same 13 tasks, same corpus, same model:

| | `default` | `retrieval` |
| --- | --- | --- |
| pass rate | 12/13 (92.3%) | **13/13 (100%)** |
| `negative` | 2/3 | **3/3** |
| `lookup` | 4/4 | 4/4 |
| billed tokens | 35,636 | 36,524 |
| amplification | 2.92x | **2.72x** |
| tool calls | 45 | **41** |

The single difference is a retrieval-first prompt. The `default` prompt lost
one task: asked for the capital of Peru, the model answered from memory in
**zero tool calls**. The `retrieval` prompt fixed exactly that task and cost
2.5% more tokens while making 9% fewer tool calls.

Getting a clean number required fixing the grader first. It rejected the
correct answer to `lookup-03` because the model wrote "as \*data\*, never as
instructions" and the check searched for the literal string "as data". The fix
is `normalize()` in `agentloop/eval.py`, which folds Markdown emphasis and
typographic punctuation before matching. Every saved report was then re-scored
offline with `scripts/regrade.py`, so the corrected verdicts cost nothing:

```
report-v1.json  12/13 passed
report-v2.json  13/13 passed   lookup-03 FIXED
```

The lesson worth keeping: when a benchmark disagrees with a visibly correct
answer, check the grader before changing the model.
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

## W2: a tool-calling service (`agentkit`)

`agentloop` is the loop, built to be read. `agentkit` is the same loop under
the conditions a service imposes: schemas generated from Pydantic models,
five real tools, one error taxonomy, an HTTP surface, and messages that can
carry an image.

```powershell
python scripts\demo_w2.py          # offline trace of every failure mode
python -m unittest tests.test_agentkit -v
```

| File | Responsibility |
| --- | --- |
| `agentkit/schema.py` | Pydantic v2 argument models; the model *is* the published contract |
| `agentkit/registry.py` | name -> schema + callable; sync and async look identical to the model |
| `agentkit/errors.py` | `ErrorKind` taxonomy and the classifier that maps exceptions onto it |
| `agentkit/dispatch.py` | validate, run, time out, classify; one call in, one result out |
| `agentkit/messages.py` | content blocks, wire encoding, image collapsing, token estimate |
| `agentkit/chat.py` | the async turn loop, plus a deterministic router for offline runs |
| `agentkit/service.py` | FastAPI app: `/agent/run`, `/tools`, `/tools/{name}` |
| `agentkit/openai_model.py` | async adapter over W1's record/replay transports |

### The three failures, handled once

- **The model invents a tool name** -> `UNKNOWN_TOOL`. The message lists what
  is available, so the next attempt can succeed.
- **The model sends a bad argument** -> `BAD_ARGUMENTS`, with the offending
  key, the problem, and the expected type. `extra="forbid"` means an invented
  argument is reported instead of silently dropped.
- **The tool hangs or fails** -> `TIMEOUT` / `UPSTREAM`, classified, retryable,
  and returned as data. Never a 500.

A tool failure is a 200 with an error payload; a bad request is a 4xx. The
first is the model's to fix, the second is the caller's, and conflating them
is what turns a recoverable mistake into an outage.

### Why dispatch is async

Every tool call in a turn is awaited concurrently, so three independent
lookups cost one round trip. Sync tools run in a worker thread and everything
is wrapped in a per-tool budget, which is why a hung tool becomes a
`timeout` result instead of a hung server. Python 3.10 has two distinct
`TimeoutError` classes (`builtins` and `asyncio`), and neither subclasses the
other -- `agentkit/errors.py` checks both, which is the difference between
handling timeouts and handling the one path that happened to be tested.

### Images cost tokens, so they are collapsed

An image block is roughly a thousand tokens, and a naive multi-turn loop
resends it every turn. `ChatMessage.collapse_images()` replaces a seen image
with its caption once it has been answered, so later turns carry the text
instead of the pixels. The demo prints the saving.

## W2 numbers

- 48 tests added, 114 total, offline, no key
- image turn: ~1006 tokens -> ~15 after collapsing
- hung tool: bounded at ~56 ms instead of 400 ms, returned as `timeout`

## Requirements

Python 3.10+. Standard library only. `from __future__ import annotations`
is used throughout, so the type hints are inert at runtime.