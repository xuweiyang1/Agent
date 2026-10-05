# agentloop

A seven-week build of an agent system, in Python. Each week adds one real
capability to the same project, and every claim in here is backed by a number
the test suite can reproduce offline.

| Package | What it is | Entry point |
| --- | --- | --- |
| `agentloop/` | a minimal agent loop in pure stdlib: model call, tool dispatch, retry with backoff, context compaction | `python demo.py` |
| `agentkit/` | the loop as a service: Pydantic tool schemas, five tools, an error taxonomy, async dispatch, FastAPI, messages that carry images | `python scripts/demo_w2.py` |
| `mcp_server/` | a filesystem MCP server with a real path sandbox, plus its cloud-function entry point | `python scripts/demo_w3.py` |
| `retrieval/` | chunking, BM25 behind a `VectorStore` protocol, and the naive RAG baseline W6 is measured against | `python scripts/rag_baseline.py` |

**Everything runs offline.** No API key is needed for any test, demo, or
baseline: the model is scripted or heuristic, the index is BM25, and network
transports have record/replay doubles. A real model is called only when a
number is being produced for the record.

## Getting started

Requires Python 3.10 or newer.

```powershell
git clone https://github.com/xuweiyang1/Agent.git
cd Agent
python -m pip install -r requirements.txt

# does it work? this is the whole check -- no key, no network, no cost
python -m unittest discover -s tests -t .
```

That command should print `OK` with 301 tests. If it does, everything below is
reproducible on your machine; if it does not, the failure is a real signal
about the environment rather than a flaky test.

Then run whichever week you care about:

```powershell
python demo.py                      # W1: the loop, with a trace
python scripts\demo_w2.py           # W2: tool calling, all three failure modes
python scripts\demo_w3.py           # W3: sandbox escapes, MCP, charts
python scripts\rag_baseline.py       # W3.5: the retrieval baseline numbers
python scripts\demo_w4.py            # W4: the planner, its branches and its checkpoint
python scripts\demo_w5.py            # W5: three memory layers, across a restart
python scripts\check_docs.py         # docs drift check: docs vs code
```

None of these need an API key. `agentloop/` needs no third-party packages at
all, so if you only want W1 you can skip the install.

### Where to look next

| If you want | Read |
| --- | --- |
| what the project is and where it is going | `docs/ROADMAP.md` |
| the rules for working on it across two machines | `docs/WORKFLOW.md` |
| a machine-readable brief for an AI agent picking this up | `AGENTS.md` |
| the decisions behind the code | the module docstrings |

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
| `graph/` | W4: the LangGraph planner -- `plan.py` (decisions), `nodes.py` (thin), `build.py` (wiring) |
| `agentloop/providers/` | OpenAI-compatible adapter (DeepSeek, Moonshot, Qwen), plus record and replay transports |

## Run it

```powershell
python demo.py
python -m unittest discover -s tests -t . -v
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
python scripts\smoke_live.py
```

`smoke_live.py` runs three questions and prints what was actually measured:
turns, retries, tool call success, tokens, and latency. Those numbers, not
impressions, are what belongs on a resume line.

Point it at another endpoint without touching code:

```powershell
python scripts\smoke_live.py --base-url https://api.moonshot.cn/v1 --model kimi-k2
```

### Record and replay

A benchmark is worthless if the model behind the endpoint changes underneath
it. `--record` saves every request and response, and `ReplayTransport` serves
those back with no network access, which is why the test suite runs offline:

```powershell
python scripts\smoke_live.py --record tests\fixtures\my_run.json
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
python scripts\eval.py --record eval\baseline.json --report eval\report-v1.json

# re-grade offline, no key and no cost, as often as you like
python scripts\eval.py --report eval\report-v2.json

# what changed between two runs
python scripts\eval.py --compare eval\report-v1.json eval\report-v2.json
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

## W3: a filesystem MCP server, and a chart tool

Two things land in W3, and they are deliberately opposite in cost.

The **MCP server** is a sandbox plus a protocol binding. The sandbox is pure
logic with no MCP import, so the security boundary is tested directly rather
than through a server. The **chart tool** is output multimodality, and it
changes nothing structural: it is a tool like the other six.

```powershell
python scripts\demo_w3.py
python -m unittest tests.test_mcp_fs tests.test_chart tests.test_cloud -v
```

| File | Responsibility |
| --- | --- |
| `mcp_server/fs.py` | the sandbox: resolve, read, write, list, search -- no MCP dependency |
| `mcp_server/adapter.py` | `MCPServer` under the `FastMCP` name, plus error translation |
| `mcp_server/server.py` | four MCP tools bound to one sandbox |
| `mcp_server/cloud.py` | one ASGI app: FastAPI + the mounted MCP transport |
| `agentkit/tools/chart.py` | M2: render a chart to a PNG file |

### The sandbox is the point

A model supplies a path. Joining it to a root naively lets `../../etc/passwd`
walk out, and a `startswith` check is defeated by a symlink planted inside the
root. The defense is order: **resolve first, then compare** -- which
normalizes `..` and follows symlinks in one step. Checking before resolving is
the same as not checking.

Refused, and each has a test: relative climb, Windows-style climb, absolute
POSIX path, absolute Windows path, null byte, and a symlink pointing out of
the root. The last one is the only case a string-prefix check passes.

A refusal is `bad_arguments` naming the root, because the caller is a model
that should correct itself, not a crash that should propagate. The known
limitation is stated in the module rather than hidden: resolve-then-open is a
time-of-check/time-of-use window, and closing it needs `O_NOFOLLOW`/`openat`.

### MCP 2.x, and the bug the adapter exists for

`pip install mcp` is now 2.x, where `FastMCP` was renamed `MCPServer` and
`from mcp.server.fastmcp import FastMCP` fails outright. `adapter.py` keeps the
familiar spelling over the new class.

The non-obvious half: the SDK distinguishes an *anticipated* failure from a
crash. Raising its own `ToolError` returns `is_error=True` with your message;
raising anything else yields only `Error executing tool <name>` and sends the
traceback to the log. Our tools raise `ToolCallError` carrying the classified
sandbox message -- without translation, every one of those would arrive as a
bare crash and the model would have nothing to act on.

### Deployment: two settings that are not optional

- **Mount the inner route at `/`.** The SDK registers its handler *inside* the
  app it returns, so mounting that app at `/mcp` serves `/mcp/mcp`. The mount
  prefix has to supply the whole path.
- **Pass `allowed_hosts`.** The SDK defaults to DNS-rebinding protection
  against localhost only, which is right for a laptop and wrong behind a cloud
  function: the platform routes by service hostname, so the default rejects
  every real request with a 421.

The mounted app's lifespan must also be forwarded, or its session manager
never starts and every request hangs instead of failing.

### Charts: cheap output, expensive input

The chart tool renders with matplotlib's Agg backend, so no display and no
font server are needed. The trap it is built around: a tool result travels
back through the transcript as text, and a PNG as a data URL is roughly a
hundred thousand characters -- which would undo every token saving in
`messages.py`. So the default result is a path and a digest (under 500
characters); inline bytes are opt-in. The demo prints both numbers.

## W5: three kinds of memory (`memory`)

Each layer exists because it loses a different thing if you try to do its job
with one of the others. A single store cannot be bounded and complete at once,
which is the whole problem.

```powershell
python scripts\demo_w5.py
python -m unittest tests.test_memory -v
```

| File | Responsibility |
| --- | --- |
| `memory/working.py` | the per-turn scratchpad: bounded by tokens, eviction counted |
| `memory/session.py` | window + rolling summary + recall, and the W4 checkpoint bridge |
| `memory/longterm.py` | structured preferences/decisions, plus semantic recall |
| `memory/store.py` | record type, the JSON and in-memory stores, the semantic store |

### The question this week answers

**"The conversation is long and the prompt blew up. Now what?"**

Not "a bigger context window". The answer is three mechanisms over one
transcript, because they lose different things:

- **A verbatim window** keeps the last turns untouched. Nothing is gained by
  summarising the thing the user just said.
- **A rolling summary** holds the shape of everything older, and stays bounded
  because it summarises its own previous summary rather than the whole history
  again. Measured, not asserted: the demo prints a compression ratio of ~0.32.
- **Recall over the evicted turns** is the part naive compaction skips. A
  summary is lossy on purpose, so the fix for the loss is not a better summary
  -- it is being able to get the original back. "What exactly did I say about
  option 2" searches the turns the window dropped.

Eviction does not depend on summarisation. Without a summariser the window is
all the prompt holds and recall still reaches everything older -- the
configuration where recall is the *only* way to recover a detail is the last
place it should silently disappear.

### Two stores for long-term memory, not one

- **Preferences and decisions are structured.** `preference("seat")` returns
  `"aisle"` or nothing, and never a 90%-match. A preference that is nearly
  right is a bug that gets blamed on the model.
- **Everything else is semantic.** "Which plan did I reject" is a sentence,
  not a field, and is only findable by meaning.

Setting a preference twice replaces it, and the replacement is removed from
*both* stores -- otherwise `recall` returns a value that exact lookup has
already retired. That failure is a test, not a comment.

Persistence is demonstrated the only way that counts: a new object over the
same file reads the preference back. An in-memory dict would prove nothing, so
the file store is the demo default and the in-memory one is the test double --
the reverse of the usual arrangement, and for a stated reason.

### Session memory is the W4 checkpoint, not a copy of it

`session_from_checkpoint` hydrates the session layer from a real planner run's
state history. Keeping a second list of turns alongside the graph would create
two histories that disagree the moment a run is resumed, because the
checkpoint is what actually survives a pause. Only the tail of the history is
hydrated, for the same reason the window exists.

### W5 numbers

- 30 tests added, 301 total, offline, no key
- session: 24 turns -> 4 verbatim, ~190 tokens summarised to ~60 (0.32)
- working: 6 results in, 4 kept, 2 evicted, 914 of 1200 tokens held
- a planner run's 8 notes hydrate into session memory and stay recallable
- a preference written in one object is read back by another over the same file

## W4: a planner that branches on what it finds (`graph`)

A weekend trip, planned in four dependent steps: search destinations, check
the forecast, convert the price, draft the itinerary. The point is not the
itinerary -- it is that each step consumes the previous one's result, and the
plan changes when a tool result says it should.

```powershell
python scripts\demo_w4.py
python -m unittest tests.test_graph -v
```

| File | Responsibility |
| --- | --- |
| `graph/plan.py` | every decision, as pure functions over plain dicts |
| `graph/state.py` | the state contract, with reducers where history must accumulate |
| `graph/nodes.py` | the thin layer: call a tool, call a decision, return a delta |
| `graph/edges.py` | the routing functions, testable without compiling a graph |
| `graph/build.py` | the wiring diagram: nodes, edges, conditional branches |
| `graph/checkpoint.py` | saver selection and the `thread_id` handle |

### Nodes stay thin, decisions stay outside

LangGraph is a scheduler, not a place to think. Each node reads state, calls
one function in `plan.py`, and returns the keys it owns. The test of whether
that held is simple: delete LangGraph and the decisions are still callable,
which is exactly how the first half of `tests/test_graph.py` tests them --
no graph, no checkpointer, no key.

The ROADMAP warned that this framework's API moves. The answer is that the
volatile surface is six short methods in `nodes.py` and the wiring in
`build.py`; everything that took thought is in files that do not import it.

### The branches that make it a planner

- **No candidates** -> rewrite the query and search again. Rewriting rather
  than re-running, because asking the same index the same question twice
  cannot help.
- **Weather says rain** -> the outdoor half of the itinerary moves indoors.
  This is the branch that makes the forecast change the plan instead of being
  printed next to it.
- **Nothing viable** (all wet, or all over budget) -> back to search with a
  broader query, up to a cap, then an explicit give-up that records why.
- **Rejected by the human** -> the destination goes into `avoid` and the loop
  re-enters search, so the next offer is genuinely different.

### Checkpointing is not a cache

The run pauses before the approval gate and resumes from the checkpoint, so a
person decides between two invocations rather than inside one. `get_state`
answers "what did it know when it paused" and `get_state_history` answers "how
did it get there" -- which is what makes a plan auditable.

One version trap, recorded because it cost real debugging time: `interrupt()`
needs the runnable config, which LangGraph propagates through a Python 3.11
task context. On 3.10 every async run failed with "Called get_config outside
of a runnable context", so the gate is a static `interrupt_before` breakpoint
instead. `nodes.py` has the full note; upgrading the interpreter is what would
allow the tidier form back.

### W4 numbers

- 26 tests added, 271 total, offline, no key
- an approved run: search -> weather -> price -> decide -> approve
- a rejected run: the same path again with a rewritten query, and a different city
- a dead search tool: three attempts, then an explicit give-up -- never an exception
- a dead weather tool: the unforecastable city is dropped, the run still plans

## W3 numbers

- 68 tests added, 182 total, offline, no key
- chart result: ~250 characters in the transcript instead of ~12 KB of base64
- six escape techniques refused, each with a test

## W2 numbers

- 48 tests added, 114 total, offline, no key
- image turn: ~1006 tokens -> ~15 after collapsing
- hung tool: bounded at ~56 ms instead of 400 ms, returned as `timeout`

## Requirements

Python 3.10 or newer. Dependencies go in with
`python -m pip install -r requirements.txt`. The one version constraint worth
stating: the MCP SDK must be 2.x, where `FastMCP` is spelled `MCPServer` (see
`mcp_server/adapter.py` for how that rename is absorbed).

`agentloop/` is the exception -- deliberately standard-library only, so the W1
loop stays readable and its tests need no install step.
`from __future__ import annotations` is used throughout, so the type hints are
inert at runtime.
