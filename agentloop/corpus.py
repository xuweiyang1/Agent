"""The knowledge base the built-in tools search.

Kept in its own module because it is data, not logic, and because the
evaluation suite grades answers against it. Making the corpus large enough
to have wrong answers available is the point: a retrieval score over four
entries measures nothing.
"""

from __future__ import annotations

CORPUS: dict[str, str] = {
    "agent-loop": (
        "An agent loop alternates model calls with tool calls until the model "
        "stops requesting tools. Each iteration sends the transcript to the "
        "model, executes whatever tools it asks for, appends the results, and "
        "repeats. The loop needs a turn ceiling because a model can request "
        "tools indefinitely."
    ),
    "tool-registry": (
        "The tool registry maps a tool name to its schema and its "
        "implementation, and validates arguments before calling it. The schema "
        "is what the model sees, so a missing required field becomes a tool "
        "error message in the transcript rather than an exception."
    ),
    "context-window": (
        "The context window is a hard budget of tokens shared by the system "
        "prompt, the conversation history, and every tool result. Unlike disk "
        "or memory, it does not grow, so anything appended pushes something "
        "else out or causes the request to be rejected."
    ),
    "backoff": (
        "Exponential backoff retries a transient failure with growing delays, "
        "and is a safety requirement once writes are involved. A retryable "
        "error is one a later attempt can succeed on, such as a rate limit or "
        "a timeout. A non-retryable error, such as a malformed request, will "
        "fail the same way every time, so retrying it only wastes the budget."
    ),
    "compaction": (
        "Compaction summarizes or drops older turns so a long run keeps "
        "fitting in the context window. It is deliberately lossy, so the cut "
        "must not separate a tool result from the request that produced it; "
        "most providers reject a transcript where a tool message has no "
        "matching call."
    ),
    "function-calling": (
        "Function calling lets a model return a structured request to run a "
        "named tool instead of prose. Arguments arrive as a JSON string, and "
        "models do occasionally emit invalid JSON, so a parser must treat a "
        "parse failure as a normal event rather than a crash."
    ),
    "tokenization": (
        "A tokenizer splits text into the units a model bills and attends "
        "over. English averages about four characters per token, while code "
        "and non-Latin scripts are less efficient, so a character count is "
        "only a rough budget estimate."
    ),
    "prompt-injection": (
        "Prompt injection is an attack where text retrieved from a tool or a "
        "document contains instructions that the model then follows. The "
        "defence is to treat tool output as data, never as instructions, and "
        "to keep a human in the loop for any action with side effects."
    ),
    "embedding-retrieval": (
        "Keyword search matches literal terms and fails when a query and a "
        "document use different words for the same idea. Embedding retrieval "
        "compares vectors instead, so paraphrases match, at the cost of a "
        "model call and a loss of the exact-match guarantee that keyword "
        "search gives."
    ),
    "evaluation": (
        "An evaluation set is a fixed list of tasks with known correct "
        "answers, used to compare two versions of a system. Without one, a "
        "prompt change is a guess, because a single convincing demo says "
        "nothing about the cases that regressed."
    ),
    "observability": (
        "Observability for an agent means recording the full transcript, the "
        "token count, the latency, and the tool outcomes of every run. The "
        "trace is what makes a failure reproducible, and a failure that cannot "
        "be replayed cannot be fixed."
    ),
    "latency-budget": (
        "Latency in an agent loop is dominated by the number of round trips, "
        "not by the size of the payload. Reducing ten tool calls to five "
        "roughly halves the wall clock time, while trimming tokens from a "
        "request mostly reduces cost."
    ),
}