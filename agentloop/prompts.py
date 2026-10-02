"""System prompts, kept as data so they can be versioned and compared.

A prompt is an experimental variable, not a constant. Putting the two
variants side by side is what makes a prompt change a measurable result
instead of an opinion, and the evaluation harness takes whichever one it is
given.
"""

from __future__ import annotations

# Permissive: the model decides when a tool would help. This is the default
# the runtime ships with, and it is what makes a confident model skip
# retrieval on questions it thinks it already knows.
DEFAULT = "You are a helpful assistant. Use tools when they help."

# Retrieval-first: every factual claim must come from the knowledge base or
# be flagged as not covered. The reason this is needed is measurable: with
# DEFAULT, a model answered an out-of-corpus question from its own memory in
# zero tool calls, which is exactly the hallucination a knowledge base is
# supposed to prevent.
RETRIEVAL_FIRST = (
    "You are a helpful assistant with access to a small local knowledge base.\n"
    "\n"
    "Rules:\n"
    "1. Before answering any factual question, search the knowledge base.\n"
    "2. Base your answer on what the knowledge base returns. Do not rely on "
    "your own memory for facts the knowledge base is meant to cover.\n"
    "3. If the knowledge base does not cover the question, say so plainly. "
    "You may then offer a general answer, but you must label it as coming "
    "from your own knowledge rather than from the knowledge base.\n"
    "4. Never state a specific fact that the knowledge base does not contain "
    "without labelling it as unverified.\n"
)

PROMPTS: dict[str, str] = {
    "default": DEFAULT,
    "retrieval": RETRIEVAL_FIRST,
}


def get(name: str) -> str:
    """Look up a prompt by name, failing loudly on a typo."""
    if name not in PROMPTS:
        raise KeyError(f"unknown prompt {name!r}; available: {', '.join(sorted(PROMPTS))}")
    return PROMPTS[name]