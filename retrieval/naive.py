"""The naive RAG pipeline: retrieve once, stuff the prompt, generate.

Deliberately dumb, and deliberately *kept*. The value of this module is not
that it works -- it is that it establishes the number the agentic retriever
has to beat. A benchmark without a baseline is an anecdote, so this file is
the thing W6 is measured against, not a draft to be replaced.

What it does not do is the whole point:
- it does not decide *whether* retrieval is needed, so it retrieves for
  questions it should answer from the conversation alone;
- it does not rewrite the query when the first attempt returns nothing
  useful, so a question phrased unlike the corpus fails;
- it does not retrieve twice, so multi-hop questions that need two facts
  from different entries cannot be satisfied;
- it does not judge its own results, so it will happily stuff the prompt
  with irrelevant chunks and invent an answer.

Each of those is a capability W6 adds, and each is measurable on the shared
task set. The ``naive`` pipeline is expected to fail ``multi_hop`` and
``negative`` -- that failure is the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .corpus_rag import RagTask
from .index import BM25Index, ScoredChunk

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the context does not contain the answer, say so plainly.

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class NaiveAnswer:
    """One naive run: the answer, the chunks it was built from, and the cost."""

    task_id: str
    category: str
    answer: str
    hits: list[ScoredChunk]
    prompt: str
    prompt_tokens: int
    calls: int = 1

    @property
    def sources(self) -> list[str]:
        seen: list[str] = []
        for scored in self.hits:
            if scored.doc_id not in seen:
                seen.append(scored.doc_id)
        return seen


def format_context(hits: Sequence[ScoredChunk], *, with_sources: bool = True) -> str:
    """Render hits into the prompt.

    Source ids are included because an answer without a citation cannot be
    checked, and because a model that can cite is a model that can be asked
    to refuse. The numbering is stable so a later agent can refer to chunk 2
    without re-deriving the list.
    """
    blocks: list[str] = []
    for position, scored in enumerate(hits, start=1):
        header = f"[{position}] {scored.doc_id}" if with_sources else f"[{position}]"
        blocks.append(f"{header}\n{scored.chunk.text}")
    return "\n\n".join(blocks) if blocks else "(no matching context)"


def estimate_tokens(text: str) -> int:
    """Four characters per token, matching the rest of the project."""
    return max(1, (len(text) + 3) // 4)


def naive_rag(
    task: RagTask,
    index: BM25Index,
    generate: Callable[[str], str],
    *,
    k: int = 5,
) -> NaiveAnswer:
    """One retrieval, one generation, no decisions in between."""
    hits = index.search(task.question, k=k)
    context = format_context(hits)
    prompt = PROMPT_TEMPLATE.format(context=context, question=task.question)
    answer = generate(prompt)
    return NaiveAnswer(
        task_id=task.id,
        category=task.category,
        answer=answer,
        hits=hits,
        prompt=prompt,
        prompt_tokens=estimate_tokens(prompt),
    )


class ExtractiveGenerator:
    """A generator that answers from the retrieved text, offline.

    Not a language model, and labelled as such. It exists so the pipeline can
    be measured end to end without an API key, and so the retrieval score is
    not confounded with a model's ability to summarise. A real model plugs
    into the same ``generate`` callable.

    It is deliberately literal: it returns the highest-scoring sentence that
    shares a term with the question. If that sentence is in the context, the
    run passes; if retrieval missed, it fails. Which is the measurement.
    """

    def __init__(self, *, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        context, question = _split_prompt(prompt)
        if not context.strip() or context.strip() == "(no matching context)":
            return "The context does not contain an answer."

        from agentloop.tools import stem_tokens

        wanted = set(stem_tokens(question))
        sentences = _sentences(context)
        scored: list[tuple[int, int, str]] = []
        for position, sentence in enumerate(sentences):
            overlap = len(wanted & set(stem_tokens(sentence)))
            if overlap:
                # Earlier sentences win ties, which keeps the output stable.
                scored.append((-overlap, position, sentence))
        if not scored:
            return "The context does not contain an answer."
        scored.sort()
        chosen = [sentence for _, _, sentence in scored[: self.max_sentences]]
        return " ".join(chosen)


def _split_prompt(prompt: str) -> tuple[str, str]:
    """Recover context and question from a rendered prompt.

    Needed only because the extractive generator is offline and cannot parse
    with a model. A real generator never does this.
    """
    context = ""
    question = ""
    if "Context:" in prompt and "Question:" in prompt:
        after = prompt.split("Context:", 1)[1]
        context, rest = after.split("Question:", 1)
        question = rest.split("Answer:", 1)[0]
    return context.strip(), question.strip()


def _sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?。！？])\s+", text)
    return [part.strip() for part in parts if part.strip()]


__all__ = [
    "ExtractiveGenerator",
    "NaiveAnswer",
    "PROMPT_TEMPLATE",
    "estimate_tokens",
    "format_context",
    "naive_rag",
]
