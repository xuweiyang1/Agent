"""``retrieve`` as a tool, not as a stage.

W3.5's pipeline calls retrieval unconditionally, once, before generation. That
is a *pipeline*: a fixed order of stages where the retrieval step cannot know
what the generation step needs. W6 replaces the stage with a tool, and the
difference is not cosmetic -- a tool can be called zero times, once, or five,
by something that has a reason.

Being a tool means being a *published contract*, so this registers through
W2's ``ToolRegistry`` with a Pydantic argument model rather than being a
helper function the agent is expected to call correctly. That choice pays off
in three places at once: the schema is rendered for the model, arguments are
validated before the body runs, and a bad call comes back as a classified
error the model can correct. None of that has to be re-implemented here.

Two properties of the result shape are deliberate:

- **Citations are first-class.** Every hit carries its ``doc_id`` and the
  chunk text, so an answer built from it can be checked. A retrieval result
  without provenance cannot support a claim.
- **``seen`` is reported.** The service remembers which documents it has
  already returned for a run, because an agent that retrieves twice and gets
  the same chunk twice has spent tokens to learn nothing. The caller can see
  that and change the query -- which is the whole behaviour W6 exists to
  demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from pydantic import Field

from agentkit.registry import ToolRegistry
from agentkit.schema import ToolArgs

from .index import ScoredChunk, VectorStore

DEFAULT_K = 5


class RetrieveArgs(ToolArgs):
    query: str = Field(
        ...,
        min_length=1,
        description="What to search for. Use the user's own wording first; "
        "if the results do not answer the question, search again with the "
        "missing terms rather than repeating this query.",
    )
    k: int = Field(DEFAULT_K, ge=1, le=20, description="Maximum number of chunks to return.")


@dataclass
class Citation:
    """One retrieved chunk, with everything needed to cite it."""

    doc_id: str
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "score": round(self.score, 4), "text": self.text}


@dataclass
class RetrieveResult:
    """One call's return value."""

    query: str
    citations: list[Citation]
    total_indexed: int
    repeated: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.citations

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [c.to_dict() for c in self.citations],
            "total_indexed": self.total_indexed,
            "repeated": list(self.repeated),
        }

    def text(self) -> str:
        """The tool result as a string, which is what a model reads."""
        if self.empty:
            return f"no results for {self.query!r}"
        lines = [f"{len(self.citations)} result(s) for {self.query!r}:"]
        for position, citation in enumerate(self.citations, start=1):
            lines.append(f"[{position}] {citation.doc_id}\n{citation.text}")
        if self.repeated:
            lines.append(f"already seen in this run: {', '.join(sorted(set(self.repeated)))}")
        return "\n\n".join(lines)


class RetrieverTool:
    """The ``retrieve`` tool over any ``VectorStore``.

    Any store, because the protocol is W3.5's and W5's semantic memory also
    satisfies it. That is the ROADMAP's "one build, two uses": the vector
    store written for long-term memory is usable as the retriever here, and a
    dense store like Chroma is a substitution rather than a rewrite.
    """

    def __init__(self, store: VectorStore, *, default_k: int = DEFAULT_K) -> None:
        self.store = store
        self.default_k = default_k
        self.calls = 0
        self.returned_tokens = 0
        self._seen: set[str] = set()

    def retrieve(self, query: str, k: int | None = None) -> RetrieveResult:
        self.calls += 1
        hits: Sequence[ScoredChunk] = self.store.search(query, k=k or self.default_k)

        citations = [Citation(hit.doc_id, hit.score, hit.chunk.text) for hit in hits]
        repeated = [c.doc_id for c in citations if c.doc_id in self._seen]
        for citation in citations:
            self._seen.add(citation.doc_id)
        self.returned_tokens += sum(len(c.text) // 4 for c in citations)

        return RetrieveResult(
            query=query,
            citations=citations,
            total_indexed=len(self.store),
            repeated=repeated,
        )

    def register(self, registry: ToolRegistry) -> "RetrieverTool":
        """Attach ``retrieve`` to a registry, using W2's schema machinery."""

        @registry.tool(
            "retrieve",
            "Search the knowledge base. Returns chunks with their source ids. "
            "Call it again with different wording if the results do not answer "
            "the question; call it not at all only if the question is about "
            "the conversation itself.",
            args_model=RetrieveArgs,
            timeout=5.0,
        )
        def retrieve(query: str, k: int = DEFAULT_K) -> dict[str, Any]:
            return self.retrieve(query, k).to_dict()

        return self

    def reset(self) -> None:
        """Forget run-scoped state, so two runs do not share a seen-set."""
        self.calls = 0
        self.returned_tokens = 0
        self._seen.clear()


__all__ = ["Citation", "RetrieveArgs", "RetrieveResult", "RetrieverTool"]
