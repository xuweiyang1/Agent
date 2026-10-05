"""Local search over a small in-memory corpus.

The ranking is ``agentloop.tools.rank_entries`` verbatim. That is not reuse
for its own sake: W1 already fixed the two bugs this code would otherwise
repeat (a stemmer so "retry" matches "retries", and snippets so the model can
often answer without a second read). W3.5 replaces this tool's *backend* with
a real index while keeping the same schema, so the model never learns a new
tool to do the same job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from agentloop.tools import rank_entries, tokenize

from ..registry import ToolRegistry
from ..schema import ToolArgs

DEFAULT_CORPUS: dict[str, str] = {
    "wifi-password": (
        "The guest wifi password is stored in the password manager under "
        "'home-guest'. It rotates on the first of the month."
    ),
    "laundry": (
        "The building laundry room is on level B1 and takes contactless "
        "payment. Quiet hours are 22:00 to 07:00."
    ),
    "insurance": (
        "Health insurance renews every April. Claims go through the portal, "
        "and receipts must be uploaded within 30 days."
    ),
    "passport": (
        "The passport expires in March of next year. Renewal needs a photo, "
        "the old passport, and a fee paid online."
    ),
}


class SearchArgs(ToolArgs):
    query: str = Field(..., min_length=1, description="Words to search for, e.g. 'passport renewal'.")
    k: int = Field(3, ge=1, le=20, description="Maximum number of hits to return.")


@dataclass
class SearchService:
    corpus: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CORPUS))

    def search(self, query: str, k: int) -> dict[str, Any]:
        hits = [
            {"id": key, "score": score, "snippet": _snippet(self.corpus[key], query)}
            for key, score in rank_entries(query, self.corpus)[:k]
        ]
        return {"query": query, "hits": hits}


def _snippet(body: str, query: str, width: int = 140) -> str:
    """Return the fragment most likely to answer the query.

    Same idea as W1's snippet, kept local because it is a presentation choice
    for this tool, not a property of the ranking.
    """
    if len(body) <= width:
        return body
    terms = {word for word in tokenize(query)}
    for word in tokenize(body):
        if word in terms:
            start = max(0, body.lower().find(word) - width // 3)
            return body[start : start + width].strip()
    return body[:width].strip()


def register(registry: ToolRegistry, service: SearchService | None = None) -> SearchService:
    """Attach the ``search`` tool."""
    svc = service or SearchService()

    @registry.tool(
        "search",
        "Search the user's local notes for a topic. Returns ranked hits with "
        "a short snippet each; read the full note only if the snippet is not "
        "enough.",
        args_model=SearchArgs,
        timeout=5.0,
    )
    def search(query: str, k: int = 3) -> dict[str, Any]:
        return svc.search(query, k)

    return svc


__all__ = ["DEFAULT_CORPUS", "SearchArgs", "SearchService", "register"]
