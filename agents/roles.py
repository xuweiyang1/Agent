"""The three roles, and what makes each one a role rather than a prompt.

A multi-agent system earns its cost only if the agents are *different* -- not
differently worded, differently scoped. Each role here owns one question that
the others are not allowed to answer:

- **Researcher:** what does the corpus say? Owns evidence, invents nothing.
- **Writer:** what is the answer, given the evidence? Owns the prose.
- **Reviewer:** what is missing? Owns the defect list, and is the only role
  whose job is to look for what is *not* there.

That last one is the structural argument for a team, and it is worth stating
precisely because it is easy to overstate. The reviewer is not smarter than
the writer; it has a different *objective*. A writer asked to produce an
answer will produce one. A reviewer asked to find omissions will find them.
Anthropic's report makes the same point about orchestration: the gain comes
from separable objectives, not from more models.

Each role is a small deterministic policy over the corpus, and that is a
deliberate trade for this week. The experiment's claim is about
*communication cost and structural capability*, both of which are properties
of the architecture and reproduce identically offline. A model-backed
implementation would change the absolute numbers and not the shape of the
result, and ``RoleModel`` is the seam where one drops in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from agentloop.context import estimate_tokens
from agentloop.tools import stem_tokens


@dataclass
class Fact:
    """One piece of evidence, tagged so coverage can be checked."""

    key: str
    text: str


@dataclass
class Brief:
    """What the task asks for: a set of facts that must all appear."""

    task_id: str
    request: str
    required: list[str]

    @property
    def size(self) -> int:
        """How many items the brief requires. The experiment's difficulty axis."""
        return len(self.required)


class RoleModel(Protocol):
    """What a role needs from whatever is thinking for it."""

    def research(self, request: str, required: Sequence[str]) -> list[Fact]: ...

    def draft(self, request: str, facts: Sequence[Fact], *, capacity: int, extra: Sequence[Fact] = ()) -> str: ...

    def review(self, request: str, draft: str, required: Sequence[str], facts: Sequence[Fact]) -> list[str]: ...


def mentions(text: str, needle: str) -> bool:
    """Whether a draft actually states a fact.

    Matched on stemmed terms rather than substrings, using W1's tokenizer, so
    "retries" counts as mentioning "retry" and a shared word like "the" cannot
    make a fact look covered. Reusing the tokenizer rather than writing a
    second one is the same call W3.5 made for the same reason.
    """
    wanted = set(stem_tokens(needle))
    if not wanted:
        return False
    return wanted <= set(stem_tokens(text))


@dataclass
class DeterministicRoles:
    """The offline role implementation, driven by a fact source.

    ``facts`` is the knowledge base keyed by the same identifiers the tasks
    use, so coverage can be checked mechanically. ``capacity`` is the one
    behavioural parameter that matters: how many items a single drafting pass
    actually states. It is the mechanism behind the whole experiment -- a
    writer with more to say than it has room for drops items, and nothing in a
    single-agent run notices.
    """

    facts: dict[str, str]
    capacity: int = 3
    _researched: int = field(default=0, init=False)

    def research(self, request: str, required: Sequence[str]) -> list[Fact]:
        """Gather exactly the evidence the brief names.

        Retrieval is *not* reimplemented here on purpose: what the researcher
        contributes is the decision of what to look for and how to present it,
        and in a model-backed version that is where the tool calls live. The
        offline version reads the same source the corpus exposes.
        """
        found: list[Fact] = []
        for key in required:
            text = self.facts.get(key)
            if text:
                found.append(Fact(key=key, text=_first_sentence(text)))
        self._researched += len(found)
        return found

    def draft(
        self,
        request: str,
        facts: Sequence[Fact],
        *,
        capacity: int | None = None,
        extra: Sequence[Fact] = (),
    ) -> str:
        """Write a briefing, stating at most ``capacity`` facts per pass.

        ``extra`` is what the reviewer sent back, and it is appended *after*
        the capacity cut rather than competing with it. That ordering is the
        mechanic being tested: a revision pass adds the missing items, so the
        review loop can reach facts that one pass structurally cannot.
        """
        room = capacity if capacity is not None else self.capacity
        chosen = list(facts)[:room]
        lines = [f"Briefing: {request}"]
        lines.extend(f"- {fact.key}: {fact.text}" for fact in chosen)
        for fact in extra:
            if not any(fact.key == seen.key for seen in chosen):
                lines.append(f"- {fact.key}: {fact.text}")
        return "\n".join(lines)

    def review(
        self,
        request: str,
        draft: str,
        required: Sequence[str],
        facts: Sequence[Fact],
    ) -> list[str]:
        """Return the required items the draft does not state.

        The return value is keys, not prose: the reviewer's output is a defect
        list a writer can act on, which is what keeps the handoff small. A
        reviewer that replies with an essay is a second writer, and the
        pipeline pays for both.
        """
        return [key for key in required if not mentions(draft, key)]


def _first_sentence(text: str, limit: int = 180) -> str:
    """Keep an evidence line short, because it crosses a wire.

    Evidence is summarised at the point it is gathered rather than at the
    point it is sent: a fact sheet that is already bounded cannot blow up
    later when someone decides to forward it.
    """
    for separator in (". ", "。", "! ", "? "):
        position = text.find(separator)
        if 0 < position < limit:
            return text[: position + 1].strip()
    return text[:limit].strip()


__all__ = [
    "Brief",
    "DeterministicRoles",
    "Fact",
    "RoleModel",
    "mentions",
]
