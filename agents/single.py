"""The single-agent baseline: one pass, no second opinion.

This is the number the team has to beat, and it is written to be the strongest
honest version of itself rather than a strawman. Three choices make it that:

- **It gathers all the evidence.** A baseline that only retrieves half of what
  the team retrieves would make the team look capable for a reason that has
  nothing to do with being a team.
- **Its capacity is the same per pass as each agent's.** If the single agent
  had less room to write than the team's writer, the comparison would measure
  the room rather than the architecture.
- **It costs one pass plus its tool use.** No review hop, no handoff -- the
  team's overhead is then visible as exactly the difference.

What it cannot do is structural, not a matter of effort: it writes once and
nothing checks the draft. A single pass has one chance to state every required
item, and the pass has a capacity. When the brief is larger than the capacity,
something is dropped -- and a single agent has no mechanism that would notice.
That is the gap the team's review loop fills, and the experiment measures where
it starts to matter.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentloop.context import estimate_tokens

from .roles import Brief, DeterministicRoles, Fact


@dataclass
class SingleResult:
    """One single-agent run."""

    task_id: str
    answer: str
    facts: list[Fact]
    passes: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passes": self.passes,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "facts": [f.key for f in self.facts],
        }


@dataclass
class SingleAgent:
    """One agent, one draft, no review.

    ``capacity`` is exposed so the experiment can vary task size against a
    fixed writing room; that is the axis the break-even point sits on.
    """

    roles: DeterministicRoles
    capacity: int | None = None

    def run(self, brief: Brief) -> SingleResult:
        facts = self.roles.research(brief.request, brief.required)
        draft = self.roles.draft(
            brief.request,
            facts,
            capacity=self.capacity if self.capacity is not None else self.roles.capacity,
        )
        # The prompt is what the agent had to read to write this: the request
        # plus the evidence it gathered. Counted, not guessed, because the
        # comparison in the report is a token comparison.
        prompt = brief.request + "\n" + "\n".join(f.text for f in facts)
        return SingleResult(
            task_id=brief.task_id,
            answer=draft,
            facts=facts,
            passes=1,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(draft),
        )


__all__ = ["SingleAgent", "SingleResult"]
