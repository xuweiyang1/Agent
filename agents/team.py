"""The minimal multi-agent team: Supervisor, Researcher, Writer, Reviewer.

Minimal is the requirement, not a compromise. The ROADMAP says the week is an
experiment rather than a team-building exercise, and Anthropic's own report
says not to reach for a team by default. So this is the smallest arrangement
that can demonstrate the two things a team is *for*:

- **Separable objectives.** The Reviewer is the only agent whose job is to
  find what is missing, and that is structurally unavailable to a single pass.
- **A revision loop.** The supervisor can route back to the writer with the
  reviewer's defect list -- bounded, and the bound is what keeps the cost
  predictable.

Everything else is left out. There is no debate, no voting, no negotiation,
and there is exactly one supervisor turn per iteration. A larger arrangement
would produce a bigger bill and the same conclusion, and the point of the week
is to put a number on *when* the bill is worth paying.

The supervisor is deliberately a router and not an agent with opinions. Its
only decisions are "who is next" and "is the review clean". That keeps the
orchestration cost visible as hops rather than hiding it inside a coordinator
that also writes prose -- the conflation that makes most multi-agent bills
unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentloop.context import estimate_tokens

from .protocol import HandoffPolicy, Mailbox, summarise_for_handoff
from .roles import Brief, DeterministicRoles, Fact

# Where the supervisors can route. A closed set on purpose: a free-form
# "next agent" is how a pipeline acquires an agent nobody can explain.
RESEARCHER = "researcher"
WRITER = "writer"
REVIEWER = "reviewer"
SUPERVISOR = "supervisor"
DONE = "done"


@dataclass
class TeamResult:
    """One team run, with the communication broken out separately."""

    task_id: str
    answer: str
    route: list[str]
    facts: list[Fact]
    iterations: int
    prompt_tokens: int
    completion_tokens: int
    communication_tokens: int
    hops: int
    by_edge: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """Everything paid: thinking plus talking.

        Kept as one property and one sum so a report cannot accidentally
        compare the team's total against the single agent's thinking-only
        figure -- the most flattering possible mistake, and the one a reader
        would not catch.
        """
        return self.prompt_tokens + self.completion_tokens + self.communication_tokens

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "iterations": self.iterations,
            "route": self.route,
            "hops": self.hops,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "communication_tokens": self.communication_tokens,
            "total_tokens": self.total_tokens,
            "by_edge": self.by_edge,
            "facts": [f.key for f in self.facts],
        }


@dataclass
class Team:
    """Supervisor + three roles, over one mailbox."""

    roles: DeterministicRoles
    capacity: int | None = None
    max_iterations: int = 2
    policy: HandoffPolicy = HandoffPolicy.SUMMARY

    def run(self, brief: Brief) -> TeamResult:
        """Researcher -> Writer -> Reviewer, with a bounded revision loop.

        ``max_iterations`` bounds the loop at the orchestration layer rather
        than inside the reviewer, because a reviewer that can demand unlimited
        revisions is a cost with no ceiling. One revision is enough to
        demonstrate the mechanism; two would make the same point twice as
        expensively.
        """
        mailbox = Mailbox(policy=self.policy)
        route = [SUPERVISOR]
        prompt_tokens = 0
        completion_tokens = 0

        # -- research ------------------------------------------------------
        route.append(RESEARCHER)
        facts = self.roles.research(brief.request, brief.required)
        prompt_tokens += estimate_tokens(brief.request + "\n" + " ".join(brief.required))
        research_note = summarise_for_handoff(
            [f"evidence: {'; '.join(f.key for f in facts)}"],
            policy=self.policy,
            transcript=mailbox.transcript(),
        )
        mailbox.send(RESEARCHER, SUPERVISOR, research_note)
        mailbox.send(SUPERVISOR, WRITER, research_note)

        # -- write, then review, then possibly revise -----------------------
        revisions: list[Fact] = []
        answer = ""
        iterations = 0
        for iterations in range(1, self.max_iterations + 1):
            route.append(WRITER)
            answer = self.roles.draft(
                brief.request,
                facts,
                capacity=self.capacity if self.capacity is not None else self.roles.capacity,
                extra=revisions,
            )
            prompt_tokens += estimate_tokens(brief.request + "\n" + " ".join(f.text for f in facts))
            completion_tokens += estimate_tokens(answer)

            written = summarise_for_handoff(
                [f"draft: {answer}"],
                policy=self.policy,
                transcript=mailbox.transcript(),
            )
            mailbox.send(WRITER, SUPERVISOR, written)
            mailbox.send(SUPERVISOR, REVIEWER, written)

            route.append(REVIEWER)
            missing = self.roles.review(brief.request, answer, brief.required, facts)
            prompt_tokens += estimate_tokens(answer + "\n" + " ".join(brief.required))
            review_note = f"missing: {', '.join(missing) if missing else 'none'}"
            mailbox.send(REVIEWER, SUPERVISOR, review_note)

            if not missing:
                break

            # The reviewer's defect list goes back as *keys*, and the writer
            # fills them from evidence it already has. That is what keeps the
            # revision hop cheap: no second retrieval, no re-read of the draft.
            revisions = [f for f in facts if f.key in set(missing)]

        route.append(DONE)
        return TeamResult(
            task_id=brief.task_id,
            answer=answer,
            route=route,
            facts=facts,
            iterations=iterations,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            communication_tokens=mailbox.tokens,
            hops=mailbox.hops,
            by_edge=mailbox.by_pair(),
        )


__all__ = [
    "DONE",
    "RESEARCHER",
    "REVIEWER",
    "SUPERVISOR",
    "Team",
    "TeamResult",
    "WRITER",
]
