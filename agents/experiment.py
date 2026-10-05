"""The single-vs-multi experiment, and where the break-even point is.

The week's acceptance criterion is "use your own data to show the multi-agent
break-even point". So the experiment is built around one axis -- **how many
items the brief requires** -- and one fixed room to write them in. That is the
only way the result can be a *point* rather than a preference.

The mechanism is deliberately plain, because a mechanism nobody can follow
produces a number nobody can defend:

- A single agent gathers all the evidence and writes **once**.
- The team gathers the same evidence, writes once, then a reviewer looks for
  what is missing, and the writer gets a bounded revision.
- Writing room per pass is **identical** for both. If the single agent had
  less room than the writer, the experiment would be measuring the room.

So the difference is purely structural: a single pass has one chance, and a
review loop has two. Below the room size that difference buys nothing and
costs the handoffs; above it, the review loop recovers items a single pass
structurally cannot. The break-even point is where the coverage gap starts
paying for the communication bill.

A second comparison is run in the same report, because "communication is
expensive" is only useful with a number attached: the same team with
``FULL_TRANSCRIPT`` handoffs instead of summaries. That is the control for the
ROADMAP's P2 rule, and it is what turns "keep handoffs small" from advice into
a measured factor.

Both are offline and deterministic, so the numbers are reproducible on any
machine for free -- which matters, because a results table you cannot re-run
is a results table that goes stale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from .protocol import HandoffPolicy
from .roles import Brief, DeterministicRoles, Fact
from .single import SingleAgent
from .team import Team

# A synthetic evidence base. Synthetic rather than the W1 corpus because the
# experiment needs *scalable* briefs: the corpus has twelve entries and the
# axis being tested is what happens as a brief outgrows the writing room.
FACT_TEMPLATE = (
    "The {topic} decision is recorded as: {topic} requires review before the "
    "release, and the {topic} owner signs it off. This sentence exists to give "
    "the fact enough length to behave like a real one when it is summarised."
)

TOPICS: tuple[str, ...] = (
    "deployment",
    "billing",
    "authentication",
    "retention",
    "telemetry",
    "migration",
    "rollback",
    "pagination",
    "caching",
    "encryption",
    "throttling",
    "auditing",
)


def build_facts(topics: Sequence[str] = TOPICS) -> dict[str, str]:
    return {topic: FACT_TEMPLATE.format(topic=topic) for topic in topics}


def build_briefs(
    *,
    sizes: Sequence[int] = tuple(range(1, 11)),
    topics: Sequence[str] = TOPICS,
) -> list[Brief]:
    """One brief per size, each requiring exactly that many facts.

    Sizes are the independent variable, so each brief is built from a distinct
    slice of topics -- sharing topics between briefs would make a larger brief
    a superset of a smaller one and the coverage curve would reflect the
    overlap rather than the size.
    """
    briefs: list[Brief] = []
    for size in sizes:
        chosen = list(topics[:size])
        briefs.append(
            Brief(
                task_id=f"brief-{size:02d}",
                request=f"Write a briefing covering {size} area(s) of the release process.",
                required=chosen,
            )
        )
    return briefs


def coverage(answer: str, required: Sequence[str]) -> float:
    """Fraction of the required facts the answer actually states."""
    if not required:
        return 1.0
    from .roles import mentions

    return sum(1 for key in required if mentions(answer, key)) / len(required)


@dataclass
class Row:
    """One brief, both strategies, all the costs."""

    size: int
    single_coverage: float
    team_coverage: float
    single_tokens: int
    team_tokens: int
    team_communication_tokens: int
    team_hops: int
    team_iterations: int

    @property
    def coverage_gain(self) -> float:
        return self.team_coverage - self.single_coverage

    @property
    def token_premium(self) -> int:
        return self.team_tokens - self.single_tokens

    @property
    def premium_ratio(self) -> float:
        """Team cost as a multiple of single-agent cost."""
        return self.team_tokens / self.single_tokens if self.single_tokens else 0.0

    @property
    def communication_share(self) -> float:
        """What fraction of the team's bill was talking rather than thinking."""
        return self.team_communication_tokens / self.team_tokens if self.team_tokens else 0.0

    def to_dict(self) -> dict:
        return {
            "size": self.size,
            "single_coverage": round(self.single_coverage, 4),
            "team_coverage": round(self.team_coverage, 4),
            "coverage_gain": round(self.coverage_gain, 4),
            "single_tokens": self.single_tokens,
            "team_tokens": self.team_tokens,
            "token_premium": self.token_premium,
            "premium_ratio": round(self.premium_ratio, 3),
            "team_communication_tokens": self.team_communication_tokens,
            "communication_share": round(self.communication_share, 4),
            "team_hops": self.team_hops,
            "team_iterations": self.team_iterations,
        }


@dataclass
class Report:
    """The experiment's outcome, as data and as a table."""

    rows: list[Row]
    capacity: int
    summary_policy_tokens: list[int] = field(default_factory=list)
    transcript_policy_tokens: list[int] = field(default_factory=list)

    @property
    def break_even(self) -> Row | None:
        """The smallest brief where the team's coverage beats the single agent.

        The acceptance criterion, computed rather than eyeballed. ``None``
        means the team never earned its cost on this data, which is a legitimate
        and reportable result -- the honest one for a well-sized workload.
        """
        for row in self.rows:
            if row.coverage_gain > 1e-9:
                return row
        return None

    def to_dict(self) -> dict:
        break_even = self.break_even
        return {
            "capacity": self.capacity,
            "break_even_size": break_even.size if break_even else None,
            "break_even": break_even.to_dict() if break_even else None,
            "rows": [row.to_dict() for row in self.rows],
            "policies": {
                "summary_tokens": self.summary_policy_tokens,
                "transcript_tokens": self.transcript_policy_tokens,
                "transcript_over_summary": [
                    round(t / s, 3) if s else 0.0
                    for s, t in zip(self.summary_policy_tokens, self.transcript_policy_tokens)
                ],
            },
        }

    def render(self) -> str:
        lines: list[str] = []
        lines.append(f"writing room per pass: {self.capacity} items")
        lines.append("")
        header = (
            f"{'brief':>5} {'single cov':>10} {'team cov':>9} {'gain':>6} "
            f"{'single tok':>10} {'team tok':>9} {'premium':>8} {'comm share':>10} {'iters':>6}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for row in self.rows:
            lines.append(
                f"{row.size:>5} {row.single_coverage:>10.2f} {row.team_coverage:>9.2f} "
                f"{row.coverage_gain:>+6.2f} {row.single_tokens:>10} {row.team_tokens:>9} "
                f"{row.premium_ratio:>7.2f}x {row.communication_share:>9.0%} {row.team_iterations:>6}"
            )

        break_even = self.break_even
        lines.append("")
        if break_even:
            lines.append(
                f"break-even: briefs requiring {break_even.size}+ items. The team gains "
                f"{break_even.coverage_gain:+.2f} coverage for {break_even.premium_ratio:.2f}x the tokens."
            )
            lines.append(
                f"  below that size the team pays {self.rows[0].premium_ratio:.2f}x for no coverage gain."
            )
        else:
            lines.append("break-even: not reached on this data -- the team never earned its cost.")

        if self.summary_policy_tokens:
            lines.append("")
            lines.append("handoff policy (team communication tokens):")
            lines.append(f"{'brief':>5} {'summary':>9} {'transcript':>11} {'factor':>8}")
            for row, summary, transcript in zip(self.rows, self.summary_policy_tokens, self.transcript_policy_tokens):
                factor = transcript / summary if summary else 0.0
                lines.append(f"{row.size:>5} {summary:>9} {transcript:>11} {factor:>7.1f}x")
        return "\n".join(lines)


def run_experiment(
    *,
    sizes: Sequence[int] = tuple(range(1, 11)),
    capacity: int = 3,
    max_iterations: int = 2,
    topics: Sequence[str] = TOPICS,
) -> Report:
    """Run both strategies across brief sizes and collect the table."""
    facts = build_facts(topics)
    briefs = build_briefs(sizes=sizes, topics=topics)

    rows: list[Row] = []
    summary_tokens: list[int] = []
    transcript_tokens: list[int] = []

    for brief in briefs:
        roles = DeterministicRoles(facts=facts, capacity=capacity)
        single = SingleAgent(roles).run(brief)
        team = Team(roles, capacity=capacity, max_iterations=max_iterations).run(brief)

        # The control. Same team, same brief, same capacity, same iteration
        # bound -- only the handoff policy differs, so the difference in
        # communication tokens is attributable to the policy and nothing else.
        roles_transcript = DeterministicRoles(facts=facts, capacity=capacity)
        verbose = Team(
            roles_transcript,
            capacity=capacity,
            max_iterations=max_iterations,
            policy=HandoffPolicy.FULL_TRANSCRIPT,
        ).run(brief)

        rows.append(
            Row(
                size=brief.size,
                single_coverage=coverage(single.answer, brief.required),
                team_coverage=coverage(team.answer, brief.required),
                single_tokens=single.total_tokens,
                team_tokens=team.total_tokens,
                team_communication_tokens=team.communication_tokens,
                team_hops=team.hops,
                team_iterations=team.iterations,
            )
        )
        summary_tokens.append(team.communication_tokens)
        transcript_tokens.append(verbose.communication_tokens)

    return Report(
        rows=rows,
        capacity=capacity,
        summary_policy_tokens=summary_tokens,
        transcript_policy_tokens=transcript_tokens,
    )


__all__ = [
    "FACT_TEMPLATE",
    "TOPICS",
    "Report",
    "Row",
    "build_briefs",
    "build_facts",
    "coverage",
    "run_experiment",
]
