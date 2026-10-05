"""W7: the minimal multi-agent team, and the experiment that judges it.

``run_experiment`` is the entry point, because this week's deliverable is a
number rather than a system. ``SingleAgent`` and ``Team`` are the two
strategies, ``Mailbox`` is the accountant that makes their communication cost
reportable, and ``build_memory``-style wiring is deliberately absent -- the
team is constructed directly so a reader can see every hop it takes.
"""

from __future__ import annotations

from .experiment import Report, Row, build_briefs, build_facts, coverage, run_experiment
from .protocol import Handoff, HandoffPolicy, Mailbox, summarise_for_handoff
from .roles import Brief, DeterministicRoles, Fact, mentions
from .single import SingleAgent, SingleResult
from .team import DONE, RESEARCHER, REVIEWER, SUPERVISOR, WRITER, Team, TeamResult

__all__ = [
    "Brief",
    "DONE",
    "DeterministicRoles",
    "Fact",
    "Handoff",
    "HandoffPolicy",
    "Mailbox",
    "RESEARCHER",
    "REVIEWER",
    "Report",
    "Row",
    "SUPERVISOR",
    "SingleAgent",
    "SingleResult",
    "Team",
    "TeamResult",
    "WRITER",
    "build_briefs",
    "build_facts",
    "coverage",
    "mentions",
    "run_experiment",
    "summarise_for_handoff",
]
