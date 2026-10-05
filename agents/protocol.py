"""What agents hand to each other, and what that costs.

Multi-agent systems are sold on capability and paid for in communication, and
the communication is where the design decisions actually are. Two of them are
made here.

**Handoffs carry a summary, not a transcript.** This is the ROADMAP's token
rule and it is the single highest-leverage decision in the module. If each
agent receives the previous agent's full transcript, the cost of a pipeline of
length *n* grows with the square of the work -- every agent re-reads everything
everyone did. A summary breaks that: each hop costs a bounded artifact instead
of an unbounded history. ``HandoffPolicy`` keeps both behaviours callable, not
because the transcript version is a good idea, but because it is the control
that makes the summary's saving measurable instead of asserted.

**Communication is accounted, not estimated.** ``Mailbox`` records every
handoff with its token cost, so "communication overhead" is a number that can
go in a report rather than an adjective. It also separates the two cost
sources that are easy to conflate: what an agent *thought* and what it *said
to another agent*. Only the second one is caused by going multi-agent, so a
comparison that lumps them together overstates the penalty.

``Handoff.to_agent`` and ``from_agent`` exist so the trace can be read as a
conversation. A cost number with no direction is not diagnosable, and the
first question anyone asks about a multi-agent bill is which hop produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from agentloop.context import estimate_tokens


class HandoffPolicy(str, Enum):
    """How much of the sender's context travels with a handoff."""

    SUMMARY = "summary"
    FULL_TRANSCRIPT = "full_transcript"


@dataclass
class Handoff:
    """One message between two agents.

    ``content`` is what the receiver actually reads. ``tokens`` is measured on
    it, not on what the sender knew, because the bill is for what is sent.
    """

    from_agent: str
    to_agent: str
    content: str
    policy: HandoffPolicy = HandoffPolicy.SUMMARY
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.content)

    def to_dict(self) -> dict:
        return {
            "from": self.from_agent,
            "to": self.to_agent,
            "policy": self.policy.value,
            "tokens": self.tokens,
            "chars": len(self.content),
        }


@dataclass
class Mailbox:
    """The channel between agents, and the accountant for it.

    A mailbox rather than direct calls, because the interesting quantity is
    the *sum* over hops and it has to be collected in one place to be
    trustworthy. If each agent counted its own sends, a bug in one of them
    would silently under-report the overhead the experiment is about.
    """

    policy: HandoffPolicy = HandoffPolicy.SUMMARY
    messages: list[Handoff] = field(default_factory=list)

    def send(self, from_agent: str, to_agent: str, content: str, **metadata: Any) -> Handoff:
        """Send one message, and record it as history.

        Under ``FULL_TRANSCRIPT`` the content has already been built with the
        prior history prepended by ``summarise_for_handoff``, so nothing extra
        happens here. The history is kept regardless, because the growth has to
        come from somewhere real: a control that only *claims* to send the
        transcript would show the policy difference as a constant.
        """
        handoff = Handoff(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            policy=self.policy,
            metadata=dict(metadata),
        )
        self.messages.append(handoff)
        return handoff

    def transcript(self) -> list[str]:
        """Everything sent so far, oldest first.

        This is what a naive ``FULL_TRANSCRIPT`` handoff carries: the receiver
        gets not just the new artifact but every artifact that preceded it.
        That is the behaviour the ROADMAP's P2 rule forbids, and the reason its
        cost grows with the number of hops rather than staying flat.
        """
        return [f"{m.from_agent}->{m.to_agent}: {m.content}" for m in self.messages]

    @property
    def tokens(self) -> int:
        """Total tokens sent between agents."""
        return sum(message.tokens for message in self.messages)

    @property
    def hops(self) -> int:
        return len(self.messages)

    def by_pair(self) -> dict[str, int]:
        """Tokens per ``sender->receiver`` edge.

        The per-edge breakdown is what turns "communication is expensive" into
        "this hop is expensive", which is the only form of the claim that can
        be acted on.
        """
        totals: dict[str, int] = {}
        for message in self.messages:
            edge = f"{message.from_agent}->{message.to_agent}"
            totals[edge] = totals.get(edge, 0) + message.tokens
        return dict(sorted(totals.items()))

    def render(self) -> str:
        lines = [f"hops={self.hops} tokens={self.tokens} policy={self.policy.value}"]
        for edge, tokens in self.by_pair().items():
            lines.append(f"  {edge:28} {tokens:>6} tokens")
        return "\n".join(lines)


def summarise_for_handoff(
    parts: Sequence[str],
    *,
    policy: HandoffPolicy,
    transcript: Sequence[str] = (),
    limit_chars: int = 800,
) -> str:
    """Build what actually crosses the wire.

    Under ``SUMMARY`` the artifact is the parts the receiver needs, truncated
    to a bound. Under ``FULL_TRANSCRIPT`` the whole history is prepended -- the
    behaviour being measured, and the reason its cost grows with the number of
    agents rather than staying flat.

    The truncation exists so a summary stays a summary even when an agent
    produces a long one. Without a bound, "send a summary" degrades into
    "send everything, called a summary", which is the failure mode this whole
    design is trying to avoid.
    """
    body = "\n".join(part for part in parts if part).strip()
    if policy is HandoffPolicy.SUMMARY:
        return body[:limit_chars]

    history = "\n".join(transcript)
    return (history + "\n" + body).strip()


__all__ = ["Handoff", "HandoffPolicy", "Mailbox", "summarise_for_handoff"]
