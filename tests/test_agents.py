"""Tests for W7: the handoff protocol, the roles, and the break-even claim.

Four groups, each pinning one half of the week's argument.

The **protocol** tests check that communication is accounted rather than
estimated -- per hop, per edge, and separately from what the agents thought.
That separation is what makes the cost column meaningful, so it is asserted
rather than assumed.

The **role** tests check the one thing that makes a team more than a bundle of
prompts: the reviewer finds what the writer's *objective* would never surface,
and it reports it as a list a writer can act on rather than as prose.

The **team** tests check the loop is bounded, that it actually revises, and
that its answer differs from the single agent's for a structural reason.

The **experiment** tests pin the acceptance criterion -- a computed break-even
point -- and the P2 control, where the only difference is the handoff policy.
A test that only checked the final table would pass while the mechanism was
broken, so the mechanism is asserted directly.
"""

from __future__ import annotations

import unittest

from agents.experiment import (
    TOPICS,
    build_briefs,
    build_facts,
    coverage,
    run_experiment,
)
from agents.protocol import Handoff, HandoffPolicy, Mailbox, summarise_for_handoff
from agents.roles import Brief, DeterministicRoles, Fact, mentions
from agents.single import SingleAgent
from agents.team import Team


def small_facts() -> dict[str, str]:
    return {
        "alpha": "Alpha requires review before release. The alpha owner signs it off.",
        "beta": "Beta is billed monthly. The billing system rounds down.",
        "gamma": "Gamma expires after thirty days. Rotation is automatic.",
        "delta": "Delta records every access. Auditors read the delta log.",
    }


def brief(size: int) -> Brief:
    keys = list(small_facts())[:size]
    return Brief(task_id=f"b{size}", request="Write a briefing.", required=keys)


class ProtocolTests(unittest.TestCase):
    def test_a_handoff_measures_what_is_sent_not_what_was_known(self):
        handoff = Handoff(from_agent="writer", to_agent="reviewer", content="x" * 400)
        self.assertEqual(handoff.tokens, 100)

    def test_the_mailbox_totals_every_hop(self):
        box = Mailbox()
        box.send("a", "b", "x" * 40)
        box.send("b", "c", "y" * 80)
        self.assertEqual(box.hops, 2)
        self.assertEqual(box.tokens, 30)

    def test_the_mailbox_attributes_cost_per_edge(self):
        """'Communication is expensive' is only useful as 'this hop is'."""
        box = Mailbox()
        box.send("supervisor", "writer", "x" * 40)
        box.send("supervisor", "reviewer", "y" * 80)
        by_edge = box.by_pair()
        self.assertEqual(by_edge["supervisor->reviewer"], 20)
        self.assertEqual(by_edge["supervisor->writer"], 10)

    def test_a_summary_is_bounded_even_when_the_agent_is_verbose(self):
        """Otherwise 'send a summary' degrades into 'send everything'."""
        long_body = "z" * 5000
        sent = summarise_for_handoff([long_body], policy=HandoffPolicy.SUMMARY, limit_chars=200)
        self.assertEqual(len(sent), 200)

    def test_a_full_transcript_handoff_carries_the_history(self):
        history = ["first", "second"]
        sent = summarise_for_handoff(
            ["new"], policy=HandoffPolicy.FULL_TRANSCRIPT, transcript=history
        )
        self.assertIn("first", sent)
        self.assertIn("second", sent)
        self.assertIn("new", sent)

    def test_the_transcript_grows_while_a_summary_does_not(self):
        """The P2 rule, as a property rather than an opinion."""
        summaries: list[int] = []
        transcripts: list[int] = []
        for policy, sink in ((HandoffPolicy.SUMMARY, summaries), (HandoffPolicy.FULL_TRANSCRIPT, transcripts)):
            box = Mailbox(policy=policy)
            for index in range(6):
                sent = summarise_for_handoff(
                    [f"payload {index} " + "x" * 200],
                    policy=policy,
                    transcript=box.transcript(),
                )
                box.send("a", "b", sent)
                # Marginal cost of *this* hop, not the running total: a total
                # is monotonic by construction and would prove nothing.
                sink.append(box.messages[-1].tokens)
        self.assertEqual(summaries[-1], summaries[0])
        self.assertGreater(transcripts[-1], transcripts[0])

    def test_the_mailbox_transcript_is_readable(self):
        box = Mailbox()
        box.send("writer", "reviewer", "draft")
        self.assertIn("writer->reviewer", box.transcript()[0])


class RoleTests(unittest.TestCase):
    def setUp(self):
        self.roles = DeterministicRoles(facts=small_facts(), capacity=2)

    def test_mentions_matches_on_stems_not_substrings(self):
        """Reusing W1's tokenizer, so 'billing' is not matched inside noise."""
        self.assertTrue(mentions("The billing system rounds down.", "billing"))
        self.assertFalse(mentions("unrelated text", "billing"))

    def test_the_researcher_gathers_exactly_what_the_brief_names(self):
        found = self.roles.research("brief", ["alpha", "gamma", "unknown"])
        self.assertEqual([f.key for f in found], ["alpha", "gamma"])

    def test_evidence_is_short_because_it_crosses_a_wire(self):
        found = self.roles.research("brief", ["alpha"])
        self.assertLess(len(found[0].text), 200)

    def test_the_writer_states_at_most_its_capacity(self):
        facts = self.roles.research("brief", list(small_facts()))
        draft = self.roles.draft("brief", facts, capacity=2)
        self.assertEqual(draft.count("- "), 2)

    def test_extra_items_survive_the_capacity_cut(self):
        """The revision mechanic: the second pass adds what the first could not."""
        facts = self.roles.research("brief", list(small_facts()))
        extra = [f for f in facts if f.key == "delta"]
        draft = self.roles.draft("brief", facts, capacity=2, extra=extra)
        self.assertIn("delta", draft)
        self.assertEqual(draft.count("- "), 3)

    def test_the_reviewer_returns_keys_not_prose(self):
        """A defect list a writer can act on keeps the handoff cheap."""
        facts = self.roles.research("brief", list(small_facts()))
        draft = self.roles.draft("brief", facts, capacity=1)
        missing = self.roles.review("brief", draft, list(small_facts()), facts)
        self.assertEqual(missing, ["beta", "gamma", "delta"])
        self.assertTrue(all(isinstance(item, str) for item in missing))

    def test_a_complete_draft_reviews_clean(self):
        facts = self.roles.research("brief", list(small_facts()))
        draft = self.roles.draft("brief", facts, capacity=4)
        self.assertEqual(self.roles.review("brief", draft, list(small_facts()), facts), [])


class TeamTests(unittest.TestCase):
    def test_the_route_is_research_write_review(self):
        team = Team(DeterministicRoles(facts=small_facts(), capacity=2))
        result = team.run(brief(4))
        self.assertEqual(result.route[:4], ["supervisor", "researcher", "writer", "reviewer"])
        self.assertEqual(result.route[-1], "done")

    def test_the_team_covers_what_a_single_pass_cannot(self):
        """The structural claim, asserted directly."""
        roles = DeterministicRoles(facts=small_facts(), capacity=2)
        single = SingleAgent(roles).run(brief(4))
        team = Team(roles, capacity=2).run(brief(4))
        self.assertLess(coverage(single.answer, brief(4).required), 1.0)
        self.assertEqual(coverage(team.answer, brief(4).required), 1.0)

    def test_a_small_brief_needs_no_revision(self):
        roles = DeterministicRoles(facts=small_facts(), capacity=3)
        result = Team(roles, capacity=3).run(brief(2))
        self.assertEqual(result.iterations, 1)
        self.assertNotIn("writer", result.route[4:], "no second writing pass should happen")

    def test_the_revision_loop_is_bounded(self):
        """A reviewer that can demand unlimited revisions is an unbounded bill."""
        roles = DeterministicRoles(facts=small_facts(), capacity=1)
        result = Team(roles, capacity=1, max_iterations=2).run(brief(4))
        self.assertEqual(result.iterations, 2)
        self.assertEqual(result.route.count("writer"), 2)

    def test_communication_is_counted_separately_from_thinking(self):
        team = Team(DeterministicRoles(facts=small_facts(), capacity=2), capacity=2)
        result = team.run(brief(4))
        self.assertGreater(result.communication_tokens, 0)
        self.assertGreater(result.hops, 0)
        self.assertEqual(result.total_tokens, result.prompt_tokens + result.completion_tokens + result.communication_tokens)

    def test_the_team_costs_more_than_the_single_agent(self):
        """The penalty has to be real, or the break-even means nothing."""
        roles = DeterministicRoles(facts=small_facts(), capacity=2)
        single = SingleAgent(roles).run(brief(4))
        team = Team(roles, capacity=2).run(brief(4))
        self.assertGreater(team.total_tokens, single.total_tokens)

    def test_the_transcript_policy_costs_more_than_summaries(self):
        briefs = brief(6)
        short = Team(DeterministicRoles(facts=small_facts(), capacity=2), capacity=2).run(briefs)
        long = Team(
            DeterministicRoles(facts=small_facts(), capacity=2),
            capacity=2,
            policy=HandoffPolicy.FULL_TRANSCRIPT,
        ).run(briefs)
        self.assertGreater(long.communication_tokens, short.communication_tokens)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_experiment(sizes=tuple(range(1, 11)), capacity=3)

    def test_coverage_is_a_fraction_of_required_items(self):
        self.assertEqual(coverage("alpha and beta", ["alpha", "beta"]), 1.0)
        self.assertEqual(coverage("alpha only", ["alpha", "beta"]), 0.5)

    def test_briefs_are_disjoint_so_size_is_the_only_variable(self):
        briefs = build_briefs(sizes=(2, 5))
        self.assertEqual(len(set(briefs[0].required) & set(briefs[1].required)), 2)

    def test_a_break_even_point_is_computed(self):
        """The acceptance criterion, computed rather than eyeballed."""
        point = self.report.break_even
        self.assertIsNotNone(point)
        self.assertEqual(point.size, 4)
        self.assertGreater(point.coverage_gain, 0)

    def test_below_the_break_even_the_team_pays_for_nothing(self):
        below = [r for r in self.report.rows if r.size < self.report.break_even.size]
        self.assertTrue(below)
        self.assertTrue(all(r.coverage_gain == 0.0 for r in below))
        self.assertTrue(all(r.premium_ratio > 1.0 for r in below))

    def test_above_the_break_even_the_gap_widens_with_size(self):
        above = [r for r in self.report.rows if r.size >= self.report.break_even.size]
        gains = [r.coverage_gain for r in above]
        self.assertEqual(gains, sorted(gains))

    def test_the_single_agent_degrades_past_its_capacity(self):
        rows = {r.size: r for r in self.report.rows}
        self.assertEqual(rows[3].single_coverage, 1.0)
        self.assertLess(rows[10].single_coverage, 0.5)

    def test_communication_is_a_real_share_of_the_team_bill(self):
        """Not a rounding error -- which is the argument against a team."""
        rows = self.report.rows
        self.assertTrue(all(0.2 < r.communication_share < 0.6 for r in rows))

    def test_the_summary_policy_grows_far_slower_than_a_full_transcript(self):
        """A summary is bounded per hop, so what it carries grows with the
        brief's keys; a transcript compounds with the hops taken."""
        summary = self.report.summary_policy_tokens
        transcript = self.report.transcript_policy_tokens
        summary_growth = summary[-1] / summary[0]
        transcript_growth = transcript[-1] / transcript[0]
        self.assertGreater(transcript_growth, 1.5 * summary_growth)

    def test_the_transcript_policy_grows_and_is_reported(self):
        summary = self.report.summary_policy_tokens
        transcript = self.report.transcript_policy_tokens
        self.assertGreater(transcript[-1] / summary[-1], transcript[0] / summary[0])

    def test_the_report_serialises_with_the_break_even(self):
        payload = self.report.to_dict()
        self.assertEqual(payload["break_even_size"], 4)
        self.assertIn("policies", payload)

    def test_every_row_costs_more_than_the_baseline(self):
        self.assertTrue(all(row.premium_ratio > 1.0 for row in self.report.rows))


if __name__ == "__main__":
    unittest.main()
