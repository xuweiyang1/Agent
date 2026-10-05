"""The whole chain: one image in, one governed action sequence out.

Every week so far has its own demo, and each demo proves one thing in
isolation. This module exists because the ROADMAP's product promise is not
"seven features", it is *one assistant*: "a photo goes in and comes out as
todos, and on the way it passed through memory". A chain is a different
claim from a pile of parts, and it can fail in a way no single week can --
a step that works alone but cannot hand its result to the next one.

So the design constraint here is not capability, it is **evidence at every
joint**. Each step publishes a small ``StepResult``: what it was given, what
it decided, and why. The demo prints that list, which means the thing you
read is the actual handoff, not a narration of one.

Three decisions worth stating, because each could have gone the other way:

- **The request is thin on purpose.** ``"plan a weekend trip"`` names intent
  and no facts. The destination, the budget and the note arrive only through
  the OCR step. If the request carried them too, the chain would still print
  "perceived the board" while actually parsing a string, and the multimodal
  claim would be false. A test pins the request for exactly this reason.
- **The chain is not a LangGraph.** W4's graph is a *planner* with branches
  and a resumable checkpoint; this is a script of fixed handoffs. Wrapping
  five sequential calls in a state machine would import the framework's cost
  and add no decision -- and the framed version would be harder to read, not
  easier. LangGraph is used where a branch is real (W4) and left out where it
  is not.
- **A step that cannot run is not a step that crashes.** Planning without a
  budget is a planner with no constraint, so the chain stops there and says
  so; a missing note degrades a todo instead of failing the run. Which steps
  are load-bearing is a property of the chain, declared in one place rather
  than discovered by a stack trace.

The exception to "no branches" is the last step, and it is deliberate: a
Worker that has all the coverage it needs should NOT be sent through a
review loop, because W7 measured that the loop costs about 2.9x and returns
nothing. So ``govern`` is the one place in this file that makes a real
decision, and it makes it on a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agentkit.dispatch import ToolInvoker
from agentkit.registry import ToolRegistry
from agentkit.tools import build_registry

from memory.store import KIND_DECISION

from .ocr import OcrEngine, OcrResult, perceive_board

# Where a chain run keeps anything it produces on disk. Under the system temp
# dir by default so importing this module never writes into the repository.
DEFAULT_ARTIFACTS = "assistant-chain"

# The trip happens on the weekend after the board is read. Expressed as a
# weekday number so the date is computed, not hard-coded -- a checked-in date
# is a demo that stops working next month.
WEEKEND_WEEKDAY = 5  # Saturday


@dataclass
class StepResult:
    """One handoff, with the evidence for it.

    ``inputs`` and ``outputs`` are the actual values passed across the joint.
    Recording them is the difference between a chain you can debug and a chain
    you can only admire: when output 4 is wrong, this says whether step 3
    computed it wrong or step 4 received it wrong.
    """

    name: str
    ok: bool
    detail: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    tokens: int = 0
    cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.name,
            "ok": self.ok,
            "detail": self.detail,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tokens": self.tokens,
            "cost": round(self.cost, 4),
        }


@dataclass
class ChainRun:
    """A finished run: every step, every artifact, and what it wrote."""

    request: str
    board_lines: list[str]
    perceived: dict[str, str]
    steps: list[StepResult] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    answer: str = ""
    todos: list[dict[str, Any]] = field(default_factory=list)
    event: dict[str, Any] = field(default_factory=dict)
    itinerary: dict[str, Any] = field(default_factory=dict)
    route: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)

    @property
    def planned_destination(self) -> str:
        """What the board asked for. Empty if the board could not be read."""
        return self.perceived.get("destination", "")

    @property
    def divergence(self) -> str:
        """Where the plan departed from the request, as one readable sentence.

        A chain that silently answers a different question than the one asked
        is the most expensive kind of bug, because everything downstream looks
        healthy. Recording the substitution -- with the reason -- is what makes
        it a decision the user can disagree with rather than a surprise they
        discover on the trip.

        The reason is read off the avoid list that was actually passed to the
        planner, not guessed from its prose. A note is not evidence; the
        constraint that was applied is.
        """
        wanted = self.planned_destination
        got = str(self.itinerary.get("destination", ""))
        if not wanted or not got or wanted.strip().lower() == got.strip().lower():
            return ""
        if wanted.strip().lower() in {name.strip().lower() for name in self.avoid}:
            why = f"{wanted} was excluded by a recorded past decision"
        else:
            why = f"nothing matching {wanted} survived the weather and budget checks"
        return f"the board named {wanted}, the plan uses {got}: {why}"

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps)

    @property
    def failed(self) -> list[StepResult]:
        return [step for step in self.steps if not step.ok]

    @property
    def tokens(self) -> int:
        return sum(step.tokens for step in self.steps)

    def step(self, name: str) -> StepResult | None:
        return next((s for s in self.steps if s.name == name), None)

    def render(self) -> str:
        lines: list[str] = []
        for index, step in enumerate(self.steps, start=1):
            mark = "ok " if step.ok else "FAIL"
            lines.append(f"{index}. [{mark}] {step.name}: {step.detail}")
            for key, value in step.outputs.items():
                shown = value if not isinstance(value, str) else value.replace("\n", " | ")
                lines.append(f"      {key}: {shown}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request,
            "board_lines": list(self.board_lines),
            "perceived": dict(self.perceived),
            "ok": self.ok,
            "tokens": self.tokens,
            "route": list(self.route),
            "artifacts": dict(self.artifacts),
            "steps": [step.to_dict() for step in self.steps],
        }


def _estimate(text: str) -> int:
    """The cheap character/4 estimate, matching the rest of the repository.

    Not a tokenizer. Every other cost number in this project uses the same
    approximation, and mixing in a different one here would make this chain's
    totals incomparable with W6's and W7's.
    """
    return max(1, (len(text) + 3) // 4)


def weekend_after(today: Any) -> Any:
    """The Saturday strictly after ``today``.

    Strictly after, so a board read on a Saturday plans the *next* weekend
    rather than today -- which is what someone writing "this weekend" on a
    Saturday afternoon almost certainly means, and avoids an event that is
    already happening.
    """
    from datetime import timedelta

    ahead = (WEEKEND_WEEKDAY - today.weekday()) % 7
    return today + timedelta(days=ahead or 7)


# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------


def perceive_step(
    *,
    engine: OcrEngine,
    board_lines: Sequence[str],
    known_lines: dict[str, str],
    image_path: Path,
    title: str = "",
    scale: int = 1,
) -> tuple[StepResult, OcrResult]:
    """Render the board and read it back.

    Nothing downstream may access ``known_lines``. The step's outputs are what
    the *pixels* yielded, and the returned ``OcrResult`` carries the
    confidence so a partial read is visible rather than silent.
    """
    result, written, image = perceive_board(
        board_lines,
        engine=engine,
        known_lines=known_lines,
        path=image_path,
        title=title,
        scale=scale,
    )
    pixels = image.size[0] * image.size[1]
    step = StepResult(
        name="perceive",
        ok=result.confidence >= 1.0,
        detail=(
            f"read {result.matched}/{result.expected} board row(s) at "
            f"confidence {result.confidence:.0%}"
        ),
        inputs={"image": written.name, "engine": result.engine, "pixels": pixels},
        outputs={"perceived": "; ".join(f"{v}" for v in result.lines)},
        # Perception is billed like an image on the wire, because that is what
        # a real vision model would receive. Counting it as free text is how
        # a multimodal chain gets to look cheaper than it is.
        tokens=pixels // 750,
        cost=len(written.read_bytes()) / 1024.0,
    )
    return step, result


def memory_step(longterm: Any, *, query: str) -> StepResult:
    """Consult long-term memory before planning, and say what it changed.

    Both halves of W5 are exercised: an exact ``preference`` lookup, which is a
    constraint the plan must respect, and a semantic ``recall``, which is how a
    past *rejection* becomes this run's ``avoid`` list. That second one is the
    whole point of remembering decisions -- without it, the assistant is
    perfectly happy to propose the city you declined last month.
    """
    prefs = longterm.preferences()
    hits = longterm.recall(query, k=3)
    rejected = [
        record.key
        for hit in hits
        if (record := longterm.semantic.record_of(hit.doc_id)) is not None
        and record.kind == KIND_DECISION
        and str(record.value.get("outcome", "")).lower() == "rejected"
    ]
    step = StepResult(
        name="memory",
        ok=True,
        detail=f"{len(prefs)} preference(s), {len(hits)} recalled, {len(rejected)} past rejection(s)",
        inputs={"query": query},
        outputs={
            "preferences": ", ".join(f"{k}={v}" for k, v in sorted(prefs.items())) or "(none)",
            "avoid": ", ".join(sorted(set(rejected))) or "(none)",
        },
        tokens=_estimate(query),
    )
    return step


async def plan_step(
    *,
    graph: Any,
    config: dict[str, Any],
    request: str,
    query: str,
    nights: int,
    budget: float,
    currency: str,
    avoid: Sequence[str],
) -> tuple[StepResult, dict[str, Any]]:
    """Run W4's planner with the facts the board supplied.

    Two joints meet here, and both are the reason this step takes arguments a
    bare planner call would not need:

    - ``query`` is the *board's* destination folded into the search text. W4's
      search is lexical, so a request of "plan a weekend trip" retrieves
      whatever ranks highest and never the city the user actually wrote down.
      Seeding ``query`` is what makes the plan about the user's trip rather
      than about the word "weekend".
    - ``avoid`` is W5's output. The planner is told which cities this user has
      already declined, so a rejection recorded last month is a constraint
      today. That is the difference between remembering a decision and
      displaying one.

    A refusal is deliberately *not* raised here when the two conflict. If the
    board names a city the user has already rejected, the planner quietly picks
    something else -- and the step's outputs record that, so the coordinator can
    explain the substitution instead of hiding it.
    """
    from graph.state import initial_state

    state = initial_state(
        request,
        query=query,
        nights=nights,
        budget=budget,
        currency=currency,
        avoid=list(avoid),
    )
    # ``ainvoke`` rather than ``invoke``: the nodes call tools, and the tool
    # registry is async. A synchronous call would need a second set of
    # synchronous node functions -- two implementations of one plan, which is
    # how the two drift.
    await graph.ainvoke(state, config)
    graph.update_state(config, {"approved": True})
    await graph.ainvoke(None, config)
    values = graph.get_state(config).values

    itinerary = values.get("itinerary") or {}
    chosen = values.get("chosen") or {}
    step = StepResult(
        name="plan",
        ok=bool(itinerary),
        detail=(
            f"chose {chosen.get('name')} for {chosen.get('cost')} {chosen.get('cost_currency')}"
            if itinerary
            else "no candidate survived the weather and budget checks"
        ),
        inputs={
            "request": request,
            "query": query,
            "budget": f"{budget:.0f} {currency}",
            "avoid": ", ".join(avoid) or "(none)",
        },
        outputs={
            "destination": itinerary.get("destination", "(none)"),
            "weather": itinerary.get("weather", ""),
            "why": chosen.get("why", ""),
            "attempts": len(values.get("attempts", [])),
        },
        tokens=sum(_estimate(note) for note in values.get("notes", [])),
    )
    return step, values


def retrieve_step(*, retriever: Any, graph_state: dict[str, Any]) -> StepResult:
    """Retrieve the corpus evidence for the chosen destination.

    W6 semantics, not a pipeline: the retriever decides whether to search, when
    the first result is not enough, and what to search for next. It runs after
    the plan because the plan is what makes the query specific -- asking before
    choosing would mean searching for "a weekend trip" and hoping.
    """
    from retrieval.corpus_rag import RagTask

    destination = str((graph_state.get("itinerary") or {}).get("destination", ""))
    question = f"What should a weekend in {destination} include?"
    task = RagTask(id="chain", category="chain", question=question)

    from retrieval.naive import ExtractiveGenerator

    run = retriever.run(task, ExtractiveGenerator())
    step = StepResult(
        name="retrieve",
        ok=bool(run.citations) if run.retrieved else True,
        detail=(
            f"{run.retrievals} retrieval(s), {len(run.citations)} citation(s)"
            if run.retrieved
            else "decided no retrieval was needed"
        ),
        inputs={"question": question},
        outputs={
            "queries": " | ".join(s.query for s in run.steps) or "(none)",
            "sources": ", ".join(run.sources) or "(none)",
            "answer": run.answer[:160],
        },
        tokens=run.prompt_tokens,
    )
    return step


async def todos_step(*, invoker: ToolInvoker, graph_state: dict[str, Any], perceived: dict[str, str]) -> tuple[StepResult, list[dict[str, Any]]]:
    """Turn the plan and the *board* into todos, through the W2 todo tool.

    Called through the registry rather than by touching ``TodoService`` so the
    chain exercises the same schema, validation and error path a model would.
    The board contributes the note and the budget; the plan contributes the
    destination and the weather. Both are needed -- that is the chain.
    """
    itinerary = graph_state.get("itinerary") or {}
    destination = str(itinerary.get("destination", "") or "the destination")
    note = perceived.get("note", "")
    budget = perceived.get("budget", "")

    wanted: list[str] = []
    if destination:
        wanted.append(f"Book travel and stay for {destination}")
    if budget:
        wanted.append(f"Pay the {budget} trip budget for {destination}")
    if note:
        wanted.append(f"Reserve the {note} in {destination}")
    if itinerary.get("outdoor_ok") is False:
        wanted.append("Pack for rain: the forecast moved the plan indoors")

    created: list[dict[str, Any]] = []
    for text in wanted:
        result = await invoker.invoke("todo", {"action": "add", "text": text}, call_id="chain")
        if result.ok:
            created.append(result.value["added"])

    step = StepResult(
        name="todos",
        ok=len(created) == len(wanted) and bool(created),
        detail=f"created {len(created)} of {len(wanted)} todo(s)",
        inputs={"destination": destination, "note": note or "(none)", "budget": budget or "(none)"},
        outputs={"todos": " | ".join(item["text"] for item in created) or "(none)"},
        tokens=sum(_estimate(item["text"]) for item in created),
    )
    return step, created


async def calendar_step(
    *,
    invoker: ToolInvoker,
    graph_state: dict[str, Any],
    perceived: dict[str, str],
    today: Any,
) -> tuple[StepResult, dict[str, Any]]:
    """Book the trip on the weekend the board named.

    The date is *computed* from the board's words rather than asked for as a
    date: "this weekend" is what a person writes on a whiteboard, and resolving
    it is part of the job. The resolved date goes into the step's outputs so
    the interpretation is inspectable rather than implicit.
    """
    from datetime import datetime, time, timedelta

    itinerary = graph_state.get("itinerary") or {}
    destination = str(itinerary.get("destination", "") or "Trip")
    nights = max(1, int(itinerary.get("nights", 2) or 2))
    first = weekend_after(today)
    title = f"Trip to {destination}"

    # One event per night rather than one event spanning the trip. The calendar
    # tool caps an event at 24 hours, and that cap is right: a month-long block
    # is not an appointment. Splitting is also what a person's calendar
    # actually looks like, so the modeling limitation and the useful shape
    # agree -- the boundary is kept at the tool's limit on purpose.
    events: list[dict[str, Any]] = []
    failures: list[str] = []
    for offset in range(nights):
        day = first + timedelta(days=offset)
        start = datetime.combine(day, time(9, 0)).isoformat(timespec="minutes")
        result = await invoker.invoke(
            "calendar",
            {"action": "create", "title": title, "start": start, "duration_minutes": 60 * 24},
            call_id="chain",
        )
        if result.ok and isinstance(result.value, dict):
            events.append(result.value.get("created", {}))
        else:
            failures.append(result.output[:120])

    step = StepResult(
        name="calendar",
        ok=bool(events) and not failures,
        detail=(
            f"booked {len(events)} night(s) from {first.isoformat()} "
            f"(weekend resolved from the board)"
            if events
            else f"calendar refused every event: {failures[0] if failures else 'no events'}"
        ),
        inputs={
            "board_trip": perceived.get("trip", "(none)"),
            "today": today.isoformat(),
            "nights": nights,
        },
        outputs={
            "events": " | ".join(f"{e.get('start')} -> {e.get('end')}" for e in events) or "(none)",
        },
        tokens=sum(_estimate(title + str(e.get("start", ""))) for e in events),
    )
    return step, (events[0] if events else {})


async def chart_step(*, invoker: ToolInvoker, graph_state: dict[str, Any]) -> StepResult:
    """Render the shortlist as a chart, through the W3 chart tool.

    This is the "output multimodality is just another tool" claim, tested at
    the joint rather than in isolation: the values come from ``quotes``, which
    the planner produced, so a chart here proves the planner's numbers are
    usable by a later step.
    """
    quotes = graph_state.get("quotes") or []
    if not quotes:
        return StepResult(name="chart", ok=False, detail="no quotes to chart")
    labels = [str(q.get("destination", "?")) for q in quotes]
    values = [float(q.get("budget_cost", 0.0)) for q in quotes]
    currency = str(quotes[0].get("budget_currency", ""))

    result = await invoker.invoke(
        "render_chart",
        {
            "kind": "bar",
            "labels": labels,
            "values": values,
            "title": f"Weekend shortlist ({currency})",
            "y_label": currency,
        },
        call_id="chain",
    )
    value = result.value if isinstance(result.value, dict) else {}
    path = str(value.get("path", ""))
    return StepResult(
        name="chart",
        ok=bool(path),
        detail=f"rendered a {len(labels)}-bar comparison" if path else f"chart failed: {result.output[:120]}",
        inputs={"labels": ", ".join(labels)},
        outputs={"path": path or "(none)", "inline_bytes": value.get("inline_chars", 0)},
        tokens=_estimate(" ".join(labels)),
    )


def persist_step(
    *,
    longterm: Any,
    graph_state: dict[str, Any],
    perceived: dict[str, str],
    session: str,
) -> StepResult:
    """Write this run's outcome back, so the next run starts better informed.

    Two writes, because they are used differently later: a *decision* (which
    becomes an avoid entry) and a *fact* (which is only findable by semantic
    recall). Writing both is what closes the loop -- W5's memory is only
    interesting if something later reads what this wrote.
    """
    itinerary = graph_state.get("itinerary") or {}
    destination = str(itinerary.get("destination", "") or "unknown")
    decision = longterm.remember_decision(
        destination,
        "approved",
        reason=f"{itinerary.get('cost')} {itinerary.get('cost_currency')} of a {perceived.get('budget', 'unknown')} budget, {itinerary.get('weather', '')}",
        session=session,
    )
    fact = longterm.remember_fact(
        f"The weekend trip goes to {destination}; the plan is "
        f"{itinerary.get('mode', 'unknown')} because of the forecast.",
        session=session,
    )
    return StepResult(
        name="persist",
        ok=True,
        detail=f"wrote {len(longterm)} record(s) to long-term memory",
        inputs={"session": session},
        outputs={"decision": decision.id, "fact": fact.id, "total": len(longterm)},
        tokens=_estimate(decision.text + fact.text),
    )


def govern_step(
    *,
    coverage: float,
    team: Any,
    brief: Any,
    single: Any,
    max_iterations: int,
) -> StepResult:
    """Apply W7's measured rule instead of running a team out of habit.

    This is the one real branch in the chain, and it is built on a number W7
    produced: below the writing room a review loop costs about 2.9x and gains
    nothing, so it is skipped; above it, the loop recovers what a single pass
    structurally cannot. Recording *why* in the step is what makes the branch
    auditable -- "we did not use a team here" is a decision, not an omission.
    """
    if coverage >= 1.0:
        return StepResult(
            name="govern",
            ok=True,
            detail="single pass: the brief is inside one pass's writing room",
            inputs={"coverage": f"{coverage:.2f}"},
            outputs={"strategy": "single", "reason": "review loop would cost ~2.9x for no coverage gain"},
            tokens=single.total_tokens if single is not None else 0,
        )

    result = team.run(brief)
    return StepResult(
        name="govern",
        ok=True,
        detail=f"revision loop: single-pass coverage was {coverage:.2f}",
        inputs={"coverage": f"{coverage:.2f}", "max_iterations": max_iterations},
        outputs={
            "strategy": "team",
            "iterations": result.iterations,
            "hops": result.hops,
            "communication_tokens": result.communication_tokens,
        },
        tokens=result.total_tokens,
    )


__all__ = [
    "ChainRun",
    "DEFAULT_ARTIFACTS",
    "StepResult",
    "calendar_step",
    "chart_step",
    "govern_step",
    "memory_step",
    "perceive_step",
    "persist_step",
    "plan_step",
    "retrieve_step",
    "todos_step",
    "weekend_after",
]
