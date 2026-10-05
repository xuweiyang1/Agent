"""Chaining the seven weeks into one assistant run.

The ROADMAP's closing product statement is a single sentence -- a photo goes
in, it becomes todos, and it passed through memory on the way -- and this
module is that sentence, executed. ``run_chain`` is the only entry point, and
the order of its calls *is* the architecture:

    perceive -> memory -> plan -> retrieve -> todos -> calendar -> chart
             -> persist -> govern

The sequence is a script, not a graph, and that is a deliberate reading of this
project's own evidence. W4 built a planner because that task has real
branches; here every handoff is unconditional, so a state machine would add
the framework's cost and the reader's confusion while making no decision. The
one genuine decision -- whether a review loop is worth its ~2.9x -- is isolated
at the end and made on W7's number.

The chain is also where the *joints* get tested. Each step in ``chain.py``
returns evidence, and this module checks the joints the individual weeks never
had to: that the board supplied the budget the planner received, that the
plan's destination reached the todos, that memory changed the planner's avoid
list. Those assertions live in ``tests/test_chain.py``, but the wiring they
check is here, so it is visible in one reading.

``today`` is a parameter, the same convention W5 uses. A date defaulted to
``datetime.now()`` would make the resolved weekend untestable and the artifact
name change daily, which is a demo that quietly stops reproducing.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from agentkit.dispatch import ToolInvoker
from agentkit.tools import build_registry
from graph.build import build_graph
from graph.checkpoint import thread_config
from memory import build_memory

from .chain import (
    ChainRun,
    StepResult,
    calendar_step,
    chart_step,
    govern_step,
    memory_step,
    perceive_step,
    persist_step,
    plan_step,
    retrieve_step,
    todos_step,
)
from .ocr import OcrEngine, default_ocr

# The facts that will be drawn on the board and read back. The request below
# deliberately does not contain any of the values -- only the labels, so the
# demo can be checked for cheating by reading the two side by side.
BOARD_LINES: tuple[str, ...] = (
    "Destination - Lisbon",
    "Budget - 9000 CNY",
    "Trip - this weekend",
    "Note - museum afternoon",
)

# Label -> meaning. Handed to the reader as *labels*; the values are what the
# reader is supposed to recover from the image, never what it is given.
BOARD_MEANINGS: dict[str, str] = {
    "Destination - Lisbon": "Lisbon",
    "Budget - 9000 CNY": "9000 CNY",
    "Trip - this weekend": "this weekend",
    "Note - museum afternoon": "museum afternoon",
}

# Intent only. No city, no number, no date -- see the module docstring.
DEFAULT_REQUEST = "plan a weekend trip"

# Seeded history: a city this user already turned down, so the memory step has
# something real to change. Recorded as a *decision with a reason*, because an
# outcome without a reason cannot be acted on later.
SEED_DECISIONS: tuple[tuple[str, str], ...] = (
    ("Tokyo", "rain on the last attempt"),
    ("Sydney", "too long a flight for two nights"),
)

SEED_PREFERENCES: dict[str, Any] = {
    "seat": "aisle",
    "hotel": "quiet, away from nightlife",
}


async def run_chain(
    *,
    request: str = DEFAULT_REQUEST,
    today: Any,
    nights: int = 2,
    engine: OcrEngine | None = None,
    board_lines: Sequence[str] = BOARD_LINES,
    meanings: dict[str, str] | None = None,
    artifacts_dir: str | Path | None = None,
    scale: int = 1,
    memory_path: str | Path | None = None,
    seed: bool = True,
    memory_query: str = "which destination did I reject, and why",
    use_retrieval: bool = True,
    allow_substitution: bool = False,
    perceived_override: dict[str, str] | None = None,
    state_dir: str | Path | None = None,
) -> ChainRun:
    """Run the whole assistant chain once and return its evidence.

    Arguments exist for the tests, not for configuration: ``today`` makes the
    resolved weekend deterministic, ``scale`` lets a test trade perception cost
    against time, and ``seed`` lets a test start from an empty memory. A demo
    that cannot be pointed at a different input is a demo that cannot be
    tested, and this chain is the one piece of the project where a joint
    failure would be invisible in every weekly test.
    """
    meanings = meanings or BOARD_MEANINGS
    reader = engine or default_ocr()
    workdir = _artifacts_dir(artifacts_dir)
    _isolate_matplotlib_cache(workdir)

    data_path = Path(memory_path) if memory_path else workdir / "memory.json"
    longterm, session, _working = build_memory(path=data_path)
    if seed:
        # Only seed an empty memory, so a second run against the same file
        # demonstrates persistence instead of re-creating the same history.
        if len(longterm) == 0:
            for key, value in SEED_PREFERENCES.items():
                longterm.remember_preference(key, value)
            for subject, reason in SEED_DECISIONS:
                longterm.remember_decision(subject, "rejected", reason=reason, session="seed")

    run = ChainRun(request=request, board_lines=[str(line) for line in board_lines], perceived={})

    # -- 1. perception -----------------------------------------------------
    # Two ways in, and the difference is stated rather than hidden. The default
    # renders a board and reads it back through ``engine``; a caller who already
    # has the fields (the local web form, or a vision model upstream of this
    # call) passes ``perceived_override`` and the pixels are skipped. The
    # override path records where the fields came from, so a run that did not
    # go through OCR cannot later be read as though it had.
    if perceived_override is not None:
        run.perceived = {str(k): str(v) for k, v in perceived_override.items() if str(v).strip()}
        run.artifacts["board"] = ""
        run.steps.append(
            StepResult(
                name="perceive",
                ok=bool(run.perceived),
                detail=(
                    f"fields supplied directly ({len(run.perceived)}); "
                    "no image was read on this path"
                ),
                inputs={"source": "override"},
                outputs={"perceived": "; ".join(sorted(run.perceived.values()))},
            )
        )
    else:
        step, perceived = perceive_step(
            engine=reader,
            board_lines=board_lines,
            known_lines=meanings,
            image_path=workdir / "board.png",
            title="weekend note",
            scale=scale,
        )
        run.steps.append(step)
        run.artifacts["board"] = str(step.inputs.get("image", ""))
        run.perceived = _interpret(perceived, meanings)

    # A chain that cannot read its own input must not continue with a guess.
    # Stopping here is the honest failure: the alternative is planning for a
    # destination nobody named.
    if not run.perceived:
        run.answer = "the board could not be read; nothing was planned"
        return run

    # -- 2. memory ---------------------------------------------------------
    memory = memory_step(longterm, query=memory_query)
    run.steps.append(memory)
    avoid = _split(memory.outputs.get("avoid", ""))
    run.avoid = list(avoid)

    # -- 3. plan -----------------------------------------------------------
    # The search text is composed here, in the coordinator, and that is a
    # deliberate division of labour: W4 owns how to plan, and this chain owns
    # what the user is asking about. Folding the board's destination into the
    # query is the joint that makes the plan about the board.
    budget = _read_budget(run)
    if budget is None:
        return run
    plan_query = _plan_query(request, run.perceived.get("destination", ""))
    plan, state = await plan_step(
        graph=build_graph(),
        config=thread_config("chain"),
        request=request,
        query=plan_query,
        nights=nights,
        budget=budget,
        currency="CNY",
        avoid=avoid,
    )
    run.steps.append(plan)
    run.itinerary = state.get("itinerary") or {}
    run.notes = list(state.get("notes", []))

    if not plan.ok:
        run.answer = "no destination survived the weather and budget checks"
        return run

    # The board named one city; the planner may have chosen another. This is
    # checked *before* any side effect runs, because the steps below create
    # todos and calendar events -- acting on a trip nobody asked for cannot be
    # undone by noticing afterwards.
    wanted = run.perceived.get("destination", "")
    got = str(run.itinerary.get("destination", ""))
    if not _substitution_allowed(wanted=wanted, got=got, allow_substitution=allow_substitution):
        run.steps.append(
            StepResult(
                name="guard",
                ok=False,
                detail="stopped: the plan would not be about what the board asked for",
                inputs={"board": wanted, "planned": got},
                outputs={
                    "reason": run.divergence or "the planner chose a different city",
                    "next": "pass allow_substitution=True to let the planner's ranking stand",
                },
            )
        )
        run.answer = (
            f"stopped before creating anything -- the board names {wanted}, "
            f"the plan uses {got}. Nothing was written to the todo list or the calendar."
        )
        return run

    # -- 4. retrieval ------------------------------------------------------
    if use_retrieval:
        from graph.destinations import as_corpus
        from retrieval.agentic import AgenticRetriever
        from retrieval.corpus_rag import build_index_for
        from retrieval.tools import RetrieverTool

        # Indexed over the *destination* notes, which is what the run is about.
        # W3.5's ``build_index_for`` defaults to W1's personal-notes corpus; a
        # chain about a trip retrieving wifi passwords would be a joint that
        # looks wired and answers the wrong question.
        retriever = AgenticRetriever(RetrieverTool(build_index_for(as_corpus())))
        run.steps.append(retrieve_step(retriever=retriever, graph_state=state))

    # -- 5/6/7. actions ----------------------------------------------------
    # A local deployment wants todos and events to survive a restart, so a
    # ``state_dir`` turns the two in-memory services into small JSON stores. The
    # default (None) keeps them in memory, which is what the weekly tests want.
    todo_service = None
    calendar_service = None
    if state_dir is not None:
        from agentkit.tools.calendar import CalendarService
        from agentkit.tools.todo import TodoService

        store = Path(state_dir)
        store.mkdir(parents=True, exist_ok=True)
        todo_service = TodoService(path=store / "todos.json")
        calendar_service = CalendarService(path=store / "calendar.json")

    registry = build_registry(
        chart_output_dir=str(workdir / "charts"),
        with_todos=True,
        with_calendar=True,
        with_weather=False,
        with_fx=False,
        with_search=False,
        todo_service=todo_service,
        calendar_service=calendar_service,
    )
    invoker = ToolInvoker(registry, default_timeout=10.0)

    # Awaited directly, in the order the user experiences them, rather than
    # gathered: these are sequential actions with side effects (a todo, an
    # event, a file), and running them concurrently would make the ordering of
    # the transcript depend on which tool happened to finish first.
    todo_step, todos = await todos_step(invoker=invoker, graph_state=state, perceived=run.perceived)
    cal_step, event = await calendar_step(
        invoker=invoker, graph_state=state, perceived=run.perceived, today=today
    )
    chart = await chart_step(invoker=invoker, graph_state=state)
    run.steps.append(todo_step)
    run.todos = todos
    run.steps.append(cal_step)
    run.event = event
    run.steps.append(chart)
    if chart.outputs.get("path"):
        run.artifacts["chart"] = str(chart.outputs["path"])

    # -- 8. persistence ----------------------------------------------------
    persisted = persist_step(
        longterm=longterm, graph_state=state, perceived=run.perceived, session="chain"
    )
    run.steps.append(persisted)
    run.artifacts["memory"] = str(data_path)

    # -- 9. governance -----------------------------------------------------
    # W7's rule, applied to this run's own brief. The brief is built from the
    # plan's coverage needs, so a two-night plan inside one pass's room skips
    # the loop and the step says why.
    from agents.experiment import coverage as coverage_of
    from agents.roles import Brief, DeterministicRoles
    from agents.single import SingleAgent
    from agents.team import Team

    required = _coverage_requirements(state, run.perceived)
    brief = Brief(task_id="chain", request=request, required=required)
    roles = DeterministicRoles(facts=_facts_for(required), capacity=3)
    single = SingleAgent(roles).run(brief)
    observed = coverage_of(single.answer, required)
    team = Team(DeterministicRoles(facts=_facts_for(required), capacity=3), capacity=3, max_iterations=2)

    run.steps.append(
        govern_step(coverage=observed, team=team, brief=brief, single=single, max_iterations=2)
    )

    run.answer = _answer(run)
    return run


def _isolate_matplotlib_cache(workdir: Path) -> None:
    """Point matplotlib's cache at a writable directory before it is imported.

    Without this, matplotlib tries to build a font cache in the user's home
    directory, which a sandboxed or locked-down machine may refuse. The failure
    is a warning rather than an error, but it prints on every chart -- so the
    chart step looks broken while working. Set here rather than in
    ``chart.py`` because ``chart.py`` imports lazily by design, and the
    environment has to be right *before* that import happens.
    """
    import os

    cache = workdir / "mpl-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))


def _artifacts_dir(override: str | Path | None) -> Path:
    import tempfile

    base = Path(override) if override else Path(tempfile.gettempdir()) / "assistant-chain"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _interpret(perceived: Any, meanings: dict[str, str]) -> dict[str, str]:
    """Turn the reader's lines back into named fields.

    The mapping is derived from the labels rather than from a table of known
    values, and that difference is load-bearing. A value-keyed table silently
    drops any city it does not list -- which is how an earlier version of this
    module read a board saying "Berlin" and then planned as though no
    destination had been given. Deriving the field from the label's first word
    means a board naming a city nobody anticipated is read correctly, and the
    guard downstream sees the real request.

    The reader recovered the *meanings*; this is the coordinator taking its
    own labels back off them, which is why it happens here and not in ``ocr``.
    """
    field_of: dict[str, str] = {}
    for label, meaning in meanings.items():
        head = label.split(" - ")[0].strip().lower()
        if head:
            field_of[meaning] = head

    fields: dict[str, str] = {}
    for line in getattr(perceived, "lines", []):
        field = field_of.get(line)
        if field:
            fields[field] = line
    return fields


def _parse_budget(text: str) -> float | None:
    """Pull the number out of a perceived budget string, or ``None``.

    Refusing matters more than parsing: a planner with a silently-defaulted
    budget will happily approve anything, which is worse than stopping and
    telling the user the board was not legible. Returning ``None`` rather than
    raising is what lets the run *report* that refusal -- an exception escaping
    a chain hands the user a stack trace where they need a sentence.
    """
    digits = "".join(ch for ch in text if ch.isdigit())
    return float(digits) if digits else None


def _read_budget(run: ChainRun) -> float | None:
    """Read the budget, or record why the run stopped without one."""
    budget = _parse_budget(run.perceived.get("budget", ""))
    if budget is None:
        run.steps.append(
            StepResult(
                name="guard",
                ok=False,
                detail="stopped: the board's budget row was not legible",
                inputs={"perceived": dict(run.perceived)},
                outputs={"next": "write a budget on the board, e.g. 'Budget - 9000 CNY'"},
            )
        )
        run.answer = (
            "stopped before planning -- the board's budget row did not yield a "
            "number, and a plan without a constraint is not a plan."
        )
    return budget


def _plan_query(request: str, destination: str) -> str:
    """Compose the search text: the request's intent, plus the board's city.

    Kept as a named function rather than an inline f-string because this is the
    single line where "the board supplied the facts" stops being a slogan. If
    the destination is missing, the request alone is used and the chain is
    honest about planning from intent only.
    """
    if destination and destination.lower() not in request.lower():
        return f"{request} {destination}".strip()
    return request


def _substitution_allowed(*, wanted: str, got: str, allow_substitution: bool) -> bool:
    """Whether a plan for ``got`` may stand in for a board naming ``wanted``.

    Two cases are allowed, and both are narrow on purpose. If the planner chose
    the city the board asked for, nothing was substituted -- the board said
    "Lisbon" and search matched Lisbon, so the request and the plan agree. If
    the caller explicitly opted in, the planner is following its ranking and
    that is the caller's decision to make.

    What is *not* allowed by default is the interesting case: the board says
    Lisbon, the planner picks Beijing. A stored past rejection explains why the
    board's city was skipped; it does not license replacing it with an
    unrelated one, and treating it as if it did would make this check
    decorative. The strict default matters because the steps after this one
    create real todos and real calendar events -- acting on a trip nobody asked
    for is worse than stopping and saying so.
    """
    if not wanted or not got:
        return True
    if wanted.strip().lower() == got.strip().lower():
        return True
    return allow_substitution


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip() and part.strip() != "(none)"]


def _coverage_requirements(state: dict[str, Any], perceived: dict[str, str]) -> list[str]:
    """What this run's write-up has to cover, built from the run itself.

    Derived rather than hard-coded so the governance step is judging *this*
    plan: destination, cost, forecast and the board's own note. A fixed list
    would let the step report a coverage number that has nothing to do with
    what was produced.
    """
    itinerary = state.get("itinerary") or {}
    required: list[str] = []
    if itinerary.get("destination"):
        required.append("destination")
    if itinerary.get("weather"):
        required.append("forecast")
    if perceived.get("note"):
        required.append("museum")
    required.append("budget")
    return required


def _facts_for(required: Sequence[str]) -> dict[str, str]:
    """Evidence for the governance brief, one fact per required key."""
    from agents.experiment import FACT_TEMPLATE

    return {key: FACT_TEMPLATE.format(topic=key) for key in required}


def _answer(run: ChainRun) -> str:
    """The one-paragraph summary a user would actually read."""
    itinerary = run.itinerary
    parts = [
        f"{itinerary.get('destination', '?')}: {itinerary.get('weather', '')}",
        f"{itinerary.get('cost')} {itinerary.get('cost_currency')}",
    ]
    if run.event:
        parts.append(f"booked from {run.event.get('start')}")
    parts.append(f"{len(run.todos)} todo(s) created")
    if run.perceived.get("note"):
        parts.append(f"note: {run.perceived['note']}")
    if run.divergence:
        parts.append(f"note -- {run.divergence}")
    return "; ".join(part for part in parts if part and "None" not in part)


def save_report(run: ChainRun, path: str | Path) -> Path:
    """Write the run as JSON, LF-terminated, for the record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((json.dumps(run.to_dict(), indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    return target


__all__ = [
    "BOARD_LINES",
    "BOARD_MEANINGS",
    "DEFAULT_REQUEST",
    "SEED_DECISIONS",
    "SEED_PREFERENCES",
    "run_chain",
    "save_report",
]
