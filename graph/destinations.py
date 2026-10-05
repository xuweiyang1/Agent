"""The destinations the planner chooses between, and the note search indexes.

One source of truth on purpose. The search tool indexes ``note``; the planner
prices and schedules the structured fields. Keeping two copies of a
destination -- one prose, one structured -- is how the two drift, and the
drift shows up as a plan that recommends a city the search cannot find.

Costs are per night in the destination's *own* currency, so the price step has
to call the fx tool for a real reason instead of a decorative one. The five
cities are exactly the ones W2's weather stub knows, which is what keeps the
whole planner runnable offline: a destination with no forecast would turn
"the weather step" into a broken step.

The prose is written to be retrievable, not to be marketing. It repeats the
terms a request would plausibly use -- weekend, sunny, rain, indoor -- because
the retriever is lexical and a synonym it has never seen is a miss.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Destination:
    """One candidate, in both the shapes the planner needs."""

    name: str
    country: str
    currency: str
    nightly_cost: float
    indoor: tuple[str, ...]
    outdoor: tuple[str, ...]
    note: str


DESTINATIONS: tuple[Destination, ...] = (
    Destination(
        name="Tokyo",
        country="Japan",
        currency="JPY",
        nightly_cost=22000.0,
        indoor=("teamLab Planets", "the Edo-Tokyo Museum", "a Shinjuku ramen crawl"),
        outdoor=("Meiji Shrine at dawn", "the Yanaka backstreets", "Ueno Park"),
        note=(
            "A weekend trip to Tokyo is dense and easy to fill: the city works "
            "in the rain as well as in the sun. Late spring is mild but often "
            "wet, so a plan should carry a museum afternoon next to the shrine "
            "morning. Indoor choices are strong, which makes it the safe "
            "weekend pick when the forecast is unsure."
        ),
    ),
    Destination(
        name="Lisbon",
        country="Portugal",
        currency="EUR",
        nightly_cost=120.0,
        indoor=("the Gulbenkian Museum", "the Time Out Market", "a fado house"),
        outdoor=("the Alfama miradouros", "a tram ride to Belem", "the Tagus waterfront"),
        note=(
            "Lisbon is the reliable sunny weekend: dry, bright and walkable, "
            "with enough hills to make the trams worth taking. It is the "
            "natural answer when the request asks for sun or for outdoor time, "
            "and it stays pleasant when other cities are wet."
        ),
    ),
    Destination(
        name="Paris",
        country="France",
        currency="EUR",
        nightly_cost=180.0,
        indoor=("the Orsay", "the Louvre's quiet wing", "a Saint-Germain cafe"),
        outdoor=("the Canal Saint-Martin", "Luxembourg Gardens", "a Seine walk"),
        note=(
            "Paris is the expensive but reliable weekend: overcast more often "
            "than not, which is a fine excuse for museums and long lunches. "
            "Outdoor plans are possible but should have an indoor second half."
        ),
    ),
    Destination(
        name="Sydney",
        country="Australia",
        currency="AUD",
        nightly_cost=250.0,
        indoor=("the Art Gallery of NSW", "the Powerhouse Museum", "the Queen Victoria Building"),
        outdoor=("the Bondi to Coogee walk", "the Manly ferry", "the Botanic Garden"),
        note=(
            "A Sydney weekend is outdoor-first: the coastal walk and the ferry "
            "are the reason to go. Strong wind changes which of those is "
            "pleasant, so a plan should keep a museum as the wet-weather half. "
            "Long flights make it a poor fit for a short weekend."
        ),
    ),
    Destination(
        name="Beijing",
        country="China",
        currency="CNY",
        nightly_cost=900.0,
        indoor=("the National Museum", "the Capital Museum", "a hutong teahouse"),
        outdoor=("the Temple of Heaven", "a Jingshan climb", "the Shichahai lakes"),
        note=(
            "Beijing is the cheapest weekend on this list and the easiest to "
            "reach from home, with clear spring weather and wide outdoor "
            "spaces. The tradeoff is air quality, so a plan should pair each "
            "outdoor morning with an indoor afternoon."
        ),
    ),
)

# Lowercased name -> Destination. Lowercased because the search tool's IDs
# come back as free strings, and a case difference must not read as "unknown
# destination".
CATALOG: dict[str, Destination] = {d.name.lower(): d for d in DESTINATIONS}

# Conditions that rule out an outdoor activity. Wind is deliberately absent:
# it changes which activity is pleasant, not whether one is possible.
OUTDOOR_BLOCKERS = frozenset({"rain", "storm", "snow", "thunderstorm"})


def as_corpus() -> dict[str, str]:
    """The mapping the search tool indexes: display name -> note."""
    return {d.name: d.note for d in DESTINATIONS}


__all__ = ["CATALOG", "DESTINATIONS", "OUTDOOR_BLOCKERS", "Destination", "as_corpus"]
