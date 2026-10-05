"""Weather, with a real async shape and an offline default.

The interesting part is not the forecast, it is the seam. ``WeatherService``
takes an optional ``fetcher``; when one is supplied it is a real network call
whose failures become ``UPSTREAM``, and when it is absent the tool returns a
deterministic stub keyed off the city name. That is what lets the whole agent
run offline in tests and still exercise the async timeout path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import Field

from ..errors import ErrorKind, ToolCallError
from ..registry import ToolRegistry
from ..schema import ToolArgs

Fetcher = Callable[[str, int], Awaitable[dict[str, Any]]]

# Deterministic stub data. Keyed by lowercased city so the same city always
# returns the same forecast, which is what makes an offline eval reproducible.
_BUILTIN: dict[str, dict[str, Any]] = {
    "beijing": {"condition": "clear", "high": 24, "low": 12},
    "shanghai": {"condition": "cloudy", "high": 26, "low": 18},
    "tokyo": {"condition": "rain", "high": 21, "low": 16},
    "paris": {"condition": "overcast", "high": 17, "low": 9},
    "lisbon": {"condition": "sunny", "high": 27, "low": 15},
    "sydney": {"condition": "windy", "high": 22, "low": 14},
}


class WeatherArgs(ToolArgs):
    city: str = Field(..., min_length=1, description="City name, e.g. 'Shanghai'.")
    days: int = Field(
        1,
        ge=1,
        le=7,
        description="Forecast horizon in days, 1 to 7.",
    )


@dataclass
class WeatherService:
    """Returns a forecast, from the network when wired up, a stub otherwise."""

    fetcher: Fetcher | None = None
    _calls: int = field(default=0, init=False)

    async def forecast(self, city: str, days: int) -> dict[str, Any]:
        self._calls += 1
        if self.fetcher is not None:
            try:
                return await self.fetcher(city, days)
            except Exception as exc:  # noqa: BLE001 - reclassified below
                raise ToolCallError(
                    f"weather provider failed: {exc}",
                    kind=ErrorKind.UPSTREAM,
                    details={"city": city},
                ) from exc

        base = _BUILTIN.get(city.strip().lower())
        if base is None:
            # A city we have no stub for is a data gap, not a model mistake:
            # the arguments were well formed, the referenced thing is missing.
            raise ToolCallError(
                f"no forecast available for {city!r}",
                kind=ErrorKind.NOT_FOUND,
                details={"known_cities": sorted(_BUILTIN)},
            )
        days = max(1, min(int(days), 7))
        forecast = [
            {
                "day": offset + 1,
                "condition": base["condition"],
                "high": base["high"] - offset,
                "low": base["low"] - offset,
            }
            for offset in range(days)
        ]
        return {"city": city, "days": days, "forecast": forecast}


def register(registry: ToolRegistry, service: WeatherService | None = None) -> WeatherService:
    """Attach the ``weather`` tool and return the service behind it."""
    svc = service or WeatherService()

    @registry.tool(
        "weather",
        "Get a short weather forecast for a city. Use it before recommending "
        "outdoor plans or packing.",
        args_model=WeatherArgs,
        timeout=5.0,
    )
    async def weather(city: str, days: int = 1) -> dict[str, Any]:
        return await svc.forecast(city, days)

    return svc


__all__ = ["WeatherArgs", "WeatherService", "register"]
