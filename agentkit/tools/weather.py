"""Weather, with a real async shape and an offline default.

The interesting part is not the forecast, it is the seam. ``WeatherService``
takes an optional ``fetcher``; when one is supplied it is a real network call
whose failures become ``UPSTREAM``, and when it is absent the tool returns a
deterministic stub keyed off the city name. That is what lets the whole agent
run offline in tests and still exercise the async timeout path.

``open_meteo_fetcher`` is the real one, and it is deliberately keyless. The
free Open-Meteo API needs no account, so "check the weather" works on a fresh
clone instead of degrading to an apology -- which is exactly what the first
version did: it matched six hardcoded English names and answered every other
city with a refusal, a data gap wearing the costume of a capability. It also
geocodes the name first, so 南充 and Chengdu both resolve without the model
having to translate.

``network_with_stub_fallback`` composes the two, because a restricted network
is a real condition here: the live call is tried, and a known city still
answers from the table when it cannot be reached. Falling back is stated in
the payload's ``source`` field rather than hidden, so neither the model nor a
reader is misled about where a number came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import Field

from ..errors import ErrorKind, ToolCallError
from ..registry import ToolRegistry
from ..schema import ToolArgs

Fetcher = Callable[[str, int], Awaitable[dict[str, Any]]]

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

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

# WMO weather interpretation codes, which is what Open-Meteo reports. Mapped
# to words because "weather_code: 61" is not an answer a person can use.
_WMO: dict[int, str] = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴",
    45: "有雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "阵雨", 81: "强阵雨", 82: "暴雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷暴伴冰雹",
}


class WeatherArgs(ToolArgs):
    city: str = Field(
        ...,
        min_length=1,
        description="City name, exactly as the user wrote it (中文或 English). "
        "It is geocoded, so no translation is needed.",
    )
    days: int = Field(
        1,
        ge=1,
        le=7,
        description="Forecast horizon in days, 1 to 7.",
    )


def _stub(city: str, days: int, base: dict[str, Any]) -> dict[str, Any]:
    """The offline answer: a plausible, deterministic forecast."""
    horizon = max(1, min(int(days), 7))
    return {
        "city": city,
        "days": horizon,
        "source": "stub",
        "forecast": [
            {
                "day": offset + 1,
                "condition": base["condition"],
                "high": base["high"] - offset,
                "low": base["low"] - offset,
            }
            for offset in range(horizon)
        ],
    }


async def open_meteo_fetcher(city: str, days: int) -> dict[str, Any]:
    """A real forecast from Open-Meteo, which needs no API key.

    Two calls, in order: geocode the name to coordinates, then ask for the
    daily forecast there. Geocoding is what makes a Chinese city name work;
    skipping it would reintroduce the "only six cities" problem in a different
    place. ``httpx`` is imported lazily so importing this module never
    requires the network stack, the same rule ``chart.py`` follows for
    matplotlib.
    """
    import httpx

    horizon = max(1, min(int(days), 7))
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            geo = await client.get(
                GEOCODE_URL,
                params={"name": city, "count": 1, "language": "zh", "format": "json"},
            )
            geo.raise_for_status()
            results = geo.json().get("results") or []
            if not results:
                raise ToolCallError(
                    f"could not find a place named {city!r}",
                    kind=ErrorKind.NOT_FOUND,
                    details={"city": city},
                )
            place = results[0]
            forecast = await client.get(
                FORECAST_URL,
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                    "timezone": "auto",
                    "forecast_days": horizon,
                },
            )
            forecast.raise_for_status()
            daily = forecast.json().get("daily") or {}
    except ToolCallError:
        raise
    except Exception as exc:  # noqa: BLE001 - reclassified below
        raise ToolCallError(
            f"weather provider unreachable: {exc}",
            kind=ErrorKind.UPSTREAM,
            details={"city": city},
        ) from exc

    codes = daily.get("weather_code") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    return {
        "city": place.get("name") or city,
        "country": place.get("country", ""),
        "days": len(codes) or horizon,
        "source": "open-meteo",
        "forecast": [
            {
                "day": index + 1,
                "condition": _WMO.get(int(code), f"code {code}"),
                "high": highs[index] if index < len(highs) else None,
                "low": lows[index] if index < len(lows) else None,
            }
            for index, code in enumerate(codes)
        ],
    }


def network_with_stub_fallback() -> Fetcher:
    """Live weather when reachable, the builtin table when it is not.

    Only network-shaped failures fall back. A city that genuinely does not
    exist stays a ``NOT_FOUND``, because answering it from a table would be
    inventing data rather than degrading gracefully.
    """

    async def fetch(city: str, days: int) -> dict[str, Any]:
        try:
            return await open_meteo_fetcher(city, days)
        except ToolCallError as exc:
            base = _BUILTIN.get(city.strip().lower())
            if base is not None and exc.kind in (ErrorKind.UPSTREAM, ErrorKind.TIMEOUT):
                return _stub(city, days, base)
            raise

    return fetch


@dataclass
class WeatherService:
    """Returns a forecast, from the network when wired up, a stub otherwise."""

    fetcher: Fetcher | None = None
    _calls: int = field(default=0, init=False)

    async def forecast(self, city: str, days: int) -> dict[str, Any]:
        self._calls += 1
        if self.fetcher is not None:
            return await self.fetcher(city, days)

        base = _BUILTIN.get(city.strip().lower())
        if base is None:
            # A city we have no stub for is a data gap, not a model mistake:
            # the arguments were well formed, the referenced thing is missing.
            raise ToolCallError(
                f"no forecast available for {city!r}",
                kind=ErrorKind.NOT_FOUND,
                details={"known_cities": sorted(_BUILTIN)},
            )
        return _stub(city, days, base)


def register(registry: ToolRegistry, service: WeatherService | None = None) -> WeatherService:
    """Attach the ``weather`` tool and return the service behind it."""
    svc = service or WeatherService()

    @registry.tool(
        "weather",
        "Get a real weather forecast for a city, in any language. Use it "
        "before recommending outdoor plans or packing.",
        args_model=WeatherArgs,
        timeout=10.0,
    )
    async def weather(city: str, days: int = 1) -> dict[str, Any]:
        return await svc.forecast(city, days)

    return svc


__all__ = [
    "Fetcher",
    "WeatherArgs",
    "WeatherService",
    "network_with_stub_fallback",
    "open_meteo_fetcher",
    "register",
]