"""Currency conversion over a static rate table.

Rates are fixed rather than fetched for one reason: a moving rate makes an
evaluation unreproducible. The lookup path is still the same shape a live
provider would need, so swapping in a fetch is a change to ``rates``, not to
the tool's contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import Field

from ..errors import ErrorKind, ToolCallError
from ..registry import ToolRegistry
from ..schema import ToolArgs

# Everything is quoted per 1 unit of the key currency, USD as the pivot.
_RATES: dict[str, float] = {
    "USD": 1.0,
    "CNY": 7.21,
    "EUR": 0.92,
    "JPY": 151.4,
    "GBP": 0.79,
    "HKD": 7.82,
    "SGD": 1.34,
    "AUD": 1.52,
}


class FxArgs(ToolArgs):
    amount: float = Field(..., gt=0, description="Amount to convert; must be positive.")
    source: str = Field(..., min_length=3, max_length=3, description="From currency, ISO 4217, e.g. 'USD'.")
    target: str = Field(..., min_length=3, max_length=3, description="To currency, ISO 4217, e.g. 'CNY'.")


@dataclass
class FxService:
    rates: dict[str, float] = field(default_factory=lambda: dict(_RATES))

    def convert(self, amount: float, source: str, target: str) -> dict[str, Any]:
        src = source.strip().upper()
        dst = target.strip().upper()
        unknown = [code for code in (src, dst) if code not in self.rates]
        if unknown:
            # Valid arguments, missing data: NOT_FOUND, and the known list is
            # in ``details`` so the model can retry with a supported code.
            raise ToolCallError(
                f"unsupported currency: {', '.join(unknown)}",
                kind=ErrorKind.NOT_FOUND,
                details={"unsupported": unknown, "supported": sorted(self.rates)},
            )
        rate = self.rates[dst] / self.rates[src]
        return {
            "amount": amount,
            "source": src,
            "target": dst,
            "rate": round(rate, 6),
            "converted": round(amount * rate, 2),
        }


def register(registry: ToolRegistry, service: FxService | None = None) -> FxService:
    """Attach the ``convert_currency`` tool."""
    svc = service or FxService()

    @registry.tool(
        "convert_currency",
        "Convert an amount between two ISO-4217 currencies. Use when a price "
        "or budget is quoted in a currency the user does not think in.",
        args_model=FxArgs,
        timeout=2.0,
    )
    def convert_currency(amount: float, source: str, target: str) -> dict[str, Any]:
        return svc.convert(amount, source, target)

    return svc


__all__ = ["FxArgs", "FxService", "register"]
