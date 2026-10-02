"""A transport that fails on purpose, so the retry path can be measured.

The retry and backoff logic only runs when a request fails, which means a
green test suite can sit on top of code that has never executed. That is
exactly the state this module exists to fix: wrapping a real transport with a
known failure rate turns "the retry code looks right" into "the retry code
ran 12 times and every run recovered".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMError


@dataclass
class FaultInjectingTransport:
    """Fails the first `fail_first` calls, then delegates to `inner`.

    `per_call` makes the failure rate proportional instead of front-loaded,
    which is closer to a real rate limit: the run keeps recovering all the way
    through rather than getting its failures over with at the start.
    """

    inner: Any
    fail_first: int = 1
    per_call: int = 0
    retryable: bool = True
    reason: str = "injected transient failure"
    calls: int = 0
    injected: int = 0
    recovered: int = 0
    _seen: int = field(default=0, init=False)

    def __call__(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self._seen += 1

        should_fail = self._seen <= self.fail_first or (
            self.per_call > 0 and self._seen % self.per_call == 0
        )
        if should_fail:
            self.injected += 1
            raise LLMError(f"{self.reason} (#{self.injected})", retryable=self.retryable)

        body = self.inner(url, headers, payload)
        if self.injected:
            self.recovered += 1
        return body