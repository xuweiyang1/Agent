"""Error taxonomy for tool dispatch.

The W1 runtime turned a tool failure into the string ``"error: ..."`` and
moved on. That was enough to keep a run alive, but it threw away the one
thing a model needs in order to correct itself: *what kind* of mistake it
made. A wrong tool name and a malformed argument require different fixes,
and a stringly-typed message cannot tell them apart.

So every failure is classified into an ``ErrorKind``. The kind is data, not
prose: it is carried in the tool result, serialized next to the message, and
is what the retry/escalation policy keys on. The message is written for the
model to read; the kind is written for the code to branch on.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum

# Python 3.10 has two distinct TimeoutError classes: the builtin one and
# ``asyncio.TimeoutError``. They are not the same type and neither subclasses
# the other, so a bare ``except TimeoutError`` misses every await timeout.
# Checking both is the difference between "timeouts are handled" and
# "timeouts are handled on the path we happened to test".
_TIMEOUT_TYPES: tuple[type[BaseException], ...] = (TimeoutError, asyncio.TimeoutError)


class ErrorKind(str, Enum):
    """One failure class per way a tool call can go wrong.

    ``UNKNOWN_TOOL``   the model invented a name that is not registered.
    ``BAD_ARGUMENTS``  the name is real but the arguments do not fit the schema.
    ``NOT_FOUND``      arguments are valid, but the referenced thing does not exist.
    ``TIMEOUT``        the tool did not finish in time; the run must not die.
    ``UPSTREAM``       the tool ran and the dependency behind it failed.
    ``INTERNAL``       a bug on our side. Never blame the model for this one.
    """

    UNKNOWN_TOOL = "unknown_tool"
    BAD_ARGUMENTS = "bad_arguments"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    UPSTREAM = "upstream"
    INTERNAL = "internal"

    @property
    def is_model_fault(self) -> bool:
        """True when the model can plausibly fix this on the next turn.

        Only these kinds are worth feeding back as a correction. A timeout
        is not the model's mistake, and an internal error is ours; both are
        reported, but neither is something the model should try to rewrite.
        """
        return self in (ErrorKind.UNKNOWN_TOOL, ErrorKind.BAD_ARGUMENTS, ErrorKind.NOT_FOUND)


@dataclass(frozen=True)
class ToolFailure:
    """A structured, serializable tool failure.

    ``to_payload`` produces exactly what goes into the transcript, so the
    model sees a stable shape it can learn from across runs, and a test can
    assert on the dict instead of pattern-matching a message string.
    """

    kind: ErrorKind
    message: str
    tool: str = ""
    details: dict = field(default_factory=dict)

    @property
    def retryable(self) -> bool:
        """Whether the *dispatch layer* may retry without the model's help."""
        return self.kind in (ErrorKind.TIMEOUT, ErrorKind.UPSTREAM)

    def to_payload(self) -> dict:
        """The dict form embedded in a tool result.

        ``details`` is intentionally preserved verbatim: for a bad argument
        it holds the offending key and the expected type, which is what turns
        "something was wrong" into "fix ``city``, it must be a string".
        """
        payload = {
            "error": self.kind.value,
            "message": self.message,
        }
        if self.tool:
            payload["tool"] = self.tool
        if self.details:
            payload["details"] = self.details
        return payload

    def to_text(self) -> str:
        """Compact one-line form for logs and transcripts without a JSON body."""
        return f"{self.kind.value}: {self.message}"


class ToolCallError(Exception):
    """Raised by a tool to signal a classified failure.

    Tools raise this instead of returning an error string, so the classifier
    has a kind to work with. The dispatcher catches it and converts it into a
    ``ToolFailure``; anything else that escapes is classified as ``INTERNAL``,
    because an unclassified exception is a bug on our side.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: ErrorKind = ErrorKind.INTERNAL,
        details: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.details = dict(details or {})

    def as_failure(self, tool: str = "") -> ToolFailure:
        return ToolFailure(self.kind, str(self), tool=tool, details=self.details)


def classify(exc: BaseException, *, tool: str = "") -> ToolFailure:
    """Map an arbitrary exception onto the taxonomy.

    Dispatch never lets a raw exception reach the transcript. The mapping is
    the contract: a timeout becomes ``TIMEOUT`` (retryable, not the model's
    fault), a ``ToolCallError`` keeps its own kind, and everything else is
    ``INTERNAL`` so it is visibly our bug rather than silently reported as
    the model's.
    """
    if isinstance(exc, ToolCallError):
        return exc.as_failure(tool)
    if isinstance(exc, _TIMEOUT_TYPES):
        return ToolFailure(
            ErrorKind.TIMEOUT,
            f"tool {tool!r} exceeded its time budget" if tool else "tool timed out",
            tool=tool,
        )
    return ToolFailure(
        ErrorKind.INTERNAL,
        f"{type(exc).__name__}: {exc}",
        tool=tool,
    )


__all__ = ["ErrorKind", "ToolFailure", "ToolCallError", "classify"]
