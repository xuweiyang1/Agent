"""Tool schemas, generated from Pydantic v2 models.

W1 hand-wrote its JSON schemas, and the docstring there argued that was
deliberate. W2 reverses that on purpose, and the reason is worth stating:
once tools take structured arguments, a hand-written schema and the function
that consumes it drift apart. The schema says ``string``, the function
annotates ``int``, and nothing catches it until a model sends the wrong type
at runtime.

Here the Pydantic model *is* the contract. ``model_json_schema`` renders it
for the model, and ``model_validate`` enforces the same rules on the way in,
so the published contract and the enforced one cannot disagree. The model
field descriptions are the docstrings the model actually reads, which is why
they are written for a reader trying to call the tool correctly.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, ValidationError

from .errors import ErrorKind, ToolCallError

# Pydantic v2 reports a machine ``type`` (``string_type``) and a prose ``msg``,
# but not the JSON type the schema asked for. Mapping the common ones back to
# JSON Schema vocabulary is what lets the error say "expected string" in the
# same words the model saw in the tool schema, instead of a synonym.
_EXPECTED_BY_ERROR: dict[str, str] = {
    "string_type": "string",
    "int_type": "integer",
    "int_parsing": "integer",
    "float_type": "number",
    "float_parsing": "number",
    "bool_type": "boolean",
    "bool_parsing": "boolean",
    "list_type": "array",
    "dict_type": "object",
    "model_type": "object",
    "missing": "a value (this field is required)",
    "extra_forbidden": "no extra keys (this argument does not exist)",
    "too_short": "at least the minimum length",
    "too_long": "at most the maximum length",
    "greater_than": "a value above the minimum",
    "less_than": "a value below the maximum",
    "string_pattern_mismatch": "a value matching the allowed pattern",
}


class ToolArgs(BaseModel):
    """Base class for every tool's argument model.

    ``extra="forbid"`` is the important default. A model that invents an
    argument name should be told so, not have the key silently dropped --
    silently ignoring it produces a plausible-looking wrong answer, which is
    the most expensive failure mode to debug.
    """

    model_config = ConfigDict(extra="forbid")


def _first_error(exc: ValidationError) -> dict[str, Any]:
    """Reduce a Pydantic error to the field, the problem, and the fix.

    Pydantic reports every problem at once, which is thorough but noisy in a
    transcript the model has to reread each turn. The first error is enough
    to unblock the next attempt, and ``expected`` is what makes the message
    actionable rather than merely negative.
    """
    errors = exc.errors()
    if not errors:
        return {"argument": "(root)", "problem": "invalid arguments"}
    first = errors[0]
    location = ".".join(str(part) for part in first.get("loc", ())) or "(root)"
    detail: dict[str, Any] = {
        "argument": location,
        "problem": first.get("msg", "invalid"),
    }
    expected = _EXPECTED_BY_ERROR.get(str(first.get("type", "")))
    if expected:
        detail["expected"] = expected
    raw_input = first.get("input")
    if isinstance(raw_input, (str, int, float, bool)):
        detail["got"] = raw_input
    if len(errors) > 1:
        detail["other_errors"] = len(errors) - 1
    return detail


def validate_args(model: type[ToolArgs], arguments: dict[str, Any], *, tool: str) -> ToolArgs:
    """Validate raw arguments, raising a classified ``BAD_ARGUMENTS`` failure.

    This is the single choke point between "what the model sent" and "what
    the tool receives", so it is also the natural place to guarantee the tool
    function never sees a malformed argument.
    """
    try:
        return model.model_validate(arguments)
    except ValidationError as exc:
        detail = _first_error(exc)
        raise ToolCallError(
            f"{tool}: argument {detail['argument']!r} is invalid "
            f"({detail['problem']})",
            kind=ErrorKind.BAD_ARGUMENTS,
            details=detail,
        ) from exc


def schema_for(model: type[ToolArgs]) -> dict[str, Any]:
    """Render one argument model as a tool ``parameters`` object.

    ``additionalProperties`` is set explicitly because the JSON Schema
    default is permissive, which would contradict ``extra="forbid"`` in the
    model. The published schema and the enforced model must say the same
    thing, or the model is being graded against rules it was never shown.
    """
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema.setdefault("type", "object")
    schema["additionalProperties"] = False
    return schema


def schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Fallback for a plain function with no argument model.

    Not every tool needs typed arguments -- a zero-argument tool is a real
    case. Rather than inventing a model for it, this returns the empty object
    schema, which is also what a model correctly reads as "call me with no
    arguments".
    """
    _ = fn  # signature inspection is enough for now; kept for API symmetry
    return {"type": "object", "properties": {}, "additionalProperties": False, "required": []}


__all__ = ["ToolArgs", "schema_for", "schema_from_signature", "validate_args"]
