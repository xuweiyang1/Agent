"""Reading a real photo into the chain's fields, when a vision model is wired.

``RuleOcr`` is the offline reader and it works because it rendered the board
itself. A photo of a real invoice has no such luck, so the local deployment
needs a reader that actually looks. This module is that reader, kept to one
job: image in, the chain's fields (destination / budget / note) out.

The split matters. ``OcrEngine`` is synchronous and template-based; a vision
call is asynchronous and paid. Rather than bend one protocol to cover both,
the web layer asks this module for fields and hands them to ``run_chain`` as
an override. The chain never learns whether the fields came from pixels it
drew or pixels a model looked at -- which is the point of a seam.

The model is asked for JSON and the answer is parsed defensively: a model that
wraps its JSON in prose is a normal occurrence, not a bug in this code, so the
first JSON object in the reply is extracted rather than the whole reply being
rejected.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from agentkit.messages import ChatMessage, TextBlock, image_from_file

# The fields the chain understands. A model that invents a new key is ignored
# rather than allowed to smuggle an unhandled field into the run.
FIELD_KEYS: tuple[str, ...] = ("destination", "budget", "note")

_PROMPT = (
    "You read the photo and return the trip facts it contains. Reply with a "
    "single JSON object and nothing else. Keys: destination (a city name), "
    "budget (the amount and currency exactly as written), note (any extra "
    "instruction, or an empty string). Omit a key if the photo does not state it."
)


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply, defensively."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean(fields: Mapping[str, Any]) -> dict[str, str]:
    """Keep only known fields, and only non-empty ones."""
    out: dict[str, str] = {}
    for key in FIELD_KEYS:
        value = fields.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text.lower() not in {"null", "none", "n/a"}:
            out[key] = text
    return out


async def read_fields(image_path: str | Path, *, model: Any) -> dict[str, str]:
    """Ask a vision model for the chain's fields, from a real image.

    ``model`` is any ``AsyncModel`` (``AsyncOpenAICompatibleModel`` in the
    local deployment). Returning ``{}`` on an unreadable photo is deliberate:
    the chain already knows how to report a blank read, and raising here would
    turn a bad photo into a 500 instead of a sentence.
    """
    message = ChatMessage(
        role="user",
        content=[
            TextBlock(_PROMPT),
            image_from_file(image_path, caption="the photo to read"),
        ],
    )
    reply = await model.acomplete([message], [])
    return _clean(_extract_json(reply.text() if hasattr(reply, "text") else str(reply.content)))


__all__ = ["FIELD_KEYS", "read_fields"]