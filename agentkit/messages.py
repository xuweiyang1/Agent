"""Messages that can carry an image, plus the encoding to the wire format.

This file exists because of one uncomfortable fact: a multimodal message is
not a string, and W1's ``Message.content: str`` cannot represent one. Adding
images therefore is not "one more tool" -- it changes the type that flows
through the whole loop, which is why it belongs in W2 rather than later.

The second decision in here is the expensive one. An image block costs orders
of magnitude more tokens than the text describing it, and in a multi-turn
conversation the raw image would be resent every single turn. So an image is
allowed in, and once it has been seen it is *collapsed* to its caption. That
single rule removes most of the token cost of being multimodal.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, Union

from agentloop.llm import ToolCall

# A conservative per-image estimate when the pixel size is unknown. Real
# accounting depends on the provider's tiling rule; what matters here is that
# the number is large and visible, so compaction triggers instead of silently
# blowing the budget.
IMAGE_TOKEN_ESTIMATE = 1000


@dataclass(frozen=True)
class TextBlock:
    text: str

    @property
    def type(self) -> str:
        return "text"

    def to_wire(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text}


@dataclass(frozen=True)
class ImageBlock:
    """An image reference: a URL, or a base64 data URL built from a file."""

    url: str
    detail: str = "auto"
    caption: str = ""

    @property
    def type(self) -> str:
        return "image_url"

    def to_wire(self) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self.url, "detail": self.detail}}


ContentBlock = Union[TextBlock, ImageBlock]


def image_from_url(url: str, *, detail: str = "auto", caption: str = "") -> ImageBlock:
    return ImageBlock(url=url, detail=detail, caption=caption)


def image_from_file(path: str | Path, *, detail: str = "auto", caption: str = "") -> ImageBlock:
    """Inline a local image as a data URL.

    Data URLs keep the demo self-contained -- no file server, no expiring
    link -- at the cost of base64 inflation. For a hosted deployment the
    right move is to upload once and pass the URL, which also lets the
    provider cache the fetch.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such image: {p}")
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    payload = base64.b64encode(p.read_bytes()).decode("ascii")
    return ImageBlock(url=f"data:{mime};base64,{payload}", detail=detail, caption=caption)


def text_content(text: str) -> list[ContentBlock]:
    return [TextBlock(text)]


@dataclass
class ChatMessage:
    """One turn that may hold text, images, or both.

    ``content`` is a string for the simple case and a block list otherwise,
    so the common path stays readable and only the multimodal path pays for
    the extra structure.
    """

    role: str
    content: str | list[ContentBlock] = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None

    def blocks(self) -> list[ContentBlock]:
        if isinstance(self.content, str):
            return [TextBlock(self.content)] if self.content else []
        return list(self.content)

    def text(self) -> str:
        """The concatenated text, ignoring images. For logs and grading."""
        return " ".join(b.text for b in self.blocks() if isinstance(b, TextBlock)).strip()

    def has_image(self) -> bool:
        return any(isinstance(b, ImageBlock) for b in self.blocks())

    def collapse_images(self) -> "ChatMessage":
        """Replace images with their captions, once they have been seen.

        Called after the model has answered a turn that included an image.
        Without this, every later turn resends the image and the run gets
        expensive fast; with it, the image contributes roughly caption-sized
        text instead of a thousand tokens.
        """
        if not self.has_image():
            return self
        parts: list[ContentBlock] = []
        for block in self.blocks():
            if isinstance(block, ImageBlock):
                label = block.caption or "[image omitted]"
                parts.append(TextBlock(f"[image: {label}]"))
            else:
                parts.append(block)
        return ChatMessage(
            role=self.role,
            content=parts,
            tool_calls=self.tool_calls,
            tool_call_id=self.tool_call_id,
        )


def estimate_message_tokens(message: ChatMessage) -> int:
    """Token estimate that accounts for images instead of ignoring them.

    W1's estimate counted characters, which reports an image-bearing turn as
    nearly free. That is the bug that makes multimodal runs surprise you on
    the bill, so images get a flat, deliberately pessimistic weight here.
    """
    total = 0
    for block in message.blocks():
        if isinstance(block, TextBlock):
            total += max(1, (len(block.text) + 3) // 4)
        else:
            total += IMAGE_TOKEN_ESTIMATE
    for call in message.tool_calls:
        total += max(1, (len(call.name) + 3) // 4)
        total += max(1, (len(str(call.arguments)) + 3) // 4)
    return total


def to_wire(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    """Encode messages for an OpenAI-compatible endpoint."""
    wire: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "tool":
            wire.append(
                {
                    "role": "tool",
                    "content": message.text(),
                    "tool_call_id": message.tool_call_id or "",
                }
            )
            continue
        if message.tool_calls:
            wire.append(
                {
                    "role": message.role,
                    "content": _wire_content(message),
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": _json(call.arguments),
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )
            continue
        wire.append({"role": message.role, "content": _wire_content(message)})
    return wire


def _wire_content(message: ChatMessage) -> Any:
    """Collapse to a plain string when there are no images.

    Several providers accept a block list for text-only messages but bill or
    cache it differently from a string, so sending the simple form when it is
    genuinely simple is the safer default.
    """
    if isinstance(message.content, str):
        return message.content
    blocks = message.blocks()
    if not blocks:
        return ""
    if all(isinstance(b, TextBlock) for b in blocks):
        return " ".join(b.text for b in blocks)
    return [block.to_wire() for block in blocks]


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False)


__all__ = [
    "ChatMessage",
    "ContentBlock",
    "IMAGE_TOKEN_ESTIMATE",
    "ImageBlock",
    "TextBlock",
    "estimate_message_tokens",
    "image_from_file",
    "image_from_url",
    "text_content",
    "to_wire",
]
