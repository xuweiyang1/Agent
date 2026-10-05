"""A tool-calling service: async tools, a real schema, and a FastAPI layer.

``agentloop`` is the minimal loop, built to be read. ``agentkit`` is the same
idea under the conditions a job actually imposes: tools declared with Pydantic
so validation is generated rather than hand-written, the five tool families a
product needs, one error taxonomy that survives a host timeout, and an HTTP
surface a client can call.
"""

from .chat import AsyncModel, HeuristicModel, ToolCallingAgent, TurnResult
from .dispatch import DispatchResult, ToolInvoker
from .errors import ErrorKind, ToolCallError, ToolFailure, classify
from .messages import (
    ChatMessage,
    ImageBlock,
    TextBlock,
    estimate_message_tokens,
    image_from_file,
    image_from_url,
    to_wire,
)
from .registry import Tool, ToolRegistry
from .schema import ToolArgs, schema_for, validate_args
from .service import RunRequest, RunResponse, create_app
from .tools import build_registry

__all__ = [
    "AsyncModel",
    "ChatMessage",
    "DispatchResult",
    "ErrorKind",
    "HeuristicModel",
    "ImageBlock",
    "RunRequest",
    "RunResponse",
    "TextBlock",
    "Tool",
    "ToolArgs",
    "ToolCallError",
    "ToolCallingAgent",
    "ToolFailure",
    "ToolInvoker",
    "ToolRegistry",
    "TurnResult",
    "build_registry",
    "classify",
    "create_app",
    "estimate_message_tokens",
    "image_from_file",
    "image_from_url",
    "schema_for",
    "to_wire",
    "validate_args",
]
