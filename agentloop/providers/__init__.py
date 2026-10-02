"""Vendor adapters.

The core loop depends only on the `Model` protocol in `agentloop.llm`, so a
provider is just an adapter from that protocol to one HTTP API. Nothing in
`runtime.py` needs to change when a provider is added.
"""

from .openai_compat import (
    OpenAICompatibleModel,
    RecordTransport,
    ReplayTransport,
    http_transport,
)

__all__ = [
    "OpenAICompatibleModel",
    "RecordTransport",
    "ReplayTransport",
    "http_transport",
]