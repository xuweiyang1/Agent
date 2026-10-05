"""A filesystem MCP server and its cloud-function entry point.

Split so the security boundary is testable on its own: ``fs`` is pure logic
with no protocol dependency, ``adapter`` absorbs the MCP 2.x rename,
``server`` binds the sandbox to MCP tools, and ``cloud`` turns the whole
thing into one ASGI app.
"""

from .adapter import FastMCP
from .cloud import create_cloud_app
from .fs import DEFAULT_MAX_RESULTS, DEFAULT_READ_BYTES, Entry, Sandbox
from .server import INSTRUCTIONS, file_system_server

__all__ = [
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_READ_BYTES",
    "Entry",
    "FastMCP",
    "INSTRUCTIONS",
    "Sandbox",
    "create_cloud_app",
    "file_system_server",
]
