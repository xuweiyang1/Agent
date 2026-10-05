"""A thin FastMCP-style facade over the MCP 2.x ``MCPServer``.

The name changed and so did some of the surface. Rather than scatter version
checks through the server, the difference is isolated here: this class keeps
the ``FastMCP`` spelling (``@server.tool()``, ``server.run()``) that every
tutorial and most training data still uses, and forwards to
``mcp.server.mcpserver.MCPServer`` underneath.

Being explicit about what this is not: it is a name-level adapter, not a
compatibility shim. It does not emulate FastMCP features the new SDK dropped;
it exposes the ones this project uses so the call sites read the familiar way
and the SDK can move without touching them.

It does one thing that is not a rename, and that is the interesting part.

**Error translation.** The SDK draws a hard line. Raising its own
``mcp...ToolError`` means "a failure I anticipated": the call returns
``is_error=True`` with *your* message, logged at INFO. Raising anything else
is treated as a crash: the client sees only ``Error executing tool <name>``,
the message is discarded, and the traceback goes to the log at ERROR. Our
tools raise ``agentkit.ToolCallError`` carrying the classified, actionable
text -- "path '..\\..' is outside the sandbox root; only paths inside
/root are allowed" -- and without translation every one of those arrives as
a bare crash. So registration wraps each tool and converts a
``ToolCallError`` into the SDK's ``ToolError``.

The wrapper has to preserve the signature, because the SDK builds the tool's
input schema by introspecting it: ``functools.wraps`` makes
``inspect.signature`` follow ``__wrapped__`` back to the original, so the
schema is unchanged. A wrapper that hid the signature would silently publish
an empty argument schema, which is worse than the bug it fixes.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError

from agentkit.errors import ToolCallError

F = TypeVar("F", bound=Callable[..., Any])


def translate_errors(fn: F) -> F:
    """Re-raise ``ToolCallError`` as the SDK's anticipated-failure type.

    A ``ToolCallError`` already says exactly what a caller needs: a kind, a
    message, and structured details. This keeps the message intact and moves
    it onto the channel where the model can read it.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except ToolCallError as exc:
                raise McpToolError(str(exc)) from exc

        return awrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except ToolCallError as exc:
            raise McpToolError(str(exc)) from exc

    return wrapper  # type: ignore[return-value]


class FastMCP(MCPServer):
    """``MCPServer`` under the name people know, with the same decorators."""

    def add_tool(  # type: ignore[override]
        self,
        fn: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Register a tool, translating its classified errors on the way.

        Wrapping here rather than in each tool means the rule is enforced by
        construction: a tool added through this class cannot accidentally
        leak an unclassified crash. Filtering ``title`` out of ``kwargs``
        keeps the FastMCP positional convention working for callers that pass
        ``(name, description)``.
        """
        kwargs.pop("title", None)
        super().add_tool(
            translate_errors(fn),
            name=name or getattr(fn, "__name__", None),
            description=description,
            **kwargs,
        )

    def tool(  # type: ignore[override]
        self,
        name: str | None = None,
        description: str | None = None,
        **kwargs: Any,
    ) -> Callable[[F], F]:
        """Register a tool, accepting FastMCP's positional ``name``.

        The base signature is ``tool(name, title, description, ...)``; the
        older spelling was ``tool(name, description, ...)``. Without this, a
        call like ``@server.tool("read", "Read a file")`` binds the
        description to ``title`` and leaves the description empty -- a
        silent, cosmetic bug that is exactly what a shim should absorb.
        """
        return super().tool(name=name, description=description, **kwargs)

    def run(self, transport: str = "stdio", **kwargs: Any) -> None:
        """Run over a transport, accepting either spelling of its name."""
        super().run(transport.replace("_", "-"), **kwargs)

    def mount(self, app: Any, path: str = "/mcp") -> None:
        """Attach the streamable-HTTP transport to an existing ASGI app.

        This is the deployment seam: a cloud function gives you an ASGI
        application, not a port, so the MCP transport has to be a sub-app.

        The inner route is ``"/"``, not ``path``. The SDK registers its
        handler at ``streamable_http_path`` *inside* the app it returns, so
        mounting that app at ``path`` would serve the endpoint at
        ``path + path`` -- ``/mcp/mcp``. Letting the mount prefix supply the
        full path is the only combination that yields exactly ``path``.
        """
        inner = self.streamable_http_app(streamable_http_path="/")
        app.mount(path, inner)


__all__ = ["FastMCP", "translate_errors"]
