"""The filesystem MCP server: four tools over one sandbox.

Every tool here is a thin binding to ``Sandbox``. That is the point of
keeping the sandbox free of protocol code: the security boundary is tested
directly, and this file only has to be right about wiring.

Errors follow the same rule as W2. A path that escapes the sandbox comes
back as a classified ``bad_arguments`` message naming the root, because the
caller is a model that should correct itself, not a crash that should
propagate. The MCP SDK turns a raised exception into an error *result*, so
nothing here needs to catch and hand-format a failure.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from agentkit.errors import ToolCallError

from .adapter import FastMCP
from .fs import DEFAULT_MAX_RESULTS, DEFAULT_READ_BYTES, Sandbox

INSTRUCTIONS = (
    "A sandboxed view of one directory tree. All paths are relative to the "
    "sandbox root; absolute paths and any path that resolves outside the "
    "root are rejected. Use list_directory to discover paths, search to find "
    "content, read to load a file, and write to save one."
)


def file_system_server(root: str | os.PathLike[str] | None = None) -> FastMCP:
    """Build the server, rooted at ``root`` or ``FS_SANDBOX_ROOT``.

    The root is a constructor argument rather than a constant so a test can
    point the server at a temporary directory, and so a deployment can choose
    it without editing code.
    """
    if root is None:
        root = os.environ.get("FS_SANDBOX_ROOT")
    if root is None:
        raise ToolCallError(
            "no sandbox root configured; pass root= or set FS_SANDBOX_ROOT",
            details={"env": "FS_SANDBOX_ROOT"},
        )

    sandbox = Sandbox(Path(root))
    server = FastMCP(
        name="fs-sandbox",
        instructions=INSTRUCTIONS,
        version="0.3.0",
    )

    @server.tool(
        name="list_directory",
        description=(
            "List the entries in a directory, or recursively beneath it. "
            "Returns each entry's path, name and kind."
        ),
    )
    def list_directory(
        path: str = ".",
        recursive: bool = False,
        pattern: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> dict[str, Any]:
        return sandbox.list_dir(path, recursive=recursive, pattern=pattern, max_results=max_results)

    @server.tool(
        name="read_file",
        description=(
            "Read a UTF-8 text file. Returns the text and whether it was "
            "truncated. Binary files come back marked binary with a digest "
            "instead of text."
        ),
    )
    def read_file(path: str, max_bytes: int = DEFAULT_READ_BYTES) -> dict[str, Any]:
        return sandbox.read(path, max_bytes=max_bytes)

    @server.tool(
        name="write_file",
        description=(
            "Write UTF-8 text to a file, creating parent directories. "
            "Refuses to replace an existing file unless overwrite is true."
        ),
    )
    def write_file(
        path: str,
        content: str,
        overwrite: bool = False,
        create_parents: bool = True,
    ) -> dict[str, Any]:
        return sandbox.write(
            path, content, overwrite=overwrite, create_parents=create_parents
        )

    @server.tool(
        name="search_files",
        description=(
            "Substring search across text files under a path. Returns the "
            "file, line number and matching text for each hit."
        ),
    )
    def search_files(
        query: str,
        path: str = ".",
        pattern: str | None = None,
        case_sensitive: bool = False,
        max_results: int = DEFAULT_MAX_RESULTS,
        context_lines: int = 0,
    ) -> dict[str, Any]:
        return sandbox.search(
            query,
            path=path,
            pattern=pattern,
            case_sensitive=case_sensitive,
            max_results=max_results,
            context_lines=context_lines,
        )

    server.sandbox = sandbox  # type: ignore[attr-defined]
    return server


def main(args: list[str] | None = None) -> None:
    """CLI entry point: ``python -m mcp_server.server --root DIR --transport stdio``."""
    import argparse

    parser = argparse.ArgumentParser(description="Run the filesystem MCP server")
    parser.add_argument("--root", default=None, help="sandbox root (or set FS_SANDBOX_ROOT)")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
    )
    options = parser.parse_args(args)
    file_system_server(options.root).run(options.transport)


if __name__ == "__main__":
    main()


__all__ = ["INSTRUCTIONS", "file_system_server", "main"]
