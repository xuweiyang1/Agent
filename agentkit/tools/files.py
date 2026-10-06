"""File system tools, wrapped around the existing MCP sandbox.

Reuses ``mcp_server.fs.Sandbox`` for the security boundary (resolve-first,
symlink-safe traversal rejection) and exposes it as agentkit tools:
list, read, write, search.

The sandbox root is set when the registry is built — defaulting to the
user's home directory, overridable via ``ASSISTANT_WORKSPACE`` env var.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field

from ..registry import ToolRegistry
from ..schema import ToolArgs
from mcp_server.fs import Sandbox


class ListArgs(ToolArgs):
    path: str = Field(".", description="Directory to list, relative to the workspace root.")


class ReadArgs(ToolArgs):
    path: str = Field(..., description="File path to read, relative to the workspace root.")
    max_bytes: int = Field(
        200_000, ge=100, le=1_000_000, description="Maximum bytes to read."
    )


class WriteArgs(ToolArgs):
    path: str = Field(..., description="File path to write, relative to the workspace root.")
    content: str = Field(..., description="Text content to write.")


class SearchArgs(ToolArgs):
    query: str = Field(..., min_length=1, description="Text to search for.")
    path: str = Field(".", description="Directory to search in, relative to workspace root.")
    max_results: int = Field(20, ge=1, le=100, description="Maximum results to return.")


def register(registry: ToolRegistry, workspace: str | Path | None = None) -> Sandbox:
    """Attach list/read/write/search tools and return the sandbox behind them."""
    root = Path(workspace or os.environ.get("ASSISTANT_WORKSPACE", str(Path.home())))
    sandbox = Sandbox(root)

    @registry.tool(
        "list_files",
        "List files and directories in the workspace. Use it to explore before reading.",
        args_model=ListArgs,
        timeout=5.0,
    )
    async def list_files(path: str = ".") -> dict[str, Any]:
        result = sandbox.list_dir(path)
        return {"path": path, "entries": [e.to_dict() for e in result["entries"]]}

    @registry.tool(
        "read_file",
        "Read a text file from the workspace. Returns the file content.",
        args_model=ReadArgs,
        timeout=10.0,
    )
    async def read_file(path: str, max_bytes: int = 200_000) -> dict[str, Any]:
        result = sandbox.read(path, max_bytes=max_bytes)
        return {
            "path": path,
            "content": result["text"],
            "bytes": len(result["text"].encode("utf-8")),
        }

    @registry.tool(
        "write_file",
        "Write a text file to the workspace. Creates directories if needed.",
        args_model=WriteArgs,
        timeout=10.0,
    )
    async def write_file(path: str, content: str) -> dict[str, Any]:
        result = sandbox.write(path, content)
        return {"path": path, "written": result.get("written", len(content))}

    @registry.tool(
        "search_files",
        "Search for text in files within the workspace. Use it to find something when you don't know the exact file.",
        args_model=SearchArgs,
        timeout=15.0,
    )
    async def search_files(
        query: str, path: str = ".", max_results: int = 20
    ) -> dict[str, Any]:
        result = sandbox.search(query, path=path, max_results=max_results)
        return {
            "query": query,
            "files_scanned": result.get("scanned", 0),
            "hits": [
                {"path": h["path"], "line": h.get("line"), "text": h.get("text", "")[:200]}
                for h in result.get("hits", [])
            ],
        }

    return sandbox


__all__ = ["register"]
