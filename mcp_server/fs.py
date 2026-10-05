"""A filesystem sandbox: the only place a path is allowed to be interpreted.

This module deliberately has no MCP import. The sandbox is a security
boundary, and a security boundary should be testable without standing up a
protocol server first -- so the functions here are plain, synchronous, and
unit-testable, and ``server.py`` is a thin binding over them.

The threat is simple to state and easy to get wrong. A model supplies a
path. If that path is joined to the root naively, ``../../etc/passwd`` walks
out, and ``startswith`` checks are defeated by both ``..`` and by a symlink
planted inside the root that points outside it. The defense is to resolve
the candidate *first* -- which normalizes ``..`` and follows symlinks -- and
then ask whether the result is still inside the resolved root. Order matters:
checking before resolving is the same as not checking.

Known limitation, stated rather than hidden: resolve-then-open is a
time-of-check/time-of-use window. A local attacker who can swap a path for a
symlink between the two calls can win the race. Closing it needs
``O_NOFOLLOW``/``openat`` on POSIX, which this project does not do because
its threat model is a model emitting bad paths, not a concurrent local
attacker.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agentkit.errors import ErrorKind, ToolCallError

# A read cap. The point is not the exact number, it is that no single tool
# call can return an unbounded amount of text into a context window.
DEFAULT_READ_BYTES = 200_000
DEFAULT_MAX_RESULTS = 50
BINARY_SNIFF_BYTES = 4096


def _traversal_error(user_path: str, root: Path) -> ToolCallError:
    """One error shape for every escape attempt.

    The message names the root, because a model that gets rejected for a
    traversing path needs to be told where it is actually allowed to look;
    a bare "denied" leaves it guessing and likely to repeat the mistake.
    """
    return ToolCallError(
        f"path {user_path!r} is outside the sandbox root; only paths inside "
        f"{root} are allowed",
        kind=ErrorKind.BAD_ARGUMENTS,
        details={"root": str(root), "rejected": user_path},
    )


@dataclass(frozen=True)
class Entry:
    """One directory listing row."""

    path: str
    name: str
    kind: str  # "file" | "dir" | "link"
    size: int = 0

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {"path": self.path, "name": self.name, "kind": self.kind}
        if self.kind == "file":
            row["size"] = self.size
        return row


class Sandbox:
    """Resolves and performs every file operation, confined to one root.

    Constructing it resolves the root once, so a root that is itself a
    symlink is normalized before any comparison happens. If that were skipped,
    every legitimate path would look like it escaped.
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        resolved = Path(root).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"sandbox root does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"sandbox root is not a directory: {resolved}")
        self.root = resolved

    def resolve(self, user_path: str) -> Path:
        """Turn a model-supplied path into a guaranteed-inside-root one.

        Four rejections, in order, because each one defeats the naive check
        that would otherwise be written:

        1. null bytes, which truncate a path in the underlying C call;
        2. absolute paths, which ignore the root entirely;
        3. the resolved target leaving the root, which is what catches both
           ``..`` and a symlink pointing out;
        4. a resolved target that is a symlink to outside -- covered by 3,
           and asserted by a test so it stays covered.
        """
        if not isinstance(user_path, str) or not user_path.strip():
            raise ToolCallError(
                "path must be a non-empty string",
                kind=ErrorKind.BAD_ARGUMENTS,
                details={"got": repr(user_path)},
            )
        if "\x00" in user_path:
            raise ToolCallError(
                "path contains a null byte",
                kind=ErrorKind.BAD_ARGUMENTS,
                details={"rejected": "null byte"},
            )

        raw = Path(user_path)
        if raw.is_absolute() or raw.drive:
            raise _traversal_error(user_path, self.root)

        candidate = (self.root / raw).resolve()
        if not candidate.is_relative_to(self.root):
            raise _traversal_error(user_path, self.root)
        return candidate

    def relative(self, path: Path) -> str:
        """Render a path back as a root-relative posix string."""
        return path.relative_to(self.root).as_posix()

    # -- operations -------------------------------------------------------

    def read(self, path: str, *, max_bytes: int = DEFAULT_READ_BYTES) -> dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            raise ToolCallError(f"no such file: {path!r}", kind=ErrorKind.NOT_FOUND)
        if target.is_dir():
            raise ToolCallError(
                f"{path!r} is a directory; use list_directory", kind=ErrorKind.BAD_ARGUMENTS
            )

        raw = target.read_bytes()
        truncated = len(raw) > max_bytes
        head = raw[:max_bytes]

        # A binary file is not an error, it is an answer: the agent asked to
        # read something and the useful reply is "this is not text, here is
        # its size and digest". Raising would waste a turn on a retry that
        # cannot succeed.
        if b"\x00" in head[:BINARY_SNIFF_BYTES]:
            return {
                "path": self.relative(target),
                "binary": True,
                "size": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()[:16],
                "text": None,
            }

        return {
            "path": self.relative(target),
            "binary": False,
            "size": len(raw),
            "truncated": truncated,
            "text": head.decode("utf-8", errors="replace"),
        }

    def write(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = False,
        create_parents: bool = True,
    ) -> dict[str, Any]:
        target = self.resolve(path)
        if target.exists() and target.is_dir():
            raise ToolCallError(
                f"{path!r} is a directory", kind=ErrorKind.BAD_ARGUMENTS
            )
        if target.exists() and not overwrite:
            # Refusing by default is the safe choice: a model retrying a
            # write should not silently destroy whatever was there.
            raise ToolCallError(
                f"{path!r} already exists; pass overwrite=true to replace it",
                kind=ErrorKind.BAD_ARGUMENTS,
                details={"path": self.relative(target)},
            )
        if not create_parents and not target.parent.exists():
            raise ToolCallError(
                f"parent directory of {path!r} does not exist",
                kind=ErrorKind.NOT_FOUND,
            )

        if create_parents:
            target.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode("utf-8")
        target.write_bytes(payload)
        return {
            "path": self.relative(target),
            "bytes_written": len(payload),
            "created": True,
            "sha256": hashlib.sha256(payload).hexdigest()[:16],
        }

    def list_dir(
        self,
        path: str = ".",
        *,
        recursive: bool = False,
        pattern: str | None = None,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> dict[str, Any]:
        target = self.resolve(path)
        if not target.exists():
            raise ToolCallError(f"no such directory: {path!r}", kind=ErrorKind.NOT_FOUND)
        if not target.is_dir():
            raise ToolCallError(f"{path!r} is not a directory", kind=ErrorKind.BAD_ARGUMENTS)

        walker: Iterable[Path] = target.rglob("*") if recursive else target.iterdir()
        entries: list[Entry] = []
        truncated = False
        for child in sorted(walker, key=lambda p: p.as_posix()):
            if pattern and not fnmatch.fnmatch(child.name, pattern):
                continue
            # A symlink is reported as a link and never followed here, so a
            # listing cannot be used to enumerate outside the root.
            if child.is_symlink():
                kind = "link"
            elif child.is_dir():
                kind = "dir"
            else:
                kind = "file"
            size = child.stat().st_size if kind == "file" else 0
            entries.append(Entry(self.relative(child), child.name, kind, size))
            if len(entries) >= max_results:
                truncated = True
                break

        return {
            "path": self.relative(target) or ".",
            "entries": [entry.to_dict() for entry in entries],
            "truncated": truncated,
        }

    def search(
        self,
        query: str,
        *,
        path: str = ".",
        pattern: str | None = None,
        case_sensitive: bool = False,
        max_results: int = DEFAULT_MAX_RESULTS,
        context_lines: int = 0,
    ) -> dict[str, Any]:
        """Substring search across text files under ``path``.

        Substring rather than the corpus ranker on purpose: a filesystem
        search is a grep, and the agent can always refine a query. Ranked
        retrieval is what ``retrieval/`` will do in W3.5; conflating the two
        would mean the MCP server depends on an index that W3.5 has not built
        yet.
        """
        if not query:
            raise ToolCallError("query must be non-empty", kind=ErrorKind.BAD_ARGUMENTS)

        start = self.resolve(path)
        if not start.exists():
            raise ToolCallError(f"no such directory: {path!r}", kind=ErrorKind.NOT_FOUND)

        needle = query if case_sensitive else query.lower()
        files = [start] if start.is_file() else sorted(start.rglob("*"))
        hits: list[dict[str, Any]] = []
        scanned = 0
        truncated = False

        for candidate in files:
            if not candidate.is_file() or candidate.is_symlink():
                continue
            if pattern and not fnmatch.fnmatch(candidate.name, pattern):
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            scanned += 1
            lines = text.splitlines()
            for number, line in enumerate(lines, start=1):
                haystack = line if case_sensitive else line.lower()
                if needle not in haystack:
                    continue
                hits.append(
                    {
                        "path": self.relative(candidate),
                        "line": number,
                        "text": line.strip()[:240],
                        "context": _context(lines, number, context_lines),
                    }
                )
                if len(hits) >= max_results:
                    truncated = True
                    break
            if truncated:
                break

        return {
            "query": query,
            "files_scanned": scanned,
            "hits": hits,
            "truncated": truncated,
        }


def _context(lines: list[str], number: int, radius: int) -> list[str]:
    if radius <= 0:
        return []
    start = max(0, number - 1 - radius)
    end = min(len(lines), number + radius)
    return [line.strip()[:240] for line in lines[start:end]]


__all__ = [
    "BINARY_SNIFF_BYTES",
    "DEFAULT_MAX_RESULTS",
    "DEFAULT_READ_BYTES",
    "Entry",
    "Sandbox",
]
