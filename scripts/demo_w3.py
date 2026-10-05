"""Run the W3 sandbox and chart tool offline, and print what they do.

    python scripts/demo_w3.py

No key, no network. The interesting output is the escape attempts: each one
is a real technique, and each is refused with a message a model could act on.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentkit import ToolInvoker, build_registry
from agentkit.errors import ToolCallError
from mcp_server import Sandbox, file_system_server


def header(text: str) -> None:
    print(f"\n=== {text} ===")


async def main() -> None:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "notes").mkdir()
    (root / "notes" / "todo.md").write_text(
        "buy milk\nrenew passport\nbook dentist\n", encoding="utf-8", newline=""
    )
    (root / "readme.txt").write_text("hello sandbox\n", encoding="utf-8", newline="")

    sandbox = Sandbox(root)
    print(f"sandbox root: {root}")

    header("1. normal use")
    print("  list      :", [row["name"] for row in sandbox.list_dir(".")["entries"]])
    print("  read      :", sandbox.read("readme.txt")["text"].strip())
    print("  search    :", sandbox.search("passport")["hits"][0]["path"])
    sandbox.write("out/saved.txt", "written by the agent\n")
    print("  write     :", sandbox.read("out/saved.txt")["text"].strip())

    header("2. escape attempts, all refused")
    attempts = [
        "../../etc/passwd",
        "notes/../../../../windows/system32",
        "/etc/passwd",
        "C:\\Windows\\System32",
        "readme.txt\x00../../etc/passwd",
    ]
    for attempt in attempts:
        try:
            sandbox.resolve(attempt)
            print(f"  {attempt!r:<45} ALLOWED  <-- this would be a bug")
        except ToolCallError as exc:
            print(f"  {attempt!r:<45} {exc.kind.value}")

    header("3. a refusal is readable by the model")
    try:
        sandbox.read("../../etc/passwd")
    except ToolCallError as exc:
        print("  kind    :", exc.kind.value)
        print("  message :", str(exc)[:110])
        print("  details :", {k: v for k, v in exc.details.items() if k == "rejected"})

    header("4. the MCP server, through the protocol")
    server = file_system_server(root)
    tools = await server.list_tools()
    print("  tools   :", [t.name for t in tools])
    result = await server.call_tool("search_files", {"query": "dentist"})
    print("  search  :", result.content[0].text.strip()[:100])
    try:
        await server.call_tool("read_file", {"path": "../../etc/passwd"})
    except Exception as exc:
        print("  refused :", type(exc).__name__, "-", str(exc)[:80])

    header("5. the chart tool (M2: output multimodality, no architecture change)")
    invoker = ToolInvoker(build_registry(chart_output_dir=str(root / "charts")))
    result = await invoker.invoke(
        "render_chart",
        {
            "kind": "bar",
            "labels": ["Mon", "Tue", "Wed", "Thu"],
            "values": [3, 5, 2, 7],
            "title": "Tasks completed",
        },
    )
    payload = result.value
    print("  path    :", Path(payload["path"]).name)
    print("  bytes   :", payload["bytes"], "(real PNG on disk)")
    print("  transcript size:", len(str(payload)), "chars -- not", payload["bytes"], "of base64")

    header("6. tool count in the agent registry")
    registry = build_registry(chart_output_dir=str(root / "charts"))
    print("  tools   :", registry.names())
    print("  note    : input multimodality changed Message; output multimodality changed nothing")

    tmp.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
