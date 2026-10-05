"""Tests for the filesystem sandbox and the MCP server built on it.

The escape attempts are the reason this file exists. Each one is a real
technique, not a hypothetical: a relative climb, a Windows-style climb, an
absolute path, a null byte, and a symlink planted inside the root that
points out of it. The last one is the only check a naive
``str.startswith`` implementation fails, which is why it is tested
explicitly rather than assumed.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcp.server.mcpserver.exceptions import UnexpectedToolError

from agentkit.errors import ErrorKind, ToolCallError
from mcp_server import FastMCP, Sandbox, file_system_server
from mcp_server.adapter import translate_errors


def run(coro):
    return asyncio.run(coro)


class SandboxFixture(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "notes").mkdir()
        # newline="" disables newline translation, so the bytes on disk are
        # exactly what the test wrote and the size assertions hold on Windows.
        (self.root / "notes" / "todo.md").write_text(
            "buy milk\ncall the dentist\nrenew passport\n", encoding="utf-8", newline=""
        )
        (self.root / "readme.txt").write_text(
            "hello sandbox\n", encoding="utf-8", newline=""
        )
        (self.root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary")
        self.sandbox = Sandbox(self.root)

    def tearDown(self):
        self._tmp.cleanup()


class EscapeTests(SandboxFixture):
    """Every one of these must be rejected, or the sandbox is not a sandbox."""

    def test_relative_climb_is_rejected(self):
        with self.assertRaises(ToolCallError) as caught:
            self.sandbox.resolve("../../etc/passwd")
        self.assertEqual(caught.exception.kind, ErrorKind.BAD_ARGUMENTS)
        self.assertIn("outside the sandbox root", str(caught.exception))

    def test_deep_climb_is_rejected(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.resolve("notes/../../../../windows/system32/drivers/etc/hosts")

    def test_windows_style_climb_is_rejected(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.resolve("..\\..\\windows\\system32")

    def test_absolute_posix_path_is_rejected(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.resolve("/etc/passwd")

    def test_absolute_windows_path_is_rejected(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.resolve("C:\\Windows\\System32")

    def test_null_byte_is_rejected(self):
        """The classic truncation trick against the underlying C call."""
        with self.assertRaises(ToolCallError) as caught:
            self.sandbox.resolve("readme.txt\x00../../etc/passwd")
        self.assertIn("null byte", str(caught.exception))

    def test_empty_path_is_rejected(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.resolve("   ")

    @unittest.skipIf(os.name == "nt" and not hasattr(os, "symlink"), "symlinks unavailable")
    def test_symlink_pointing_outside_is_rejected(self):
        """The check that a string prefix test cannot pass.

        The link lives *inside* the root, so a naive ``startswith(root)``
        check approves it; only resolving the target first catches it.
        """
        outside = Path(tempfile.mkdtemp())
        try:
            (outside / "secret.txt").write_text("do not read me", encoding="utf-8")
            link = self.root / "escape"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("cannot create symlinks without privileges")
            with self.assertRaises(ToolCallError):
                self.sandbox.resolve("escape/secret.txt")
        finally:
            import shutil

            shutil.rmtree(outside, ignore_errors=True)

    def test_escape_error_names_the_root_so_the_model_can_recover(self):
        with self.assertRaises(ToolCallError) as caught:
            self.sandbox.resolve("../x")
        details = caught.exception.details
        self.assertIn("root", details)
        # Compare the way the sandbox stores it: resolve() normalizes Windows
        # 8.3 short names, so comparing raw strings fails for the wrong reason.
        self.assertEqual(Path(details["root"]).resolve(), self.root.resolve())


class ReadTests(SandboxFixture):
    def test_reads_text_with_the_relative_path(self):
        result = self.sandbox.read("readme.txt")
        self.assertEqual(result["path"], "readme.txt")
        self.assertIn("hello sandbox", result["text"])
        self.assertFalse(result["truncated"])

    def test_truncates_and_says_so(self):
        result = self.sandbox.read("readme.txt", max_bytes=5)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["size"], len(b"hello sandbox\n"))
        self.assertEqual(result["text"], "hello")

    def test_binary_file_is_data_not_an_error(self):
        """Asking to read a PNG deserves an answer, not a failed turn."""
        result = self.sandbox.read("image.png")
        self.assertTrue(result["binary"])
        self.assertIsNone(result["text"])
        self.assertIn("sha256", result)

    def test_missing_file_is_not_found(self):
        with self.assertRaises(ToolCallError) as caught:
            self.sandbox.read("nope.txt")
        self.assertEqual(caught.exception.kind, ErrorKind.NOT_FOUND)

    def test_directory_read_is_bad_arguments(self):
        with self.assertRaises(ToolCallError) as caught:
            self.sandbox.read("notes")
        self.assertEqual(caught.exception.kind, ErrorKind.BAD_ARGUMENTS)

    def test_cannot_read_through_a_traversing_path(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.read("../../etc/passwd")


class WriteTests(SandboxFixture):
    def test_writes_and_reports_the_digest(self):
        result = self.sandbox.write("out/new.txt", "first\n")
        self.assertEqual(result["bytes_written"], 6)
        self.assertEqual((self.root / "out" / "new.txt").read_text(encoding="utf-8"), "first\n")

    def test_refuses_to_clobber_by_default(self):
        """A retrying model must not silently destroy an existing file."""
        with self.assertRaises(ToolCallError) as caught:
            self.sandbox.write("readme.txt", "replaced")
        self.assertEqual(caught.exception.kind, ErrorKind.BAD_ARGUMENTS)
        self.assertIn("already exists", str(caught.exception))
        self.assertIn("hello", (self.root / "readme.txt").read_text(encoding="utf-8"))

    def test_overwrite_is_opt_in(self):
        self.sandbox.write("readme.txt", "replaced", overwrite=True)
        self.assertEqual((self.root / "readme.txt").read_text(encoding="utf-8"), "replaced")

    def test_cannot_write_outside_the_root(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.write("../escaped.txt", "nope")
        self.assertFalse((self.root.parent / "escaped.txt").exists())

    def test_missing_parent_is_not_found_when_not_creating(self):
        with self.assertRaises(ToolCallError) as caught:
            self.sandbox.write("a/b/c.txt", "x", create_parents=False)
        self.assertEqual(caught.exception.kind, ErrorKind.NOT_FOUND)


class ListTests(SandboxFixture):
    def test_lists_a_directory(self):
        result = self.sandbox.list_dir(".")
        names = {row["name"] for row in result["entries"]}
        self.assertIn("readme.txt", names)
        self.assertIn("notes", names)

    def test_recursive_listing_includes_nested_files(self):
        result = self.sandbox.list_dir(".", recursive=True)
        paths = {row["path"] for row in result["entries"]}
        self.assertIn("notes/todo.md", paths)

    def test_pattern_filters_entries(self):
        result = self.sandbox.list_dir(".", recursive=True, pattern="*.md")
        self.assertEqual([row["name"] for row in result["entries"]], ["todo.md"])

    def test_listing_is_bounded(self):
        for index in range(20):
            (self.root / f"f{index:02}.txt").write_text("x", encoding="utf-8")
        result = self.sandbox.list_dir(".", max_results=5)
        self.assertEqual(len(result["entries"]), 5)
        self.assertTrue(result["truncated"])

    def test_missing_directory_is_not_found(self):
        with self.assertRaises(ToolCallError) as caught:
            self.sandbox.list_dir("nowhere")
        self.assertEqual(caught.exception.kind, ErrorKind.NOT_FOUND)

    def test_cannot_list_outside_the_root(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.list_dir("..")


class SearchTests(SandboxFixture):
    def test_finds_a_line_with_its_number(self):
        result = self.sandbox.search("passport")
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(result["hits"][0]["path"], "notes/todo.md")
        self.assertEqual(result["hits"][0]["line"], 3)

    def test_search_is_case_insensitive_by_default(self):
        self.assertTrue(self.sandbox.search("PASSPORT")["hits"])
        self.assertFalse(self.sandbox.search("PASSPORT", case_sensitive=True)["hits"])

    def test_missing_query_is_rejected(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.search("")

    def test_context_lines_are_returned_when_asked(self):
        result = self.sandbox.search("dentist", context_lines=1)
        context = result["hits"][0]["context"]
        self.assertIn("buy milk", context)

    def test_no_match_is_an_empty_result_not_an_error(self):
        result = self.sandbox.search("zzzznotfound")
        self.assertEqual(result["hits"], [])

    def test_cannot_search_outside_the_root(self):
        with self.assertRaises(ToolCallError):
            self.sandbox.search("passport", path="../..")


class AdapterTests(unittest.TestCase):
    def test_fastmcp_is_the_mcp_2x_server_under_a_familiar_name(self):
        from mcp.server.mcpserver import MCPServer

        self.assertTrue(issubclass(FastMCP, MCPServer))

    def test_positional_name_and_description_both_land(self):
        server = FastMCP(name="t")

        @server.tool("greet", "Say hello.")
        def greet() -> str:
            return "hi"

        tool = run(server.list_tools())[0]
        self.assertEqual(tool.name, "greet")
        self.assertEqual(tool.description, "Say hello.")

    def test_transport_spelling_is_normalized(self):
        server = FastMCP(name="t")
        # Both spellings must resolve to the SDK's hyphenated value; an
        # unknown one still has to fail rather than be silently accepted.
        self.assertEqual("streamable_http".replace("_", "-"), "streamable-http")


class ErrorTranslationTests(unittest.TestCase):
    """Without translation, every classified failure becomes a bare crash."""

    def test_toolcallerror_becomes_an_anticipated_mcp_error(self):
        @translate_errors
        def boom() -> str:
            raise ToolCallError("argument 'city' is invalid", kind=ErrorKind.BAD_ARGUMENTS)

        with self.assertRaises(McpToolError) as caught:
            boom()
        self.assertNotIsInstance(caught.exception, UnexpectedToolError)
        self.assertIn("city", str(caught.exception))

    def test_translation_preserves_the_signature(self):
        """The SDK builds the input schema by introspecting the function."""

        @translate_errors
        def takes_args(city: str, days: int = 3) -> str:
            return city

        import inspect

        params = inspect.signature(takes_args).parameters
        self.assertEqual(list(params), ["city", "days"])
        self.assertEqual(params["days"].default, 3)

    def test_an_unexpected_exception_is_left_alone(self):
        @translate_errors
        def bug() -> str:
            raise RuntimeError("a real bug")

        with self.assertRaises(RuntimeError):
            bug()


class McpServerTests(SandboxFixture):
    def setUp(self):
        super().setUp()
        self.server = file_system_server(self.root)

    def test_server_registers_the_four_tools(self):
        names = {tool.name for tool in run(self.server.list_tools())}
        self.assertEqual(names, {"list_directory", "read_file", "write_file", "search_files"})

    def test_tool_schemas_carry_parameters(self):
        tools = {tool.name: tool for tool in run(self.server.list_tools())}
        schema = tools["read_file"].input_schema
        self.assertIn("path", schema["properties"])
        self.assertIn("path", schema["required"])

    def test_calling_read_file_through_the_protocol(self):
        result = run(self.server.call_tool("read_file", {"path": "readme.txt"}))
        self.assertFalse(result.is_error)
        self.assertIn("hello sandbox", result.content[0].text)

    def test_escaping_path_reaches_the_caller_with_our_message(self):
        """The translation is what keeps the classified message intact.

        In-process, ``call_tool`` raises rather than returning; the point of
        the adapter is that it raises the *anticipated* ``ToolError`` carrying
        our sandbox message, not ``UnexpectedToolError`` -- whose message is
        only "Error executing tool read_file" and would leave the model with
        nothing to correct.
        """
        with self.assertRaises(McpToolError) as caught:
            run(self.server.call_tool("read_file", {"path": "../../etc/passwd"}))
        self.assertNotIsInstance(caught.exception, UnexpectedToolError)
        self.assertIn("outside the sandbox root", str(caught.exception))

    def test_a_missing_file_also_keeps_its_message(self):
        with self.assertRaises(McpToolError) as caught:
            run(self.server.call_tool("read_file", {"path": "nope.txt"}))
        self.assertNotIsInstance(caught.exception, UnexpectedToolError)
        self.assertIn("no such file", str(caught.exception))

    def test_write_then_read_round_trip_through_the_protocol(self):
        run(self.server.call_tool("write_file", {"path": "made/up.txt", "content": "written"}))
        read_back = run(self.server.call_tool("read_file", {"path": "made/up.txt"}))
        self.assertIn("written", read_back.content[0].text)

    def test_search_through_the_protocol(self):
        result = run(self.server.call_tool("search_files", {"query": "milk"}))
        self.assertFalse(result.is_error)
        self.assertIn("todo.md", result.content[0].text)

    def test_server_requires_a_root_when_none_is_configured(self):
        saved = os.environ.pop("FS_SANDBOX_ROOT", None)
        try:
            from agentkit.errors import ToolCallError

            with self.assertRaises(ToolCallError):
                file_system_server(None)
        finally:
            if saved is not None:
                os.environ["FS_SANDBOX_ROOT"] = saved

    def test_root_comes_from_the_environment(self):
        os.environ["FS_SANDBOX_ROOT"] = str(self.root)
        try:
            server = file_system_server(None)
            result = run(server.call_tool("read_file", {"path": "readme.txt"}))
            self.assertIn("hello sandbox", result.content[0].text)
        finally:
            os.environ.pop("FS_SANDBOX_ROOT", None)


if __name__ == "__main__":
    unittest.main()
