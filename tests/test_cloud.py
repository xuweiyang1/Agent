"""Tests for the cloud-function entry point.

The assertion that matters is that the mounted MCP transport actually works
inside the host app, because the failure mode when the lifespan is not
forwarded is a request that hangs forever rather than an error -- which is
invisible in a unit test that only checks the route table.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from mcp_server.cloud import create_cloud_app


class CloudAppTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "note.txt").write_text("cloud content\n", encoding="utf-8", newline="")
        # A host allowlist is required behind a real host; TestClient speaks
        # as "testserver", which the localhost-only SDK default rejects.
        self.app = create_cloud_app(
            root=self.root, mcp_path="/mcp", allowed_hosts=["testserver", "localhost"]
        )

    def tearDown(self):
        self._tmp.cleanup()

    def test_health_endpoint_answers(self):
        with TestClient(self.app) as client:
            body = client.get("/healthz").json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["mcp_path"], "/mcp")

    def test_health_does_not_leak_the_absolute_root(self):
        with TestClient(self.app) as client:
            body = client.get("/healthz").json()
        self.assertNotIn(str(self.root), str(body))

    def test_mcp_transport_is_mounted_and_lifespan_runs(self):
        """A POST to the MCP path must be answered, not hang.

        This is the lifespan check: if the mounted app's startup never ran,
        the session manager is uninitialized and the request stalls.
        """
        with TestClient(self.app) as client:
            response = client.post(
                "/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0"},
                    },
                },
                headers={"accept": "application/json, text/event-stream"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("jsonrpc", response.text)

    def test_mcp_endpoint_is_not_served_at_a_doubled_path(self):
        """Regression: mounting the SDK app at its own route gave /mcp/mcp."""
        with TestClient(self.app) as client:
            self.assertEqual(
                client.post("/mcp/mcp", json={}, headers={"accept": "application/json"}).status_code,
                404,
            )

    def test_localhost_only_default_rejects_a_foreign_host(self):
        """The 421 this produces is the reason allowed_hosts exists."""
        app = create_cloud_app(root=self.root)
        with TestClient(app) as client:
            response = client.post(
                "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                headers={"accept": "application/json, text/event-stream"},
            )
        self.assertEqual(response.status_code, 421)

    def test_app_exposes_the_server_for_inspection(self):
        self.assertTrue(hasattr(self.app.state, "mcp_server"))
        tools = {
            tool.name
            for tool in __import__("asyncio").run(
                self.app.state.mcp_server.list_tools()
            )
        }
        self.assertEqual(
            tools, {"list_directory", "read_file", "write_file", "search_files"}
        )


if __name__ == "__main__":
    unittest.main()
