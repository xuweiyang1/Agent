"""Cloud-function deployment: one ASGI app carrying both surfaces.

A function host (Cloud Run, Lambda behind an adapter, a container platform)
hands you an ASGI callable and expects it to own the whole request space.
So the MCP transport has to be mounted inside the app rather than started as
its own server, and the FastAPI health/tool endpoints live alongside it.

The lifespan is the part that is easy to miss. The mounted MCP app has its
own startup/shutdown, and if the outer app does not run it, the session
manager is never initialized -- every MCP request then hangs instead of
erroring, which is a miserable thing to debug from a cloud log. The
``@asynccontextmanager`` below is the fix, not boilerplate.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from mcp.server.transport_security import TransportSecuritySettings

from .server import file_system_server

# A cloud host sets PORT; locally it defaults to something conventional.
DEFAULT_PORT = int(os.environ.get("PORT", "8080"))


def create_cloud_app(
    *,
    root: str | os.PathLike[str] | None = None,
    mcp_path: str = "/mcp",
    allowed_hosts: list[str] | None = None,
) -> FastAPI:
    """Build the deployed app: FastAPI at ``/``, MCP at ``mcp_path``.

    ``allowed_hosts`` is the one security setting a deployment cannot skip.
    The SDK defaults to enabling DNS-rebinding protection against localhost
    hostnames only, which is right for a laptop and actively wrong behind a
    cloud function: the platform routes to the container by a service hostname,
    so the default rejects every real request with a 421. Passing the service's
    hostnames here is the required configuration step, not an optional knob.
    """
    server = file_system_server(root)
    # The inner route is "/" because the mount prefix supplies the path; see
    # FastMCP.mount for the /mcp/mcp failure this avoids.
    if allowed_hosts is None:
        transport_security = None  # SDK default: localhost-only, fine for local dev
    else:
        transport_security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=[f"https://{host}" for host in allowed_hosts if ":" not in host]
            + list(allowed_hosts),
        )
    inner = server.streamable_http_app(
        streamable_http_path="/", transport_security=transport_security
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Run the MCP sub-app's lifespan, or its session manager never starts.
        async with inner.router.lifespan_context(app):
            yield

    app = FastAPI(title="fs-mcp", version="0.3.0", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "fs-mcp",
            "mcp_path": mcp_path,
            # The root is deliberately not returned: a health endpoint should
            # not leak the filesystem layout to an unauthenticated caller.
            "rooted": server.sandbox.root.name,
        }

    app.mount(mcp_path, inner)
    app.state.mcp_server = server
    return app


def main(argv: list[str] | None = None) -> None:
    """Serve locally with uvicorn; the cloud host calls ``create_cloud_app``."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Serve the filesystem MCP app")
    parser.add_argument("--root", default=None, help="sandbox root (or set FS_SANDBOX_ROOT)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--mcp-path", default="/mcp")
    options = parser.parse_args(argv)

    app = create_cloud_app(root=options.root, mcp_path=options.mcp_path)
    uvicorn.run(app, host=options.host, port=options.port)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_PORT", "create_cloud_app", "main"]
