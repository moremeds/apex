"""Auth, DNS-rebinding protection, real-socket lifecycle and cancellation.

No lake data needed: an empty ``LakeServices()`` is enough to exercise the transport
itself (bearer auth, Host/Origin checks, a real uvicorn socket, and MCP-level
cancellation), independent of what the tools read.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx2
import pytest
import uvicorn
from mcp import Client, MCPError
from mcp.client.streamable_http import streamable_http_client

import src.application.lake.catalog as catalog_module
from src.application.lake.services import LakeServices
from src.mcp_server.server import build_app

API_KEY = "mcp-test-key"
ALLOWED_HOST = "testserver"


def test_build_app_requires_an_api_key() -> None:
    with pytest.raises(ValueError):
        build_app(LakeServices(), "", ["testserver"])


@asynccontextmanager
async def _transport_http() -> AsyncIterator[httpx2.AsyncClient]:
    """A plain ASGI HTTP client with no auth header pre-attached, for the header
    matrix below; the app's lifespan is started manually since ``ASGITransport``
    does not drive it on its own. A context manager rather than a ``yield``-based
    fixture: see the module docstring in ``test_tools.py`` for why (anyio cancel
    scopes need enter/exit in the same asyncio Task)."""
    app = build_app(LakeServices(), API_KEY, [ALLOWED_HOST])
    async with app._app.router.lifespan_context(app._app):
        transport = httpx2.ASGITransport(app=app)
        async with httpx2.AsyncClient(
            transport=transport, base_url=f"http://{ALLOWED_HOST}"
        ) as http_client:
            yield http_client


async def test_healthz_needs_no_auth() -> None:
    async with _transport_http() as http:
        resp = await http.get("/healthz")
    assert resp.status_code == 200


async def test_missing_authorization_is_401() -> None:
    async with _transport_http() as http:
        resp = await http.post("/mcp", json={})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_wrong_key_is_401() -> None:
    async with _transport_http() as http:
        resp = await http.post("/mcp", json={}, headers={"authorization": "Bearer nope"})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_wrong_scheme_is_401() -> None:
    async with _transport_http() as http:
        resp = await http.post("/mcp", json={}, headers={"authorization": f"Basic {API_KEY}"})
    assert resp.status_code == 401


async def test_bad_host_is_421() -> None:
    async with _transport_http() as http:
        resp = await http.post(
            "/mcp",
            json={},
            headers={"authorization": f"Bearer {API_KEY}", "host": "evil.example"},
        )
    assert resp.status_code == 421


async def test_bad_origin_is_403() -> None:
    async with _transport_http() as http:
        resp = await http.post(
            "/mcp",
            json={},
            headers={
                "authorization": f"Bearer {API_KEY}",
                "origin": "http://evil.example",
            },
        )
    assert resp.status_code == 403


async def test_allowed_origin_and_bearer_succeed_end_to_end() -> None:
    app = build_app(LakeServices(), API_KEY, [ALLOWED_HOST])
    async with app._app.router.lifespan_context(app._app):
        async with httpx2.AsyncClient(
            transport=httpx2.ASGITransport(app=app),
            base_url=f"http://{ALLOWED_HOST}",
            headers={
                "authorization": f"Bearer {API_KEY}",
                "origin": f"http://{ALLOWED_HOST}",
            },
        ) as http_client:
            async with Client(
                streamable_http_client(f"http://{ALLOWED_HOST}/mcp", http_client=http_client)
            ) as client:
                tools = (await client.list_tools()).tools
                assert len(tools) == 20


# -- real socket ----------------------------------------------------------------


class _RunningServer:
    def __init__(self, server: uvicorn.Server, task: "asyncio.Task[None]", base_url: str) -> None:
        self.server = server
        self.task = task
        self.base_url = base_url


async def _start_uvicorn(app: object) -> _RunningServer:
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="on")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return _RunningServer(server, task, f"http://127.0.0.1:{port}")


async def _stop_uvicorn(running: _RunningServer) -> None:
    running.server.should_exit = True
    await asyncio.wait_for(running.task, timeout=5)


async def test_real_socket_serves_tools_and_shuts_down_cleanly() -> None:
    # Wildcard port pattern: the real port is only known after the OS assigns one.
    app = build_app(LakeServices(), API_KEY, ["127.0.0.1:*"])
    running = await _start_uvicorn(app)
    try:
        async with httpx2.AsyncClient(
            base_url=running.base_url, headers={"authorization": f"Bearer {API_KEY}"}
        ) as http_client:
            async with Client(
                streamable_http_client(f"{running.base_url}/mcp", http_client=http_client)
            ) as client:
                tools = (await client.list_tools()).tools
                assert len(tools) == 20
                result = await client.call_tool("list_asset_classes", {})
                assert not result.is_error
    finally:
        await _stop_uvicorn(running)


async def test_cancellation_leaves_the_server_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool call abandoned on a client read timeout must not wedge the server: a
    later, independent call succeeds and shutdown still completes."""
    blocked = asyncio.Event()

    async def _hang(*args: object, **kwargs: object) -> None:
        await blocked.wait()

    monkeypatch.setattr(catalog_module, "lake_status", _hang)

    app = build_app(LakeServices(), API_KEY, ["127.0.0.1:*"])
    running = await _start_uvicorn(app)
    try:
        async with httpx2.AsyncClient(
            base_url=running.base_url, headers={"authorization": f"Bearer {API_KEY}"}
        ) as http_client:
            async with Client(
                streamable_http_client(f"{running.base_url}/mcp", http_client=http_client)
            ) as client:
                with pytest.raises(MCPError):
                    await client.call_tool("get_lake_status", {}, read_timeout_seconds=0.2)
                result = await client.call_tool("list_asset_classes", {})
                assert not result.is_error
    finally:
        blocked.set()
        await _stop_uvicorn(running)
