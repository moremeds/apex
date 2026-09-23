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
from mcp_types import REQUEST_TIMEOUT
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import src.application.lake.catalog as catalog_module
from src.application.lake.services import LakeServices
from src.mcp_server.server import build_app

API_KEY = "mcp-test-key"
ALLOWED_HOST = "testserver"


def test_build_app_requires_an_api_key() -> None:
    with pytest.raises(ValueError):
        build_app(LakeServices(), "", ["testserver"])


def test_websocket_connection_is_refused() -> None:
    """BearerAuth refuses every websocket scope outright (there is no websocket
    surface); only ``lifespan`` and ``/healthz`` bypass the check."""
    app = build_app(LakeServices(), API_KEY, [ALLOWED_HOST])
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/mcp"):
                pass
        assert excinfo.value.code == 1008


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


async def test_client_timeout_leaves_the_server_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client read timeout must not wedge the server: the raised ``MCPError``
    really is a request timeout, and a later independent call succeeds. Shutdown
    must complete while the first call is *still hung* server-side -- the whole-call
    deadline (set short here) ends it on its own, so this test never calls
    ``blocked.set()`` before shutdown; that would hide the deadline actually doing
    the job the server is supposed to do without our help."""
    monkeypatch.setenv("APEX_MCP_CALL_TIMEOUT_SECONDS", "1.5")
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
                with pytest.raises(MCPError) as excinfo:
                    await client.call_tool("get_lake_status", {}, read_timeout_seconds=0.2)
                # A real request timeout, not an arbitrary server-side error under
                # the same broad exception type.
                assert excinfo.value.code == REQUEST_TIMEOUT, (
                    f"expected a request-timeout MCPError (code {REQUEST_TIMEOUT}), got "
                    f"code={excinfo.value.code} message={excinfo.value.message!r}"
                )
                result = await client.call_tool("list_asset_classes", {})
                assert not result.is_error
    finally:
        # No blocked.set(): the 1.5s whole-call deadline ends the still-hung server
        # task by itself, well within _stop_uvicorn's 5s shutdown wait.
        await _stop_uvicorn(running)


@pytest.mark.xfail(
    reason=(
        "src/SDK finding: mcp 2.2.0 stateless mode does not cancel a tool call the "
        "client has abandoned via read_timeout_seconds. `cancelled` (set only from "
        "inside `except asyncio.CancelledError` in the blocked tool -- proof of a "
        "real server-side cancellation, not merely the client giving up) must fire "
        "well before the independent 3s whole-call deadline configured here; it does "
        "not. Likely cause: `build_app` always builds with `stateless_http=True` "
        "(src/mcp_server/server.py), and in stateless mode there is no persistent "
        "channel left open for the client's courtesy `notifications/cancelled` to "
        "reach the server once the original request's own connection has been "
        "abandoned client-side. Do not weaken this assertion to make it pass."
    ),
    strict=True,
)
async def test_client_timeout_cancels_the_server_task_before_the_call_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A deadline long enough that only a real client-driven cancellation -- not this
    # independent safety net -- could satisfy the 1s wait below.
    monkeypatch.setenv("APEX_MCP_CALL_TIMEOUT_SECONDS", "3")
    cancelled = asyncio.Event()

    async def _hang(*args: object, **kwargs: object) -> None:
        try:
            await asyncio.Event().wait()  # never externally set
        except asyncio.CancelledError:
            cancelled.set()
            raise

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
                await asyncio.wait_for(cancelled.wait(), timeout=1.0)
    finally:
        # Cleanup regardless of the assertion above: if the client-driven
        # cancellation never happened, the 3s whole-call deadline still ends the
        # hang on its own before _stop_uvicorn's 5s shutdown wait elapses.
        await _stop_uvicorn(running)
