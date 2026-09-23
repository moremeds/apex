"""Apex market-data MCP: the Livewire lake, read-only, over Streamable HTTP.

A separate process from REST (``python -m src.mcp_server.server``). It builds only the
lake readers -- no PostgreSQL, no streaming pipeline, no REST app -- and refuses to
boot without ``APEX_MCP_API_KEY``: there is no unauthenticated mode.

The package is ``mcp_server``, not ``mcp``: a ``src/mcp`` package would shadow the SDK
for anything run with ``src/`` on its path.
"""

import logging
import os
import secrets
from pathlib import Path
from typing import List, Optional

import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from src.application.lake.services import LakeServices
from src.infrastructure.adapters.livewire.coverage import CoverageCatalog
from src.infrastructure.adapters.livewire.membership import MembershipReader
from src.infrastructure.adapters.livewire.ohlc_provider import LivewireOhlcProvider
from src.infrastructure.adapters.livewire.pit_revisions import PitRevisionReader
from src.infrastructure.adapters.livewire.reference import LivewireReferenceReader
from src.infrastructure.adapters.livewire.repairs import RepairsReader
from src.infrastructure.adapters.livewire.revisions import RevisionManifestReader
from src.mcp_server import tools_bars, tools_discovery, tools_identity, tools_revisions
from src.mcp_server._common import LakeMCPServer

logger = logging.getLogger(__name__)

HEALTH_PATH = "/healthz"
MCP_PATH = "/mcp"

INSTRUCTIONS = (
    "Read-only access to the Livewire market-data lake: discovery, bars and yields, "
    "identity/actions/membership, Silver and PIT revisions, coverage and gaps. "
    "Series come back as columns + rows, oldest first, capped (see each tool's limit); "
    "`truncated` means older rows exist -- narrow the window or page. Errors are JSON "
    '{"error": {"code", "message", "symbol"?, "asset_class"?, "details"?}} with the '
    "same codes as the REST API; bad arguments are invalid_parameter with "
    'details.source="arguments".'
)


def _path(name: str) -> Optional[Path]:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


def services_from_env() -> LakeServices:
    """The same lake sources REST reads, from the same env vars; unset ones stay None."""
    bronze = _path("APEX_LIVEWIRE_ROOT")
    silver = _path("APEX_LIVEWIRE_SILVER_ROOT")
    price_mode = os.environ.get("APEX_LIVEWIRE_PRICE_MODE", "raw")
    if price_mode not in ("raw", "adjusted"):
        raise ValueError(f"unsupported Livewire price mode: {price_mode!r}")
    coverage = _path("APEX_LIVEWIRE_COVERAGE_DB")
    provider = (
        LivewireOhlcProvider(
            bronze_root=bronze,
            silver_root=silver,
            price_mode=price_mode,  # type: ignore[arg-type]  # checked above
            delisted_root=_path("APEX_LIVEWIRE_DELISTED_ROOT"),
        )
        if bronze is not None
        else None
    )
    return LakeServices(
        provider=provider,
        catalog=CoverageCatalog(coverage) if coverage is not None else None,
        membership=MembershipReader.from_env(),
        reference=LivewireReferenceReader.from_env(),
        silver=RevisionManifestReader(silver) if silver is not None else None,
        pit=PitRevisionReader(silver) if silver is not None else None,
        repairs=RepairsReader.from_env(),
    )


def build_server(services: LakeServices) -> MCPServer:
    server = LakeMCPServer("apex-lake", instructions=INSTRUCTIONS)
    for group in (tools_discovery, tools_bars, tools_identity, tools_revisions):
        group.register(server, services)

    @server.custom_route(HEALTH_PATH, methods=["GET"])
    async def healthz(request: Request) -> Response:
        # Liveness only: no data, no secrets. Readiness is an authenticated tools call.
        return JSONResponse({"status": "ok"})

    return server


class BearerAuth:
    """Every request but the liveness probe needs ``Authorization: Bearer <key>``."""

    def __init__(self, app: ASGIApp, api_key: str) -> None:
        self._app = app
        self._key = api_key.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan" or (
            scope["type"] == "http" and scope["path"] == HEALTH_PATH
        ):
            await self._app(scope, receive, send)
            return
        if scope["type"] != "http":  # no websocket surface: refuse before the app sees it
            await send({"type": "websocket.close", "code": 1008})
            return
        header = dict(scope["headers"]).get(b"authorization", b"")
        scheme, _, token = header.partition(b" ")
        if scheme.lower() != b"bearer" or not secrets.compare_digest(token.strip(), self._key):
            body = {
                "error": {
                    "code": "unauthorized",
                    "message": "missing or invalid bearer token",
                }
            }
            await JSONResponse(body, status_code=401, headers={"WWW-Authenticate": "Bearer"})(
                scope, receive, send
            )
            return
        await self._app(scope, receive, send)


def _csv(name: str) -> List[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def build_app(services: LakeServices, api_key: str, allowed_hosts: List[str]) -> ASGIApp:
    """Stateless JSON Streamable HTTP behind bearer auth, with the SDK's Host/Origin
    checks on (DNS-rebinding protection); no CORS, so browsers get nothing."""
    if not api_key:
        raise ValueError("APEX_MCP_API_KEY must be set: the MCP server has no open mode")
    app = build_server(services).streamable_http_app(
        streamable_http_path=MCP_PATH,
        json_response=True,
        stateless_http=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=allowed_hosts,
            allowed_origins=[f"http://{host}" for host in allowed_hosts]
            + [f"https://{host}" for host in allowed_hosts],
        ),
    )
    return BearerAuth(app, api_key)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("APEX_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("APEX_MCP_PORT", "8333"))
    # Host header values clients may use, e.g. "apex-mini:8333,100.x.y.z:8333".
    allowed = _csv("APEX_MCP_ALLOWED_HOSTS") or [
        f"127.0.0.1:{port}",
        f"localhost:{port}",
    ]
    app = build_app(services_from_env(), os.environ.get("APEX_MCP_API_KEY", ""), allowed)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
