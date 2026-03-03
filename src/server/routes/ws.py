"""WebSocket route — /ws endpoint for real-time market data streaming."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.server.ws_hub import WebSocketHub

logger = logging.getLogger(__name__)


def create_ws_router(hub: WebSocketHub | None = None) -> APIRouter:
    """Create a router with /ws endpoint.

    The hub is resolved at request time from ``app.state.hub`` (set by lifespan).
    The *hub* parameter is only used as a fallback for unit tests that don't
    run the lifespan.
    """
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        active_hub: WebSocketHub = getattr(websocket.app.state, "hub", None) or hub  # type: ignore[assignment]
        if active_hub is None:
            await websocket.close(code=1011, reason="Hub not available")
            return
        await websocket.accept()
        active_hub.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                await active_hub.handle_command(websocket, data)
        except WebSocketDisconnect:
            active_hub.disconnect(websocket)
        except Exception:
            logger.exception("WebSocket error")
            active_hub.disconnect(websocket)

    return router
