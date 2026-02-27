"""WebSocket route — /ws endpoint for real-time market data streaming."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.server.ws_hub import WebSocketHub

logger = logging.getLogger(__name__)


def create_ws_router(hub: WebSocketHub) -> APIRouter:
    """Create a router with /ws endpoint bound to the given hub."""
    router = APIRouter()

    @router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        hub.connect(websocket)
        try:
            while True:
                data = await websocket.receive_json()
                await hub.handle_command(websocket, data)
        except WebSocketDisconnect:
            hub.disconnect(websocket)
        except Exception:
            logger.exception("WebSocket error")
            hub.disconnect(websocket)

    return router
