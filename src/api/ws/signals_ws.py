"""WS endpoint: argon subscribes to tickers; apex pushes signal payloads.

Frame protocol (argon -> apex):
    {"action": "subscribe",   "ticker": "AAPL"}
    {"action": "unsubscribe", "ticker": "AAPL"}
apex -> argon: ack {"status": ..., "ticker": ...}, then an initial snapshot payload,
then live signal_service_payload frames as signals fire.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.payload.builder import build_payload
from src.api.payload.validate import validate_payload

router = APIRouter()


@router.websocket("/ws/signals")
async def signals_ws(ws: WebSocket) -> None:
    await ws.accept()
    hub = ws.app.state.signal_hub
    mgr = ws.app.state.subscription_manager
    repo = getattr(ws.app.state, "signal_repo", None)
    try:
        while True:
            msg = await ws.receive_json()
            ticker = msg.get("ticker", "")
            action = msg.get("action")
            if action == "subscribe" and ticker:
                hub.register(ws, ticker)
                await mgr.subscribe(ticker)
                await ws.send_json({"status": "subscribed", "ticker": ticker})
                # Initial snapshot so argon can render immediately (spec 3.1).
                # MVP: recent persisted signals. NOTE: enriching this with the full
                # historical bars + indicator series requires an indicator-snapshot
                # API on TASignalService -- deferred.
                if repo is not None:
                    rows = await repo.fetch_signals(ticker)
                    snapshot = build_payload(rows, generated_at=datetime.now(timezone.utc))
                    validate_payload(snapshot)
                    await ws.send_json(snapshot)
            elif action == "unsubscribe" and ticker:
                # Decrement the manager once per ticker actually removed from the hub.
                for removed in hub.unregister(ws, ticker):
                    await mgr.unsubscribe(removed)
                await ws.send_json({"status": "unsubscribed", "ticker": ticker})
            else:
                await ws.send_json({"status": "error", "detail": "bad frame"})
    except WebSocketDisconnect:
        # Decrement EVERY ticker the socket still held (no refcount leak).
        for removed in hub.unregister(ws):
            await mgr.unsubscribe(removed)
