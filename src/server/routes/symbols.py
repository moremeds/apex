"""REST routes — /api/symbols and /api/history/{symbol}."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)


def create_symbols_router(
    quote_adapter: Any = None,
    historical_adapter: Any = None,
) -> APIRouter:
    """Create router for symbol list and history endpoints.

    Dependencies can be passed directly (for tests) or resolved from
    request.app.state at request time (for production with lifespan).
    """
    router = APIRouter(prefix="/api")

    def _get_quote_adapter(request: Request) -> Any:
        return quote_adapter or getattr(request.app.state, "quote_adapter", None)

    def _get_historical_adapter(request: Request) -> Any:
        return historical_adapter or getattr(request.app.state, "historical_adapter", None)

    @router.get("/symbols")
    async def list_symbols(request: Request) -> dict:
        """List active symbols with latest quote data."""
        qa = _get_quote_adapter(request)
        symbols: Dict[str, Any] = {}

        if qa is not None:
            all_quotes = qa.get_all_quotes()
            for sym, tick in all_quotes.items():
                symbols[sym] = {
                    "symbol": sym,
                    "last": tick.last,
                    "bid": tick.bid,
                    "ask": tick.ask,
                    "volume": tick.volume,
                    "timestamp": tick.timestamp.isoformat() if tick.timestamp else None,
                    "source": tick.source,
                }

        return {"symbols": symbols, "count": len(symbols)}

    @router.get("/history/{symbol}")
    async def get_history(
        request: Request,
        symbol: str,
        tf: str = Query(default="1d", description="Timeframe (1m, 5m, 1h, 4h, 1d)"),
        bars: int = Query(default=500, ge=1, le=5000, description="Number of bars"),
    ) -> dict:
        """Get historical OHLCV bars for a symbol.

        Data resolution: Longbridge adapter → indicator engine → DuckDB → 503.
        """
        ha = _get_historical_adapter(request)
        unsupported_tf = False

        # 1. Try Longbridge adapter
        if ha is not None:
            if not ha.supports_timeframe(tf):
                unsupported_tf = True
            else:
                try:
                    now = datetime.now(timezone.utc)
                    tf_minutes = {
                        "1m": 1,
                        "5m": 5,
                        "15m": 15,
                        "30m": 30,
                        "1h": 60,
                        "4h": 240,
                        "1d": 1440,
                        "1w": 10080,
                    }
                    minutes_back = tf_minutes.get(tf, 1440) * bars * 1.5
                    start = now - timedelta(minutes=minutes_back)
                    bar_data_list = await ha.fetch_bars(symbol, tf, start, now)
                    result_bars = [
                        {
                            "t": b.timestamp.isoformat() if b.timestamp else None,
                            "o": b.open,
                            "h": b.high,
                            "l": b.low,
                            "c": b.close,
                            "v": b.volume,
                        }
                        for b in bar_data_list[-bars:]
                    ]
                    return {
                        "symbol": symbol,
                        "timeframe": tf,
                        "bars": result_bars,
                        "count": len(result_bars),
                    }
                except Exception as e:
                    logger.warning(
                        "Longbridge history failed for %s/%s: %s, trying fallbacks", symbol, tf, e
                    )

        # 2. Fallback: indicator engine bars (bootstrapped from R2)
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline:
            engine = pipeline._indicator_engine
            bar_deque = engine.get_history(symbol, tf)
            if bar_deque and len(bar_deque) > 0:
                result_bars = [
                    {
                        "t": (
                            b["timestamp"].isoformat()
                            if hasattr(b["timestamp"], "isoformat")
                            else str(b["timestamp"])
                        ),
                        "o": b["open"],
                        "h": b["high"],
                        "l": b["low"],
                        "c": b["close"],
                        "v": b["volume"],
                    }
                    for b in bar_deque[-bars:]
                ]
                return {
                    "symbol": symbol,
                    "timeframe": tf,
                    "bars": result_bars,
                    "count": len(result_bars),
                }

        # 3. Fallback: DuckDB bars
        persistence = getattr(request.app.state, "persistence", None)
        if persistence:
            db_bars = await asyncio.to_thread(persistence.query_bars, symbol, tf, bars)
            if db_bars:
                result_bars = [
                    {
                        "t": (
                            b["ts"].isoformat()
                            if hasattr(b.get("ts"), "isoformat")
                            else str(b.get("ts", ""))
                        ),
                        "o": b["o"],
                        "h": b["h"],
                        "l": b["l"],
                        "c": b["c"],
                        "v": b["v"],
                    }
                    for b in db_bars
                ]
                return {
                    "symbol": symbol,
                    "timeframe": tf,
                    "bars": result_bars,
                    "count": len(result_bars),
                }

        if unsupported_tf:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported timeframe: {tf}. Supported: {ha.get_supported_timeframes()}",
            )
        raise HTTPException(status_code=503, detail="Historical data not available")

    return router
