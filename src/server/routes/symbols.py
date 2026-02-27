"""REST routes — /api/symbols and /api/history/{symbol}."""

from __future__ import annotations

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
        """Get historical OHLCV bars for a symbol."""
        ha = _get_historical_adapter(request)
        if ha is None:
            raise HTTPException(status_code=503, detail="Historical data not available")

        if not ha.supports_timeframe(tf):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported timeframe: {tf}. Supported: {ha.get_supported_timeframes()}",
            )

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

        try:
            bar_data_list = await ha.fetch_bars(symbol, tf, start, now)
        except Exception as e:
            logger.error("Failed to fetch history for %s/%s: %s", symbol, tf, e)
            raise HTTPException(status_code=502, detail=f"Failed to fetch history: {e}")

        result_bars = []
        for b in bar_data_list[-bars:]:
            result_bars.append(
                {
                    "t": b.timestamp.isoformat() if b.timestamp else None,
                    "o": b.open,
                    "h": b.high,
                    "l": b.low,
                    "c": b.close,
                    "v": b.volume,
                }
            )

        return {
            "symbol": symbol,
            "timeframe": tf,
            "bars": result_bars,
            "count": len(result_bars),
        }

    return router
