"""REST routes — /api/symbols and /api/history/{symbol}."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)


def create_symbols_router(
    quote_adapter: Any = None,
    historical_adapter: Any = None,
    pipeline: Any = None,
) -> APIRouter:
    """Create router for symbol list and history endpoints.

    Args:
        quote_adapter: QuoteProvider instance for latest quotes.
        historical_adapter: HistoricalSourcePort for historical bars.
        pipeline: ServerPipeline for indicator state.
    """
    router = APIRouter(prefix="/api")

    @router.get("/symbols")
    async def list_symbols() -> dict:
        """List active symbols with latest quote data."""
        symbols: Dict[str, Any] = {}

        if quote_adapter is not None:
            all_quotes = quote_adapter.get_all_quotes()
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
        else:
            # No adapter — return subscribed symbols from config if available
            pass

        return {"symbols": symbols, "count": len(symbols)}

    @router.get("/history/{symbol}")
    async def get_history(
        symbol: str,
        tf: str = Query(default="1d", description="Timeframe (1m, 5m, 1h, 4h, 1d)"),
        bars: int = Query(default=500, ge=1, le=5000, description="Number of bars"),
    ) -> dict:
        """Get historical OHLCV bars for a symbol."""
        if historical_adapter is None:
            raise HTTPException(status_code=503, detail="Historical data not available")

        if not historical_adapter.supports_timeframe(tf):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported timeframe: {tf}. Supported: {historical_adapter.get_supported_timeframes()}",
            )

        # Fetch bars — use a wide date range, adapter will limit
        now = datetime.now(timezone.utc)
        # Estimate how far back we need based on timeframe and bar count
        from datetime import timedelta

        tf_minutes = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
        }
        minutes_back = tf_minutes.get(tf, 1440) * bars * 1.5  # 1.5x for market gaps
        start = now - timedelta(minutes=minutes_back)

        try:
            bar_data_list = await historical_adapter.fetch_bars(symbol, tf, start, now)
        except Exception as e:
            logger.error("Failed to fetch history for %s/%s: %s", symbol, tf, e)
            raise HTTPException(status_code=502, detail=f"Failed to fetch history: {e}")

        # Convert BarData → dicts, take last N bars
        result_bars = []
        for b in bar_data_list[-bars:]:
            result_bars.append({
                "t": b.timestamp.isoformat() if b.timestamp else None,
                "o": b.open,
                "h": b.high,
                "l": b.low,
                "c": b.close,
                "v": b.volume,
            })

        return {
            "symbol": symbol,
            "timeframe": tf,
            "bars": result_bars,
            "count": len(result_bars),
        }

    return router
