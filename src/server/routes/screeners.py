"""REST routes — /api/screeners and /api/backtest (R2 proxy with caching).

Data resolution order:
1. In-memory cache (5 min TTL)
2. R2 (Cloudflare storage) if credentials configured
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)


class _CachedProxy:
    """TTL cache with R2 as the sole data source."""

    # R2 stores these files under meta/ prefix
    _R2_KEY_MAP: dict[str, str] = {
        "universe.json": "meta/universe.json",
        "data_quality.json": "meta/data_quality.json",
    }

    def __init__(self, r2_client: Any, ttl_sec: int = 300):
        self._r2 = r2_client
        self._ttl = ttl_sec
        self._cache: dict[str, tuple[float, Any]] = {}  # key → (expires_at, data)

    def get(self, key: str, r2_override: Any = None) -> Any:
        now = time.monotonic()
        if key in self._cache:
            expires_at, data = self._cache[key]
            if now < expires_at:
                return data

        r2 = r2_override or self._r2
        if r2 is not None:
            try:
                data = r2.get_json(key)
                if data is not None:
                    self._cache[key] = (now + self._ttl, data)
                    return data
            except Exception as e:
                logger.error("R2 fetch failed for %s: %s", key, e)
                # Return stale cache if available
                if key in self._cache:
                    return self._cache[key][1]

        return None

    def set_cache(self, key: str, data: Any) -> None:
        """Inject data into cache with standard TTL."""
        self._cache[key] = (time.monotonic() + self._ttl, data)

    def merge_cache(self, key: str, partial: dict) -> None:
        """Merge partial data into existing cached dict.

        Useful for updating only the 'momentum' section of screeners.json
        without overwriting the 'pead' section.
        """
        existing = self.get(key) or {}
        if not isinstance(existing, dict):
            existing = {}
        self.set_cache(key, {**existing, **partial})

    async def get_with_fallback(self, key: str, r2_override: Any = None) -> Any:
        """Try cache → R2 → stale cache."""
        # 1. Cache
        now = time.monotonic()
        if key in self._cache:
            expires_at, data = self._cache[key]
            if now < expires_at:
                return data

        # 2. R2 (some files live under meta/ prefix in R2)
        r2 = r2_override or self._r2
        r2_key = self._R2_KEY_MAP.get(key, key)
        if r2 is not None:
            try:
                data = await asyncio.to_thread(r2.get_json, r2_key)
                if data is not None:
                    self._cache[key] = (now + self._ttl, data)
                    return data
            except Exception as e:
                logger.error("R2 fetch failed for %s: %s", key, e)

        # 3. Stale cache as last resort
        if key in self._cache:
            return self._cache[key][1]

        return None


def create_screeners_router(
    r2_client: Any = None,
    cache_ttl: int = 300,
    proxy: _CachedProxy | None = None,
) -> APIRouter:
    """Create router for screener and backtest proxy endpoints.

    Args:
        r2_client: R2Client instance for fetching JSON from Cloudflare R2.
        cache_ttl: Cache TTL in seconds (default 5 minutes).
        proxy: Optional pre-created proxy (for sharing with jobs route via app.state).
    """
    router = APIRouter(prefix="/api")
    if proxy is None:
        proxy = _CachedProxy(r2_client, ttl_sec=cache_ttl)

    def _r2_from_request(request: Request) -> Any:
        return getattr(request.app.state, "r2_client", None)

    @router.get("/screeners")
    async def get_screeners(request: Request) -> dict:
        """Get screener results (R2 with cache)."""
        data = await proxy.get_with_fallback("screeners.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Screener data not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/backtest")
    async def get_backtest(request: Request) -> dict:
        """Get strategy comparison results (R2 with cache)."""
        data = await proxy.get_with_fallback("strategies.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Backtest data not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/signal-data/{symbol}")
    async def get_signal_data(symbol: str, request: Request, tf: str = Query(default="1d")) -> dict:
        """Per-symbol signal data (R2 with cache), extended with fresh bars."""
        key = f"{symbol}_{tf}.json"
        data = await proxy.get_with_fallback(key, _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=404, detail=f"No signal data for {symbol}/{tf}")
        if not isinstance(data, dict):
            data = {"data": data}

        # Deep-copy before mutation to avoid contaminating the cache
        data = copy.deepcopy(data)

        # Extend chart with fresh bars from indicator engine
        pipeline_obj = getattr(request.app.state, "pipeline", None)
        chart_data = data.get("chart_data")
        if pipeline_obj is not None and chart_data and chart_data.get("timestamps"):
            engine = pipeline_obj._indicator_engine
            bar_deque = engine.get_history(symbol, tf)
            if bar_deque:
                last_ts_str = chart_data["timestamps"][-1]
                from datetime import datetime

                try:
                    last_ts = datetime.fromisoformat(last_ts_str.replace("Z", "+00:00"))
                    if last_ts.tzinfo:
                        last_ts = last_ts.replace(tzinfo=None)
                except Exception:
                    last_ts = None

                if last_ts is not None:
                    new_bars = [
                        b for b in bar_deque
                        if b.get("timestamp") is not None and b["timestamp"] > last_ts
                    ]
                    for bar in new_bars:
                        ts = bar["timestamp"]
                        chart_data["timestamps"].append(
                            ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                        )
                        chart_data["open"].append(bar["open"])
                        chart_data["high"].append(bar["high"])
                        chart_data["low"].append(bar["low"])
                        chart_data["close"].append(bar["close"])
                        chart_data["volume"].append(bar["volume"])
                        # Append null for all indicator sub-arrays
                        for section_key in ("overlays", "rsi", "macd", "oscillators", "volume_ind", "dual_macd", "price_levels"):
                            section = chart_data.get(section_key, {})
                            if isinstance(section, dict):
                                for _ind_name, ind_data in section.items():
                                    if isinstance(ind_data, list):
                                        ind_data.append(None)
                                    elif isinstance(ind_data, dict):
                                        for _sub_key, sub_arr in ind_data.items():
                                            if isinstance(sub_arr, list):
                                                sub_arr.append(None)
                    if new_bars:
                        data["bar_count"] = len(chart_data["timestamps"])

        return data

    @router.get("/summary")
    async def get_summary(request: Request) -> dict:
        """Summary.json — ETF data, regime, generated_at (R2 with cache).

        Overlays live regime data from pipeline if available.
        """
        data = await proxy.get_with_fallback("summary.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Summary data not available")
        if not isinstance(data, dict):
            data = {"data": data}

        # Deep-copy before mutation to avoid contaminating the cache
        data = copy.deepcopy(data)

        # Overlay prices from R2-bootstrapped + gap-filled indicator engine.
        # Track prev_close per symbol for correct daily_change_pct computation.
        pipeline = getattr(request.app.state, "pipeline", None)
        prev_close_map: dict[str, float] = {}
        if pipeline is not None:
            try:
                engine = pipeline._indicator_engine
                for ticker in data.get("tickers", []):
                    sym = ticker.get("symbol", "")
                    bar_deque = engine.get_history(sym, "1d")
                    if bar_deque and len(bar_deque) >= 2:
                        latest = bar_deque[-1]
                        prev = bar_deque[-2]
                        latest_close = latest["close"]
                        prev_close = prev["close"]
                        if prev_close > 0:
                            prev_close_map[sym] = prev_close
                        if latest_close > 0 and prev_close > 0:
                            ticker["close"] = latest_close
                            ticker["daily_change_pct"] = round(
                                ((latest_close - prev_close) / prev_close) * 100, 2
                            )
            except Exception:
                pass

        # Overlay live prices from quote adapter (takes precedence over R2).
        # Only during market hours — on weekends/holidays Longbridge returns stale data.
        from datetime import datetime, timezone

        qa = getattr(request.app.state, "quote_adapter", None)
        if qa is not None:
            now_utc = datetime.now(timezone.utc)
            is_weekday = now_utc.weekday() < 5
            if is_weekday:
                try:
                    for ticker in data.get("tickers", []):
                        sym = ticker.get("symbol", "")
                        quote = (
                            qa.get_latest_quote(sym) if hasattr(qa, "get_latest_quote") else None
                        )
                        if quote and hasattr(quote, "last") and quote.last:
                            # Use prev_close from bar history (not from ticker dict,
                            # which may already be overlaid with today's close)
                            prev_close = prev_close_map.get(sym)
                            ticker["close"] = quote.last
                            if prev_close and prev_close > 0:
                                ticker["daily_change_pct"] = round(
                                    ((quote.last - prev_close) / prev_close) * 100, 2
                                )
                            if hasattr(quote, "timestamp") and quote.timestamp:
                                ticker["timestamp"] = quote.timestamp.isoformat()
                except Exception:
                    pass

        # Overlay live regime from pipeline indicator engine.
        # get_regime_states() now includes composite_score — avoids a second
        # expensive get_all_indicator_states() call per request.
        if pipeline is not None:
            try:
                regimes = pipeline.get_regime_states("1d")
                if regimes:
                    for ticker in data.get("tickers", []):
                        sym = ticker.get("symbol", "")
                        r = regimes.get(sym)
                        if r:
                            ticker["regime"] = r["regime"]
                            ticker["regime_name"] = r["regime_name"]
                            ticker["confidence"] = r["confidence"]
                            cs = r.get("composite_score")
                            if cs is not None:
                                ticker["composite_score_avg"] = round(cs, 1)
                    # Also overlay market benchmarks
                    market = data.get("market", {})
                    for bench_sym, bench_data in market.get("benchmarks", {}).items():
                        r = regimes.get(bench_sym)
                        if r:
                            bench_data["regime"] = r["regime"]
                            bench_data["confidence"] = r["confidence"]
            except Exception:
                pass  # Fall back to cached regime data

        # Recompute regime counts from overlaid ticker data
        counts: dict[str, int] = {}
        for ticker in data.get("tickers", []):
            r = ticker.get("regime")
            if r:
                counts[r] = counts.get(r, 0) + 1
        if counts:
            data["regime_counts"] = counts

        return data

    @router.get("/score-history")
    async def get_score_history(request: Request) -> dict:
        """Score history for sparklines (DuckDB live → R2 fallback)."""
        persistence = getattr(request.app.state, "persistence", None)
        if persistence:
            try:
                data = await asyncio.to_thread(persistence.get_score_history)
                if data and data.get("snapshots"):
                    return data
            except Exception:
                logger.debug("DuckDB score history failed, falling back to R2", exc_info=True)
        data = await proxy.get_with_fallback("score_history.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Score history not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/indicators")
    async def get_indicators(request: Request) -> dict:
        """Indicator definitions + rules (R2 with cache)."""
        data = await proxy.get_with_fallback("indicators.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Indicators data not available")
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/universe")
    async def get_universe(request: Request) -> dict:
        """Universe.json — tier, sector enrichment (R2 with cache)."""
        data = await proxy.get_with_fallback("universe.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Universe data not available")
        return data if isinstance(data, dict) else {"data": data}

    return router
