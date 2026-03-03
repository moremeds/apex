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
            return {}
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/backtest")
    async def get_backtest(request: Request) -> dict:
        """Get strategy comparison results (R2 with cache)."""
        data = await proxy.get_with_fallback("strategies.json", _r2_from_request(request))
        if data is None:
            return {}
        return data if isinstance(data, dict) else {"data": data}

    @router.get("/signal-data/{symbol}")
    async def get_signal_data(symbol: str, request: Request, tf: str = Query(default="1d")) -> dict:
        """Per-symbol signal data with on-demand indicator computation.

        Data resolution: cache → compute from engine → DuckDB bars → R2 fallback.
        """
        key = f"{symbol}_{tf}.json"
        data = proxy.get(key)

        # Compute from indicator engine (bars already bootstrapped from R2)
        if data is None:
            pipeline_obj = getattr(request.app.state, "pipeline", None)
            if pipeline_obj:
                data = await asyncio.to_thread(_compute_signal_data, pipeline_obj, symbol, tf)
                if data:
                    proxy.set_cache(key, data)

        # DuckDB bars fallback (if engine not ready yet)
        if data is None:
            persistence = getattr(request.app.state, "persistence", None)
            if persistence:
                bars = await asyncio.to_thread(persistence.query_bars, symbol, tf, 500)
                if bars:
                    data = _compute_signal_data_from_bars(bars, symbol, tf)
                    if data:
                        proxy.set_cache(key, data)

        # Legacy R2 fallback
        if data is None:
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
                        b
                        for b in bar_deque
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
                        for section_key in (
                            "overlays",
                            "rsi",
                            "macd",
                            "oscillators",
                            "volume_ind",
                            "dual_macd",
                            "price_levels",
                        ):
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

        Data resolution: cache → DuckDB → R2 fallback.
        Overlays live regime data from pipeline if available.
        """
        data = proxy.get("summary.json")

        # Try DuckDB summary (persisted during bootstrap)
        if data is None:
            persistence = getattr(request.app.state, "persistence", None)
            if persistence:
                try:
                    data = await asyncio.to_thread(persistence.get_summary)
                    if data:
                        proxy.set_cache("summary.json", data)
                except Exception:
                    logger.debug("DuckDB summary read failed", exc_info=True)

        # R2 fallback
        if data is None:
            data = await proxy.get_with_fallback("summary.json", _r2_from_request(request))
        if data is None:
            raise HTTPException(status_code=503, detail="Summary data not available")
        if not isinstance(data, dict):
            data = {"data": data}

        # Deep-copy before mutation to avoid contaminating the cache
        data = copy.deepcopy(data)

        # Overlay prices from R2-bootstrapped + gap-filled indicator engine.
        from src.server.main import _get_prev_close

        pipeline = getattr(request.app.state, "pipeline", None)
        prev_close_map: dict[str, float] = {}
        if pipeline is not None:
            try:
                engine = pipeline._indicator_engine
                for ticker in data.get("tickers", []):
                    sym = ticker.get("symbol", "")
                    bar_deque = engine.get_history(sym, "1d")
                    close = bar_deque[-1]["close"] if bar_deque else 0.0
                    prev_close = _get_prev_close(engine, sym)

                    if prev_close and prev_close > 0:
                        prev_close_map[sym] = prev_close
                        ticker["prev_close"] = prev_close
                    if close > 0 and prev_close and prev_close > 0:
                        ticker["close"] = close
                        ticker["daily_change_pct"] = round(
                            ((close - prev_close) / prev_close) * 100, 2
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
                            # Use prev_close from bar history (correct last trading day close)
                            prev_close = prev_close_map.get(sym)
                            ticker["close"] = quote.last
                            if prev_close and prev_close > 0:
                                ticker["prev_close"] = prev_close
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
        """Indicator definitions + rules.

        Data resolution: cache → compute from engine → R2 fallback → 503.
        """
        data = proxy.get("indicators.json")

        # Compute from engine if no cache
        if data is None:
            pipeline = getattr(request.app.state, "pipeline", None)
            if pipeline:
                data = await asyncio.to_thread(_build_indicators_metadata, pipeline)
                if data:
                    proxy.set_cache("indicators.json", data)

        # R2 fallback
        if data is None:
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


def _format_ts(ts: Any) -> str:
    """Format a timestamp to ISO string."""
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


def _compute_signal_data(pipeline: Any, symbol: str, tf: str) -> dict | None:
    """Compute signal chart data from indicator engine bars + all indicators."""
    import pandas as pd

    from src.server.pipeline import STRATEGY_INDICATORS, _map_regime_to_flex

    engine = pipeline._indicator_engine
    bar_deque = engine.get_history(symbol, tf)
    if not bar_deque or len(bar_deque) < 10:
        return None

    df = pd.DataFrame(bar_deque)

    # Build base chart_data with OHLCV
    timestamps = [_format_ts(b["timestamp"]) for b in bar_deque]
    chart_data: dict[str, Any] = {
        "timestamps": timestamps,
        "open": df["open"].tolist(),
        "high": df["high"].tolist(),
        "low": df["low"].tolist(),
        "close": df["close"].tolist(),
        "volume": df["volume"].tolist(),
        "overlays": {},
        "rsi": {},
        "macd": {},
        "dual_macd": {},
        "oscillators": {},
        "volume_ind": {},
        "price_levels": {},
    }

    # Strategy history accumulators
    dual_macd_history: list[dict] = []
    trend_pulse_history: list[dict] = []
    regime_flex_history: list[dict] = []

    # Run ALL registered indicators on the bar DataFrame
    for indicator in engine._indicators:
        try:
            if len(df) < indicator.warmup_periods:
                continue
            result_df = indicator.calculate(df, indicator.default_params)
            if result_df is None or result_df.empty:
                continue
            ind_name = indicator.name
            for col in result_df.columns:
                if col in ("open", "high", "low", "close", "volume", "timestamp"):
                    continue
                values = [None if pd.isna(v) else round(float(v), 4) for v in result_df[col]]
                full_key = f"{ind_name}_{col}"
                _place_indicator_in_chart(chart_data, ind_name, full_key, values)

            # Extract per-bar strategy states for history tables
            if ind_name in STRATEGY_INDICATORS:
                warmup = max(indicator.warmup_periods, 1)
                for i in range(warmup, len(result_df)):
                    try:
                        current = result_df.iloc[i]
                        previous = result_df.iloc[i - 1]
                        state = indicator.get_state(current, previous, indicator.default_params)
                        if not state:
                            continue
                        ts = bar_deque[i].get("timestamp")
                        date_str = _format_ts(ts) if ts else ""
                        state["date"] = date_str

                        if ind_name == "dual_macd":
                            dual_macd_history.append(state)
                        elif ind_name == "trend_pulse":
                            trend_pulse_history.append(state)
                        elif ind_name == "regime_detector":
                            regime_flex_history.append(_map_regime_to_flex(state))
                    except Exception:
                        continue
        except Exception:
            continue

    # Query persisted signals
    signals: list[dict] = []
    persistence = getattr(pipeline, "_persistence", None)
    if persistence:
        try:
            raw = persistence.query_signals(symbol=symbol, timeframe=tf, limit=200)
            signals = [
                {
                    "timestamp": _format_ts(s.get("ts", "")),
                    "rule": s.get("rule", ""),
                    "direction": s.get("direction", ""),
                    "indicator": s.get("indicator", ""),
                }
                for s in raw
            ]
        except Exception:
            pass

    return {
        "symbol": symbol,
        "timeframe": tf,
        "bar_count": len(bar_deque),
        "chart_data": chart_data,
        "signals": signals,
        "dual_macd_history": list(reversed(dual_macd_history[-50:])),
        "trend_pulse_history": list(reversed(trend_pulse_history[-50:])),
        "regime_flex_history": list(reversed(regime_flex_history[-50:])),
    }


def _compute_signal_data_from_bars(bars: list[dict], symbol: str, tf: str) -> dict | None:
    """Compute signal data from DuckDB bar rows (fallback when engine empty)."""
    if not bars or len(bars) < 10:
        return None

    timestamps = [_format_ts(b.get("ts") or b.get("timestamp", "")) for b in bars]
    chart_data: dict[str, Any] = {
        "timestamps": timestamps,
        "open": [b.get("o", b.get("open", 0)) for b in bars],
        "high": [b.get("h", b.get("high", 0)) for b in bars],
        "low": [b.get("l", b.get("low", 0)) for b in bars],
        "close": [b.get("c", b.get("close", 0)) for b in bars],
        "volume": [b.get("v", b.get("volume", 0)) for b in bars],
        "overlays": {},
        "rsi": {},
        "macd": {},
        "dual_macd": {},
        "oscillators": {},
        "volume_ind": {},
        "price_levels": {},
    }
    return {
        "symbol": symbol,
        "timeframe": tf,
        "bar_count": len(bars),
        "chart_data": chart_data,
        "signals": [],
    }


def _place_indicator_in_chart(chart_data: dict, ind_name: str, full_key: str, values: list) -> None:
    """Route indicator values to the right chart section."""
    if ind_name in ("ema", "sma", "bollinger", "supertrend", "ichimoku", "keltner"):
        chart_data["overlays"][full_key] = values
    elif ind_name == "rsi":
        chart_data["rsi"][full_key] = values
    elif ind_name == "macd":
        chart_data["macd"][full_key] = values
    elif ind_name == "dual_macd":
        chart_data["dual_macd"][full_key] = values
    elif ind_name in ("atr", "adx", "stochastic", "cci", "williams_r", "awesome"):
        chart_data["oscillators"][full_key] = values
    elif ind_name in ("obv", "mfi", "vwap", "cmf"):
        chart_data["volume_ind"][full_key] = values
    else:
        chart_data["oscillators"][full_key] = values


def _build_indicators_metadata(pipeline: Any) -> dict | None:
    """Build indicator metadata from registered indicators in the engine.

    Returns format expected by IndicatorsSection:
    {"categories": [{"name": "Overlay", "indicators": [{"name", "description", "params", "warmup_periods", "rules"}]}]}
    """
    engine = pipeline._indicator_engine
    if not hasattr(engine, "_indicators") or not engine._indicators:
        return None

    _OVERLAY = {"ema", "sma", "bollinger", "supertrend", "ichimoku", "keltner", "vwap"}
    _OSCILLATOR = {
        "rsi",
        "macd",
        "dual_macd",
        "stochastic",
        "cci",
        "williams_r",
        "awesome",
        "rsi_harmonics",
        "kdj",
        "momentum",
        "roc",
        "aroon",
        "trix",
        "vortex",
        "zerolag",
    }
    _VOLUME = {"obv", "mfi", "cmf", "cvd", "ad", "volume_ratio"}
    _TREND = {"adx", "regime_detector"}
    _VOLATILITY = {"atr", "bollinger", "squeeze", "hvol", "chaikin_vol"}
    _PATTERN = {"fibonacci", "pivot", "candlestick", "chart_patterns", "support_resistance"}

    def _classify(name: str) -> str:
        if name in _OVERLAY:
            return "Overlay"
        if name in _OSCILLATOR:
            return "Oscillator"
        if name in _VOLUME:
            return "Volume"
        if name in _TREND:
            return "Trend"
        if name in _VOLATILITY:
            return "Volatility"
        if name in _PATTERN:
            return "Pattern"
        return "Other"

    categories_map: dict[str, list] = {}
    for ind in engine._indicators:
        name = getattr(ind, "name", str(ind))
        params = getattr(ind, "default_params", {})
        warmup = getattr(ind, "warmup_periods", 0)
        cat = _classify(name)

        if cat not in categories_map:
            categories_map[cat] = []
        categories_map[cat].append(
            {
                "name": name,
                "description": f"{name} indicator",
                "params": {k: v for k, v in params.items() if not k.startswith("_")},
                "warmup_periods": warmup,
                "rules": [],
            }
        )

    category_order = ["Overlay", "Oscillator", "Volume", "Trend", "Volatility", "Pattern", "Other"]
    categories = [
        {"name": cat_name, "indicators": categories_map[cat_name]}
        for cat_name in category_order
        if cat_name in categories_map
    ]

    return {"categories": categories}
