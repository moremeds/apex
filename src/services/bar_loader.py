"""Unified historical bar loader with configurable source priority.

Single entry point for all offline/batch bar loading in APEX runners.
Reads source_priority from config/base.yaml and tries each source in order,
falling back on failure.

Supported sync sources: "fmp", "yahoo".
"ib" is async-only (requires live IB Gateway) and not available here;
use src.services.historical_data_service.HistoricalDataService for IB.

Usage:
    from src.services.bar_loader import load_bars

    bars = load_bars(["AAPL", "MSFT"], timeframe="1d", days=500)
    # bars["AAPL"] -> pd.DataFrame with columns: open, high, low, close, volume
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "base.yaml"

# Sources that require an async event loop / live connection and cannot
# be used from synchronous runner code.
_ASYNC_ONLY_SOURCES = frozenset({"ib"})


def _read_config() -> dict[str, Any]:
    """Read historical_data section from config/base.yaml."""
    try:
        with open(_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        result: dict[str, Any] = cfg.get("historical_data", {})
        return result
    except Exception as e:
        logger.warning(f"Failed to read config: {e}")
        return {}


def _read_source_priority() -> list[str]:
    """Read historical_data.source_priority from config, filtered to enabled."""
    hist = _read_config()
    priority: list[str] = hist.get("source_priority", ["fmp", "yahoo"])
    sources_cfg = hist.get("sources", {})
    return [s for s in priority if sources_cfg.get(s, {}).get("enabled", True)]


def _get_rate_delay(source: str) -> float:
    """Read per-request delay from config for a source (seconds).

    Converts rate_limit_per_sec to a sleep interval.
    Falls back to sensible defaults if config is missing.
    """
    defaults = {"fmp": 0.3, "yahoo": 1.0}
    hist = _read_config()
    source_cfg = hist.get("sources", {}).get(source, {})
    rate = source_cfg.get("rate_limit_per_sec", None)
    if rate and rate > 0:
        return 1.0 / float(rate)
    return defaults.get(source, 0.0)


# ── Source fetchers ─────────────────────────────────────────────


def _fetch_fmp(
    symbols: list[str],
    timeframe: str,
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    """Fetch bars from FMP. Respects config rate_limit_per_sec."""
    try:
        from src.infrastructure.adapters.fmp.historical_adapter import FMPHistoricalAdapter

        delay = _get_rate_delay("fmp")
        adapter = FMPHistoricalAdapter(request_delay=delay)
        return adapter.fetch_bars_batch(symbols, timeframe, start_date, end_date)
    except (ValueError, ImportError) as e:
        logger.debug(f"FMP unavailable: {e}")
        return {}
    except Exception as e:
        logger.warning(f"FMP fetch failed: {e}")
        return {}


def _fetch_yahoo(
    symbols: list[str],
    timeframe: str,
    start_date: date,
    end_date: date,
) -> dict[str, pd.DataFrame]:
    """Fetch bars from Yahoo Finance using batch download.

    Uses yf.download() for multi-symbol batch fetching (single HTTP request
    per batch of ~500 symbols) instead of per-symbol requests.

    yfinance returns split- and dividend-adjusted prices by default
    (auto_adjust=True since yfinance 0.2.x).
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not installed")
        return {}

    interval_map = {
        "1d": "1d",
        "1w": "1wk",
        "4h": "1h",
        "2h": "1h",
        "1h": "1h",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1m": "1m",
    }
    interval = interval_map.get(timeframe, "1d")

    result: dict[str, pd.DataFrame] = {}

    # Batch download — yf.download handles multiple tickers in one call
    try:
        raw = yf.download(
            symbols,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval=interval,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        if raw.empty:
            return {}

        # yf.download returns different shapes for 1 vs N symbols
        if len(symbols) == 1:
            sym = symbols[0]
            df = raw.copy()
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            for col in ["dividends", "stock splits", "adj close"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            df = df.dropna(subset=["close"])
            if not df.empty:
                result[sym] = _maybe_resample(df, timeframe, interval)
        else:
            for sym in symbols:
                try:
                    if sym not in raw.columns.get_level_values(0):
                        continue
                    df = raw[sym].copy()
                    df.columns = [c.lower() for c in df.columns]
                    for col in ["dividends", "stock splits", "adj close"]:
                        if col in df.columns:
                            df = df.drop(columns=[col])
                    df = df.dropna(subset=["close"])
                    if not df.empty:
                        result[sym] = _maybe_resample(df, timeframe, interval)
                except Exception as e:
                    logger.debug(f"Yahoo parse failed for {sym}: {e}")

    except Exception as e:
        logger.warning(f"Yahoo batch download failed: {e}")

    return result


def _maybe_resample(df: pd.DataFrame, timeframe: str, interval: str) -> pd.DataFrame:
    """Aggregate intraday bars if the target timeframe differs from download interval."""
    if timeframe in ("4h", "2h") and interval == "1h":
        df = (
            df.resample(timeframe)
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
    return df


# ── Public API ──────────────────────────────────────────────────

# Registry of synchronous fetchers. "ib" is deliberately absent because
# it requires an async event loop + live IB Gateway connection.
_SOURCE_FETCHERS: dict[str, Callable[[list[str], str, date, date], dict[str, pd.DataFrame]]] = {
    "fmp": _fetch_fmp,
    "yahoo": _fetch_yahoo,
}


def load_bars(
    symbols: list[str],
    timeframe: str = "1d",
    days: int = 500,
    end_date: Optional[date] = None,
    source_priority: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Load historical bars using configured source priority.

    Reads priority from config/base.yaml -> historical_data.source_priority
    unless overridden. Falls back to next source on failure.

    Supported sync sources: "fmp", "yahoo".
    "ib" is logged and skipped (it requires an async IB Gateway connection;
    use HistoricalDataService for IB data).

    Args:
        symbols: List of ticker symbols to load.
        timeframe: Bar timeframe (1d, 4h, 1h, 5m, etc.).
        days: Number of calendar days of history.
        end_date: End date (default: today).
        source_priority: Override config priority (e.g. ["yahoo"] for yahoo-only).

    Returns:
        Dict mapping symbol -> OHLCV DataFrame (open, high, low, close, volume).
        Prices are adjusted (split + dividend) for both FMP and Yahoo sources.
    """
    if not symbols:
        return {}

    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=days)

    priority = source_priority or _read_source_priority()
    result: Dict[str, pd.DataFrame] = {}
    remaining = list(symbols)

    for source in priority:
        if not remaining:
            break

        if source in _ASYNC_ONLY_SOURCES:
            logger.warning(
                f"Source '{source}' requires an async connection and cannot be used "
                f"in load_bars(). Use HistoricalDataService for '{source}' data. "
                f"Skipping to next source."
            )
            continue

        fetcher = _SOURCE_FETCHERS.get(source)
        if fetcher is None:
            logger.warning(f"Unknown source '{source}' in source_priority — skipping.")
            continue

        try:
            fetched = fetcher(remaining, timeframe, start_date, end_date)
            if fetched:
                result.update(fetched)
                remaining = [s for s in remaining if s not in fetched]
                logger.info(
                    f"[{source}] fetched {len(fetched)} symbols, " f"{len(remaining)} remaining"
                )
        except Exception as e:
            logger.warning(f"[{source}] failed: {e}")

    if remaining:
        logger.warning(f"No data for {len(remaining)} symbols: {remaining[:10]}")

    return result
