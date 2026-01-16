"""
BarPreloader - Historical bar loading for indicator warmup.

Handles:
- Startup preload: Load historical bars from Parquet, inject into IndicatorEngine
- Periodic refresh: Update disk cache without injection (live bars handle that)

Extracted from SignalCoordinator for single responsibility.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ....utils.logging_setup import get_logger
from ....utils.timezone import now_utc

if TYPE_CHECKING:
    from ....domain.signals import IndicatorEngine

logger = get_logger(__name__)


class BarPreloader:
    """
    Preloads historical bars for indicator warmup.

    Startup flow:
    1. Query HistoricalDataManager for bars (gap detection + download)
    2. Convert bars to dict format
    3. Inject into IndicatorEngine for warmup
    4. Compute indicators on history

    Refresh flow:
    1. Query HistoricalDataManager (updates Parquet files)
    2. No injection - live BAR_CLOSE events handle new bars
    """

    __slots__ = (
        '_historical_data_manager',
        '_indicator_engine',
        '_timeframes',
        '_preload_config',
        '_last_cache_refresh',
    )

    def __init__(
        self,
        historical_data_manager: Any,
        indicator_engine: "IndicatorEngine",
        timeframes: List[str],
        preload_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize bar preloader.

        Args:
            historical_data_manager: Manager for historical bar data (Parquet/DuckDB).
            indicator_engine: Engine to inject bars into for warmup.
            timeframes: List of timeframes to preload.
            preload_config: Configuration dict with:
                - lookback_days: Default lookback for periodic refresh (default: 365)
                - cache_refresh_hours: Hours between refreshes (default: 24)
                - slow_preload_warn_sec: Warn threshold (default: 30)

        Note:
            Startup preload uses get_max_history_days() per timeframe (source-limited),
            while periodic refresh uses lookback_days (config-limited).
        """
        self._historical_data_manager = historical_data_manager
        self._indicator_engine = indicator_engine
        self._timeframes = timeframes
        self._preload_config = preload_config or {}
        self._last_cache_refresh: Optional[datetime] = None

    @property
    def last_cache_refresh(self) -> Optional[datetime]:
        """When the cache was last refreshed."""
        return self._last_cache_refresh

    async def preload_startup(self, symbols: List[str]) -> Dict[str, int]:
        """
        Preload historical bars on STARTUP.

        Performs gap detection via DuckDB, downloads missing data from IB/Yahoo,
        stores to Parquet, and injects into IndicatorEngine for warmup.

        Args:
            symbols: List of symbols to preload (e.g., ["AAPL", "TSLA"])

        Returns:
            Dict mapping symbol -> number of bars injected
        """
        if not self._indicator_engine:
            logger.warning("Cannot preload bars: indicator engine not initialized")
            return {}

        if not self._historical_data_manager:
            logger.warning("Cannot preload bars: historical data manager not configured")
            return {}

        if not symbols:
            logger.debug("No symbols to preload")
            return {}

        slow_warn_sec = self._preload_config.get("slow_preload_warn_sec", 30)
        end_dt = now_utc()

        results: Dict[str, int] = {}
        start_time = time.monotonic()

        # Build timeframe-specific lookback based on source limits
        timeframe_lookbacks = {}
        for tf in self._timeframes:
            max_days = self._historical_data_manager.get_max_history_days(tf)
            timeframe_lookbacks[tf] = max_days

        logger.info(
            "Starting bar cache preload (startup) - max history per timeframe",
            extra={
                "symbols_count": len(symbols),
                "symbols": symbols[:10] if len(symbols) > 10 else symbols,
                "timeframes": self._timeframes,
                "timeframe_lookbacks": timeframe_lookbacks,
                "end": end_dt.isoformat(),
            },
        )

        failed_symbols: list[tuple[str, str, str]] = []  # (symbol, timeframe, error)

        for timeframe in self._timeframes:
            lookback_days = timeframe_lookbacks.get(timeframe, 365)
            start_dt = end_dt - timedelta(days=lookback_days)

            for symbol in symbols:
                try:
                    bars = await self._historical_data_manager.ensure_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_dt,
                        end=end_dt,
                    )

                    if bars:
                        bar_dicts = [
                            {
                                "timestamp": b.bar_start,
                                "open": b.open,
                                "high": b.high,
                                "low": b.low,
                                "close": b.close,
                                "volume": b.volume or 0,
                            }
                            for b in bars
                        ]
                        count = self._indicator_engine.inject_historical_bars(
                            symbol, timeframe, bar_dicts
                        )
                        results[symbol] = results.get(symbol, 0) + count

                        indicators_computed = await self._indicator_engine.compute_on_history(
                            symbol, timeframe
                        )

                        logger.debug(
                            "Preloaded bars for symbol",
                            extra={
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "bars_injected": count,
                                "indicators_computed": indicators_computed,
                                "date_range": f"{bars[0].bar_start} to {bars[-1].bar_start}",
                            },
                        )
                    else:
                        logger.warning(
                            "No bars returned for symbol",
                            extra={"symbol": symbol, "timeframe": timeframe},
                        )
                        failed_symbols.append((symbol, timeframe, "No bars returned"))

                except Exception as e:
                    logger.error(
                        "Failed to preload bars for symbol (continuing)",
                        extra={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )
                    failed_symbols.append((symbol, timeframe, str(e)))

        # Report all failed symbols at the end
        if failed_symbols:
            logger.error(
                f"Preload failed for {len(failed_symbols)} symbol/timeframe combinations",
                extra={"failed": [f"{s}/{tf}: {err}" for s, tf, err in failed_symbols]},
            )

        elapsed_sec = time.monotonic() - start_time
        self._last_cache_refresh = now_utc()

        log_level = logging.WARNING if elapsed_sec > slow_warn_sec else logging.INFO
        logger.log(
            log_level,
            "Bar cache preload complete",
            extra={
                "symbols_loaded": len(results),
                "total_bars_injected": sum(results.values()),
                "elapsed_sec": round(elapsed_sec, 2),
                "slow_threshold_sec": slow_warn_sec,
                "cache_refreshed_at": self._last_cache_refresh.isoformat(),
            },
        )

        return results

    async def refresh_disk_cache(self, symbols: List[str]) -> bool:
        """
        Refresh Parquet cache (PERIODIC, disk-only).

        Updates disk cache with new bars from broker.
        Does NOT inject into IndicatorEngine - live BAR_CLOSE events handle that.

        Args:
            symbols: List of symbols to refresh

        Returns:
            True if refresh was performed, False if skipped (not due yet)
        """
        refresh_hours = self._preload_config.get("cache_refresh_hours", 24)

        if self._last_cache_refresh:
            elapsed = now_utc() - self._last_cache_refresh
            if elapsed.total_seconds() < refresh_hours * 3600:
                logger.debug(
                    "Cache refresh skipped (not due yet)",
                    extra={
                        "last_refresh": self._last_cache_refresh.isoformat(),
                        "next_refresh_hours": round(
                            refresh_hours - elapsed.total_seconds() / 3600, 2
                        ),
                    },
                )
                return False

        if not self._historical_data_manager:
            logger.debug("Cache refresh skipped: no historical data manager")
            return False

        if not symbols:
            logger.debug("Cache refresh skipped: no symbols")
            return False

        logger.info(
            "Triggering scheduled disk cache refresh (no injection)",
            extra={
                "symbols_count": len(symbols),
                "refresh_interval_hours": refresh_hours,
            },
        )

        lookback_days = self._preload_config.get("lookback_days", 365)
        end_dt = now_utc()
        start_dt = end_dt - timedelta(days=lookback_days)

        for timeframe in self._timeframes:
            for symbol in symbols:
                try:
                    await self._historical_data_manager.ensure_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_dt,
                        end=end_dt,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to refresh cache for symbol",
                        extra={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "error": str(e),
                        },
                    )

        self._last_cache_refresh = now_utc()

        logger.info(
            "Disk cache refresh complete",
            extra={"cache_refreshed_at": self._last_cache_refresh.isoformat()},
        )
        return True
