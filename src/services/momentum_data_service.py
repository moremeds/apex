"""Momentum screener data service.

Cache-first pattern: screening reads from ParquetHistoricalStore and cached
universe JSON. Explicit `update_*` commands fetch from FMP + yfinance.

Data storage:
- Universe: data/cache/index_constituents.json (FMP index membership)
- OHLCV: data/historical/{SYMBOL}/1d.parquet (existing ParquetHistoricalStore)
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_UNIVERSE_CACHE = PROJECT_ROOT / "data/cache/index_constituents.json"


class MomentumDataService:
    """Service for managing universe + OHLCV data for momentum screening.

    Uses ParquetHistoricalStore for price data (existing infrastructure)
    and a JSON cache for index constituency.
    """

    def __init__(
        self,
        universe_cache_path: Path | None = None,
        historical_base_dir: Path | None = None,
    ) -> None:
        self._universe_cache = universe_cache_path or DEFAULT_UNIVERSE_CACHE
        self._hist_dir = historical_base_dir or (PROJECT_ROOT / "data/historical")

    # ── Universe Management ───────────────────────────────────────────

    def update_universe(
        self,
        indices: list[str] | None = None,
        russell_proxy: bool = True,
        cap_min: float = 300_000_000,
        cap_max: float = 10_000_000_000,
        api_key: str | None = None,
    ) -> list[str]:
        """Fetch index constituents via FMP and cache to JSON.

        Args:
            indices: Index names to fetch ("sp500", "nasdaq").
            russell_proxy: Include Russell 2000 proxy.
            cap_min: Min market cap for Russell proxy.
            cap_max: Max market cap for Russell proxy.
            api_key: Optional FMP API key override.

        Returns:
            Combined deduplicated symbol list.
        """
        from src.infrastructure.adapters.fmp.index_constituents import (
            FMPIndexConstituentsAdapter,
        )

        adapter = FMPIndexConstituentsAdapter(api_key=api_key)
        symbols = adapter.get_combined_universe(
            indices=indices,
            russell_proxy=russell_proxy,
            cap_min=cap_min,
            cap_max=cap_max,
        )

        # Save cache
        self._universe_cache.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "symbols": symbols,
            "count": len(symbols),
        }
        self._universe_cache.write_text(json.dumps(cache_data, indent=2))
        logger.info(f"Universe cache updated: {len(symbols)} symbols → {self._universe_cache}")
        return symbols

    def get_universe(self) -> list[str]:
        """Read cached universe from JSON. Returns empty list if no cache."""
        if not self._universe_cache.exists():
            logger.warning(f"Universe cache not found at {self._universe_cache}")
            return []

        try:
            data = json.loads(self._universe_cache.read_text())
            symbols: list[str] = data.get("symbols", [])
            updated = data.get("updated_at", "unknown")
            logger.info(f"Loaded universe: {len(symbols)} symbols (updated: {updated})")
            return symbols
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load universe cache: {e}")
            return []

    # ── OHLCV Data Management ─────────────────────────────────────────

    def update_ohlcv(
        self,
        symbols: list[str],
        months: int = 15,
        batch_size: int = 500,
    ) -> int:
        """Download daily OHLCV via yfinance and store in Parquet.

        Uses yfinance batch download for efficiency, then writes each symbol
        to the ParquetHistoricalStore.

        Args:
            symbols: List of symbols to fetch.
            months: Months of history to fetch (15 = ~13mo lookback + buffer).
            batch_size: Symbols per yfinance batch call.

        Returns:
            Number of symbols successfully written.
        """
        import yfinance as yf

        from src.domain.events.domain_events import BarData
        from src.infrastructure.stores.parquet_historical_store import (
            ParquetHistoricalStore,
        )

        store = ParquetHistoricalStore(self._hist_dir)
        end_date = date.today()
        start_date = end_date - timedelta(days=months * 31)

        success_count = 0
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch = symbols[batch_start : batch_start + batch_size]
            logger.info(
                f"Downloading batch {batch_idx + 1}/{total_batches} " f"({len(batch)} symbols)"
            )

            try:
                df = yf.download(
                    batch,
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    group_by="ticker",
                    auto_adjust=True,
                    threads=True,
                )
            except Exception as e:
                logger.error(f"yfinance batch download failed: {e}")
                continue

            if df.empty:
                continue

            for symbol in batch:
                try:
                    if len(batch) == 1:
                        sym_df = df
                    else:
                        sym_df = df[symbol] if symbol in df.columns.get_level_values(0) else None

                    if sym_df is None or sym_df.empty:
                        continue

                    sym_df = sym_df.dropna(subset=["Close"])
                    if sym_df.empty:
                        continue

                    bars = self._df_to_bars(sym_df, symbol)
                    if bars:
                        store.write_bars(symbol, "1d", bars, mode="upsert")
                        success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to process {symbol}: {e}")

        logger.info(f"OHLCV update complete: {success_count}/{len(symbols)} symbols written")
        return success_count

    def get_daily_data(self, symbol: str, start: date, end: date) -> pd.DataFrame | None:
        """Read daily OHLCV from Parquet store as DataFrame.

        Returns DataFrame with columns: close, volume, or None if no data.
        """
        parquet_path = self._hist_dir / symbol.upper() / "1d.parquet"
        if not parquet_path.exists():
            return None

        try:
            df = pd.read_parquet(parquet_path)
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
                df.index = pd.to_datetime(df.index).tz_localize(None)

            # Filter date range
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            df = df[(df.index >= start_ts) & (df.index <= end_ts)]

            if df.empty:
                return None

            result = pd.DataFrame({"close": df["close"], "volume": df["volume"]})
            return result.sort_index()
        except Exception as e:
            logger.warning(f"Failed to read {symbol} daily data: {e}")
            return None

    def get_bulk_closes(
        self,
        symbols: list[str],
        end: date,
        lookback_days: int = 300,
    ) -> dict[str, np.ndarray]:
        """Batch read closing prices for screener input.

        Args:
            symbols: Symbols to read.
            end: End date.
            lookback_days: Calendar days to look back (should exceed trading day requirement).

        Returns:
            Dict of symbol -> numpy array of daily closes (chronological).
        """
        start = end - timedelta(days=lookback_days)
        result: dict[str, np.ndarray] = {}

        for symbol in symbols:
            df = self.get_daily_data(symbol, start, end)
            if df is not None and not df.empty:
                result[symbol] = df["close"].values

        logger.info(
            f"Loaded closes: {len(result)}/{len(symbols)} symbols "
            f"({start.isoformat()} to {end.isoformat()})"
        )
        return result

    def get_bulk_volumes(
        self,
        symbols: list[str],
        end: date,
        lookback_days: int = 300,
    ) -> dict[str, np.ndarray]:
        """Batch read daily volumes for screener input.

        Args:
            symbols: Symbols to read.
            end: End date.
            lookback_days: Calendar days to look back.

        Returns:
            Dict of symbol -> numpy array of daily volumes (chronological).
        """
        start = end - timedelta(days=lookback_days)
        result: dict[str, np.ndarray] = {}

        for symbol in symbols:
            df = self.get_daily_data(symbol, start, end)
            if df is not None and not df.empty:
                result[symbol] = df["volume"].values

        return result

    def get_market_caps(self, symbols: list[str]) -> dict[str, float]:
        """Read market caps from the existing MarketCapService cache.

        Falls back to 0.0 for missing symbols.
        """
        from src.services.market_cap_service import MarketCapService

        service = MarketCapService()
        all_caps = service.get_all_cached_caps()
        return {s: all_caps.get(s, 0.0) for s in symbols}

    @staticmethod
    def _df_to_bars(df: pd.DataFrame, symbol: str) -> list[Any]:
        """Convert yfinance DataFrame to BarData list."""
        from zoneinfo import ZoneInfo

        from src.domain.events.domain_events import BarData

        UTC = ZoneInfo("UTC")
        bars = []

        for idx, row in df.iterrows():
            ts = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)

            bars.append(
                BarData(
                    symbol=symbol,
                    timeframe="1d",
                    open=float(row.get("Open", 0)) if pd.notna(row.get("Open")) else None,
                    high=float(row.get("High", 0)) if pd.notna(row.get("High")) else None,
                    low=float(row.get("Low", 0)) if pd.notna(row.get("Low")) else None,
                    close=float(row["Close"]),
                    volume=int(row.get("Volume", 0)) if pd.notna(row.get("Volume")) else 0,
                    vwap=None,
                    trade_count=None,
                    bar_start=ts,
                    source="yfinance",
                    timestamp=ts,
                )
            )

        return bars
