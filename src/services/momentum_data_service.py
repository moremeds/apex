"""Momentum screener data service.

Cache-first pattern: screening reads from ParquetHistoricalStore and cached
universe JSON. Explicit `update_*` commands fetch via unified bar loader
(FMP -> Yahoo -> IB priority).

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
        fallback_max_symbols: int = 800,
        api_key: str | None = None,
    ) -> list[str]:
        """Fetch index constituents via FMP and cache to JSON.

        Args:
            indices: Index names to fetch ("sp500", "nasdaq").
            russell_proxy: Include Russell 2000 proxy.
            cap_min: Min market cap for Russell proxy.
            cap_max: Max market cap for Russell proxy.
            fallback_max_symbols: Max symbols from company-screener fallback.
            api_key: Optional FMP API key override.

        Returns:
            Combined deduplicated symbol list.
        """
        from src.infrastructure.adapters.fmp.index_constituents import (
            FMPIndexConstituentsAdapter,
        )

        print("  Fetching universe from FMP...", end=" ", flush=True)
        adapter = FMPIndexConstituentsAdapter(api_key=api_key)
        symbols = adapter.get_combined_universe(
            indices=indices,
            russell_proxy=russell_proxy,
            cap_min=cap_min,
            cap_max=cap_max,
            fallback_max_symbols=fallback_max_symbols,
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
        print(f"{len(symbols)} symbols")
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
        """Incrementally update daily OHLCV in Parquet.

        Smart refresh: checks each symbol's last Parquet date and only
        fetches new bars. Symbols with no existing data get a full download.

        Args:
            symbols: List of symbols to fetch.
            months: Months of history for symbols missing data entirely.
            batch_size: Symbols per batch call.

        Returns:
            Number of symbols successfully written.
        """
        from src.infrastructure.stores.parquet_historical_store import (
            ParquetHistoricalStore,
        )

        store = ParquetHistoricalStore(self._hist_dir)
        end_date = date.today()

        # Partition symbols by freshness
        stale: list[str] = []
        missing: list[str] = []
        fresh_count = 0

        for sym in symbols:
            parquet_path = self._hist_dir / sym.upper() / "1d.parquet"
            if not parquet_path.exists():
                missing.append(sym)
                continue
            try:
                df = pd.read_parquet(parquet_path, columns=["timestamp"])
                if df.empty:
                    missing.append(sym)
                    continue
                last_date = pd.Timestamp(df["timestamp"].iloc[-1]).date()
                if last_date < end_date - timedelta(days=1):
                    stale.append(sym)
                else:
                    fresh_count += 1
            except Exception:
                missing.append(sym)

        print(
            f"  OHLCV: {fresh_count} fresh, {len(stale)} stale, "
            f"{len(missing)} missing (of {len(symbols)} total)"
        )

        success_count = 0

        # Incremental: stale symbols only need recent bars (14 days buffer)
        if stale:
            print(f"  Updating {len(stale)} stale symbols (14 days)...", flush=True)
            success_count += self._batch_fetch_and_store(
                stale, store, days=14, batch_size=batch_size
            )

        # Full download: symbols with no Parquet data
        if missing:
            print(
                f"  Downloading {len(missing)} new symbols ({months} months)...",
                flush=True,
            )
            success_count += self._batch_fetch_and_store(
                missing, store, days=months * 31, batch_size=batch_size
            )

        if fresh_count == len(symbols):
            print("  All symbols up to date — nothing to fetch")
        else:
            print(f"  Done: {success_count} updated, {fresh_count} already fresh")

        return success_count

    def _batch_fetch_and_store(
        self,
        symbols: list[str],
        store: Any,
        days: int,
        batch_size: int = 500,
    ) -> int:
        """Fetch bars in batches and upsert into Parquet store.

        When fetching > 250 symbols, uses Yahoo batch download directly
        (single HTTP call per 500-symbol batch) instead of FMP's serial
        per-symbol path which incurs 0.3s rate-limit sleep per symbol.
        """
        from src.services.bar_loader import load_bars

        # Yahoo's yf.download() handles 500 symbols in one HTTP call;
        # FMP loops per-symbol with 0.3s sleep → ~3 min/batch vs ~5s/batch.
        # Keep FMP as fallback so Yahoo outages don't lose data entirely.
        use_yahoo_batch = len(symbols) > 250
        source_override: list[str] | None = ["yahoo", "fmp"] if use_yahoo_batch else None
        if use_yahoo_batch:
            logger.info(f"Large batch ({len(symbols)} symbols): using Yahoo-first batch download")

        end_date = date.today()
        success_count = 0
        total_batches = (len(symbols) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch = symbols[batch_start : batch_start + batch_size]
            print(
                f"    Batch {batch_idx + 1}/{total_batches} " f"({len(batch)} symbols)...",
                end=" ",
                flush=True,
            )

            bars_dict = load_bars(
                batch,
                timeframe="1d",
                days=days,
                end_date=end_date,
                source_priority=source_override,
            )

            batch_ok = 0
            for symbol, df in bars_dict.items():
                try:
                    if df.empty:
                        continue
                    df_compat = df.copy()
                    df_compat.columns = [c.capitalize() for c in df_compat.columns]
                    bar_list = self._df_to_bars(df_compat, symbol)
                    if bar_list:
                        store.write_bars(symbol, "1d", bar_list, mode="upsert")
                        success_count += 1
                        batch_ok += 1
                except Exception as e:
                    logger.warning(f"Failed to process {symbol}: {e}")
            print(f"{batch_ok}/{len(batch)} ok")

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

    def get_bulk_close_series(
        self,
        symbols: list[str],
        end: date,
        lookback_days: int = 300,
    ) -> dict[str, pd.Series]:
        """Batch read closing prices as date-indexed Series (for PIT backtest).

        Returns:
            Dict of symbol -> pd.Series with DatetimeIndex of daily closes.
        """
        start = end - timedelta(days=lookback_days)
        result: dict[str, pd.Series] = {}
        for symbol in symbols:
            df = self.get_daily_data(symbol, start, end)
            if df is not None and not df.empty:
                result[symbol] = df["close"]
        logger.info(
            f"Loaded close series: {len(result)}/{len(symbols)} symbols "
            f"({start.isoformat()} to {end.isoformat()})"
        )
        return result

    def get_bulk_volume_series(
        self,
        symbols: list[str],
        end: date,
        lookback_days: int = 300,
    ) -> dict[str, pd.Series]:
        """Batch read daily volumes as date-indexed Series (for PIT backtest).

        Returns:
            Dict of symbol -> pd.Series with DatetimeIndex of daily volumes.
        """
        start = end - timedelta(days=lookback_days)
        result: dict[str, pd.Series] = {}
        for symbol in symbols:
            df = self.get_daily_data(symbol, start, end)
            if df is not None and not df.empty:
                result[symbol] = df["volume"]
        return result

    def get_data_as_of_date(self, symbols: list[str], sample_size: int = 5) -> date | None:
        """Determine the most recent data date by probing Parquet files.

        Uses a deterministic sample (sorted, first N) so results are
        reproducible across runs.

        Returns:
            The most recent data date found, or None if no data.
        """
        sampled = sorted(symbols)[:sample_size] if symbols else []
        latest: date | None = None

        for symbol in sampled:
            parquet_path = self._hist_dir / symbol.upper() / "1d.parquet"
            if not parquet_path.exists():
                continue
            try:
                df = pd.read_parquet(parquet_path)
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                if df.empty:
                    continue
                last_ts = pd.Timestamp(df.index[-1])
                last_date = last_ts.date()
                if latest is None or last_date > latest:
                    latest = last_date
            except Exception as e:
                logger.warning(f"Failed to read data_as_of for {symbol}: {e}")

        return latest

    def get_upcoming_earnings(self, symbols: list[str], lookahead_days: int = 7) -> dict[str, date]:
        """Fetch upcoming earnings dates for given symbols.

        Calls FMP earnings calendar and filters to provided symbol list.
        Fail-open: returns empty dict on any error.

        Args:
            symbols: Symbols to check.
            lookahead_days: Days ahead to scan for earnings.

        Returns:
            Dict of symbol -> earnings date for symbols reporting soon.
        """
        try:
            from src.infrastructure.adapters.earnings.fmp_earnings import (
                FMPEarningsAdapter,
            )

            adapter = FMPEarningsAdapter()
            today = date.today()
            end = today + timedelta(days=lookahead_days)
            calendar = adapter.fetch_earnings_calendar(today, end)

            symbol_set = set(symbols)
            result: dict[str, date] = {}
            for entry in calendar:
                sym = entry.get("symbol", "")
                if sym in symbol_set and "date" in entry:
                    try:
                        result[sym] = date.fromisoformat(entry["date"])
                    except (ValueError, TypeError):
                        continue

            logger.info(f"Upcoming earnings: {len(result)} symbols within {lookahead_days} days")
            return result
        except Exception as e:
            logger.warning(f"Earnings calendar fetch failed (fail-open): {e}")
            return {}

    def get_market_caps(self, symbols: list[str]) -> dict[str, float]:
        """Read market caps from the existing MarketCapService cache.

        Falls back to 0.0 for missing symbols.
        """
        from src.services.market_cap_service import MarketCapService

        service = MarketCapService()
        all_caps = service.get_all_cached_caps()
        return {s: all_caps.get(s, 0.0) for s in symbols}

    @staticmethod
    def _df_to_bars(df: pd.DataFrame, symbol: str, source: str = "bar_loader") -> list[Any]:
        """Convert OHLCV DataFrame to BarData list. Expects capitalized columns."""
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
                    source=source,
                    timestamp=ts,
                )
            )

        return bars
