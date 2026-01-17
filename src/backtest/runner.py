#!/usr/bin/env python3
"""
Unified Backtest Runner

Main entry point for all backtesting operations:
- Single backtests (bar-by-bar with ApexEngine)
- Systematic experiments (vectorized with VectorBTEngine)

Usage:
    # Single backtest (ApexEngine - full execution simulation)
    python -m src.backtest.runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30

    # Systematic experiment (VectorBTEngine - fast parameter optimization)
    python -m src.backtest.runner --spec config/backtest/examples/ta_metrics.yaml

    # Force specific engine
    python -m src.backtest.runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30 --engine vectorbt

    # List strategies
    python -m src.backtest.runner --list-strategies
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import yaml

from ..domain.backtest.backtest_result import BacktestResult
from ..domain.backtest.backtest_spec import BacktestSpec
from ..domain.strategy.registry import get_strategy_class, list_strategies
from .core import RunResult, RunSpec
from .data.feeds import (
    CachedBarDataFeed,
    CsvDataFeed,
    HistoricalStoreDataFeed,
    ParquetDataFeed,
    StreamingCsvDataFeed,
    StreamingParquetDataFeed,
)
from .execution.simulated import FillModel

# Check if backtrader is available
try:
    import backtrader as bt

    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "base.yaml"


def _normalize_parquet_filter_dt(value: datetime, ts_type: "pa.TimestampType") -> datetime:
    from datetime import timezone

    if ts_type.tz is None:
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    if ts_type.tz == "UTC":
        return value.astimezone(timezone.utc)

    try:
        from zoneinfo import ZoneInfo

        return value.astimezone(ZoneInfo(ts_type.tz))
    except Exception:
        return value.astimezone(timezone.utc)


def _build_parquet_timestamp_filters(
    schema: "pa.Schema",
    start_dt: datetime,
    end_dt: datetime,
) -> Optional[list[tuple[str, str, Any]]]:
    import pyarrow as pa

    if "timestamp" not in schema.names:
        return None

    ts_type = schema.field("timestamp").type
    if not pa.types.is_timestamp(ts_type):
        return None

    start_value = pa.scalar(_normalize_parquet_filter_dt(start_dt, ts_type), type=ts_type)
    end_value = pa.scalar(_normalize_parquet_filter_dt(end_dt, ts_type), type=ts_type)

    return [("timestamp", ">=", start_value), ("timestamp", "<=", end_value)]


def _to_utc_timestamp(value: datetime) -> "pd.Timestamp":
    import pandas as pd

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _read_parquet_cached_data(
    parquet_path: Path,
    start_dt: datetime,
    end_dt: datetime,
) -> "pd.DataFrame":
    import pandas as pd
    import pyarrow.parquet as pq

    filters = None
    try:
        schema = pq.read_schema(parquet_path)
        filters = _build_parquet_timestamp_filters(schema, start_dt, end_dt)
    except Exception:
        filters = None

    table = None
    if filters:
        try:
            table = pq.read_table(parquet_path, filters=filters)
        except Exception:
            logger.debug(
                "Filtered Parquet read failed for %s; retrying without filters",
                parquet_path,
            )
            table = None

    if table is None:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if "timestamp" in df.columns and not df.empty:
            timestamps = pd.to_datetime(df["timestamp"], utc=True)
            start_ts = _to_utc_timestamp(start_dt)
            end_ts = _to_utc_timestamp(end_dt)
            df = df.loc[(timestamps >= start_ts) & (timestamps <= end_ts)].copy()
            df["timestamp"] = timestamps
        return df

    return table.to_pandas()


def load_ib_config(config_path: Optional[Path] = None):
    """Load IB config from base.yaml."""
    from config.models import IbClientIdsConfig, IbConfig

    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return None

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        ib_cfg = config.get("brokers", {}).get("ibkr", {})
        if not ib_cfg.get("enabled"):
            logger.warning("IB not enabled in config")
            return None

        client_ids_cfg = ib_cfg.get("client_ids", {})
        client_ids = IbClientIdsConfig(
            execution=client_ids_cfg.get("execution", 1),
            monitoring=client_ids_cfg.get("monitoring", 2),
            historical_pool=client_ids_cfg.get("historical_pool", [3, 4, 5, 6, 7, 8, 9, 10]),
        )

        return IbConfig(
            enabled=True,
            host=ib_cfg.get("host", "127.0.0.1"),
            port=ib_cfg.get("port", 7497),
            client_ids=client_ids,
            provides_market_data=ib_cfg.get("provides_market_data", True),
        )
    except Exception as e:
        logger.warning(f"Failed to load IB config: {e}")
        return None


def load_historical_data_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load historical data config from base.yaml."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        historical_cfg = config.get("historical_data", {})
        storage_cfg = historical_cfg.get("storage", {})

        return {
            "base_dir": storage_cfg.get("base_dir", "data/historical"),
            "source_priority": historical_cfg.get("source_priority", ["ib", "yahoo"]),
            "sources": historical_cfg.get("sources", {}),
        }
    except Exception as e:
        logger.warning(f"Failed to load historical data config: {e}")
        return {}


# =============================================================================
# Single Backtest Runner (ApexEngine)
# =============================================================================


class SingleBacktestRunner:
    """
    Runner for single backtests using ApexEngine (bar-by-bar, event-driven).

    Use for:
    - Full execution simulation with order matching
    - Testing strategy logic with realistic fills
    - Small to medium datasets
    """

    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100000.0,
        data_source: str = "historical",
        data_dir: str = "./data",
        bar_size: str = "1d",
        secondary_timeframes: Optional[List[str]] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        fill_model: str = "immediate",
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
        cached_bars: Optional[Dict[str, List]] = None,
        streaming: bool = True,
        coverage_mode: Optional[str] = None,
        historical_dir: Optional[str] = None,
        source_priority: Optional[List[str]] = None,
    ):
        self.strategy_name = strategy_name
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data_source = data_source
        self.data_dir = data_dir
        self.bar_size = bar_size
        self.secondary_timeframes = secondary_timeframes or []
        self.strategy_params = strategy_params or {}
        self.fill_model = FillModel(fill_model)
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.cached_bars = cached_bars
        self.streaming = streaming
        self.coverage_mode = coverage_mode
        self.historical_dir = historical_dir
        self.source_priority = source_priority
        self._spec: Optional[BacktestSpec] = None

    @classmethod
    def from_spec(cls, spec_path: str) -> "SingleBacktestRunner":
        """Create runner from spec file."""
        spec = BacktestSpec.from_yaml(spec_path)
        errors = spec.validate()
        if errors:
            raise ValueError(f"Invalid spec: {errors}")

        streaming = spec.data.streaming if hasattr(spec.data, "streaming") else True
        secondary_timeframes = getattr(spec.data, "secondary_timeframes", None) or []

        runner = cls(
            strategy_name=spec.strategy.name,
            symbols=spec.get_symbols(),
            start_date=spec.data.start_date or date(2024, 1, 1),
            end_date=spec.data.end_date or date(2024, 12, 31),
            initial_capital=spec.execution.initial_capital,
            data_source=spec.data.source,
            data_dir=spec.data.csv_dir or spec.data.parquet_dir or "./data",
            bar_size=spec.data.bar_size,
            secondary_timeframes=secondary_timeframes,
            strategy_params=spec.strategy.params,
            streaming=streaming,
            coverage_mode=spec.data.coverage_mode,
            historical_dir=spec.data.historical_dir,
            source_priority=spec.data.source_priority,
        )
        runner._spec = spec
        return runner

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SingleBacktestRunner":
        """Create runner from CLI arguments."""
        if hasattr(args, "spec") and args.spec:
            return cls.from_spec(args.spec)

        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
        symbols = [s.strip() for s in args.symbols.split(",")]

        strategy_params = {}
        if hasattr(args, "params") and args.params:
            for param in args.params:
                key, value = param.split("=")
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                strategy_params[key] = value

        source_priority = None
        if getattr(args, "source_priority", None):
            source_priority = [
                s.strip().lower() for s in args.source_priority.split(",") if s.strip()
            ]

        return cls(
            strategy_name=args.strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=getattr(args, "capital", 100000),
            data_source=getattr(args, "data_source", "historical"),
            data_dir=getattr(args, "data_dir", "./data"),
            bar_size=getattr(args, "bar_size", "1d"),
            strategy_params=strategy_params,
            fill_model=getattr(args, "fill_model", "immediate"),
            slippage_bps=getattr(args, "slippage", 5.0),
            commission_per_share=getattr(args, "commission", 0.005),
            streaming=getattr(args, "streaming", True),
            coverage_mode=getattr(args, "coverage_mode", None),
            historical_dir=getattr(args, "historical_dir", None),
            source_priority=source_priority,
        )

    async def run(self) -> BacktestResult:
        """Run the backtest."""
        # Import to register strategies
        from ..domain.strategy.examples import (  # noqa
            BuyAndHoldStrategy,
            MovingAverageCrossStrategy,
        )
        from .execution.engines.backtest_engine import BacktestConfig, BacktestEngine

        self._print_config()
        await self._ensure_historical_coverage()

        config = BacktestConfig(
            start_date=self.start_date,
            end_date=self.end_date,
            symbols=self.symbols,
            initial_capital=self.initial_capital,
            bar_size=self.bar_size,
            strategy_name=self.strategy_name,
            strategy_params=self.strategy_params,
            fill_model=self.fill_model,
            slippage_bps=self.slippage_bps,
            commission_per_share=self.commission_per_share,
        )

        engine = BacktestEngine(config)

        strategy_class = get_strategy_class(self.strategy_name)
        if strategy_class is None:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. Available: {list_strategies()}"
            )
        engine.set_strategy(strategy_class, params=self.strategy_params)

        data_feed = self._create_data_feed()
        engine.set_data_feed(data_feed)

        result = await engine.run()
        result.print_summary()

        if self._spec and self._spec.reporting.get("persist_to_db"):
            logger.info(f"Backtest result saved: {result.backtest_id}")

        return result

    def _create_data_feed(self):
        """Create appropriate data feed based on data_source."""
        if self.data_source == "cached":
            if not self.cached_bars:
                raise RuntimeError("'cached' data source requires cached_bars parameter.")
            return CachedBarDataFeed(
                bars_by_symbol=self.cached_bars,
                start_date=self.start_date,
                end_date=self.end_date,
            )
        elif self.data_source == "historical":
            historical_cfg = load_historical_data_config()
            base_dir = self.historical_dir or historical_cfg.get("base_dir", "data/historical")
            return HistoricalStoreDataFeed(
                base_dir=base_dir,
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                bar_size=self.bar_size,
                secondary_timeframes=self.secondary_timeframes,
            )
        elif self.data_source == "csv":
            if self.streaming:
                return StreamingCsvDataFeed(
                    csv_dir=self.data_dir,
                    symbols=self.symbols,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    bar_size=self.bar_size,
                )
            else:
                return CsvDataFeed(
                    csv_dir=self.data_dir,
                    symbols=self.symbols,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    bar_size=self.bar_size,
                )
        elif self.data_source == "parquet":
            if self.streaming:
                return StreamingParquetDataFeed(
                    parquet_dir=self.data_dir,
                    symbols=self.symbols,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    bar_size=self.bar_size,
                )
            else:
                return ParquetDataFeed(
                    parquet_dir=self.data_dir,
                    symbols=self.symbols,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    bar_size=self.bar_size,
                )
        else:
            raise ValueError(
                f"Unknown data source: {self.data_source}. Use 'cached', 'historical', 'csv', or 'parquet'."
            )

    def _print_config(self) -> None:
        """Log backtest configuration using structured logging."""
        config_info = {
            "strategy": self.strategy_name,
            "symbols": self.symbols,
            "start": str(self.start_date),
            "end": str(self.end_date),
            "capital": self.initial_capital,
            "data_source": self.data_source,
            "bar_size": self.bar_size,
            "fill_model": self.fill_model.value,
        }

        if self.data_source == "historical":
            historical_cfg = load_historical_data_config()
            config_info["historical_dir"] = self.historical_dir or historical_cfg.get(
                "base_dir", "data/historical"
            )
            config_info["sources"] = self.source_priority or historical_cfg.get(
                "source_priority", ["ib", "yahoo"]
            )
            config_info["coverage_mode"] = self.coverage_mode or "download"

        if self.strategy_params:
            config_info["params"] = self.strategy_params

        logger.info(
            f"Backtest config: {self.strategy_name} on {','.join(self.symbols)} "
            f"({self.start_date} to {self.end_date})"
        )
        logger.debug(f"Full config: {config_info}")

    async def _ensure_historical_coverage(self) -> None:
        """Ensure historical data coverage before running backtest."""
        if self.data_source != "historical":
            return

        mode = self.coverage_mode or "download"
        if mode == "off":
            logger.info("Coverage check disabled (mode=off)")
            return

        valid_modes = {"off", "check", "download"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid coverage_mode: {mode}. Must be one of: {valid_modes}")

        from ..services.historical_data_manager import HistoricalDataManager

        historical_cfg = load_historical_data_config()
        base_dir = Path(self.historical_dir or historical_cfg.get("base_dir", "data/historical"))
        source_priority = self.source_priority or historical_cfg.get(
            "source_priority", ["ib", "yahoo"]
        )

        logger.info(f"Coverage check: mode={mode}, base_dir={base_dir}, sources={source_priority}")

        manager = HistoricalDataManager(base_dir=base_dir, source_priority=source_priority)
        ib_adapter = None

        try:
            if "ib" in source_priority and mode == "download":
                ib_config = load_ib_config()
                if ib_config:
                    from ..infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

                    client_id = (
                        ib_config.client_ids.historical_pool[0]
                        if ib_config.client_ids.historical_pool
                        else 10
                    )
                    ib_adapter = IbHistoricalAdapter(
                        host=ib_config.host,
                        port=ib_config.port,
                        client_id=client_id,
                    )
                    try:
                        await ib_adapter.connect()
                        manager.set_ib_source(ib_adapter)
                        logger.info(
                            f"IB historical source connected: {ib_config.host}:{ib_config.port}"
                        )
                    except Exception as e:
                        logger.warning(f"IB connection failed, will use fallback sources: {e}")
                        ib_adapter = None

            start_dt = datetime.combine(self.start_date, datetime.min.time())
            end_dt = datetime.combine(self.end_date, datetime.max.time())

            if mode == "check":
                for symbol in self.symbols:
                    gaps = manager.find_missing_ranges(symbol, self.bar_size, start_dt, end_dt)
                    if gaps:
                        gap_summary = ", ".join(f"{g.start.date()}-{g.end.date()}" for g in gaps)
                        raise RuntimeError(
                            f"Missing coverage for {symbol}/{self.bar_size}: {len(gaps)} gap(s) [{gap_summary}]. "
                            "Use --coverage-mode download to fetch missing data."
                        )
                logger.info(f"Coverage check passed for {len(self.symbols)} symbol(s)")

            elif mode == "download":
                results = await manager.download_symbols(
                    symbols=self.symbols,
                    timeframe=self.bar_size,
                    start=start_dt,
                    end=end_dt,
                )
                downloaded = [r for r in results if r.bars_downloaded > 0]
                cached = [r for r in results if r.source == "cached"]
                failed = [r for r in results if not r.success]

                if downloaded:
                    total_bars = sum(r.bars_downloaded for r in downloaded)
                    logger.info(f"Downloaded {total_bars} bars for {len(downloaded)} symbol(s)")
                if cached:
                    logger.info(f"{len(cached)} symbol(s) already cached")
                if failed:
                    failed_symbols = [r.symbol for r in failed]
                    raise RuntimeError(f"Failed to download data for: {', '.join(failed_symbols)}")

        finally:
            try:
                manager.close()
            except Exception as e:
                logger.warning(f"Error closing manager: {e}")

            if ib_adapter:
                try:
                    await ib_adapter.disconnect()
                except Exception:
                    pass


# =============================================================================
# Systematic Experiment Runner (VectorBTEngine)
# =============================================================================


async def prefetch_data(
    symbols: List[str],
    start_date,
    end_date,
    max_retries: int = 3,
    timeframe: str = "1d",
    historical_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Pre-fetch data for systematic experiments, using local cache when available.

    First checks the local Parquet store for cached data, then fetches from IB
    only for symbols that are missing or have incomplete data.

    Args:
        symbols: List of symbols to fetch
        start_date: Start date for data
        end_date: End date for data
        max_retries: Number of retry attempts for IB fetching
        timeframe: Bar timeframe (default: 1d)
        historical_dir: Base directory for historical data (default: data/historical)

    Returns:
        Dict[symbol, DataFrame] with OHLCV data indexed by timestamp
    """
    from pathlib import Path

    import pandas as pd

    logger.info(f"Pre-fetching data for {len(symbols)} symbols...")

    # Parse dates - make timezone-aware (UTC) to match Parquet storage format
    from datetime import timezone

    if hasattr(start_date, "isoformat"):
        start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    else:
        start_dt = datetime.fromisoformat(str(start_date)) if start_date else datetime(2020, 1, 1)
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)

    if hasattr(end_date, "isoformat"):
        end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
    else:
        end_dt = datetime.fromisoformat(str(end_date)) if end_date else datetime.now(timezone.utc)
        if end_dt.tzinfo is None:
            end_dt = end_dt.replace(tzinfo=timezone.utc)

    # Resolve historical data directory
    historical_cfg = load_historical_data_config()
    base_dir = Path(historical_dir or historical_cfg.get("base_dir", "data/historical"))

    results: Dict[str, Any] = {}
    symbols_to_fetch: List[str] = []

    # Step 1: Check local Parquet cache for each symbol
    logger.info(f"Checking local cache in {base_dir}...")
    for symbol in symbols:
        parquet_path = base_dir / symbol.upper() / f"{timeframe}.parquet"
        if parquet_path.exists():
            try:
                df = _read_parquet_cached_data(parquet_path, start_dt, end_dt)

                if not df.empty:
                    # Convert timestamp column to index
                    if "timestamp" in df.columns:
                        df.set_index("timestamp", inplace=True)
                    df.sort_index(inplace=True)

                    # Check coverage - we need at least 80% of expected trading days
                    expected_days = (end_dt - start_dt).days * 252 // 365  # Rough estimate
                    actual_days = len(df)
                    coverage = actual_days / expected_days if expected_days > 0 else 0

                    if coverage >= 0.8:
                        results[symbol] = df
                        logger.info(f"  {symbol}: loaded {len(df)} bars from cache")
                        continue
                    else:
                        logger.info(
                            f"  {symbol}: cache has {actual_days} bars (coverage {coverage:.0%}), will refresh"
                        )
            except Exception as e:
                logger.warning(f"  {symbol}: cache read failed ({e}), will fetch")

        symbols_to_fetch.append(symbol)

    # Step 2: Return early if all symbols are cached
    if not symbols_to_fetch:
        logger.info(f"All {len(symbols)} symbols loaded from cache")
        return results

    logger.info(f"Fetching {len(symbols_to_fetch)} symbols from IB: {', '.join(symbols_to_fetch)}")

    # Step 3: Fetch missing symbols from IB
    from .data.providers import IbBacktestDataProvider

    logger.info("Waiting 5s for IB connection readiness...")
    await asyncio.sleep(5)

    last_error = None
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                wait_time = 15 * attempt
                logger.info(f"Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                await asyncio.sleep(wait_time)

            provider = IbBacktestDataProvider(
                host="127.0.0.1",
                port=4001,
                client_id=4,
                rate_limit=True,
            )

            await provider.connect()
            try:
                fetched_data = await provider.fetch_bars(
                    symbols=symbols_to_fetch,
                    start=start_dt,
                    end=end_dt,
                    timeframe=timeframe,
                )
            finally:
                await provider.disconnect()

            # Step 4: Save fetched data to Parquet cache for future use
            for symbol, df in fetched_data.items():
                if not df.empty:
                    results[symbol] = df
                    # Save to cache
                    try:
                        parquet_path = base_dir / symbol.upper() / f"{timeframe}.parquet"
                        parquet_path.parent.mkdir(parents=True, exist_ok=True)

                        # Reset index for storage
                        df_to_save = df.reset_index() if df.index.name else df.copy()
                        if "timestamp" not in df_to_save.columns and df.index.name == "timestamp":
                            df_to_save = df.reset_index()

                        df_to_save.to_parquet(parquet_path, compression="snappy")
                        logger.info(f"  {symbol}: cached {len(df)} bars to {parquet_path}")
                    except Exception as e:
                        logger.warning(f"  {symbol}: failed to cache ({e})")
                else:
                    results[symbol] = pd.DataFrame()

            break  # Success, exit retry loop

        except Exception as e:
            last_error = e
            logger.warning(f"Pre-fetch attempt {attempt + 1} failed: {e}")
    else:
        # All retries failed
        if symbols_to_fetch and not any(s in results for s in symbols_to_fetch):
            raise RuntimeError(
                f"Failed to pre-fetch data after {max_retries} attempts: {last_error}"
            )

    successful = sum(1 for df in results.values() if not df.empty)
    cached_count = len(symbols) - len(symbols_to_fetch)
    fetched_count = successful - cached_count

    logger.info(
        f"Pre-fetch complete: {successful}/{len(symbols)} symbols with data "
        f"({cached_count} cached, {fetched_count} fetched)"
    )
    return results


# Module-level globals for multiprocessing worker state
# Each worker process has its own copy (no shared state issues)
_vectorbt_cached_data: Optional[Dict[str, Any]] = None
_vectorbt_secondary_data: Optional[Dict[str, Dict[str, Any]]] = None  # {symbol: {timeframe: df}}
_vectorbt_engine: Optional[Any] = None  # VectorBTEngine, typed as Any to avoid circular import


def _init_vectorbt_worker(
    cached_data: Optional[Dict[str, Any]],
    config_dict: Dict[str, Any],
    secondary_data: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """
    Worker initializer for VectorBT backtests.

    Called once per worker process to set up the engine and cached data.
    This avoids pickling issues with closures by using module-level globals.

    Args:
        cached_data: Primary timeframe data {symbol: DataFrame}
        config_dict: VectorBTConfig as dict
        secondary_data: Secondary timeframe data {symbol: {timeframe: DataFrame}}
    """
    global _vectorbt_cached_data, _vectorbt_secondary_data, _vectorbt_engine
    from .execution.engines import VectorBTConfig, VectorBTEngine

    _vectorbt_cached_data = cached_data
    _vectorbt_secondary_data = secondary_data
    _vectorbt_engine = VectorBTEngine(VectorBTConfig(**config_dict))


def _run_vectorbt_backtest(spec: "RunSpec") -> "RunResult":
    """
    Top-level backtest function for multiprocessing.

    This function runs in worker processes after _init_vectorbt_worker has
    set up the engine. It's a plain function (not a closure) so it pickles correctly.
    """
    if _vectorbt_engine is None:
        raise RuntimeError("VectorBT engine not initialized. Call _init_vectorbt_worker first.")

    symbol_data = _vectorbt_cached_data.get(spec.symbol) if _vectorbt_cached_data else None

    # Get secondary timeframe data for this symbol
    symbol_secondary = None
    if _vectorbt_secondary_data and spec.symbol in _vectorbt_secondary_data:
        symbol_secondary = _vectorbt_secondary_data[spec.symbol]

    return _vectorbt_engine.run(spec, data=symbol_data, secondary_data=symbol_secondary)


def create_vectorbt_backtest_fn(
    cached_data: Optional[Dict[str, Any]] = None,
    secondary_data: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Create a backtest function using VectorBT engine.

    Returns a top-level function with multiprocessing metadata attached
    so ParallelRunner can properly initialize worker processes.

    Args:
        cached_data: Primary timeframe data {symbol: DataFrame}
        secondary_data: Secondary timeframe data {symbol: {timeframe: DataFrame}}
    """
    from dataclasses import asdict

    from .execution.engines import VectorBTConfig

    if cached_data:
        config = VectorBTConfig(data_source="local")
    else:
        config = VectorBTConfig(data_source="ib", ib_port=4001)

    config_dict = asdict(config)

    # Initialize in the main process for sequential execution
    _init_vectorbt_worker(cached_data, config_dict, secondary_data)

    # Attach multiprocessing metadata for ParallelRunner to use
    _run_vectorbt_backtest._mp_initializer = _init_vectorbt_worker  # type: ignore
    _run_vectorbt_backtest._mp_initargs = (cached_data, config_dict, secondary_data)  # type: ignore
    _run_vectorbt_backtest._mp_context = "spawn"  # type: ignore - safest for cross-platform

    return _run_vectorbt_backtest


# =============================================================================
# ApexEngine Backtest Functions (for MTF / apex_only strategies)
# =============================================================================

# Module-level globals for ApexEngine worker state
_apex_cached_data: Optional[Dict[str, Dict[str, Any]]] = None  # {symbol: {timeframe: DataFrame}}
_apex_engine: Optional[Any] = None


def is_apex_required(strategy_name: str) -> bool:
    """
    Check if a strategy requires ApexEngine (cannot run in VectorBT).

    A strategy requires ApexEngine if:
    - It's marked apex_only: true in manifest.yaml
    - It has no vectorized signal generator (signals: null)

    Note: multi_timeframe strategies CAN run in VectorBT if they have
    a SignalGenerator that accepts secondary_data parameter.

    Args:
        strategy_name: Strategy name to check

    Returns:
        True if strategy requires ApexEngine
    """
    manifest_path = Path(__file__).parent.parent / "domain/strategy/manifest.yaml"
    if not manifest_path.exists():
        return False

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}

    strategies = manifest.get("strategies", {})
    if strategy_name not in strategies:
        return False

    entry = strategies[strategy_name]
    return entry.get("apex_only", False) or entry.get("signals") is None


def _init_apex_worker(
    cached_data: Optional[Dict[str, Dict[str, Any]]],
    config_dict: Dict[str, Any],
) -> None:
    """
    Worker initializer for ApexEngine backtests.

    Called once per worker process to set up the engine and cached data.
    """
    global _apex_cached_data, _apex_engine
    from .execution.engines import ApexEngine, ApexEngineConfig

    _apex_cached_data = cached_data
    _apex_engine = ApexEngine(ApexEngineConfig(**config_dict))


def _run_apex_backtest(spec: "RunSpec") -> "RunResult":
    """
    Top-level backtest function for ApexEngine multiprocessing.

    This runs bar-by-bar event-driven backtests via ApexEngine.
    """
    if _apex_engine is None:
        raise RuntimeError("ApexEngine not initialized. Call _init_apex_worker first.")

    # ApexEngine uses HistoricalStoreDataFeed internally, not cached data
    # The cached_data is kept for potential future use (pre-loaded DataFrames)
    return _apex_engine.run(spec, data=None)


def create_apex_backtest_fn(
    cached_data: Optional[Dict[str, Dict[str, Any]]] = None,
    data_source: str = "historical",
):
    """
    Create a backtest function using ApexEngine.

    Used for MTF (multi-timeframe) strategies and apex_only strategies
    that cannot run in vectorized VectorBT mode.

    Args:
        cached_data: Optional pre-loaded data {symbol: {timeframe: DataFrame}}
        data_source: Data source type ("historical" for parquet files)

    Returns:
        Backtest function with multiprocessing metadata
    """
    from dataclasses import asdict

    from .execution.engines import ApexEngineConfig

    config = ApexEngineConfig(
        data_source=data_source,
        bar_size="1d",  # Primary timeframe; secondary set per-spec
    )
    config_dict = asdict(config)

    # Initialize in main process for sequential execution
    _init_apex_worker(cached_data, config_dict)

    # Attach multiprocessing metadata
    _run_apex_backtest._mp_initializer = _init_apex_worker  # type: ignore
    _run_apex_backtest._mp_initargs = (cached_data, config_dict)  # type: ignore
    _run_apex_backtest._mp_context = "spawn"  # type: ignore

    return _run_apex_backtest


async def run_systematic_experiment(
    spec_path: str,
    output_dir: str = "results/experiments",
    parallel: int = 0,  # 0 = auto-scale based on workload
    dry_run: bool = False,
    generate_report: bool = True,  # Default ON
):
    """Run a systematic backtest experiment."""
    from . import ExperimentSpec, RunnerConfig, SystematicRunner

    spec = ExperimentSpec.from_yaml(spec_path)
    logger.info(f"Loaded experiment: {spec.name}")
    logger.info(f"  Strategy: {spec.strategy}")
    logger.info(f"  Experiment ID: {spec.experiment_id}")

    param_combinations = spec.expand_parameter_grid()
    symbols = spec.get_symbols()

    logger.info(f"  Parameter combinations: {len(param_combinations)}")
    logger.info(f"  Symbols: {len(symbols)} ({', '.join(symbols)})")
    logger.info(f"  Folds: {spec.temporal.folds}")

    total_runs = len(param_combinations) * len(symbols) * spec.temporal.folds
    logger.info(f"  Total runs: {total_runs}")

    if dry_run:
        logger.info("Dry run - would execute the above")
        logger.info("First 5 parameter combinations:")
        for i, params in enumerate(param_combinations[:5]):
            logger.info(f"  {i+1}: {params}")
        if len(param_combinations) > 5:
            logger.info(f"  ... and {len(param_combinations) - 5} more")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    db_path = output_path / f"{spec.experiment_id}.db"

    logger.info(f"  Output: {db_path}")

    config = RunnerConfig(
        db_path=str(db_path),
        parallel_workers=parallel,
        skip_existing=True,
    )
    runner = SystematicRunner(config=config)

    # Prefetch data for all timeframes (primary + secondary)
    primary_tf = spec.data.primary_timeframe or "1d"
    secondary_tfs = spec.data.secondary_timeframes or []

    logger.info(f"  Primary timeframe: {primary_tf}")
    if secondary_tfs:
        logger.info(f"  Secondary timeframes: {secondary_tfs}")

    # Load primary timeframe data
    cached_data = await prefetch_data(
        symbols=symbols,
        start_date=spec.temporal.start_date,
        end_date=spec.temporal.end_date,
        timeframe=primary_tf,
    )

    # Load secondary timeframe data (MTF support)
    secondary_data: Dict[str, Dict[str, Any]] = {}
    for tf in secondary_tfs:
        tf_data = await prefetch_data(
            symbols=symbols,
            start_date=spec.temporal.start_date,
            end_date=spec.temporal.end_date,
            timeframe=tf,
        )
        for symbol, df in tf_data.items():
            if symbol not in secondary_data:
                secondary_data[symbol] = {}
            secondary_data[symbol][tf] = df

    # Create backtest function with MTF data
    backtest_fn = create_vectorbt_backtest_fn(
        cached_data=cached_data,
        secondary_data=secondary_data if secondary_tfs else None,
    )

    start_time = datetime.now()
    logger.info("Starting experiment execution...")

    try:
        experiment_id = runner.run(
            spec,
            backtest_fn=backtest_fn,
            on_trial_complete=lambda t: logger.debug(
                f"  Trial {t.trial_index}: score={t.trial_score:.3f}"
            ),
        )

        duration = (datetime.now() - start_time).total_seconds()

        result = runner.get_experiment_result(experiment_id)
        result.total_duration_seconds = duration
        result.print_summary()

        top_trials = runner.get_top_trials(experiment_id, limit=5)
        logger.info("Top 5 trials by score:")
        for i, trial in enumerate(top_trials, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in trial["params"].items())
            logger.info(
                f"  {i}. Score={trial['trial_score']:.3f} "
                f"Sharpe={trial['median_sharpe']:.2f} "
                f"MaxDD={trial['median_max_dd']:.1%} "
                f"Params=[{params_str}]"
            )

        logger.info(f"Experiment complete! Results saved to: {db_path}")

        if generate_report:
            try:
                pass

                report_path = _generate_experiment_report(runner, experiment_id, spec, output_path)
                print(f"\nHTML Report: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to generate HTML report: {e}")

    finally:
        runner.close()


def _query_per_symbol_metrics(
    runner, experiment_id: str, best_trial_id: str
) -> Dict[str, Dict[str, Any]]:
    """Query per-symbol aggregated metrics from the runs table for the best trial."""
    rows = runner._db.fetchall(
        """
        SELECT
            symbol,
            AVG(sharpe) as sharpe,
            AVG(sortino) as sortino,
            AVG(calmar) as calmar,
            AVG(total_return) as total_return,
            AVG(cagr) as cagr,
            AVG(max_drawdown) as max_drawdown,
            SUM(total_trades) as total_trades,
            AVG(win_rate) as win_rate,
            AVG(profit_factor) as profit_factor,
            AVG(expectancy) as expectancy,
            AVG(best_trade_pct) as best_trade_pct,
            AVG(worst_trade_pct) as worst_trade_pct,
            AVG(avg_win_pct) as avg_win_pct,
            AVG(avg_loss_pct) as avg_loss_pct,
            AVG(avg_trade_duration_days) as avg_trade_duration_days,
            COUNT(*) as run_count
        FROM runs
        WHERE experiment_id = ? AND trial_id = ?
        GROUP BY symbol
        ORDER BY symbol
        """,
        (experiment_id, best_trial_id),
    )

    per_symbol = {}
    for row in rows:
        per_symbol[row[0]] = {
            "sharpe": row[1] or 0,
            "sortino": row[2] or 0,
            "calmar": row[3] or 0,
            "total_return": row[4] or 0,
            "cagr": row[5] or 0,
            "max_drawdown": row[6] or 0,
            "total_trades": int(row[7] or 0),
            "win_rate": row[8] or 0,
            "profit_factor": row[9] or 0,
            "expectancy": row[10] or 0,
            "best_trade_pct": row[11] or 0,
            "worst_trade_pct": row[12] or 0,
            "avg_win_pct": row[13] or 0,
            "avg_loss_pct": row[14] or 0,
            "avg_trade_duration_days": row[15] or 0,
            "run_count": int(row[16] or 0),
        }

    return per_symbol


def _query_per_window_metrics(
    runner, experiment_id: str, best_trial_id: str
) -> List[Dict[str, Any]]:
    """Query per-window (fold) metrics from the runs table for the best trial."""
    rows = runner._db.fetchall(
        """
        SELECT
            window_id,
            AVG(CASE WHEN is_train = true THEN sharpe END) as is_sharpe,
            AVG(CASE WHEN is_oos = true THEN sharpe END) as oos_sharpe,
            AVG(CASE WHEN is_train = true THEN total_return END) as is_return,
            AVG(CASE WHEN is_oos = true THEN total_return END) as oos_return,
            AVG(CASE WHEN is_train = true THEN max_drawdown END) as is_max_dd,
            AVG(CASE WHEN is_oos = true THEN max_drawdown END) as oos_max_dd,
            MIN(started_at) as start_time,
            MAX(completed_at) as end_time
        FROM runs
        WHERE experiment_id = ? AND trial_id = ?
        GROUP BY window_id
        ORDER BY window_id
        """,
        (experiment_id, best_trial_id),
    )

    per_window = []
    for row in rows:
        is_sharpe = row[1] or 0
        oos_sharpe = row[2] or 0
        degradation = 0.0
        if is_sharpe != 0:
            degradation = (is_sharpe - oos_sharpe) / abs(is_sharpe) if is_sharpe else 0

        per_window.append(
            {
                "window_id": row[0],
                "is_sharpe": is_sharpe,
                "oos_sharpe": oos_sharpe,
                "is_return": row[3] or 0,
                "oos_return": row[4] or 0,
                "is_max_dd": row[5] or 0,
                "oos_max_dd": row[6] or 0,
                "degradation": degradation,
                "start_time": str(row[7]) if row[7] else "",
                "end_time": str(row[8]) if row[8] else "",
            }
        )

    return per_window


def _build_equity_curve(
    runner, experiment_id: str, best_trial_id: str, initial_capital: float = 100000.0
) -> List[Dict[str, Any]]:
    """Build a simulated equity curve from OOS returns per window."""
    rows = runner._db.fetchall(
        """
        SELECT
            window_id,
            AVG(total_return) as avg_return,
            MAX(completed_at) as end_date
        FROM runs
        WHERE experiment_id = ? AND trial_id = ? AND is_oos = true
        GROUP BY window_id
        ORDER BY window_id
        """,
        (experiment_id, best_trial_id),
    )

    equity_curve = []
    equity = initial_capital

    for row in rows:
        avg_return = row[1] or 0
        end_date = row[2]
        equity = equity * (1 + avg_return)
        equity_curve.append(
            {
                "date": str(end_date)[:10] if end_date else row[0],
                "equity": round(equity, 2),
                "return": round(avg_return * 100, 2),
            }
        )

    return equity_curve


def _query_trade_summary(runner, experiment_id: str, best_trial_id: str) -> List[Dict[str, Any]]:
    """Query trade summary per run (individual trades not stored, return run-level summaries)."""
    rows = runner._db.fetchall(
        """
        SELECT
            run_id,
            symbol,
            window_id,
            is_oos,
            total_trades,
            win_rate,
            profit_factor,
            total_return,
            best_trade_pct,
            worst_trade_pct,
            avg_win_pct,
            avg_loss_pct,
            avg_trade_duration_days
        FROM runs
        WHERE experiment_id = ? AND trial_id = ? AND total_trades > 0
        ORDER BY window_id, symbol
        LIMIT 200
        """,
        (experiment_id, best_trial_id),
    )

    trades = []
    for row in rows:
        trades.append(
            {
                "run_id": row[0][:12] if row[0] else "",
                "symbol": row[1],
                "window": row[2],
                "is_oos": "OOS" if row[3] else "IS",
                "trade_count": int(row[4] or 0),
                "win_rate": round((row[5] or 0) * 100, 1),
                "profit_factor": round(row[6] or 0, 2),
                "return_pct": round((row[7] or 0) * 100, 2),
                "best_pct": round((row[8] or 0) * 100, 2),
                "worst_pct": round((row[9] or 0) * 100, 2),
                "avg_win_pct": round((row[10] or 0) * 100, 2),
                "avg_loss_pct": round((row[11] or 0) * 100, 2),
                "avg_duration": round(row[12] or 0, 1),
            }
        )

    return trades


def _generate_experiment_report(runner, experiment_id: str, spec, output_dir: Path) -> Path:
    """Generate HTML report for completed experiment."""
    import numpy as np

    from .analysis.reporting import HTMLReportGenerator, ReportConfig, ReportData

    logger.info("Generating HTML report...")

    result = runner.get_experiment_result(experiment_id)
    top_trials = runner.get_top_trials(experiment_id, limit=50)
    symbols = spec.get_symbols()

    # Get best trial ID for querying run-level data
    best_trial_id = top_trials[0]["trial_id"] if top_trials else None

    # Aggregate metrics from top trials
    agg_metrics = {}
    if top_trials:
        metric_values: Dict[str, List[float]] = {}
        for trial in top_trials:
            for our_key, trial_key in [
                ("sharpe", "median_sharpe"),
                ("max_drawdown", "median_max_dd"),
                ("total_return", "median_total_return"),
            ]:
                if trial_key in trial and trial[trial_key] is not None:
                    if our_key not in metric_values:
                        metric_values[our_key] = []
                    metric_values[our_key].append(float(trial[trial_key]))
        agg_metrics = {k: float(np.median(v)) for k, v in metric_values.items() if v}

    best_params = top_trials[0]["params"] if top_trials else {}
    best_score = top_trials[0].get("trial_score", 0.0) if top_trials else 0.0

    # Query per-symbol metrics from runs table
    per_symbol = {}
    per_window = []
    equity_curve = []
    trades = []

    if best_trial_id:
        try:
            per_symbol = _query_per_symbol_metrics(runner, experiment_id, best_trial_id)
            per_window = _query_per_window_metrics(runner, experiment_id, best_trial_id)
            equity_curve = _build_equity_curve(runner, experiment_id, best_trial_id)
            trades = _query_trade_summary(runner, experiment_id, best_trial_id)
            logger.info(
                f"Loaded report data: {len(per_symbol)} symbols, {len(per_window)} windows, {len(trades)} trade summaries"
            )
        except Exception as e:
            logger.warning(f"Failed to load detailed report data: {e}")

    # Ensure all symbols have entries (even if no data)
    for s in symbols:
        if s not in per_symbol:
            per_symbol[s] = {}

    report_data = ReportData(
        experiment_id=experiment_id,
        strategy_name=spec.strategy,
        code_version=spec.reproducibility.code_version if spec.reproducibility else "",
        data_version=spec.reproducibility.data_version if spec.reproducibility else "",
        start_date=str(spec.temporal.start_date) if spec.temporal.start_date else "auto",
        end_date=str(spec.temporal.end_date) if spec.temporal.end_date else "auto",
        symbols=symbols,
        n_folds=spec.temporal.folds,
        train_days=spec.temporal.train_days,
        test_days=spec.temporal.test_days,
        total_trials=result.total_trials,
        best_params=best_params,
        best_trial_score=best_score,
        metrics=agg_metrics,
        validation={
            "successful_trials": result.successful_trials,
            "success_rate": (
                result.successful_trials / result.total_trials if result.total_trials > 0 else 0
            ),
            "total_runs": result.total_runs,
            "successful_runs": result.successful_runs,
            "pbo": result.pbo if result.pbo is not None else 0.0,
            "dsr": result.dsr if result.dsr is not None else 0.0,
        },
        per_symbol=per_symbol,
        per_window=per_window,
        equity_curve=equity_curve,
        trades=trades,
    )

    report_config = ReportConfig(title=f"Backtest Report: {spec.name}", theme="light")
    generator = HTMLReportGenerator(config=report_config)

    report_path = output_dir / f"{experiment_id}_report.html"
    generated_path = generator.generate(report_data, report_path)

    logger.info(f"HTML report generated: {generated_path}")
    return generated_path


# =============================================================================
# Backtrader Runner (Alternative Engine)
# =============================================================================


class BacktraderRunner:
    """Runner using Backtrader engine. Requires: pip install backtrader"""

    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100000.0,
        data_source: str = "csv",
        data_dir: str = "./data",
        bar_size: str = "1d",
        strategy_params: Optional[Dict[str, Any]] = None,
        commission: float = 0.001,
    ):
        if not BACKTRADER_AVAILABLE:
            raise ImportError("backtrader not installed. Run: pip install backtrader")

        self.strategy_name = strategy_name
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data_source = data_source
        self.data_dir = data_dir
        self.bar_size = bar_size
        self.strategy_params = strategy_params or {}
        self.commission = commission
        self._spec: Optional[BacktestSpec] = None

    async def run(self) -> BacktestResult:
        """Run the backtest using Backtrader engine."""
        import time

        from ..domain.reality import RealityModelPack, get_preset_pack
        from .execution.backtrader_adapter import run_backtest_with_backtrader

        self._print_config()

        reality_pack = None
        if self._spec:
            if self._spec.reality_model:
                try:
                    reality_pack = RealityModelPack.from_config(self._spec.reality_model)
                except Exception as e:
                    logger.error(f"Failed to load reality_model from spec: {e}")

            if (
                reality_pack is None
                and hasattr(self._spec.execution, "reality_pack")
                and self._spec.execution.reality_pack
            ):
                try:
                    reality_pack = get_preset_pack(self._spec.execution.reality_pack)
                except Exception as e:
                    logger.error(f"Failed to load reality_pack preset: {e}")

        start_time = time.time()

        strategy_class = get_strategy_class(self.strategy_name)
        if strategy_class is None:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. Available: {list_strategies()}"
            )

        data_feeds = self._create_data_feeds()

        results = run_backtest_with_backtrader(
            apex_strategy_class=strategy_class,
            data_feeds=data_feeds,
            initial_cash=self.initial_capital,
            commission=self.commission,
            strategy_params=self.strategy_params,
            reality_pack=reality_pack,
        )

        run_duration = time.time() - start_time
        result = self._convert_result(results, run_duration)
        result.print_summary()

        return result

    def _create_data_feeds(self) -> List[Any]:
        """Create Backtrader data feeds."""
        feeds = []

        for symbol in self.symbols:
            if self.data_source == "csv":
                csv_path = Path(self.data_dir) / f"{symbol}.csv"
                if not csv_path.exists():
                    raise FileNotFoundError(f"CSV file not found: {csv_path}")

                feed = bt.feeds.GenericCSVData(
                    dataname=str(csv_path),
                    dtformat="%Y-%m-%d",
                    fromdate=datetime.combine(self.start_date, datetime.min.time()),
                    todate=datetime.combine(self.end_date, datetime.max.time()),
                    datetime=0,
                    open=1,
                    high=2,
                    low=3,
                    close=4,
                    volume=5,
                    openinterest=-1,
                )
                feed._name = symbol
                feeds.append(feed)

            elif self.data_source == "yahoo":
                feed = bt.feeds.YahooFinanceData(
                    dataname=symbol,
                    fromdate=datetime.combine(self.start_date, datetime.min.time()),
                    todate=datetime.combine(self.end_date, datetime.max.time()),
                )
                feed._name = symbol
                feeds.append(feed)

            else:
                raise ValueError(f"Unsupported data source for Backtrader: {self.data_source}")

        return feeds

    def _convert_result(self, bt_results: Dict[str, Any], run_duration: float) -> BacktestResult:
        """Convert Backtrader results to BacktestResult."""
        from ..domain.backtest.backtest_result import (
            CostMetrics,
            PerformanceMetrics,
            RiskMetrics,
            TradeMetrics,
        )

        initial = self.initial_capital
        final = bt_results.get("final_value", initial)
        total_return = (final - initial) / initial if initial > 0 else 0

        trading_days = (self.end_date - self.start_date).days * 252 // 365

        performance = PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return * 100,
            cagr=self._calculate_cagr(initial, final, trading_days),
            annualized_return=total_return * 252 / max(trading_days, 1),
        )

        sharpe = bt_results.get("sharpe_ratio") or 0.0
        max_dd = bt_results.get("max_drawdown") or 0.0
        risk = RiskMetrics(max_drawdown=max_dd, max_drawdown_duration_days=0, sharpe_ratio=sharpe)

        total_trades = bt_results.get("total_trades", 0)
        trades = TradeMetrics(total_trades=total_trades)

        estimated_commission = total_trades * 2 * 100 * self.commission
        costs = CostMetrics(
            total_commission=estimated_commission,
            cost_pct_of_capital=(estimated_commission / initial * 100) if initial > 0 else 0,
        )

        return BacktestResult(
            strategy_name=self.strategy_name,
            strategy_id=f"backtrader-{self.strategy_name}",
            start_date=self.start_date,
            end_date=self.end_date,
            trading_days=trading_days,
            initial_capital=initial,
            final_capital=final,
            symbols=self.symbols,
            performance=performance,
            risk=risk,
            trades=trades,
            costs=costs,
            equity_curve=[],
            run_duration_seconds=run_duration,
            engine="backtrader",
        )

    def _calculate_cagr(self, initial: float, final: float, days: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        if initial <= 0 or days <= 0:
            return 0.0
        years = days / 252
        if years <= 0:
            return 0.0
        return ((final / initial) ** (1 / years) - 1) * 100

    def _print_config(self) -> None:
        """Log backtest configuration using structured logging."""
        config_info = {
            "engine": "backtrader",
            "strategy": self.strategy_name,
            "symbols": self.symbols,
            "start": str(self.start_date),
            "end": str(self.end_date),
            "capital": self.initial_capital,
            "data_source": self.data_source,
            "bar_size": self.bar_size,
            "commission": self.commission,
        }

        if self.strategy_params:
            config_info["params"] = self.strategy_params

        logger.info(
            f"Backtrader config: {self.strategy_name} on {','.join(self.symbols)} "
            f"({self.start_date} to {self.end_date})"
        )
        logger.debug(f"Full config: {config_info}")


# =============================================================================
# CLI (delegated to cli/ module)
# =============================================================================

# Re-export CLI functions for backward compatibility
from .cli import main

if __name__ == "__main__":
    main()
