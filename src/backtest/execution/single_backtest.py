"""
Single Backtest Runner using ApexEngine (bar-by-bar, event-driven).

Use for:
- Full execution simulation with order matching
- Testing strategy logic with realistic fills
- Small to medium datasets
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...domain.backtest.backtest_result import BacktestResult
from ...domain.backtest.backtest_spec import BacktestSpec
from ...domain.strategy.registry import get_strategy_class, list_strategies
from ..config import load_historical_data_config, load_ib_config
from ..data.feeds import (
    CachedBarDataFeed,
    CsvDataFeed,
    HistoricalStoreDataFeed,
    ParquetDataFeed,
    StreamingCsvDataFeed,
    StreamingParquetDataFeed,
)
from .simulated import FillModel

logger = logging.getLogger(__name__)


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
        from ...domain.strategy.examples import (  # noqa
            BuyAndHoldStrategy,
            MovingAverageCrossStrategy,
        )
        from .engines.backtest_engine import BacktestConfig, BacktestEngine

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

    def _create_data_feed(self) -> Any:
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

        from ...services.historical_data_manager import HistoricalDataManager

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
                    from ...infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

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
