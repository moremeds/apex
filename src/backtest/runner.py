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
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..domain.backtest.backtest_result import BacktestResult
from ..domain.backtest.backtest_spec import BacktestSpec
from ..domain.strategy.registry import get_strategy_class, get_strategy_info, list_strategies
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

        runner = cls(
            strategy_name=spec.strategy.name,
            symbols=spec.get_symbols(),
            start_date=spec.data.start_date or date(2024, 1, 1),
            end_date=spec.data.end_date or date(2024, 12, 31),
            initial_capital=spec.execution.initial_capital,
            data_source=spec.data.source,
            data_dir=spec.data.csv_dir or spec.data.parquet_dir or "./data",
            bar_size=spec.data.bar_size,
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
            source_priority = [s.strip().lower() for s in args.source_priority.split(",") if s.strip()]

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
        from .execution.engines.backtest_engine import BacktestConfig, BacktestEngine

        # Import to register strategies
        from ..domain.strategy.examples import BuyAndHoldStrategy, MovingAverageCrossStrategy  # noqa

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
            raise ValueError(f"Unknown strategy: {self.strategy_name}. Available: {list_strategies()}")
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
            raise ValueError(f"Unknown data source: {self.data_source}. Use 'cached', 'historical', 'csv', or 'parquet'.")

    def _print_config(self) -> None:
        """Print backtest configuration."""
        print(f"\n{'=' * 60}")
        print("BACKTEST CONFIGURATION (ApexEngine)")
        print(f"{'=' * 60}")
        print(f"Strategy:     {self.strategy_name}")
        print(f"Symbols:      {', '.join(self.symbols)}")
        print(f"Period:       {self.start_date} to {self.end_date}")
        print(f"Capital:      ${self.initial_capital:,.2f}")
        print(f"Data Source:  {self.data_source}")
        if self.data_source == "historical":
            historical_cfg = load_historical_data_config()
            base_dir = self.historical_dir or historical_cfg.get("base_dir", "data/historical")
            sources = self.source_priority or historical_cfg.get("source_priority", ["ib", "yahoo"])
            coverage_mode = self.coverage_mode or "download"
            print(f"Coverage:     mode={coverage_mode}, dir={base_dir}")
            print(f"Sources:      {', '.join(sources)}")
        elif self.data_source in ("csv", "parquet"):
            print(f"Streaming:    {self.streaming}")
        print(f"Bar Size:     {self.bar_size}")
        print(f"Fill Model:   {self.fill_model.value}")
        if self.strategy_params:
            print(f"Parameters:   {self.strategy_params}")
        print(f"{'=' * 60}\n")

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
        source_priority = self.source_priority or historical_cfg.get("source_priority", ["ib", "yahoo"])

        logger.info(f"Coverage check: mode={mode}, base_dir={base_dir}, sources={source_priority}")

        manager = HistoricalDataManager(base_dir=base_dir, source_priority=source_priority)
        ib_adapter = None

        try:
            if "ib" in source_priority and mode == "download":
                ib_config = load_ib_config()
                if ib_config:
                    from ..infrastructure.adapters.ib.historical_adapter import IbHistoricalAdapter

                    client_id = ib_config.client_ids.historical_pool[0] if ib_config.client_ids.historical_pool else 10
                    ib_adapter = IbHistoricalAdapter(
                        host=ib_config.host,
                        port=ib_config.port,
                        client_id=client_id,
                    )
                    try:
                        await ib_adapter.connect()
                        manager.set_ib_source(ib_adapter)
                        logger.info(f"IB historical source connected: {ib_config.host}:{ib_config.port}")
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


async def prefetch_data(symbols: List[str], start_date, end_date, max_retries: int = 3) -> Dict[str, Any]:
    """Pre-fetch all data from IB once before running the experiment.

    This is an async function to avoid nested event loop issues when called
    from within an async context (e.g., main_async).
    """
    from .data.providers import IbBacktestDataProvider

    logger.info(f"Pre-fetching data for {len(symbols)} symbols...")

    if hasattr(start_date, "isoformat"):
        start_dt = datetime.combine(start_date, datetime.min.time())
    else:
        start_dt = datetime.fromisoformat(str(start_date)) if start_date else datetime(2020, 1, 1)

    if hasattr(end_date, "isoformat"):
        end_dt = datetime.combine(end_date, datetime.max.time())
    else:
        end_dt = datetime.fromisoformat(str(end_date)) if end_date else datetime.now()

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

            # Use async method directly - connect, fetch, disconnect
            await provider.connect()
            try:
                data = await provider.fetch_bars(
                    symbols=symbols,
                    start=start_dt,
                    end=end_dt,
                    timeframe="1d",
                )
            finally:
                await provider.disconnect()

            successful = sum(1 for df in data.values() if not df.empty)
            logger.info(f"Pre-fetch complete: {successful}/{len(symbols)} symbols with data")
            return data

        except Exception as e:
            last_error = e
            logger.warning(f"Pre-fetch attempt {attempt + 1} failed: {e}")

    raise RuntimeError(f"Failed to pre-fetch data after {max_retries} attempts: {last_error}")


def create_vectorbt_backtest_fn(cached_data: Optional[Dict[str, Any]] = None):
    """Create a backtest function using VectorBT engine."""
    from .execution.engines import VectorBTConfig, VectorBTEngine

    if cached_data:
        config = VectorBTConfig(data_source="local")
    else:
        config = VectorBTConfig(data_source="ib", ib_port=4001)

    engine = VectorBTEngine(config)

    def backtest_fn(spec):
        symbol_data = cached_data.get(spec.symbol) if cached_data else None
        return engine.run(spec, data=symbol_data)

    return backtest_fn


async def run_systematic_experiment(
    spec_path: str,
    output_dir: str = "results/experiments",
    parallel: int = 1,
    dry_run: bool = False,
    generate_report: bool = False,
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
        print("\nFirst 5 parameter combinations:")
        for i, params in enumerate(param_combinations[:5]):
            print(f"  {i+1}: {params}")
        if len(param_combinations) > 5:
            print(f"  ... and {len(param_combinations) - 5} more")
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

    cached_data = await prefetch_data(
        symbols=symbols,
        start_date=spec.temporal.start_date,
        end_date=spec.temporal.end_date,
    )

    backtest_fn = create_vectorbt_backtest_fn(cached_data=cached_data)

    start_time = datetime.now()
    logger.info("Starting experiment execution...")

    try:
        experiment_id = runner.run(
            spec,
            backtest_fn=backtest_fn,
            on_trial_complete=lambda t: logger.debug(f"  Trial {t.trial_index}: score={t.trial_score:.3f}"),
        )

        duration = (datetime.now() - start_time).total_seconds()

        result = runner.get_experiment_result(experiment_id)
        result.total_duration_seconds = duration
        result.print_summary()

        print("\nTop 5 Trials by Score:")
        print("-" * 80)
        top_trials = runner.get_top_trials(experiment_id, limit=5)
        for i, trial in enumerate(top_trials, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in trial["params"].items())
            print(
                f"  {i}. Score: {trial['trial_score']:.3f} | "
                f"Sharpe: {trial['median_sharpe']:.2f} | "
                f"MaxDD: {trial['median_max_dd']:.1%}"
            )
            print(f"     Params: {params_str}")

        logger.info(f"Experiment complete! Results saved to: {db_path}")

        if generate_report:
            try:
                from .analysis.reporting import HTMLReportGenerator, ReportConfig, ReportData

                report_path = _generate_experiment_report(runner, experiment_id, spec, output_path)
                print(f"\nHTML Report: {report_path}")
            except Exception as e:
                logger.warning(f"Failed to generate HTML report: {e}")

    finally:
        runner.close()


def _generate_experiment_report(runner, experiment_id: str, spec, output_dir: Path) -> Path:
    """Generate HTML report for completed experiment."""
    import numpy as np

    from .analysis.reporting import HTMLReportGenerator, ReportConfig, ReportData

    logger.info("Generating HTML report...")

    result = runner.get_experiment_result(experiment_id)
    top_trials = runner.get_top_trials(experiment_id, limit=50)
    symbols = spec.get_symbols()

    # Aggregate metrics
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
            "success_rate": result.successful_trials / result.total_trials if result.total_trials > 0 else 0,
            "total_runs": result.total_runs,
            "successful_runs": result.successful_runs,
            "pbo": result.pbo if result.pbo is not None else 0.0,
            "dsr": result.dsr if result.dsr is not None else 0.0,
        },
        per_symbol={s: {} for s in symbols},
        per_window=[],
        equity_curve=[],
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

            if reality_pack is None and hasattr(self._spec.execution, "reality_pack") and self._spec.execution.reality_pack:
                try:
                    reality_pack = get_preset_pack(self._spec.execution.reality_pack)
                except Exception as e:
                    logger.error(f"Failed to load reality_pack preset: {e}")

        start_time = time.time()

        strategy_class = get_strategy_class(self.strategy_name)
        if strategy_class is None:
            raise ValueError(f"Unknown strategy: {self.strategy_name}. Available: {list_strategies()}")

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
        from ..domain.backtest.backtest_result import CostMetrics, PerformanceMetrics, RiskMetrics, TradeMetrics

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
        """Print backtest configuration."""
        print(f"\n{'=' * 60}")
        print("BACKTEST CONFIGURATION (Backtrader Engine)")
        print(f"{'=' * 60}")
        print(f"Strategy:     {self.strategy_name}")
        print(f"Symbols:      {', '.join(self.symbols)}")
        print(f"Period:       {self.start_date} to {self.end_date}")
        print(f"Capital:      ${self.initial_capital:,.2f}")
        print(f"Data Source:  {self.data_source}")
        print(f"Bar Size:     {self.bar_size}")
        print(f"Commission:   {self.commission * 100:.2f}%")
        if self.strategy_params:
            print(f"Parameters:   {self.strategy_params}")
        print(f"{'=' * 60}\n")


# =============================================================================
# CLI
# =============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Backtest Runner for APEX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single backtest (ApexEngine - full simulation)
    python -m src.backtest.runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30

    # Systematic experiment (VectorBTEngine - fast optimization)
    python -m src.backtest.runner --spec config/backtest/examples/ta_metrics.yaml

    # Force VectorBT engine for single backtest
    python -m src.backtest.runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30 --engine vectorbt

    # Offline mode (fail if data gaps)
    python -m src.backtest.runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30 --coverage-mode check

    # Use Backtrader engine
    python -m src.backtest.runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30 --engine backtrader

    # List available strategies
    python -m src.backtest.runner --list-strategies
        """,
    )

    # Single backtest options
    parser.add_argument("--strategy", type=str, help="Strategy name (e.g., ma_cross)")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g., AAPL,MSFT)")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital")

    # Systematic experiment options
    parser.add_argument("--spec", type=str, help="Path to experiment YAML spec")
    parser.add_argument("--output", type=str, default="results/experiments", help="Output directory")
    parser.add_argument("--parallel", type=int, default=1, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")

    # Engine selection
    parser.add_argument(
        "--engine",
        type=str,
        choices=["apex", "vectorbt", "backtrader"],
        help="Engine type (default: apex for single, vectorbt for experiment)",
    )

    # Data options
    parser.add_argument(
        "--data-source",
        type=str,
        default="historical",
        choices=["historical", "csv", "parquet"],
        help="Data source",
    )
    parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--bar-size", type=str, default="1d", help="Bar size")
    parser.add_argument("--params", type=str, nargs="*", help="Strategy params as key=value")
    parser.add_argument("--fill-model", type=str, default="immediate", choices=["immediate", "slippage"])
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps")
    parser.add_argument("--commission", type=float, default=0.005, help="Commission per share")

    # Streaming options
    parser.add_argument("--streaming", action="store_true", default=True, help="Use streaming feeds")
    parser.add_argument("--no-streaming", action="store_false", dest="streaming", help="Use full-load feeds")

    # Historical coverage options
    parser.add_argument(
        "--coverage-mode",
        type=str,
        choices=["download", "check", "off"],
        help="Coverage mode for historical source",
    )
    parser.add_argument("--historical-dir", type=str, help="Historical data directory")
    parser.add_argument("--source-priority", type=str, help="Source priority (e.g., ib,yahoo)")

    # Utility options
    parser.add_argument("--list-strategies", action="store_true", help="List strategies and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser


async def main_async():
    """Async main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.list_strategies:
        print("\nAvailable strategies:")
        for name in sorted(list_strategies()):
            info = get_strategy_info(name)
            desc = info.get("description", "No description") if info else "No description"
            print(f"  {name:20s} - {desc}")
        return

    # Systematic experiment mode
    if args.spec and not args.strategy:
        await run_systematic_experiment(
            spec_path=args.spec,
            output_dir=args.output,
            parallel=args.parallel,
            dry_run=args.dry_run,
            generate_report=args.report,
        )
        return

    # Single backtest mode
    if not args.strategy:
        parser.error("--strategy or --spec required")
    if not args.symbols:
        parser.error("--symbols required")
    if not args.start or not args.end:
        parser.error("--start and --end required")

    try:
        engine_type = args.engine or "apex"

        if engine_type == "backtrader":
            runner = BacktraderRunner(
                strategy_name=args.strategy,
                symbols=[s.strip() for s in args.symbols.split(",")],
                start_date=datetime.strptime(args.start, "%Y-%m-%d").date(),
                end_date=datetime.strptime(args.end, "%Y-%m-%d").date(),
                initial_capital=args.capital,
                data_source=args.data_source,
                data_dir=args.data_dir,
                bar_size=args.bar_size,
                commission=args.commission,
            )
            result = await runner.run()
        elif engine_type == "vectorbt":
            # Use VectorBT for single backtest
            from .core import RunSpec
            from .execution.engines import VectorBTConfig, VectorBTEngine

            symbols = [s.strip() for s in args.symbols.split(",")]
            start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
            end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

            config = VectorBTConfig(data_source="ib", ib_port=4001)
            engine = VectorBTEngine(config)

            # Run each symbol
            for symbol in symbols:
                spec = RunSpec(
                    strategy=args.strategy,
                    symbol=symbol,
                    start=datetime.combine(start_date, datetime.min.time()),
                    end=datetime.combine(end_date, datetime.max.time()),
                    params={},
                )
                result = engine.run(spec)
                print(f"\n{symbol}: Return={result.total_return:.2%}, Sharpe={result.sharpe:.2f}")
            return
        else:
            # Default: ApexEngine
            runner = SingleBacktestRunner.from_args(args)
            result = await runner.run()

        sys.exit(0 if result.is_profitable else 1)

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
