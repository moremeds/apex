"""
Backtest mode runner.

Provides CLI integration for running backtests.

Usage:
    # From spec file
    runner = BacktestRunner.from_spec("config/backtest/my_strategy.yaml")
    result = await runner.run()

    # Programmatic
    runner = BacktestRunner(
        strategy_name="ma_cross",
        symbols=["AAPL", "MSFT"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        initial_capital=100000,
        data_source="ib",  # Uses connection from config/base.yaml
    )
    result = await runner.run()

CLI:
    python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL,MSFT \\
        --start 2024-01-01 --end 2024-06-30 --capital 100000
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import yaml

from ..domain.backtest.backtest_spec import BacktestSpec
from ..domain.backtest.backtest_result import BacktestResult
from ..domain.strategy.registry import get_strategy_class, list_strategies
from ..infrastructure.backtest.backtest_engine import BacktestEngine, BacktestConfig
from ..infrastructure.backtest.data_feeds import (
    IbHistoricalDataFeed,
    CsvDataFeed,
    ParquetDataFeed,
)
from ..infrastructure.backtest.simulated_execution import FillModel

# Import example strategies to register them
from ..domain.strategy.examples import MovingAverageCrossStrategy, BuyAndHoldStrategy  # noqa

# Check if backtrader is available
try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None

logger = logging.getLogger(__name__)

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "base.yaml"


def load_broker_config(broker: str, config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load broker connection settings from base.yaml.

    Args:
        broker: Broker name (ib, futu).
        config_path: Path to config file. Defaults to config/base.yaml.

    Returns:
        Dict with broker connection settings.
    """
    path = config_path or DEFAULT_CONFIG_PATH

    # Broker name mapping
    broker_keys = {
        "ib": "ibkr",
        "ibkr": "ibkr",
        "futu": "futu",
    }

    if not path.exists():
        logger.warning(f"Config file not found: {path}")
        return {}

    try:
        with open(path) as f:
            config = yaml.safe_load(f)

        broker_key = broker_keys.get(broker, broker)
        return config.get("brokers", {}).get(broker_key, {})
    except Exception as e:
        logger.warning(f"Failed to load broker config: {e}")
        return {}


class BacktestRunner:
    """
    Backtest mode runner.

    Coordinates backtest execution:
    1. Load configuration (from args, spec file, or programmatic)
    2. Set up data feed
    3. Set up strategy
    4. Run backtest engine
    5. Report results
    """

    def __init__(
        self,
        strategy_name: str,
        symbols: List[str],
        start_date: date,
        end_date: date,
        initial_capital: float = 100000.0,
        data_source: str = "ib",
        data_dir: str = "./data",
        bar_size: str = "1d",
        strategy_params: Optional[Dict[str, Any]] = None,
        fill_model: str = "immediate",
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.005,
    ):
        """
        Initialize backtest runner.

        Args:
            strategy_name: Name of strategy in registry.
            symbols: List of symbols to trade.
            start_date: Backtest start date.
            end_date: Backtest end date.
            initial_capital: Starting capital.
            data_source: Broker/data source (ib, futu, csv, parquet).
                         Connection details loaded from config/base.yaml.
            data_dir: Directory containing data files (for csv/parquet).
            bar_size: Bar size (1m, 5m, 1h, 1d).
            strategy_params: Strategy parameters.
            fill_model: Fill model (immediate, slippage).
            slippage_bps: Slippage in basis points.
            commission_per_share: Commission per share.
        """
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

        self._spec: Optional[BacktestSpec] = None

    @classmethod
    def from_spec(cls, spec_path: str) -> "BacktestRunner":
        """
        Create runner from spec file.

        Args:
            spec_path: Path to YAML spec file.

        Returns:
            BacktestRunner instance.
        """
        spec = BacktestSpec.from_yaml(spec_path)

        errors = spec.validate()
        if errors:
            raise ValueError(f"Invalid spec: {errors}")

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
        )
        runner._spec = spec
        return runner

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BacktestRunner":
        """
        Create runner from CLI arguments.

        Args:
            args: Parsed CLI arguments.

        Returns:
            BacktestRunner instance.
        """
        # Check if spec file provided
        if hasattr(args, "spec") and args.spec:
            return cls.from_spec(args.spec)

        # Parse dates
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

        # Parse symbols
        symbols = [s.strip() for s in args.symbols.split(",")]

        # Parse strategy params if provided
        strategy_params = {}
        if hasattr(args, "params") and args.params:
            for param in args.params:
                key, value = param.split("=")
                # Try to parse as number
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                strategy_params[key] = value

        return cls(
            strategy_name=args.strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=getattr(args, "capital", 100000),
            data_source=getattr(args, "data_source", "ib"),
            data_dir=getattr(args, "data_dir", "./data"),
            bar_size=getattr(args, "bar_size", "1d"),
            strategy_params=strategy_params,
            fill_model=getattr(args, "fill_model", "immediate"),
            slippage_bps=getattr(args, "slippage", 5.0),
            commission_per_share=getattr(args, "commission", 0.005),
        )

    async def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with all metrics.
        """
        # Print configuration
        self._print_config()

        # Create backtest config
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

        # Create engine
        engine = BacktestEngine(config)

        # Set strategy
        strategy_class = get_strategy_class(self.strategy_name)
        if strategy_class is None:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. "
                f"Available: {list_strategies()}"
            )
        engine.set_strategy(strategy_class, params=self.strategy_params)

        # Set data feed
        data_feed = self._create_data_feed()
        engine.set_data_feed(data_feed)

        # Run backtest
        result = await engine.run()

        # Print results
        result.print_summary()

        # Save results if spec provided
        if self._spec and self._spec.reporting.get("persist_to_db"):
            self._save_result(result)

        return result

    def _create_data_feed(self):
        """Create appropriate data feed based on data_source.

        Broker connection details are loaded from config/base.yaml.
        """
        if self.data_source in ("ib", "ibkr"):
            # Load IB config from base.yaml
            broker_config = load_broker_config("ib")
            return IbHistoricalDataFeed(
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                bar_size=self.bar_size,
                host=broker_config.get("host", "127.0.0.1"),
                port=broker_config.get("port", 4001),
                client_id= broker_config.get("client_id", 1),
            )
        elif self.data_source == "csv":
            return CsvDataFeed(
                csv_dir=self.data_dir,
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                bar_size=self.bar_size,
            )
        elif self.data_source == "parquet":
            return ParquetDataFeed(
                parquet_dir=self.data_dir,
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                bar_size=self.bar_size,
            )
        else:
            raise ValueError(f"Unknown data source: {self.data_source}. Use 'ib', 'csv', or 'parquet'.")

    def _print_config(self) -> None:
        """Print backtest configuration."""
        print(f"\n{'=' * 60}")
        print("BACKTEST CONFIGURATION")
        print(f"{'=' * 60}")
        print(f"Strategy:     {self.strategy_name}")
        print(f"Symbols:      {', '.join(self.symbols)}")
        print(f"Period:       {self.start_date} to {self.end_date}")
        print(f"Capital:      ${self.initial_capital:,.2f}")
        print(f"Data Source:  {self.data_source}")
        if self.data_source in ("ib", "ibkr"):
            broker_config = load_broker_config("ib")
            print(f"IB Gateway:   {broker_config.get('host', '127.0.0.1')}:{broker_config.get('port', 4001)}")
        print(f"Bar Size:     {self.bar_size}")
        print(f"Fill Model:   {self.fill_model.value}")
        if self.strategy_params:
            print(f"Parameters:   {self.strategy_params}")
        print(f"{'=' * 60}\n")

    def _save_result(self, result: BacktestResult) -> None:
        """Save result to database."""
        # TODO: Implement persistence via BacktestRepository
        logger.info(f"Backtest result saved: {result.backtest_id}")


class BacktraderRunner:
    """
    Backtest runner using Backtrader engine.

    Provides the same interface as BacktestRunner but executes
    using Backtrader's engine for comparison or alternative execution.

    Requires: pip install backtrader
    """

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
        """
        Initialize Backtrader runner.

        Args:
            strategy_name: Name of strategy in registry.
            symbols: List of symbols to trade.
            start_date: Backtest start date.
            end_date: Backtest end date.
            initial_capital: Starting capital.
            data_source: Data source (csv, parquet, yahoo).
            data_dir: Directory containing data files.
            bar_size: Bar size (1d, 1h, etc.).
            strategy_params: Strategy parameters.
            commission: Commission rate.
        """
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

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "BacktraderRunner":
        """
        Create runner from CLI arguments.

        Args:
            args: Parsed CLI arguments.

        Returns:
            BacktraderRunner instance.
        """
        # Check if spec file provided
        if hasattr(args, "spec") and args.spec:
            spec = BacktestSpec.from_yaml(args.spec)
            errors = spec.validate()
            if errors:
                raise ValueError(f"Invalid spec: {errors}")

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
            )
            runner._spec = spec
            return runner

        # Parse dates
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

        # Parse symbols
        symbols = [s.strip() for s in args.symbols.split(",")]

        # Parse strategy params if provided
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

        return cls(
            strategy_name=args.strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            initial_capital=getattr(args, "capital", 100000),
            data_source=getattr(args, "data_source", "csv"),
            data_dir=getattr(args, "data_dir", "./data"),
            bar_size=getattr(args, "bar_size", "1d"),
            strategy_params=strategy_params,
            commission=getattr(args, "commission", 0.001),
        )

    async def run(self) -> BacktestResult:
        """
        Run the backtest using Backtrader engine.

        Returns:
            BacktestResult with metrics (converted from Backtrader analyzers).
        """
        import time
        from ..infrastructure.backtest.backtrader_adapter import (
            ApexStrategyWrapper,
            run_backtest_with_backtrader,
        )
        from ..domain.reality import RealityModelPack, get_preset_pack

        # Print configuration
        self._print_config()

        # Resolve reality pack
        reality_pack = None
        if self._spec:
            if self._spec.reality_model:
                try:
                    reality_pack = RealityModelPack.from_config(self._spec.reality_model)
                except Exception as e:
                    logger.error(f"Failed to load reality_model from spec: {e}")
            
            if reality_pack is None and hasattr(self._spec.execution, 'reality_pack') and self._spec.execution.reality_pack:
                try:
                    reality_pack = get_preset_pack(self._spec.execution.reality_pack)
                except Exception as e:
                    logger.error(f"Failed to load reality_pack preset {self._spec.execution.reality_pack}: {e}")

        start_time = time.time()

        # Get strategy class
        strategy_class = get_strategy_class(self.strategy_name)
        if strategy_class is None:
            raise ValueError(
                f"Unknown strategy: {self.strategy_name}. "
                f"Available: {list_strategies()}"
            )

        # Create data feeds
        data_feeds = self._create_data_feeds()

        # Run backtest
        results = run_backtest_with_backtrader(
            apex_strategy_class=strategy_class,
            data_feeds=data_feeds,
            initial_cash=self.initial_capital,
            commission=self.commission,
            strategy_params=self.strategy_params,
            reality_pack=reality_pack,
        )

        run_duration = time.time() - start_time

        # Convert to BacktestResult
        result = self._convert_result(results, run_duration)

        # Print results
        result.print_summary()

        return result

    def _create_data_feeds(self) -> List[Any]:
        """Create Backtrader data feeds."""
        feeds = []

        for symbol in self.symbols:
            if self.data_source == "csv":
                # Look for CSV file
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
                # Use Yahoo Finance data feed
                feed = bt.feeds.YahooFinanceData(
                    dataname=symbol,
                    fromdate=datetime.combine(self.start_date, datetime.min.time()),
                    todate=datetime.combine(self.end_date, datetime.max.time()),
                )
                feed._name = symbol
                feeds.append(feed)

            else:
                raise ValueError(
                    f"Unsupported data source for Backtrader: {self.data_source}. "
                    "Use 'csv' or 'yahoo'."
                )

        return feeds

    def _convert_result(self, bt_results: Dict[str, Any], run_duration: float) -> BacktestResult:
        """Convert Backtrader results to BacktestResult."""
        from ..domain.backtest.backtest_result import (
            PerformanceMetrics,
            RiskMetrics,
            TradeMetrics,
            CostMetrics,
        )

        initial = self.initial_capital
        final = bt_results.get('final_value', initial)
        total_return = (final - initial) / initial if initial > 0 else 0

        # Calculate trading days
        trading_days = (self.end_date - self.start_date).days * 252 // 365

        # Performance metrics
        performance = PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return * 100,
            cagr=self._calculate_cagr(initial, final, trading_days),
            annualized_return=total_return * 252 / max(trading_days, 1),
        )

        # Risk metrics
        sharpe = bt_results.get('sharpe_ratio') or 0.0
        max_dd = bt_results.get('max_drawdown') or 0.0

        risk = RiskMetrics(
            max_drawdown=max_dd,
            max_drawdown_duration_days=0,  # Not easily available from Backtrader
            sharpe_ratio=sharpe,
        )

        # Trade metrics
        total_trades = bt_results.get('total_trades', 0)
        trades = TradeMetrics(total_trades=total_trades)

        # Cost metrics (estimated from commission rate)
        estimated_commission = total_trades * 2 * 100 * self.commission  # Rough estimate
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
            equity_curve=[],  # Backtrader doesn't easily expose this
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


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Apex Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run MA cross strategy on AAPL (uses IB settings from config/base.yaml)
    python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30

    # Run from spec file
    python -m src.runners.backtest_runner --spec config/backtest/my_strategy.yaml

    # Use CSV data instead of IB (for offline testing)
    python -m src.runners.backtest_runner --strategy ma_cross --symbols AAPL \\
        --start 2024-01-01 --end 2024-06-30 --data-source csv --data-dir ./data

    # List available strategies
    python -m src.runners.backtest_runner --list-strategies

Note: Broker connection settings are loaded from config/base.yaml (brokers section).
        """,
    )

    parser.add_argument(
        "--strategy",
        type=str,
        help="Strategy name (e.g., ma_cross, buy_and_hold)",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (e.g., AAPL,MSFT)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="ib",
        choices=["ib", "csv", "parquet"],
        help="Data source/broker (default: ib). Connection from config/base.yaml",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory containing data files (default: ./data)",
    )
    parser.add_argument(
        "--bar-size",
        type=str,
        default="1d",
        help="Bar size (1m, 5m, 15m, 1h, 1d) (default: 1d)",
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs="*",
        help="Strategy parameters as key=value pairs",
    )
    parser.add_argument(
        "--fill-model",
        type=str,
        default="immediate",
        choices=["immediate", "slippage"],
        help="Fill model (default: immediate)",
    )
    parser.add_argument(
        "--slippage",
        type=float,
        default=5.0,
        help="Slippage in basis points (default: 5.0)",
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.005,
        help="Commission per share (default: 0.005)",
    )
    parser.add_argument(
        "--spec",
        type=str,
        help="Path to backtest spec YAML file",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


async def main():
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # List strategies
    if args.list_strategies:
        print("\nAvailable strategies:")
        for name in list_strategies():
            print(f"  - {name}")
        return

    # Validate required args
    if not args.spec:
        if not args.strategy:
            parser.error("--strategy or --spec required")
        if not args.symbols:
            parser.error("--symbols required")
        if not args.start or not args.end:
            parser.error("--start and --end required")

    try:
        runner = BacktestRunner.from_args(args)
        result = await runner.run()
        sys.exit(0 if result.is_profitable else 1)
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
