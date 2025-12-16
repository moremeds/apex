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
