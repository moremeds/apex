"""
Backtest CLI Argument Parser.

Defines all command-line arguments for the unified backtest runner.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Optional, Sequence


class DateAction(argparse.Action):
    """Custom action to parse and validate date arguments at parse time."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: Optional[str] = None,
    ) -> None:
        if values is None:
            setattr(namespace, self.dest, None)
            return

        value_str = str(values)
        try:
            parsed_date = datetime.strptime(value_str, "%Y-%m-%d").date()
            setattr(namespace, self.dest, parsed_date)
        except ValueError:
            parser.error(
                f"{option_string}: invalid date format '{value_str}' "
                f"(expected YYYY-MM-DD, e.g., 2024-01-15)"
            )


class PositiveFloatAction(argparse.Action):
    """Custom action to validate positive float values at parse time."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: Optional[str] = None,
    ) -> None:
        if values is None:
            setattr(namespace, self.dest, None)
            return

        try:
            value_str = str(values)
            value = float(value_str)
            if value <= 0:
                parser.error(f"{option_string}: value must be positive (got {value})")
            setattr(namespace, self.dest, value)
        except (TypeError, ValueError):
            parser.error(f"{option_string}: invalid number '{values}'")


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
    parser.add_argument("--start", action=DateAction, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", action=DateAction, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--capital",
        action=PositiveFloatAction,
        default=100000.0,
        help="Initial capital (must be positive, default: 100000)",
    )

    # Systematic experiment options
    parser.add_argument("--spec", type=str, help="Path to experiment YAML spec")
    parser.add_argument(
        "--output", type=str, default="results/experiments", help="Output directory"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Parallel workers (0=auto-scale based on tickers/folds, capped at 16)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation")

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
    parser.add_argument(
        "--fill-model", type=str, default="immediate", choices=["immediate", "slippage"]
    )
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps")
    parser.add_argument("--commission", type=float, default=0.005, help="Commission per share")

    # Streaming options
    parser.add_argument(
        "--streaming", action="store_true", default=True, help="Use streaming feeds"
    )
    parser.add_argument(
        "--no-streaming", action="store_false", dest="streaming", help="Use full-load feeds"
    )

    # Historical coverage options
    parser.add_argument(
        "--coverage-mode",
        type=str,
        choices=["download", "check", "off"],
        help="Coverage mode for historical source",
    )
    parser.add_argument("--historical-dir", type=str, help="Historical data directory")
    parser.add_argument("--source-priority", type=str, help="Source priority (e.g., ib,yahoo)")

    # Behavioral gate validation
    parser.add_argument(
        "--behavioral",
        action="store_true",
        help="Run DualMACD behavioral gate validation (spec required or defaults used)",
    )
    parser.add_argument(
        "--behavioral-cases",
        action="store_true",
        help="Run predefined behavioral case studies",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="After behavioral run, auto-cluster symbols into gate policy buckets (dry-run)",
    )
    parser.add_argument(
        "--slope-lookback",
        type=int,
        default=3,
        help="DualMACD slope lookback (default: 3)",
    )
    parser.add_argument(
        "--hist-norm-window",
        type=int,
        default=252,
        help="DualMACD histogram normalization window (default: 252)",
    )

    # Utility options
    parser.add_argument("--list-strategies", action="store_true", help="List strategies and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    return parser
