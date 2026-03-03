"""
Signal Pipeline Configuration.

Contains the configuration dataclass and argument parser for the signal pipeline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _extract_symbols_from_universe(universe: Dict[str, Any], fallback: List[str]) -> List[str]:
    """
    Extract symbols from various universe YAML formats.

    Supports:
    1. training_universe + holdout_universe (model training format)
    2. Flat symbols list
    3. Groups format (universe.yaml style)
    4. Regime verification format (market + sectors with ETF/stocks)
    """
    symbols: List[str] = []
    seen: set = set()

    def add_symbol(sym: str) -> None:
        if sym and sym not in seen:
            symbols.append(sym)
            seen.add(sym)

    # 1. training_universe + holdout_universe
    for sym in universe.get("training_universe", []):
        add_symbol(sym)
    for sym in universe.get("holdout_universe", []):
        add_symbol(sym)

    # 2. Flat symbols list
    if not symbols:
        for sym in universe.get("symbols", []):
            add_symbol(sym)

    # 3. Groups format (universe.yaml style)
    if not symbols and "groups" in universe:
        for group_config in universe["groups"].values():
            if isinstance(group_config, dict) and group_config.get("enabled", True):
                for sym in group_config.get("symbols", []):
                    add_symbol(sym)

    # 4. Regime verification format (market + sectors)
    if not symbols:
        # Market-level ETFs (list of dicts with 'symbol' key)
        for item in universe.get("market", []):
            if isinstance(item, dict):
                add_symbol(item.get("symbol", ""))
            elif isinstance(item, str):
                add_symbol(item)

        # Sector ETFs and stocks
        for sector_config in universe.get("sectors", {}).values():
            if isinstance(sector_config, dict):
                add_symbol(sector_config.get("etf", ""))
                for sym in sector_config.get("stocks", []):
                    add_symbol(sym)

    # 5. Quick test subset (flat list)
    if not symbols:
        for sym in universe.get("quick_test", []):
            add_symbol(sym)

    # Final fallback
    return symbols if symbols else fallback


@dataclass
class SignalPipelineConfig:
    """Configuration for signal pipeline."""

    symbols: List[str]
    timeframes: List[str]
    max_workers: int = 4

    # Live mode
    live: bool = False

    # Backfill mode
    backfill: bool = False
    backfill_days: int = 365

    # Validate bars mode (PR-01)
    validate_bars: bool = False  # Output BarValidationReport showing bar count breakdown

    # Market cap update mode (PR-C)
    update_market_caps: bool = False  # Update market cap cache from yfinance
    universe_path: Optional[str] = None  # Path to universe.yaml for market cap symbols

    # Persistence
    with_persistence: bool = False

    # Output
    verbose: bool = False
    stats_interval: int = 10
    json_output: bool = False  # Output results as JSON (for --validate-bars)

    # Performance tuning (CI optimization)
    preload_concurrency: int = 1  # Concurrent symbol downloads during preload (1 = serial)

    # Model training options
    train_models: bool = False  # Train models before signal generation
    retrain_models: bool = False  # Walk-forward retrain (mutually exclusive with train_models)
    model_symbols: Optional[List[str]] = None  # Symbols for training (default: main symbols)
    model_days: int = 750  # Days of history for training
    training_code_signature: str = "tp-v1"  # Bump to force retrain after feature/model changes
    force_retrain: bool = False  # Force update even if not better
    eval_only: bool = False  # Evaluation only, no promotion
    model_output_dir: Optional[str] = None  # Override model output directory
    dry_run: bool = False  # No email, no model promotion
    train_concurrency: int = 2  # Parallel training workers


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for signal runner."""
    parser = argparse.ArgumentParser(
        description="Standalone TA signal pipeline runner with model training support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live signal processing
  python -m src.domain.signals.pipeline --live --symbols AAPL TSLA QQQ

  # Train models before generating signals
  python -m src.domain.signals.pipeline --live --symbols AAPL --train-models \\
      --model-symbols SPY QQQ AAPL

  # Validate bar counts
  python -m src.domain.signals.pipeline --validate-bars --symbols AAPL SPY

  # Update market cap cache
  python -m src.domain.signals.pipeline --update-market-caps --universe config/universe.yaml

  # Backfill historical signals
  python -m src.domain.signals.pipeline --backfill --symbols AAPL --days 365
        """,
    )

    # Mode selection (at least one required, but can combine training with live/backfill)
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument(
        "--live",
        action="store_true",
        help="Run on live market data (headless mode)",
    )
    mode_group.add_argument(
        "--backfill",
        action="store_true",
        help="Process historical bars for indicator warmup",
    )
    mode_group.add_argument(
        "--validate-bars",
        action="store_true",
        help="Output BarValidationReport showing bar count breakdown (PR-01)",
    )
    mode_group.add_argument(
        "--update-market-caps",
        action="store_true",
        help="Update market cap cache from yfinance (PR-C). Use with --universe.",
    )

    # Symbol/timeframe configuration
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["AAPL"],
        help="Space-separated symbols (default: AAPL)",
    )
    parser.add_argument(
        "--universe",
        type=str,
        metavar="PATH",
        help="Load symbols from universe YAML file (overrides --symbols)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1d"],
        help="Space-separated timeframes (default: 1d)",
    )

    # Backfill options
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history for backfill (default: 365)",
    )

    # Persistence options
    parser.add_argument(
        "--with-persistence",
        action="store_true",
        help="Enable database persistence for signals",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for --validate-bars mode)",
    )

    # General options
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Thread pool size for indicator calculations (default: 4)",
    )
    parser.add_argument(
        "--stats-interval",
        type=int,
        default=10,
        help="Seconds between stats output in live mode (default: 10)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output (show individual signals)",
    )

    # =========================================================================
    # Performance Tuning (CI optimization)
    # =========================================================================
    perf_group = parser.add_argument_group("Performance Tuning")
    perf_group.add_argument(
        "--preload-concurrency",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent symbol downloads during bar preload (default: 1 = serial). "
        "Uses asyncio.Semaphore for rate limiting.",
    )
    # =========================================================================
    # Model Training Options
    # =========================================================================
    training_group = parser.add_argument_group("Model Training Options")

    # Mutually exclusive: --train-models vs --retrain-models
    train_mode = training_group.add_mutually_exclusive_group()
    train_mode.add_argument(
        "--train-models",
        action="store_true",
        help="Train models before signal generation",
    )
    train_mode.add_argument(
        "--retrain-models",
        action="store_true",
        help="Retrain models with walk-forward validation (mutually exclusive with --train-models)",
    )

    training_group.add_argument(
        "--model-symbols",
        nargs="+",
        metavar="SYMBOL",
        help="Symbols for model training (default: uses --symbols or SPY QQQ AAPL NVDA TSLA)",
    )

    training_group.add_argument(
        "--model-days",
        type=int,
        default=750,
        metavar="DAYS",
        help="Days of history for training (default: 750)",
    )
    training_group.add_argument(
        "--training-code-signature",
        type=str,
        default="tp-v1",
        metavar="SIG",
        help="Training code signature used for skip-unchanged checks (default: tp-v1)",
    )

    training_group.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force model update even if not better than baseline",
    )

    training_group.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluation only - do not promote models (safe for testing)",
    )

    training_group.add_argument(
        "--model-output-dir",
        type=str,
        metavar="DIR",
        help="Override model output directory (default: models/turning_point)",
    )

    training_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run - no email, no model promotion (for debugging)",
    )

    training_group.add_argument(
        "--train-concurrency",
        type=int,
        default=2,
        metavar="N",
        help="Parallel training workers (default: 2)",
    )

    return parser


def parse_config(args: argparse.Namespace) -> SignalPipelineConfig:
    """
    Parse command-line arguments into SignalPipelineConfig.

    Args:
        args: Parsed argparse namespace

    Returns:
        SignalPipelineConfig instance
    """
    # Load symbols from universe file if provided
    symbols = args.symbols
    if args.universe:
        from pathlib import Path

        import yaml

        # Project root for resolving relative paths
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent

        universe_path = Path(args.universe)
        if not universe_path.is_absolute():
            universe_path = project_root / universe_path

        if universe_path.exists():
            with open(universe_path) as f:
                universe = yaml.safe_load(f)

            symbols = _extract_symbols_from_universe(universe, args.symbols)
            print(f"Loaded {len(symbols)} symbols from {args.universe}")
        else:
            print(f"WARNING: Universe file not found: {universe_path}")

    return SignalPipelineConfig(
        symbols=symbols,
        timeframes=args.timeframes,
        max_workers=args.max_workers,
        live=args.live,
        backfill=args.backfill,
        backfill_days=args.days,
        validate_bars=args.validate_bars,
        update_market_caps=args.update_market_caps,
        universe_path=args.universe,
        with_persistence=args.with_persistence,
        verbose=args.verbose,
        stats_interval=args.stats_interval,
        json_output=args.json,
        # Performance tuning
        preload_concurrency=args.preload_concurrency,
        # Training options
        train_models=args.train_models,
        retrain_models=args.retrain_models,
        model_symbols=args.model_symbols,
        model_days=args.model_days,
        training_code_signature=args.training_code_signature,
        force_retrain=args.force_retrain,
        eval_only=args.eval_only,
        model_output_dir=args.model_output_dir,
        dry_run=args.dry_run,
        train_concurrency=args.train_concurrency,
    )
