"""
Signal Pipeline Configuration.

Contains the configuration dataclass and argument parser for the signal pipeline.
Extracted from signal_runner.py for better modularity.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional


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

    # Persistence
    with_persistence: bool = False

    # Output
    verbose: bool = False
    stats_interval: int = 10
    html_output: Optional[str] = None  # Path for HTML report generation
    json_output: bool = False  # Output results as JSON (for --validate-bars)
    output_format: str = "singlefile"  # "singlefile" (legacy HTML) or "package" (PR-02)
    deploy_github: bool = False  # Deploy package to GitHub Pages
    github_repo: Optional[str] = None  # GitHub repo for deployment (e.g., "user/signal-reports")

    # Model training options
    train_models: bool = False  # Train models before signal generation
    retrain_models: bool = False  # Walk-forward retrain (mutually exclusive with train_models)
    model_symbols: Optional[List[str]] = None  # Symbols for training (default: main symbols)
    model_days: int = 750  # Days of history for training
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
  # Run with HTML report (default output: results/signals/signal_report.html)
  python -m src.runners.signal_runner --live --symbols AAPL TSLA QQQ

  # Custom HTML output path
  python -m src.runners.signal_runner --live --symbols AAPL --html-output my_report.html

  # Validate bar counts (PR-01: solves "350 vs 252" mystery)
  python -m src.runners.signal_runner --validate-bars --symbols AAPL SPY

  # Validate bars with JSON output for scripting
  python -m src.runners.signal_runner --validate-bars --symbols AAPL --json

  # Generate package format with lazy loading (PR-02)
  python -m src.runners.signal_runner --live --symbols AAPL SPY --format package

  # Deploy package to GitHub Pages (auto-deploys to gh-pages branch)
  python -m src.runners.signal_runner --live --symbols AAPL SPY --format package --deploy github

  # Deploy to a specific GitHub repo
  python -m src.runners.signal_runner --live --symbols AAPL SPY --format package --deploy github --github-repo user/signal-reports

  # Train models before generating signals
  python -m src.runners.signal_runner --live --symbols AAPL --train-models \\
      --model-symbols SPY QQQ AAPL

  # Retrain models with walk-forward validation (CI use case)
  python -m src.runners.signal_runner --live --symbols SPY --retrain-models \\
      --model-symbols SPY QQQ --dry-run

  # Training only (no signal generation)
  python -m src.runners.signal_runner --train-models --model-symbols SPY QQQ

  # Backfill historical signals
  python -m src.runners.signal_runner --backfill --symbols AAPL --days 365
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

    # HTML report generation (default: enabled)
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation",
    )
    parser.add_argument(
        "--html-output",
        type=str,
        default="results/signals/signal_report.html",
        metavar="PATH",
        help="HTML report output path (default: results/signals/signal_report.html)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON (for --validate-bars mode)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["singlefile", "package"],
        default="singlefile",
        metavar="FORMAT",
        help="Output format: singlefile (legacy HTML) or package (PR-02 lazy loading)",
    )

    # GitHub Pages deployment
    parser.add_argument(
        "--deploy",
        type=str,
        choices=["github"],
        metavar="TARGET",
        help="Deploy package to target (github = GitHub Pages)",
    )
    parser.add_argument(
        "--github-repo",
        type=str,
        metavar="REPO",
        help="GitHub repo for deployment (e.g., 'user/signal-reports'). If not set, uses current repo's gh-pages branch.",
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
        import yaml
        from pathlib import Path
        universe_path = Path(args.universe)
        if universe_path.exists():
            with open(universe_path) as f:
                universe = yaml.safe_load(f)
            # Combine training and holdout universes for signal report
            symbols = universe.get("training_universe", []) + universe.get("holdout_universe", [])
            if not symbols:
                # Fallback: try flat list
                symbols = universe.get("symbols", args.symbols)
            print(f"Loaded {len(symbols)} symbols from {args.universe}")

    return SignalPipelineConfig(
        symbols=symbols,
        timeframes=args.timeframes,
        max_workers=args.max_workers,
        live=args.live,
        backfill=args.backfill,
        backfill_days=args.days,
        validate_bars=args.validate_bars,
        with_persistence=args.with_persistence,
        verbose=args.verbose,
        stats_interval=args.stats_interval,
        html_output=None if args.no_html else args.html_output,
        json_output=args.json,
        output_format=args.format,
        deploy_github=args.deploy == "github",
        github_repo=args.github_repo,
        # Training options
        train_models=args.train_models,
        retrain_models=args.retrain_models,
        model_symbols=args.model_symbols,
        model_days=args.model_days,
        force_retrain=args.force_retrain,
        eval_only=args.eval_only,
        model_output_dir=args.model_output_dir,
        dry_run=args.dry_run,
        train_concurrency=args.train_concurrency,
    )
