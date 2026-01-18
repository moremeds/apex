#!/usr/bin/env python3
"""
DEPRECATED: Use signal_runner with --retrain-models instead.

This script is deprecated and will be removed in a future release.
Please migrate to the new unified CLI:

    # Retrain single symbol
    python -m src.runners.signal_runner --retrain-models --model-symbols SPY

    # Retrain multiple symbols
    python -m src.runners.signal_runner --retrain-models --model-symbols SPY QQQ AAPL

    # Force retrain (update even if not better)
    python -m src.runners.signal_runner --retrain-models \\
        --model-symbols SPY --force-retrain

    # Evaluation only (no model update)
    python -m src.runners.signal_runner --retrain-models \\
        --model-symbols SPY --eval-only

This script remains for backward compatibility but delegates to the new
TurningPointTrainingService with hexagonal architecture.

Original usage (still works):
    python scripts/retrain_turning_point_models.py --symbol SPY
    python scripts/retrain_turning_point_models.py --symbols SPY QQQ AAPL
    python scripts/retrain_turning_point_models.py --symbol SPY --force
    python scripts/retrain_turning_point_models.py --all
    python scripts/retrain_turning_point_models.py --symbol SPY --eval-only
"""

import argparse
import sys
import warnings
from pathlib import Path

# Emit deprecation warning
warnings.warn(
    "This script is deprecated. Use: "
    "python -m src.runners.signal_runner --retrain-models --model-symbols SYMBOL",
    DeprecationWarning,
    stacklevel=1,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain.signals.indicators.regime.turning_point.experiment import (
    TurningPointExperiment,
    WalkForwardConfig,
    batch_retrain,
    retrain_model,
)


def main():
    parser = argparse.ArgumentParser(
        description="Retrain Turning Point Models (DEPRECATED - use signal_runner instead)",
        epilog="""
DEPRECATED: This script is deprecated. Please use:
    python -m src.runners.signal_runner --retrain-models --model-symbols SYMBOL
        """,
    )
    parser.add_argument("--symbol", type=str, help="Single symbol to retrain")
    parser.add_argument(
        "--symbols", nargs="+", type=str, help="Multiple symbols to retrain"
    )
    parser.add_argument(
        "--all", action="store_true", help="Retrain all existing models"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force update even if not better"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run evaluation only, don't update models",
    )
    parser.add_argument(
        "--train-days",
        type=int,
        default=504,
        help="Training window in days (default: 504 = 2 years)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=63,
        help="Test window in days (default: 63 = 3 months)",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=4,
        help="Number of walk-forward windows (default: 4)",
    )
    args = parser.parse_args()

    # Print deprecation notice prominently
    print("\n" + "!" * 60)
    print("! DEPRECATION WARNING")
    print("!" * 60)
    print("! This script is deprecated. Please use:")
    print("!   python -m src.runners.signal_runner --retrain-models \\")
    if args.symbol:
        print(f"!       --model-symbols {args.symbol}")
    elif args.symbols:
        print(f"!       --model-symbols {' '.join(args.symbols)}")
    else:
        print("!       --model-symbols SPY QQQ AAPL NVDA TSLA")
    if args.force:
        print("!       --force-retrain")
    if args.eval_only:
        print("!       --eval-only")
    print("!" * 60 + "\n")

    # Determine symbols to process
    if args.all:
        # Find all existing models
        model_dir = Path("models/turning_point")
        if model_dir.exists():
            symbols = [
                p.stem.split("_")[0].upper()
                for p in model_dir.glob("*_logistic.pkl")
            ]
            if not symbols:
                print("No existing models found.")
                return 1
            print(f"Found {len(symbols)} existing models: {', '.join(symbols)}")
        else:
            print("No models directory found.")
            return 1
    elif args.symbols:
        symbols = [s.upper() for s in args.symbols]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        parser.print_help()
        return 1

    print("\n" + "=" * 60)
    print("TURNING POINT MODEL RETRAINING")
    print("=" * 60)
    print(f"Symbols:      {', '.join(symbols)}")
    print(f"Train Window: {args.train_days} days")
    print(f"Test Window:  {args.test_days} days")
    print(f"Walk-Forward: {args.n_windows} windows")
    print(f"Force Update: {args.force}")
    print(f"Eval Only:    {args.eval_only}")
    print("=" * 60 + "\n")

    # Configure experiment
    config = WalkForwardConfig(
        train_days=args.train_days,
        test_days=args.test_days,
        n_windows=args.n_windows,
    )

    if args.eval_only:
        # Run evaluation only
        experiment = TurningPointExperiment(config)
        for symbol in symbols:
            print(f"\n{'─' * 60}")
            print(f"Evaluating {symbol}...")
            print(f"{'─' * 60}")
            try:
                result = experiment.run(symbol, days=1500)
                result.print_summary()
            except Exception as e:
                print(f"Evaluation failed for {symbol}: {e}")
    else:
        # Run retraining
        if len(symbols) == 1:
            model, result = retrain_model(symbols[0], force=args.force)
            if model:
                print(f"\n✓ Model updated for {symbols[0]}")
            else:
                print(f"\n✗ Model not updated for {symbols[0]}")
        else:
            results = batch_retrain(symbols, force=args.force)
            print("\n" + "=" * 60)
            print("BATCH RETRAINING SUMMARY")
            print("=" * 60)
            for symbol, result in results.items():
                status = "✓ Updated" if result.model_path else "✗ Not updated"
                reason = result.baseline_comparison.get("reason", "N/A") if result.baseline_comparison else "N/A"
                print(f"  {symbol}: {status} ({reason})")

    print("\n" + "=" * 60)
    print("RETRAINING COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
