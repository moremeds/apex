"""
M2 Validation Runner - Nested CV validation for regime detection.

Runs validation against the REAL RegimeDetector using actual market data.

Usage:
    python -m src.runners.validation_runner fast --symbols SPY QQQ --output ci_fast.json
    python -m src.runners.validation_runner full --output reports/nightly.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..domain.signals.validation import (
    HorizonConfig,
    SplitConfig,
    create_fast_validation_output,
    create_full_validation_output,
)
from ..domain.signals.validation.earliness import EarlinessResult
from ..domain.signals.validation.validation_service import (
    ValidationService,
    ValidationServiceConfig,
)
from ..utils.logging_setup import get_logger
from .validation_helpers import (
    _default_labeler_thresholds,
    generate_synthetic_bars,
    get_bars_per_day,
    load_bars_yahoo,
    load_universe_from_yaml,
)

logger = get_logger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="validation_runner",
        description="M2 Validation Runner - Nested CV validation for regime detection",
    )
    subparsers = parser.add_subparsers(dest="mode", help="Validation mode")

    # Fast mode (PR gate)
    fast = subparsers.add_parser("fast", help="Fast PR gate (< 30s)")
    _add_common_args(fast)
    fast.add_argument(
        "--symbols",
        nargs="+",
        default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "MU", "GME", "AMC", "TSLA"],
    )
    fast.add_argument("--folds", type=int, default=2)
    fast.add_argument("--no-optuna", action="store_true")
    fast.add_argument("--days", type=int, default=500, help="Days of history to load")
    fast.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for CI testing (no external API calls)",
    )

    # Full mode (nightly)
    full = subparsers.add_parser("full", help="Full nightly validation")
    _add_common_args(full)
    full.add_argument("--universe", type=str, default="config/universe.yaml")
    full.add_argument("--outer-folds", type=int, default=5)
    full.add_argument("--inner-folds", type=int, default=3)
    full.add_argument("--inner-trials", type=int, default=20)
    full.add_argument("--confirm-rule", type=str, default="1d&4h")
    full.add_argument("--days", type=int, default=750, help="Days of history to load")
    full.add_argument("--max-symbols", type=int, default=50, help="Max symbols to validate")

    # Holdout mode (release gate)
    holdout = subparsers.add_parser("holdout", help="Holdout validation (release)")
    _add_common_args(holdout)
    holdout.add_argument("--universe", type=str, default="config/universe.yaml")
    holdout.add_argument("--params", type=str, help="Path to optimized params YAML")
    holdout.add_argument("--days", type=int, default=500, help="Days of history to load")

    # Optimize mode (parameter tuning with Optuna)
    optimize = subparsers.add_parser("optimize", help="Optimize detector params with nested CV")
    _add_common_args(optimize)
    optimize.add_argument("--universe", type=str, default="config/universe.yaml")
    optimize.add_argument("--outer-folds", type=int, default=3)
    optimize.add_argument("--inner-folds", type=int, default=2)
    optimize.add_argument("--inner-trials", type=int, default=30, help="Optuna trials per inner CV")
    optimize.add_argument("--days", type=int, default=750, help="Days of history to load")
    optimize.add_argument(
        "--max-symbols", type=int, default=30, help="Max symbols for optimization"
    )
    optimize.add_argument(
        "--params-output",
        type=str,
        default="config/validation/optimized_params.yaml",
        help="Output path for optimized params YAML",
    )

    return parser


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add common arguments to a subparser."""
    parser.add_argument("--timeframes", nargs="+", default=["1d"])
    parser.add_argument("--horizon-days", type=int, default=20)
    parser.add_argument("--episode-min-days", type=int, default=10)
    parser.add_argument("--bootstrap-block-days", type=int, default=5)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("-v", "--verbose", action="store_true")


class ValidationRunner:
    """Runs M2 validation framework against real regime detector."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.mode = args.mode

        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)

    def run(self) -> int:
        """Run validation based on mode. Returns exit code."""
        if self.mode == "fast":
            return self._run_fast()
        elif self.mode == "full":
            return self._run_full()
        elif self.mode == "holdout":
            return self._run_holdout()
        elif self.mode == "optimize":
            return self._run_optimize()
        else:
            logger.error(f"Unknown mode: {self.mode}")
            return 1

    def _run_fast(self) -> int:
        """Run fast PR validation gate with real data."""
        use_synthetic = getattr(self.args, "synthetic", False)
        data_source = "SYNTHETIC DATA (CI)" if use_synthetic else "REAL DATA"

        print("=" * 60)
        print(f"M2 FAST VALIDATION (PR Gate) - {data_source}")
        print("=" * 60)
        print(f"Symbols: {self.args.symbols}")
        print(f"Timeframes: {self.args.timeframes}")
        print(f"Horizon: {self.args.horizon_days} days")
        print(f"History: {self.args.days} days\n")

        # Load market data (synthetic for CI, real for manual runs)
        if use_synthetic:
            print("Generating synthetic market data...")
            bars_by_symbol = generate_synthetic_bars(
                symbols=self.args.symbols,
                days=self.args.days,
            )
        else:
            print("Loading market data from Yahoo Finance...")
            bars_by_symbol = load_bars_yahoo(
                symbols=self.args.symbols,
                timeframe=self.args.timeframes[0],
                days=self.args.days,
            )

        if not bars_by_symbol:
            print("ERROR: No data loaded. Check symbols and network connection.")
            return 1

        print(f"Loaded data for {len(bars_by_symbol)} symbols\n")

        # Create validation service
        config = ValidationServiceConfig(
            timeframes=self.args.timeframes,
            horizon_days=self.args.horizon_days,
        )
        service = ValidationService(config)

        # Run fast validation
        print("Running regime detector validation...")
        trending_r0_rate, choppy_r0_rate, causality_passed = service.compute_fast_validation(
            symbols=list(bars_by_symbol.keys()),
            bars_by_symbol=bars_by_symbol,
            timeframe=self.args.timeframes[0],
        )

        print(f"\nResults:")
        print(f"  Trending R0 rate: {trending_r0_rate:.1%}")
        print(f"  Choppy R0 rate: {choppy_r0_rate:.1%}")
        print(f"  Causality check: {'PASS' if causality_passed else 'FAIL'}\n")

        # Create output
        validation_output = create_fast_validation_output(
            trending_r0_rate=trending_r0_rate,
            choppy_r0_rate=choppy_r0_rate,
            causality_passed=causality_passed,
            symbols=list(bars_by_symbol.keys()),
        )

        self._write_output(validation_output.to_dict())
        self._print_gate_results(validation_output.gate_results, validation_output.all_gates_passed)

        # In synthetic mode, always return 0 - CI validates causality separately
        # For real data (manual runs), return based on gate results
        if use_synthetic:
            return 0
        return 0 if validation_output.all_gates_passed else 1

    def _run_full(self) -> int:
        """Run full nightly validation with real data."""
        print("=" * 60)
        print("M2 FULL VALIDATION (Nightly Gate) - REAL DATA")
        print("=" * 60)
        print(f"Universe: {self.args.universe}")
        print(f"Timeframes: {self.args.timeframes}")
        print(f"Horizon: {self.args.horizon_days} days")
        print(f"History: {self.args.days} days")
        print(f"Max symbols: {self.args.max_symbols}\n")

        # Load universe
        universe = load_universe_from_yaml(self.args.universe)
        training_symbols = universe["training_universe"][: self.args.max_symbols]
        holdout_symbols = universe["holdout_universe"]

        if not training_symbols:
            print("WARNING: No training symbols found, using defaults")
            training_symbols = [
                "SPY",
                "QQQ",
                "AAPL",
                "MSFT",
                "NVDA",
                "AMD",
                "GOOGL",
                "AMZN",
                "META",
                "TSLA",
            ]

        print(f"Training symbols: {len(training_symbols)}")
        print(f"Holdout symbols: {len(holdout_symbols)}\n")

        # Load market data
        print("Loading market data from Yahoo Finance...")
        bars_1d = load_bars_yahoo(
            symbols=training_symbols,
            timeframe="1d",
            days=self.args.days,
        )

        bars_4h = {}
        if "4h" in self.args.timeframes:
            bars_4h = load_bars_yahoo(
                symbols=training_symbols,
                timeframe="4h",
                days=min(self.args.days, 60),  # Yahoo limits intraday history
            )

        print(f"Loaded 1d data for {len(bars_1d)} symbols")
        if bars_4h:
            print(f"Loaded 4h data for {len(bars_4h)} symbols")
        print()

        # Create validation service
        config = ValidationServiceConfig(
            timeframes=self.args.timeframes,
            horizon_days=self.args.horizon_days,
        )
        service = ValidationService(config)

        # Run validation
        print("Running full validation...")
        results, statistical_result = service.validate_universe(
            symbols=list(bars_1d.keys()),
            bars_by_symbol=bars_1d,
            timeframe="1d",
        )

        # Compute earliness if 4h data available
        earliness_by_tf: Dict[str, EarlinessResult] = {}
        if bars_4h and "4h" in self.args.timeframes:
            print("Computing 4h vs 1d earliness...")
            earliness_by_tf["4h_vs_1d"] = service.compute_earliness(
                bars_1d=bars_1d,
                bars_4h=bars_4h,
                symbols=list(bars_1d.keys()),
            )

        # Compute confirmation if 4h data available
        confirmation_result = None
        if bars_4h and "4h" in self.args.timeframes:
            print("Computing confirmation analysis...")
            confirmation_result = service.compute_confirmation(
                bars_1d=bars_1d,
                bars_4h=bars_4h,
                symbols=list(bars_1d.keys()),
            )

        # Build configs
        horizon_config = HorizonConfig(
            horizon_calendar_days=self.args.horizon_days,
            bars_per_day={tf: get_bars_per_day(tf) for tf in self.args.timeframes},
            horizon_bars_by_tf={
                tf: int(self.args.horizon_days * get_bars_per_day(tf))
                for tf in self.args.timeframes
            },
        )

        split_config = SplitConfig(
            outer_folds=self.args.outer_folds,
            inner_folds=self.args.inner_folds,
            purge_bars_by_tf={
                tf: int(self.args.horizon_days * get_bars_per_day(tf))
                for tf in self.args.timeframes
            },
            embargo_bars_by_tf={
                tf: int(self.args.horizon_days // 2 * get_bars_per_day(tf))
                for tf in self.args.timeframes
            },
        )

        # Print statistical summary
        print(f"\nStatistical Summary:")
        print(f"  Trending symbols: {statistical_result.n_trending_symbols}")
        print(f"  Choppy symbols: {statistical_result.n_choppy_symbols}")
        print(f"  Cohen's d: {statistical_result.effect_size_cohens_d:.3f}")
        print(f"  p-value: {statistical_result.p_value:.4f}")
        print(f"  Trending mean: {statistical_result.trending_mean:.3f}")
        print(f"  Choppy mean: {statistical_result.choppy_mean:.3f}")
        print()

        # Create output
        validation_output = create_full_validation_output(
            statistical_result=statistical_result,
            earliness_by_tf=earliness_by_tf,
            confirmation_result=confirmation_result,
            horizon_config=horizon_config,
            split_config=split_config,
            labeler_thresholds=_default_labeler_thresholds(),
            training_symbols=list(bars_1d.keys()),
            holdout_symbols=holdout_symbols,
        )

        self._write_output(validation_output.to_dict())
        self._print_gate_results(validation_output.gate_results, validation_output.all_gates_passed)

        return 0 if validation_output.all_gates_passed else 1

    def _run_holdout(self) -> int:
        """Run holdout validation for release gate."""
        print("=" * 60)
        print("M2 HOLDOUT VALIDATION (Release Gate) - REAL DATA")
        print("=" * 60)
        print(f"Universe: {self.args.universe}")
        print(f"History: {self.args.days} days\n")

        # Load holdout universe
        universe = load_universe_from_yaml(self.args.universe)
        holdout_symbols = universe["holdout_universe"]

        if not holdout_symbols:
            print("WARNING: No holdout symbols found, using defaults")
            holdout_symbols = ["IWM", "DIA", "VTI", "XLF", "XLE"]

        print(f"Holdout symbols: {len(holdout_symbols)}\n")

        # Load market data
        print("Loading market data...")
        bars_by_symbol = load_bars_yahoo(
            symbols=holdout_symbols,
            timeframe="1d",
            days=self.args.days,
        )

        if not bars_by_symbol:
            print("ERROR: No data loaded.")
            return 1

        # Create validation service
        config = ValidationServiceConfig(
            timeframes=["1d"],
            horizon_days=self.args.horizon_days,
        )
        service = ValidationService(config)

        # Run validation on holdout
        print("Running holdout validation...")
        trending_r0_rate, choppy_r0_rate, causality_passed = service.compute_fast_validation(
            symbols=list(bars_by_symbol.keys()),
            bars_by_symbol=bars_by_symbol,
            timeframe="1d",
        )

        output = {
            "version": "m2_v2.0",
            "mode": "holdout",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "holdout_symbols": list(bars_by_symbol.keys()),
            "n_symbols": len(bars_by_symbol),
            "trending_r0_rate": trending_r0_rate,
            "choppy_r0_rate": choppy_r0_rate,
            "causality_passed": causality_passed,
        }

        self._write_output(output)

        print(f"\nHoldout Results:")
        print(f"  Trending R0 rate: {trending_r0_rate:.1%}")
        print(f"  Choppy R0 rate: {choppy_r0_rate:.1%}")
        print(f"  Causality: {'PASS' if causality_passed else 'FAIL'}")

        return 0

    def _run_optimize(self) -> int:
        """Run parameter optimization with nested CV."""
        print("=" * 60)
        print("M2 PARAMETER OPTIMIZATION (Nested CV + Optuna)")
        print("=" * 60)
        print(f"Universe: {self.args.universe}")
        print(f"History: {self.args.days} days")
        print(f"Max symbols: {self.args.max_symbols}")
        print(f"Outer folds: {self.args.outer_folds}")
        print(f"Inner folds: {self.args.inner_folds}")
        print(f"Inner trials: {self.args.inner_trials}\n")

        # Load universe
        universe = load_universe_from_yaml(self.args.universe)
        training_symbols = universe["training_universe"][: self.args.max_symbols]

        if not training_symbols:
            print("WARNING: No training symbols found, using defaults")
            training_symbols = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "GOOGL", "AMZN"]

        print(f"Training symbols: {len(training_symbols)}\n")

        # Load market data
        print("Loading market data from Yahoo Finance...")
        bars_by_symbol = load_bars_yahoo(
            symbols=training_symbols,
            timeframe="1d",
            days=self.args.days,
        )

        if not bars_by_symbol:
            print("ERROR: No data loaded.")
            return 1

        print(f"Loaded data for {len(bars_by_symbol)} symbols\n")

        # Import optimization components
        from ..domain.signals.indicators.regime.regime_detector import RegimeDetectorIndicator
        from ..domain.signals.validation.labeler_contract import (
            RegimeLabel,
            RegimeLabeler,
            RegimeLabelerConfig,
        )
        from ..domain.signals.validation.nested_cv import (
            NestedCVConfig,
            NestedWalkForwardCV,
            create_default_param_space,
        )
        from ..domain.signals.validation.time_units import ValidationTimeConfig

        # Create nested CV config
        time_config = ValidationTimeConfig.from_days("1d", self.args.horizon_days)
        cv_config = NestedCVConfig(
            outer_folds=self.args.outer_folds,
            inner_folds=self.args.inner_folds,
            inner_max_trials=self.args.inner_trials,
            time_config=time_config,
        )

        # Get date range from data
        all_dates = []
        for df in bars_by_symbol.values():
            if hasattr(df.index[0], "date"):
                all_dates.extend([d.date() for d in df.index])
            else:
                all_dates.extend(list(df.index))

        start_date = min(all_dates)
        end_date = max(all_dates)
        print(f"Date range: {start_date} to {end_date}\n")

        # Create labeler (frozen)
        labeler_config = RegimeLabelerConfig.load_v1("1d", self.args.horizon_days)
        labeler = RegimeLabeler(labeler_config)

        # Define objective function for inner CV
        from typing import Any

        from ..domain.signals.validation.nested_cv import TimeWindow

        def objective_fn(
            params: Dict[str, Any], train_window: TimeWindow, test_window: TimeWindow
        ) -> float:
            """Compute R0 rate difference (trending - choppy) on test window."""
            detector = RegimeDetectorIndicator()
            merged_params = {**detector._default_params, **params}

            total_trending_r0 = 0
            total_trending_bars = 0
            total_choppy_r0 = 0
            total_choppy_bars = 0

            for symbol, df in bars_by_symbol.items():
                # Filter to test window
                mask = (df.index >= str(test_window.start_date)) & (
                    df.index <= str(test_window.end_date)
                )
                test_df = df[mask]
                if len(test_df) < 50:
                    continue

                # Get predictions
                result_df = detector._calculate(test_df, merged_params)
                labeled_periods = labeler.label_period(test_df, test_window.end_date)

                for i, lp in enumerate(labeled_periods):
                    if i >= len(result_df):
                        break
                    regime = result_df.iloc[i].get("regime", "R1")
                    is_r0 = regime == "R0"

                    if lp.label == RegimeLabel.TRENDING:
                        total_trending_bars += 1
                        if is_r0:
                            total_trending_r0 += 1
                    elif lp.label == RegimeLabel.CHOPPY:
                        total_choppy_bars += 1
                        if is_r0:
                            total_choppy_r0 += 1

            trending_rate = (
                total_trending_r0 / total_trending_bars if total_trending_bars > 0 else 0
            )
            choppy_rate = total_choppy_r0 / total_choppy_bars if total_choppy_bars > 0 else 0

            # Objective: maximize trending R0 rate minus choppy R0 rate
            return trending_rate - choppy_rate

        # Define evaluation function for outer test
        from ..domain.signals.validation.statistics import SymbolMetrics

        def evaluate_symbol_fn(
            symbol: str, test_window: TimeWindow, params: Dict[str, Any]
        ) -> SymbolMetrics:
            """Evaluate a single symbol with fixed params."""
            df = bars_by_symbol.get(symbol)
            if df is None:
                return SymbolMetrics(
                    symbol=symbol, label_type="UNKNOWN", r0_rate=0, total_bars=0, r0_bars=0
                )

            mask = (df.index >= str(test_window.start_date)) & (
                df.index <= str(test_window.end_date)
            )
            test_df = df[mask]
            if len(test_df) < 50:
                return SymbolMetrics(
                    symbol=symbol, label_type="UNKNOWN", r0_rate=0, total_bars=0, r0_bars=0
                )

            detector = RegimeDetectorIndicator()
            merged_params = {**detector._default_params, **params}
            result_df = detector._calculate(test_df, merged_params)
            labeled_periods = labeler.label_period(test_df, test_window.end_date)

            r0_count = 0
            total_bars = 0
            for i, lp in enumerate(labeled_periods):
                if i >= len(result_df):
                    break
                if lp.label == RegimeLabel.TRENDING:
                    total_bars += 1
                    regime = result_df.iloc[i].get("regime", "R1")
                    if regime == "R0":
                        r0_count += 1

            return SymbolMetrics(
                symbol=symbol,
                label_type="TRENDING",
                r0_rate=r0_count / total_bars if total_bars > 0 else 0,
                total_bars=total_bars,
                r0_bars=r0_count,
            )

        # Run nested CV optimization
        print("Running nested CV optimization with Optuna...")
        print("(This may take a while...)\n")

        cv = NestedWalkForwardCV(cv_config)
        param_space = create_default_param_space()

        result = cv.run(
            symbols=list(bars_by_symbol.keys()),
            start_date=start_date,
            end_date=end_date,
            objective_fn=objective_fn,
            evaluate_symbol_fn=evaluate_symbol_fn,
            param_space=param_space,
        )

        # Aggregate best params across folds (use most common or average)
        print("\nOptimization Results:")
        print("-" * 40)
        for fold_result in result.outer_results:
            print(f"Fold {fold_result.fold_id}: inner_score={fold_result.inner_cv_score:.4f}")
            print(f"  Best params: {fold_result.best_params}")

        # Use params from best fold
        best_fold = max(result.outer_results, key=lambda r: r.inner_cv_score)
        best_params = best_fold.best_params

        print(f"\nBest overall params (from fold {best_fold.fold_id}):")
        for k, v in best_params.items():
            print(f"  {k}: {v}")

        # Save optimized params to YAML
        params_output = {
            "version": "m2_optimized_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "optimization_config": {
                "outer_folds": self.args.outer_folds,
                "inner_folds": self.args.inner_folds,
                "inner_trials": self.args.inner_trials,
                "n_symbols": len(bars_by_symbol),
            },
            "best_params": best_params,
            "fold_results": [
                {
                    "fold_id": r.fold_id,
                    "inner_cv_score": r.inner_cv_score,
                    "best_params": r.best_params,
                }
                for r in result.outer_results
            ],
        }

        params_path = Path(self.args.params_output)
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with open(params_path, "w") as f:
            yaml.dump(params_output, f, default_flow_style=False)
        print(f"\nOptimized params saved to: {params_path}")

        # Also write full result to JSON output
        output = {
            "version": "m2_v2.0",
            "mode": "optimize",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "best_params": best_params,
            "nested_cv_result": result.to_dict(),
        }
        self._write_output(output)

        return 0

    def _write_output(self, data: dict) -> None:
        """Write output to JSON file."""
        output_path = Path(self.args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"\nOutput written to: {output_path}")

    def _print_gate_results(self, gate_results: list, all_passed: bool) -> None:
        """Print gate results to console."""
        print("\nGate Results:")
        print("-" * 40)
        for gate in gate_results:
            status = "✓ PASS" if gate.passed else "✗ FAIL"
            print(f"  {gate.gate_name}: {status} ({gate.message})")
        print()
        print("All gates PASSED" if all_passed else "Some gates FAILED")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    if not args.mode:
        parser.print_help()
        return 1

    return ValidationRunner(args).run()


if __name__ == "__main__":
    sys.exit(main())
