#!/usr/bin/env python
"""
Validate Turning Point Detection Against Ground-Truth Samples.

PR-05 Deliverable: Generic TP validation script for M2.

This script:
1. Loads ground-truth turning point samples from YAML
2. Runs the TurningPointModel against historical data
3. Computes timing metrics (early/late, MAE, precision, recall)
4. Reports pass/fail for G9 gate (TP timing within bounds)

Usage:
    python scripts/validate_turning_points.py \
        --config tests/fixtures/turning_point_samples.yaml --all

    python scripts/validate_turning_points.py --symbol TSLA \
        --config tests/fixtures/turning_point_samples.yaml

    python scripts/validate_turning_points.py \
        --config tests/fixtures/turning_point_samples.yaml --all --verbose
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import yaml

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TPSample:
    """Ground-truth turning point sample."""

    symbol: str
    date: date
    tp_type: Literal["top", "bottom"]
    confidence_threshold: float
    notes: str = ""


@dataclass
class TimingMetrics:
    """Timing metrics from validation."""

    total_samples: int = 0
    detected: int = 0
    missed: int = 0

    # Early warnings (negative = early)
    early_warnings: int = 0
    mean_lead_bars: float = 0.0

    # Confirmed (positive = late)
    confirmed: int = 0
    mean_lag_bars: float = 0.0

    # Overall
    mean_timing_offset: float = 0.0
    timing_mae: float = 0.0  # Mean absolute error

    # Out of bounds (G9 gate)
    out_of_bounds_early: int = 0
    out_of_bounds_late: int = 0

    # Precision/Recall
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        """Detection accuracy."""
        return self.detected / self.total_samples if self.total_samples > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def g9_pass(self) -> bool:
        """G9 gate: all predictions within timing bounds."""
        return self.out_of_bounds_early == 0 and self.out_of_bounds_late == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "detected": self.detected,
            "missed": self.missed,
            "accuracy": round(self.accuracy, 3),
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "early_warnings": self.early_warnings,
            "mean_lead_bars": round(self.mean_lead_bars, 2),
            "confirmed": self.confirmed,
            "mean_lag_bars": round(self.mean_lag_bars, 2),
            "mean_timing_offset": round(self.mean_timing_offset, 2),
            "timing_mae": round(self.timing_mae, 2),
            "out_of_bounds_early": self.out_of_bounds_early,
            "out_of_bounds_late": self.out_of_bounds_late,
            "g9_pass": self.g9_pass,
        }


@dataclass
class ValidationResult:
    """Result from validating a single symbol."""

    symbol: str
    samples: List[TPSample]
    predictions: List[Dict[str, Any]] = field(default_factory=list)
    metrics: TimingMetrics = field(default_factory=TimingMetrics)
    errors: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0 and self.metrics.g9_pass


@dataclass
class ValidationConfig:
    """Configuration loaded from YAML."""

    schema_version: str
    timing_bounds: Dict[str, int]
    symbols: Dict[str, List[Dict[str, Any]]]
    validation: Dict[str, Any]

    @classmethod
    def from_yaml(cls, path: Path) -> "ValidationConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            schema_version=data.get("schema_version", "unknown"),
            timing_bounds=data.get("timing_bounds", {"max_lead_bars": 5, "max_lag_bars": 3}),
            symbols=data.get("symbols", {}),
            validation=data.get("validation", {}),
        )


def load_samples(config: ValidationConfig) -> List[TPSample]:
    """Load all TP samples from config."""
    samples = []
    for symbol, entries in config.symbols.items():
        for entry in entries:
            try:
                sample_date = entry["date"]
                if isinstance(sample_date, str):
                    sample_date = datetime.strptime(sample_date, "%Y-%m-%d").date()
                samples.append(
                    TPSample(
                        symbol=symbol,
                        date=sample_date,
                        tp_type=entry["type"],
                        confidence_threshold=entry.get("confidence_threshold", 0.7),
                        notes=entry.get("notes", ""),
                    )
                )
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid sample entry for {symbol}: {e}")
    return samples


def get_samples_for_symbol(samples: List[TPSample], symbol: str) -> List[TPSample]:
    """Filter samples for a specific symbol."""
    return [s for s in samples if s.symbol == symbol]


def load_price_data(
    symbol: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """
    Load historical price data for symbol.

    Uses Yahoo adapter for historical data.
    """
    try:
        from src.infrastructure.adapters.yahoo.yahoo_adapter import YahooMarketDataAdapter

        adapter = YahooMarketDataAdapter()

        # Convert to datetime for API
        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        bars = adapter.get_historical_bars(
            symbol=symbol,
            start=start_dt,
            end=end_dt,
            timeframe="1d",
        )

        if not bars:
            logger.warning(f"No data returned for {symbol}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame([b.to_dict() for b in bars])
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df = df.set_index("date").sort_index()
        return df

    except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {e}")
        return None


def _features_from_dataframe(df: pd.DataFrame) -> "TurningPointFeatures":
    """
    Extract TurningPointFeatures from DataFrame for the last bar.

    Uses extract_features function and converts last row to TurningPointFeatures.
    """
    from src.domain.signals.indicators.regime.turning_point.features import (
        TurningPointFeatures,
        extract_features,
    )

    feature_df = extract_features(df)
    if feature_df.empty:
        return TurningPointFeatures()

    last_row = feature_df.iloc[-1]
    return TurningPointFeatures(
        price_vs_ma20=float(last_row["price_vs_ma20"]),
        price_vs_ma50=float(last_row["price_vs_ma50"]),
        price_vs_ma200=float(last_row["price_vs_ma200"]),
        ma20_slope=float(last_row["ma20_slope"]),
        ma50_slope=float(last_row["ma50_slope"]),
        ma20_vs_ma50=float(last_row["ma20_vs_ma50"]),
        atr_pct_63=float(last_row["atr_pct_63"]),
        atr_pct_252=float(last_row["atr_pct_252"]),
        atr_expansion_rate=float(last_row["atr_expansion_rate"]),
        vol_regime=int(last_row["vol_regime"]),
        chop_pct_252=float(last_row["chop_pct_252"]),
        adx_value=float(last_row["adx_value"]),
        range_position=float(last_row["range_position"]),
        ext_atr_units=float(last_row["ext_atr_units"]),
        ext_zscore=float(last_row["ext_zscore"]),
        rsi_14=float(last_row["rsi_14"]),
        roc_5=float(last_row["roc_5"]),
        roc_10=float(last_row["roc_10"]),
        roc_20=float(last_row["roc_20"]),
        delta_atr_pct=float(last_row["delta_atr_pct"]),
        delta_chop_pct=float(last_row["delta_chop_pct"]),
        delta_ext=float(last_row["delta_ext"]),
    )


def run_model_on_data(
    symbol: str,
    df: pd.DataFrame,
    model_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Run TurningPointModel on price data.

    Returns list of predictions with dates and outputs.
    """
    try:
        from src.domain.signals.indicators.regime.turning_point.model import TurningPointModel

        # Load or create model
        if model_path and model_path.exists():
            model = TurningPointModel.load(model_path)
        else:
            # Try default model location
            default_path = Path("models/turning_point/latest.pkl")
            if default_path.exists():
                model = TurningPointModel.load(default_path)
            else:
                logger.warning("No trained model found, using untrained model")
                model = TurningPointModel()
                model.is_fitted = False

        if not model.is_fitted:
            logger.warning("Model not fitted - predictions will be NONE")
            return []

        predictions = []
        dates = list(df.index)

        for i, dt in enumerate(dates):
            # Need enough history for features
            if i < 50:
                continue

            # Extract features from historical window
            window_df = df.iloc[: i + 1]
            features = _features_from_dataframe(window_df)

            # Run prediction
            output = model.predict(features)

            predictions.append(
                {
                    "date": dt,
                    "turn_state": output.turn_state.value,
                    "turn_confidence": output.turn_confidence,
                    "detection_type": output.detection_type,
                    "bars_from_actual_tp": output.bars_from_actual_tp,
                }
            )

        return predictions

    except ImportError as e:
        logger.error(f"Import error: {e}")
        return []
    except Exception as e:
        logger.error(f"Model error for {symbol}: {e}")
        return []


def compute_timing_metrics(
    samples: List[TPSample],
    predictions: List[Dict[str, Any]],
    config: ValidationConfig,
) -> TimingMetrics:
    """
    Compute timing metrics comparing predictions to ground truth.

    For each ground-truth sample:
    1. Search within window for matching prediction
    2. Compute bars offset (negative = early, positive = late)
    3. Check if within G9 bounds
    """
    max_lead = config.timing_bounds.get("max_lead_bars", 5)
    max_lag = config.timing_bounds.get("max_lag_bars", 3)
    search_window = config.validation.get("search_window_days", 10)

    metrics = TimingMetrics(total_samples=len(samples))

    if not predictions:
        metrics.missed = len(samples)
        metrics.false_negatives = len(samples)
        return metrics

    # Create prediction lookup by date
    pred_by_date = {p["date"]: p for p in predictions}
    pred_dates = sorted(pred_by_date.keys())

    lead_bars = []
    lag_bars = []
    all_offsets = []

    for sample in samples:
        # Expected signal type
        expected_state = "top_risk" if sample.tp_type == "top" else "bottom_risk"

        # Search window around ground-truth date
        search_start = sample.date - timedelta(days=search_window)
        search_end = sample.date + timedelta(days=search_window)

        # Find matching prediction
        matching_pred = None
        min_offset = float("inf")

        for pred_date in pred_dates:
            if search_start <= pred_date <= search_end:
                pred = pred_by_date[pred_date]

                # Check if signal matches expected type
                if pred["turn_state"] == expected_state:
                    # Check confidence threshold
                    if pred["turn_confidence"] >= sample.confidence_threshold:
                        # Calculate offset in trading days
                        offset = (pred_date - sample.date).days

                        if abs(offset) < abs(min_offset):
                            min_offset = offset
                            matching_pred = pred
                            matching_pred["bars_offset"] = offset

        if matching_pred:
            metrics.detected += 1
            metrics.true_positives += 1

            offset = matching_pred["bars_offset"]
            all_offsets.append(offset)

            # PR-05: Populate timing fields on the prediction object
            matching_pred["bars_from_actual_tp"] = offset
            matching_pred["detection_type"] = "early_warning" if offset < 0 else "confirmed"

            if offset < 0:
                # Early warning (before actual TP)
                metrics.early_warnings += 1
                lead_bars.append(abs(offset))

                if abs(offset) > max_lead:
                    metrics.out_of_bounds_early += 1
            else:
                # Confirmed (at or after actual TP)
                metrics.confirmed += 1
                lag_bars.append(offset)

                if offset > max_lag:
                    metrics.out_of_bounds_late += 1
        else:
            metrics.missed += 1
            metrics.false_negatives += 1

    # Compute aggregate metrics
    if lead_bars:
        metrics.mean_lead_bars = float(np.mean(lead_bars))

    if lag_bars:
        metrics.mean_lag_bars = float(np.mean(lag_bars))

    if all_offsets:
        metrics.mean_timing_offset = float(np.mean(all_offsets))
        metrics.timing_mae = float(np.mean(np.abs(all_offsets)))

    return metrics


def validate_symbol(
    symbol: str,
    samples: List[TPSample],
    config: ValidationConfig,
    model_path: Optional[Path] = None,
    verbose: bool = False,
) -> ValidationResult:
    """Validate turning point detection for a single symbol."""
    result = ValidationResult(symbol=symbol, samples=samples)

    if not samples:
        result.errors.append(f"No samples defined for {symbol}")
        return result

    # Determine date range needed
    dates = [s.date for s in samples]
    min_date = min(dates) - timedelta(days=100)  # Need history for features
    max_date = max(dates) + timedelta(days=30)

    if verbose:
        logger.info(f"Loading data for {symbol}: {min_date} to {max_date}")

    # Load price data
    df = load_price_data(symbol, min_date, max_date)
    if df is None or df.empty:
        result.errors.append(f"Failed to load price data for {symbol}")
        return result

    # Run model
    predictions = run_model_on_data(symbol, df, model_path)
    result.predictions = predictions

    if not predictions:
        result.errors.append(f"No predictions generated for {symbol} (model may not be fitted)")
        # Still compute metrics (will show all missed)
        result.metrics = compute_timing_metrics(samples, predictions, config)
        return result

    # Compute metrics
    result.metrics = compute_timing_metrics(samples, predictions, config)

    if verbose:
        logger.info(
            f"{symbol}: {result.metrics.detected}/{result.metrics.total_samples} detected, "
            f"MAE={result.metrics.timing_mae:.1f} bars"
        )

    return result


def validate_all(
    config: ValidationConfig,
    model_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, ValidationResult]:
    """Validate all symbols in config."""
    all_samples = load_samples(config)
    symbols = list(set(s.symbol for s in all_samples))

    results = {}
    for symbol in sorted(symbols):
        symbol_samples = get_samples_for_symbol(all_samples, symbol)
        results[symbol] = validate_symbol(symbol, symbol_samples, config, model_path, verbose)

    return results


def print_report(results: Dict[str, ValidationResult], config: ValidationConfig) -> None:
    """Print validation report."""
    print("=" * 70)
    print("TURNING POINT VALIDATION REPORT")
    print("=" * 70)
    print()

    max_lead = config.timing_bounds.get("max_lead_bars", 5)
    max_lag = config.timing_bounds.get("max_lag_bars", 3)
    print(f"Timing bounds: -{max_lead} to +{max_lag} bars")
    print()

    total_samples = 0
    total_detected = 0
    total_out_of_bounds = 0
    all_offsets = []

    print(f"{'Symbol':<8} {'Samples':>8} {'Detected':>10} {'Accuracy':>10} {'MAE':>8} {'G9':>6}")
    print("-" * 70)

    for symbol, result in sorted(results.items()):
        m = result.metrics
        total_samples += m.total_samples
        total_detected += m.detected
        total_out_of_bounds += m.out_of_bounds_early + m.out_of_bounds_late

        status = "PASS" if m.g9_pass else "FAIL"
        print(
            f"{symbol:<8} {m.total_samples:>8} {m.detected:>10} "
            f"{m.accuracy:>9.1%} {m.timing_mae:>7.1f} {status:>6}"
        )

        if result.errors:
            for err in result.errors:
                print(f"    ERROR: {err}")

    print("-" * 70)

    overall_accuracy = total_detected / total_samples if total_samples > 0 else 0
    g9_overall = total_out_of_bounds == 0

    print(f"{'TOTAL':<8} {total_samples:>8} {total_detected:>10} {overall_accuracy:>9.1%}")
    print()
    print(f"G9 Gate: {'PASS' if g9_overall else 'FAIL'} ({total_out_of_bounds} out of bounds)")
    print("=" * 70)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="validate_turning_points",
        description="Validate turning point detection against ground-truth samples",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with TP samples",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Validate single symbol",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all symbols in config",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to trained model file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results",
    )

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    if not args.symbol and not args.all:
        parser.print_help()
        return 1

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    config = ValidationConfig.from_yaml(config_path)
    model_path = Path(args.model) if args.model else None

    if args.all:
        results = validate_all(config, model_path, args.verbose)
        print_report(results, config)

        if args.output:
            import json

            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_data = {
                symbol: {
                    "metrics": r.metrics.to_dict(),
                    "errors": r.errors,
                    "passed": r.passed,
                    # PR-05: Include predictions with timing fields
                    "predictions": [
                        {
                            "date": str(p.get("date")),
                            "turn_state": p.get("turn_state"),
                            "turn_confidence": p.get("turn_confidence"),
                            "detection_type": p.get("detection_type", "none"),
                            "bars_from_actual_tp": p.get("bars_from_actual_tp"),
                        }
                        for p in r.predictions
                        if p.get("turn_state") != "none"  # Only include non-NONE predictions
                    ],
                }
                for symbol, r in results.items()
            }
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"\nResults written to: {output_path}")

        # Return non-zero if any G9 failures
        g9_failures = sum(
            r.metrics.out_of_bounds_early + r.metrics.out_of_bounds_late for r in results.values()
        )
        return 1 if g9_failures > 0 else 0

    elif args.symbol:
        all_samples = load_samples(config)
        symbol_samples = get_samples_for_symbol(all_samples, args.symbol)

        if not symbol_samples:
            print(f"ERROR: No samples found for {args.symbol}")
            return 1

        result = validate_symbol(args.symbol, symbol_samples, config, model_path, args.verbose)
        print_report({args.symbol: result}, config)
        return 0 if result.passed else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
