"""
Train Regime Weights - Learn weights from diverse universe with proper train/test split.

Usage:
    python -m src.domain.signals.indicators.regime.train_weights --days 1000

Uses config/universe.yaml for train/test split:
    - Training: model_training subset (~35 symbols)
    - Testing: validation.holdout subset (~19 symbols, never seen during training)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class TrainTestSplit:
    """Train/test symbol split from universe config."""

    train_symbols: List[str]
    test_symbols: List[str]
    overlap: List[str]  # Should be empty

    @classmethod
    def from_universe(cls, config_path: str = "config/universe.yaml") -> "TrainTestSplit":
        """Load train/test split from universe config."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        train_symbols = config.get("subsets", {}).get("model_training", [])
        test_symbols = config.get("validation", {}).get("holdout", [])

        overlap = list(set(train_symbols) & set(test_symbols))

        return cls(
            train_symbols=train_symbols,
            test_symbols=test_symbols,
            overlap=overlap,
        )


@dataclass
class TrainingResult:
    """Result from weight training."""

    weights: Dict[str, float]
    train_accuracy: float
    test_accuracy: float
    train_r1_occupancy: float
    test_r1_occupancy: float
    train_symbols_used: int
    test_symbols_used: int
    feature_importance: Dict[str, float]
    validation_passed: bool
    failures: List[str]


def fetch_historical_data(
    symbols: List[str],
    days: int = 1000,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical OHLCV data for symbols.

    Args:
        symbols: List of symbols to fetch
        days: Number of days of history

    Returns:
        Dict mapping symbol to DataFrame with OHLCV
    """
    import yfinance as yf

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    data = {}
    failed = []

    logger.info(f"Fetching {len(symbols)} symbols, {days} days history...")

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")

            if df.empty or len(df) < 252:  # Need at least 1 year
                failed.append(symbol)
                continue

            # Normalize column names
            df.columns = [c.lower() for c in df.columns]
            if "adj close" in df.columns:
                df["close"] = df["adj close"]

            data[symbol] = df[["open", "high", "low", "close", "volume"]].copy()

        except Exception as e:
            logger.warning(f"Failed to fetch {symbol}: {e}")
            failed.append(symbol)

    logger.info(f"Fetched {len(data)} symbols, {len(failed)} failed")
    return data


def pool_data_for_training(
    data: Dict[str, pd.DataFrame],
    benchmark_symbol: str = "SPY",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Pool data from multiple symbols for training.

    Returns:
        Tuple of (factors_df, prices_df, labels)
    """
    from .factor_normalizer import compute_normalized_factors
    from .weight_learner import TargetLabelGenerator

    all_factors = []
    all_labels = []
    all_prices = []

    benchmark_df = data.get(benchmark_symbol)
    label_gen = TargetLabelGenerator(forward_bars=10)

    for symbol, df in data.items():
        try:
            # Compute factors
            factors = compute_normalized_factors(df, benchmark_df)
            factors_df = factors.to_dataframe()
            factors_df["symbol"] = symbol

            # Create labels
            labels = label_gen.create_risk_labels(df, drawdown_threshold_pct=5.0)

            # Align
            combined = factors_df.join(labels).dropna()
            if len(combined) < 100:
                continue

            all_factors.append(combined.drop(columns=["risk_label"]))
            all_labels.append(combined["risk_label"])
            all_prices.append(df["close"].loc[combined.index])

        except Exception as e:
            logger.debug(f"Failed to process {symbol}: {e}")

    if not all_factors:
        raise ValueError("No valid data after processing")

    factors_pooled = pd.concat(all_factors, ignore_index=False)
    labels_pooled = pd.concat(all_labels, ignore_index=False)
    prices_pooled = pd.concat(all_prices, ignore_index=False)

    return factors_pooled, prices_pooled, labels_pooled


def train_and_validate(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    method: str = "logistic",
) -> TrainingResult:
    """
    Train weights on training data and validate on test data.

    Args:
        train_data: Dict of symbol -> DataFrame for training
        test_data: Dict of symbol -> DataFrame for testing
        method: "logistic" or "optimize"

    Returns:
        TrainingResult with weights and metrics
    """
    from .composite_scorer import CompositeRegimeScorer
    from .regime_validation import FailureCriteria, RegimeValidator
    from .weight_learner import WeightLearner

    # Pool training data
    logger.info(f"Pooling {len(train_data)} training symbols...")
    train_factors, train_prices, train_labels = pool_data_for_training(train_data)
    logger.info(f"Training samples: {len(train_factors)}")

    # Learn weights
    logger.info(f"Learning weights using {method}...")
    learner = WeightLearner(method=method)
    # All 5 Phase 5 factors (must match CompositeWeights)
    factor_cols = [
        "trend",
        "trend_short",
        "momentum",
        "volatility",
        "breadth",
    ]
    available_cols = [c for c in factor_cols if c in train_factors.columns]

    result = learner.fit(train_factors[available_cols], train_labels, available_cols)
    logger.info(f"Learned weights: {result.weights.to_dict()}")
    logger.info(f"Train accuracy: {result.train_accuracy:.1%}")

    # Create scorer with learned weights
    scorer = CompositeRegimeScorer(weights=result.weights)

    # Evaluate on training data
    train_regimes = []
    for symbol, df in train_data.items():
        try:
            scored = scorer.score_and_classify(df)
            train_regimes.append(scored["regime"])
        except Exception:
            pass

    train_regimes_all = pd.concat(train_regimes) if train_regimes else pd.Series(dtype=str)

    # Evaluate on test data (out-of-sample)
    logger.info(f"Validating on {len(test_data)} test symbols...")
    test_regimes = []
    test_returns = []
    for symbol, df in test_data.items():
        try:
            scored = scorer.score_and_classify(df)
            test_regimes.append(scored["regime"])
            test_returns.append(df["close"].pct_change())
        except Exception:
            pass

    test_regimes_all = pd.concat(test_regimes) if test_regimes else pd.Series(dtype=str)
    test_returns_all = pd.concat(test_returns) if test_returns else pd.Series(dtype=float)

    # Compute R1 occupancy
    train_r1 = (train_regimes_all == "R1").mean() if len(train_regimes_all) > 0 else 0.0
    test_r1 = (test_regimes_all == "R1").mean() if len(test_regimes_all) > 0 else 0.0

    # Validate test results
    validator = RegimeValidator(FailureCriteria(max_r1_occupancy=0.70))
    validation = validator.validate(test_regimes_all, test_returns_all)

    return TrainingResult(
        weights=result.weights.to_dict(),
        train_accuracy=result.train_accuracy,
        test_accuracy=0.0,  # Would need labels for test
        train_r1_occupancy=float(train_r1),
        test_r1_occupancy=float(test_r1),
        train_symbols_used=len(train_data),
        test_symbols_used=len(test_data),
        feature_importance=result.feature_importance,
        validation_passed=validation.passed,
        failures=validation.failures,
    )


def save_weights(weights: Dict[str, float], output_path: str) -> None:
    """Save learned weights to YAML file."""
    # Convert numpy floats to plain Python floats for clean YAML output
    clean_weights = {k: float(v) for k, v in weights.items()}

    output = {
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "method": "logistic_regression",
        "weights": clean_weights,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False)

    logger.info(f"Saved weights to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train regime weights")
    parser.add_argument("--days", type=int, default=1000, help="Days of history")
    parser.add_argument("--method", default="logistic", choices=["logistic", "optimize"])
    parser.add_argument("--output", default="config/regime_weights.yaml", help="Output path")
    parser.add_argument("--dry-run", action="store_true", help="Don't save weights")
    args = parser.parse_args()

    # Load train/test split
    split = TrainTestSplit.from_universe()
    logger.info(f"Train symbols: {len(split.train_symbols)}")
    logger.info(f"Test symbols: {len(split.test_symbols)}")

    if split.overlap:
        logger.warning(f"Overlap detected: {split.overlap}")

    # Fetch data
    train_data = fetch_historical_data(split.train_symbols, args.days)
    test_data = fetch_historical_data(split.test_symbols, args.days)

    if len(train_data) < 10:
        logger.error("Insufficient training data")
        return

    # Train and validate
    result = train_and_validate(train_data, test_data, args.method)

    # Report
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training symbols used: {result.train_symbols_used}")
    print(f"Test symbols used: {result.test_symbols_used}")
    print()
    print("Learned Weights:")
    for k, v in result.weights.items():
        print(f"  {k}: {v:.3f}")
    print()
    print(f"Train accuracy: {result.train_accuracy:.1%}")
    print(f"Train R1 occupancy: {result.train_r1_occupancy:.1%}")
    print(f"Test R1 occupancy: {result.test_r1_occupancy:.1%}")
    print()
    print(f"Validation passed: {result.validation_passed}")
    if result.failures:
        print("Failures:")
        for f in result.failures:
            print(f"  - {f}")
    print("=" * 60)

    # Save weights
    if not args.dry_run:
        save_weights(result.weights, args.output)


if __name__ == "__main__":
    main()
