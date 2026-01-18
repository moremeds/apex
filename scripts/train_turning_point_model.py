#!/usr/bin/env python3
"""
DEPRECATED: Use signal_runner with --train-models instead.

This script is deprecated and will be removed in a future release.
Please migrate to the new unified CLI:

    # Train single symbol
    python -m src.runners.signal_runner --train-models --model-symbols SPY

    # Train multiple symbols
    python -m src.runners.signal_runner --train-models --model-symbols SPY QQQ AAPL

    # Train with custom parameters
    python -m src.runners.signal_runner --train-models \\
        --model-symbols SPY --model-days 750

This script remains for backward compatibility but delegates to the new
TurningPointTrainingService with hexagonal architecture.

Original usage (still works):
    python scripts/train_turning_point_model.py --symbol SPY --days 750
    python scripts/train_turning_point_model.py --symbol QQQ --model-type lightgbm
"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Emit deprecation warning
warnings.warn(
    "This script is deprecated. Use: "
    "python -m src.runners.signal_runner --train-models --model-symbols SYMBOL",
    DeprecationWarning,
    stacklevel=1,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd


def fetch_historical_data(symbol: str, days: int) -> pd.DataFrame:
    """Fetch historical OHLCV data from Yahoo Finance."""
    import yfinance as yf

    print(f"Fetching {days} days of {symbol} data from Yahoo Finance...")
    ticker = yf.Ticker(symbol)

    # Fetch with some buffer for warmup
    df = ticker.history(period=f"{days + 50}d", interval="1d")
    df.columns = df.columns.str.lower()

    if len(df) < 100:
        raise ValueError(f"Insufficient data: only {len(df)} bars")

    print(f"  Loaded {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Train Turning Point Model (DEPRECATED - use signal_runner instead)",
        epilog="""
DEPRECATED: This script is deprecated. Please use:
    python -m src.runners.signal_runner --train-models --model-symbols SYMBOL
        """,
    )
    parser.add_argument("--symbol", default="SPY", help="Symbol to train on (default: SPY)")
    parser.add_argument("--days", type=int, default=750, help="Days of history (default: 750)")
    parser.add_argument("--model-type", choices=["logistic", "lightgbm"], default="logistic",
                       help="Model type (default: logistic)")
    parser.add_argument("--cv-splits", type=int, default=5, help="CV splits (default: 5)")
    parser.add_argument("--output-dir", default="models/turning_point", help="Output directory")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period (default: 14)")
    parser.add_argument("--zigzag-threshold", type=float, default=2.0, help="ZigZag ATR threshold")
    parser.add_argument("--risk-horizon", type=int, default=10, help="Risk label horizon in bars")
    parser.add_argument("--risk-threshold", type=float, default=1.5, help="Risk ATR threshold")
    args = parser.parse_args()

    # Print deprecation notice prominently
    print("\n" + "!" * 60)
    print("! DEPRECATION WARNING")
    print("!" * 60)
    print("! This script is deprecated. Please use:")
    print("!   python -m src.runners.signal_runner --train-models \\")
    print(f"!       --model-symbols {args.symbol} --model-days {args.days}")
    print("!" * 60 + "\n")

    print("\n" + "=" * 60)
    print("TURNING POINT MODEL TRAINING")
    print("=" * 60)
    print(f"Symbol:         {args.symbol}")
    print(f"Days:           {args.days}")
    print(f"Model Type:     {args.model_type}")
    print(f"CV Splits:      {args.cv_splits}")
    print(f"Risk Horizon:   {args.risk_horizon} bars")
    print("=" * 60 + "\n")

    # Import after args parsing to avoid slow startup
    from src.domain.signals.indicators.regime.turning_point import (
        TurningPointLabeler,
        TurningPointModel,
        PurgedTimeSeriesSplit,
    )
    from src.domain.signals.indicators.regime.turning_point.features import extract_features

    # 1. Fetch data
    df = fetch_historical_data(args.symbol, args.days)

    # 2. Generate labels
    print("\nGenerating labels...")
    labeler = TurningPointLabeler(
        atr_period=args.atr_period,
        zigzag_threshold=args.zigzag_threshold,
        risk_horizon=args.risk_horizon,
        risk_threshold=args.risk_threshold,
    )

    y_top, y_bottom, pivots = labeler.generate_combined_labels(df)
    print(f"  TOP_RISK:    {y_top.sum()} positive ({100*y_top.mean():.1f}%)")
    print(f"  BOTTOM_RISK: {y_bottom.sum()} positive ({100*y_bottom.mean():.1f}%)")
    print(f"  ZigZag pivots detected: {len(pivots)}")

    # 3. Extract features
    print("\nExtracting features...")
    features_df = extract_features(df)
    print(f"  Feature matrix: {features_df.shape}")

    # 4. Align data (remove NaN rows)
    # Features have warmup period, labels have horizon at end
    valid_mask = ~features_df.isna().any(axis=1)
    valid_idx = features_df.index[valid_mask]

    # Also exclude last risk_horizon bars (labels are undefined)
    valid_idx = valid_idx[:-args.risk_horizon]

    X = features_df.loc[valid_idx].values
    y_top_arr = y_top.loc[valid_idx].values
    y_bottom_arr = y_bottom.loc[valid_idx].values

    print(f"  Training samples: {len(X)}")
    print(f"  TOP_RISK positive rate: {100*y_top_arr.mean():.1f}%")
    print(f"  BOTTOM_RISK positive rate: {100*y_bottom_arr.mean():.1f}%")

    # 5. Verify CV properties
    print("\nVerifying Purged + Embargo CV...")
    splitter = PurgedTimeSeriesSplit(
        n_splits=args.cv_splits,
        label_horizon=args.risk_horizon,
        embargo=2,
    )

    passed_no_overlap, violations = splitter.verify_no_overlap(X, y_top_arr)
    if passed_no_overlap:
        print("  No-Overlap Property: VERIFIED")
    else:
        print(f"  No-Overlap Property: FAILED - {violations}")
        return 1

    passed_embargo, violations = splitter.verify_embargo(X, y_top_arr)
    if passed_embargo:
        print("  Embargo Property: VERIFIED")
    else:
        print(f"  Embargo Property: FAILED - {violations}")
        return 1

    # 6. Train model
    print(f"\nTraining {args.model_type} model...")
    model = TurningPointModel(
        model_type=args.model_type,
        confidence_threshold=0.7,
    )

    top_metrics, bottom_metrics = model.train(
        X=X,
        y_top=y_top_arr,
        y_bottom=y_bottom_arr,
        cv_splits=args.cv_splits,
        label_horizon=args.risk_horizon,
        embargo=2,
    )

    # 7. Report metrics
    print("\n" + "-" * 60)
    print("TRAINING RESULTS")
    print("-" * 60)

    print("\nTOP_RISK Model:")
    print(f"  Samples:      {top_metrics.n_samples} ({top_metrics.n_positive} positive)")
    print(f"  ROC-AUC:      {top_metrics.cv_roc_auc_mean:.4f} +/- {top_metrics.cv_roc_auc_std:.4f}")
    print(f"  PR-AUC:       {top_metrics.cv_pr_auc_mean:.4f} +/- {top_metrics.cv_pr_auc_std:.4f}")
    if top_metrics.calibration:
        print(f"  Brier Score:  {top_metrics.calibration.brier_score:.4f}")
        print(f"  ECE:          {top_metrics.calibration.expected_calibration_error:.4f}")

    # Top features
    if top_metrics.feature_importance:
        sorted_features = sorted(top_metrics.feature_importance.items(), key=lambda x: -x[1])[:5]
        print("  Top Features:")
        for name, imp in sorted_features:
            print(f"    - {name}: {imp:.4f}")

    print("\nBOTTOM_RISK Model:")
    print(f"  Samples:      {bottom_metrics.n_samples} ({bottom_metrics.n_positive} positive)")
    print(f"  ROC-AUC:      {bottom_metrics.cv_roc_auc_mean:.4f} +/- {bottom_metrics.cv_roc_auc_std:.4f}")
    print(f"  PR-AUC:       {bottom_metrics.cv_pr_auc_mean:.4f} +/- {bottom_metrics.cv_pr_auc_std:.4f}")
    if bottom_metrics.calibration:
        print(f"  Brier Score:  {bottom_metrics.calibration.brier_score:.4f}")
        print(f"  ECE:          {bottom_metrics.calibration.expected_calibration_error:.4f}")

    if bottom_metrics.feature_importance:
        sorted_features = sorted(bottom_metrics.feature_importance.items(), key=lambda x: -x[1])[:5]
        print("  Top Features:")
        for name, imp in sorted_features:
            print(f"    - {name}: {imp:.4f}")

    # 8. Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{args.symbol.lower()}_{args.model_type}.pkl"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # 9. Show recent turning points
    print("\n" + "-" * 60)
    print("RECENT TURNING POINTS")
    print("-" * 60)

    history = labeler.get_recent_turning_points(df, n_recent=5)
    print(history.summary())

    # 10. Test inference
    print("\n" + "-" * 60)
    print("INFERENCE TEST")
    print("-" * 60)

    from src.domain.signals.indicators.regime.turning_point.features import TurningPointFeatures

    # Get last row features
    last_features = TurningPointFeatures(
        price_vs_ma20=features_df.iloc[-1]["price_vs_ma20"],
        price_vs_ma50=features_df.iloc[-1]["price_vs_ma50"],
        price_vs_ma200=features_df.iloc[-1]["price_vs_ma200"],
        ma20_slope=features_df.iloc[-1]["ma20_slope"],
        ma50_slope=features_df.iloc[-1]["ma50_slope"],
        ma20_vs_ma50=features_df.iloc[-1]["ma20_vs_ma50"],
        atr_pct_63=features_df.iloc[-1]["atr_pct_63"],
        atr_pct_252=features_df.iloc[-1]["atr_pct_252"],
        atr_expansion_rate=features_df.iloc[-1]["atr_expansion_rate"],
        vol_regime=int(features_df.iloc[-1]["vol_regime"]),
        chop_pct_252=features_df.iloc[-1]["chop_pct_252"],
        adx_value=features_df.iloc[-1]["adx_value"],
        range_position=features_df.iloc[-1]["range_position"],
        ext_atr_units=features_df.iloc[-1]["ext_atr_units"],
        ext_zscore=features_df.iloc[-1]["ext_zscore"],
        rsi_14=features_df.iloc[-1]["rsi_14"],
        roc_5=features_df.iloc[-1]["roc_5"],
        roc_10=features_df.iloc[-1]["roc_10"],
        roc_20=features_df.iloc[-1]["roc_20"],
        delta_atr_pct=features_df.iloc[-1]["delta_atr_pct"],
        delta_chop_pct=features_df.iloc[-1]["delta_chop_pct"],
        delta_ext=features_df.iloc[-1]["delta_ext"],
    )

    output = model.predict(last_features)
    print(f"Current bar prediction:")
    print(f"  Turn State:   {output.turn_state.value.upper()}")
    print(f"  Confidence:   {output.turn_confidence:.1%}")
    print(f"  Inference:    {output.inference_time_ms:.2f}ms")
    if output.top_features:
        print("  Top Features:")
        for name, contrib in output.top_features:
            sign = "+" if contrib >= 0 else ""
            print(f"    - {name}: {sign}{contrib:.4f}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
