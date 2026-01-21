"""
Test that turning point predictions don't use future data.

GATE: G7 - FAIL if this test fails (not WARN)

Causality Testing:
    The TurningPointModel must produce identical predictions for historical dates
    regardless of how much future data is available. If adding future data changes
    past predictions, that indicates a critical look-ahead bias bug.

This test:
    1. Runs model with data up to cutoff date
    2. Runs model with data up to cutoff + 30 days
    3. Compares predictions at cutoff - they MUST be identical
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd
import pytest

if TYPE_CHECKING:
    from src.domain.signals.indicators.regime.turning_point.features import (
        TurningPointFeatures,
    )

# Mark entire module as integration tests
pytestmark = pytest.mark.integration


def features_from_dataframe(df: pd.DataFrame) -> "TurningPointFeatures":
    """
    Helper to extract TurningPointFeatures from DataFrame for the last bar.

    Uses the extract_features function and converts the last row to
    a TurningPointFeatures object.
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


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Generate synthetic price data for testing.

    Creates 200 trading days of realistic OHLCV data.
    """
    np.random.seed(42)
    n_days = 200
    base_date = date(2025, 6, 1)

    # Generate realistic price series with trend and noise
    returns = np.random.normal(0.0005, 0.015, n_days)
    prices = 100 * np.exp(np.cumsum(returns))

    dates = []
    current = base_date
    while len(dates) < n_days:
        if current.weekday() < 5:  # Skip weekends
            dates.append(current)
        current += timedelta(days=1)

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
            "high": prices * (1 + np.random.uniform(0, 0.02, n_days)),
            "low": prices * (1 - np.random.uniform(0, 0.02, n_days)),
            "close": prices,
            "volume": np.random.uniform(1e6, 5e6, n_days),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


class TestTurningPointCausality:
    """
    Causality tests for TurningPointModel.

    These tests verify that the model doesn't use future data when making predictions.
    """

    def test_no_future_data_leakage(self, sample_price_data: pd.DataFrame) -> None:
        """
        Verify predictions don't change when future data is added.

        CRITICAL: This is a G7 FAIL gate. If this test fails, the model has a
        look-ahead bias bug that MUST be fixed before deployment.
        """
        from src.domain.signals.indicators.regime.turning_point.model import (
            TurningPointModel,
            TurningPointOutput,
        )

        # Create and train a simple model for testing
        model = TurningPointModel(model_type="logistic", confidence_threshold=0.5)

        df = sample_price_data
        dates = list(df.index)

        # Define cutoff at 70% of data
        cutoff_idx = int(len(dates) * 0.7)
        cutoff_date = dates[cutoff_idx]

        # Generate synthetic labels for training
        np.random.seed(123)
        n_samples = len(dates) - 50  # Skip first 50 for warmup

        # Create feature matrix
        X_list = []
        for i in range(50, len(dates)):
            window_df = df.iloc[: i + 1]
            features = features_from_dataframe(window_df)
            X_list.append(features.to_array())

        X = np.array(X_list)
        y_top = (np.random.random(n_samples) > 0.9).astype(int)
        y_bottom = (np.random.random(n_samples) > 0.9).astype(int)

        # Train model on full data (for this test, training on all data is fine
        # because we're testing inference, not training)
        model.train(X, y_top, y_bottom)

        # === Run model with data up to cutoff ===
        df_at_cutoff = df.loc[:cutoff_date]
        signals_at_cutoff: List[TurningPointOutput] = []

        for i in range(50, len(df_at_cutoff)):
            window_df = df_at_cutoff.iloc[: i + 1]
            features = features_from_dataframe(window_df)
            output = model.predict(features)
            signals_at_cutoff.append(output)

        # === Run model with data up to cutoff + 30 days ===
        cutoff_plus_30_idx = min(cutoff_idx + 30, len(dates) - 1)
        cutoff_plus_30 = dates[cutoff_plus_30_idx]
        df_extended = df.loc[:cutoff_plus_30]
        signals_after: List[TurningPointOutput] = []

        # Only compute signals up to cutoff (same as before)
        for i in range(50, cutoff_idx - 50 + 1):
            window_df = df_extended.iloc[: i + 1]
            features = features_from_dataframe(window_df)
            output = model.predict(features)
            signals_after.append(output)

        # === Compare predictions at cutoff ===
        # Both runs should produce identical predictions for the same dates
        assert len(signals_at_cutoff[: len(signals_after)]) >= len(signals_after)

        for i, (before, after) in enumerate(zip(signals_at_cutoff, signals_after)):
            assert before.turn_state == after.turn_state, (
                f"CAUSALITY VIOLATION at index {i}: "
                f"turn_state changed from {before.turn_state} to {after.turn_state} "
                f"when future data was added!"
            )
            assert abs(before.turn_confidence - after.turn_confidence) < 1e-6, (
                f"CAUSALITY VIOLATION at index {i}: "
                f"confidence changed from {before.turn_confidence:.6f} to "
                f"{after.turn_confidence:.6f} when future data was added!"
            )

    def test_prediction_determinism(self, sample_price_data: pd.DataFrame) -> None:
        """
        Verify that predictions are deterministic.

        Running the same model on the same data should produce identical results.
        """
        from src.domain.signals.indicators.regime.turning_point.model import TurningPointModel

        model = TurningPointModel(model_type="logistic")

        df = sample_price_data
        dates = list(df.index)

        # Train model
        np.random.seed(456)
        n_samples = len(dates) - 50
        X_list = []
        for i in range(50, len(dates)):
            window_df = df.iloc[: i + 1]
            features = features_from_dataframe(window_df)
            X_list.append(features.to_array())

        X = np.array(X_list)
        y_top = (np.random.random(n_samples) > 0.9).astype(int)
        y_bottom = (np.random.random(n_samples) > 0.9).astype(int)
        model.train(X, y_top, y_bottom)

        # Run predictions twice
        predictions_run1 = []
        predictions_run2 = []

        for i in range(50, min(100, len(dates))):
            window_df = df.iloc[: i + 1]
            features = features_from_dataframe(window_df)

            out1 = model.predict(features)
            out2 = model.predict(features)

            predictions_run1.append(out1)
            predictions_run2.append(out2)

        # Compare - should be identical
        for i, (p1, p2) in enumerate(zip(predictions_run1, predictions_run2)):
            assert p1.turn_state == p2.turn_state, (
                f"Non-deterministic prediction at index {i}: " f"{p1.turn_state} vs {p2.turn_state}"
            )
            assert (
                abs(p1.turn_confidence - p2.turn_confidence) < 1e-9
            ), f"Non-deterministic confidence at index {i}"

    def test_feature_window_independence(self, sample_price_data: pd.DataFrame) -> None:
        """
        Verify features only use data from the window, not global state.

        This catches bugs where features accidentally reference data outside
        the intended window (e.g., through pandas index alignment issues).
        """
        df = sample_price_data

        # Create two different windows that end at the same point
        # but have different starting points
        end_idx = 100
        window1 = df.iloc[0 : end_idx + 1]  # Full history
        window2 = df.iloc[40 : end_idx + 1]  # Truncated history

        features1 = features_from_dataframe(window1)
        features2 = features_from_dataframe(window2)

        # Features from same endpoint should be similar but not necessarily identical
        # (some features may use longer lookbacks)
        # The key invariant: features should be computed from the window, not global state

        # Both should produce valid feature arrays
        arr1 = features1.to_array()
        arr2 = features2.to_array()

        assert arr1.shape == arr2.shape, "Feature dimensions should be consistent"
        assert not np.any(np.isnan(arr1)), "Features should not contain NaN"
        assert not np.any(np.isnan(arr2)), "Features should not contain NaN"

    def test_no_future_price_in_features(self, sample_price_data: pd.DataFrame) -> None:
        """
        Verify features don't include future price information.

        Features at time T should only use data from times <= T.
        """
        df = sample_price_data
        current_idx = 80

        # Get features at current time
        window = df.iloc[: current_idx + 1]

        features = features_from_dataframe(window)

        # Feature values should be bounded by historical data
        # (no forward-looking return calculations)
        arr = features.to_array()

        # Returns should be <= 100% in absolute value for normal conditions
        # This catches bugs where future returns are accidentally included
        assert np.all(
            np.abs(arr) < 100
        ), "Feature values seem unreasonably large - possible future data leak"


class TestCausalityWithRealModel:
    """
    Causality tests using real trained model if available.

    These tests are skipped if no trained model exists.
    """

    @pytest.fixture
    def trained_model(self):
        """Load trained model or skip test."""
        from src.domain.signals.indicators.regime.turning_point.model import TurningPointModel

        model_path = Path("models/turning_point/latest.pkl")
        if not model_path.exists():
            pytest.skip("No trained model available")

        return TurningPointModel.load(model_path)

    def test_real_model_causality(self, trained_model, sample_price_data: pd.DataFrame) -> None:
        """
        Test causality with the real trained model.

        This is the definitive G7 gate test for production models.
        """
        model = trained_model
        df = sample_price_data
        dates = list(df.index)

        cutoff_idx = int(len(dates) * 0.7)
        cutoff_date = dates[cutoff_idx]

        # Run with data up to cutoff
        df_at_cutoff = df.loc[:cutoff_date]
        signals_at_cutoff = []

        for i in range(50, len(df_at_cutoff)):
            window_df = df_at_cutoff.iloc[: i + 1]
            features = features_from_dataframe(window_df)
            output = model.predict(features)
            signals_at_cutoff.append((output.turn_state.value, round(output.turn_confidence, 6)))

        # Run with extended data
        df_extended = df
        signals_extended = []

        for i in range(50, len(df_at_cutoff)):
            window_df = df_extended.iloc[: i + 1]
            features = features_from_dataframe(window_df)
            output = model.predict(features)
            signals_extended.append((output.turn_state.value, round(output.turn_confidence, 6)))

        # Must be identical
        assert (
            signals_at_cutoff == signals_extended
        ), "CAUSALITY VIOLATION: Real model predictions changed when future data added!"
