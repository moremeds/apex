"""
Regime detector stability and edge-case tests on synthetic data.

The former per-ticker sensitivity cases (MU/NVDA trending, GME/AMC choppy, VIX) fetched
bars from Yahoo at test time and skipped on any failure, so they never ran in CI; they
were removed with the Yahoo live adapter. Regime behaviour on real data is gated by
`src.verification.regime_verifier`.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.slow]


class TestRegimeStability:
    """Tests for regime detection stability."""

    def test_regime_not_oscillating(self) -> None:
        """
        Regime shouldn't oscillate rapidly between states.

        This tests that hysteresis is working correctly.
        """
        # Generate synthetic stable uptrend data
        np.random.seed(42)
        n_days = 100
        base_date = date(2025, 1, 1)

        # Create smooth uptrend
        trend = np.linspace(100, 150, n_days)
        noise = np.random.normal(0, 1, n_days)
        prices = trend + noise

        dates = []
        current = base_date
        while len(dates) < n_days:
            if current.weekday() < 5:
                dates.append(current)
            current += timedelta(days=1)

        df = pd.DataFrame(
            {
                "open": prices * 0.998,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": np.random.uniform(1e6, 5e6, n_days),
            },
            index=dates,
        )

        try:
            from src.domain.signals.indicators.regime.regime_detector import (
                RegimeDetectorIndicator,
            )

            detector = RegimeDetectorIndicator()

            regimes = []
            for i in range(50, len(df)):
                history = df.iloc[: i + 1]
                result_df = detector.calculate(history, {})
                if not result_df.empty:
                    regimes.append(result_df["regime"].iloc[-1])

            # Count regime changes
            changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
            change_rate = changes / len(regimes) if regimes else 0

            # Should not change more than 40% of the time for stable data
            # (composite scorer with benchmark breadth factor increases transitions)
            assert change_rate < 0.40, (
                f"Regime oscillating too much: {change_rate:.1%} change rate\n"
                f"This suggests hysteresis is not working correctly"
            )

        except ImportError:
            pytest.skip("RegimeDetectorIndicator not available")


class TestRegimeEdgeCases:
    """Edge case tests for regime detection."""

    def test_empty_dataframe(self) -> None:
        """Regime detector should handle empty data gracefully."""
        try:
            from src.domain.signals.indicators.regime.regime_detector import (
                RegimeDetectorIndicator,
            )

            detector = RegimeDetectorIndicator()
            df = pd.DataFrame()

            # Empty DataFrame should raise ValueError due to missing required fields
            with pytest.raises(ValueError, match="requires fields"):
                detector.calculate(df, {})

        except ImportError:
            pytest.skip("RegimeDetectorIndicator not available")

    def test_insufficient_history(self) -> None:
        """Regime detector should handle insufficient history gracefully."""
        try:
            from src.domain.signals.indicators.regime.regime_detector import (
                RegimeDetectorIndicator,
            )

            detector = RegimeDetectorIndicator()

            # Only 10 days of data (need 200 for MA200)
            df = pd.DataFrame(
                {
                    "open": [100] * 10,
                    "high": [101] * 10,
                    "low": [99] * 10,
                    "close": [100] * 10,
                    "volume": [1e6] * 10,
                },
                index=pd.date_range("2025-01-01", periods=10),
            )

            result_df = detector.calculate(df, {})
            # Should return DataFrame with regime column (may use fallback values)
            assert "regime" in result_df.columns

        except ImportError:
            pytest.skip("RegimeDetectorIndicator not available")
