"""
Regime Sensitivity Validation Tests.

PR-04 Deliverable: Test that regime detector correctly classifies known market conditions.

Validates that:
- Trending stocks (MU, NVDA) get R0 >= 85% of the time
- Choppy stocks (GME, AMC) get R0 <= 15% of the time
- VIX is almost never R0 (<= 5%)

These tests ensure the regime detector doesn't produce false positives
(calling choppy markets "healthy uptrends") or false negatives
(failing to recognize legitimate uptrends).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import pytest
import yaml

# Mark module as slow tests (require data loading)
pytestmark = [pytest.mark.slow]


@dataclass
class RegimeSample:
    """Sample for regime sensitivity testing."""

    symbol: str
    start_date: date
    end_date: date
    expected_regime: str
    threshold: float  # min_r0_rate or max_r0_rate
    threshold_type: Literal["min", "max"]
    notes: str = ""


def load_sensitivity_samples(fixture_path: Optional[Path] = None) -> Dict[str, List[RegimeSample]]:
    """Load samples from YAML fixture."""
    if fixture_path is None:
        fixture_path = Path("tests/fixtures/regime_sensitivity_samples.yaml")

    if not fixture_path.exists():
        return {"trending": [], "choppy": [], "vix": [], "downtrend": []}

    with open(fixture_path) as f:
        data = yaml.safe_load(f)

    samples: Dict[str, List[RegimeSample]] = {
        "trending": [],
        "choppy": [],
        "vix": [],
        "downtrend": [],
    }

    # Load trending samples
    for entry in data.get("trending_samples", []):
        period = entry.get("period", {})
        start = period.get("start")
        end = period.get("end")
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d").date()
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d").date()

        samples["trending"].append(
            RegimeSample(
                symbol=entry["symbol"],
                start_date=start,
                end_date=end,
                expected_regime="R0",
                threshold=entry.get("min_r0_rate", 0.85),
                threshold_type="min",
                notes=entry.get("notes", ""),
            )
        )

    # Load choppy samples
    for entry in data.get("choppy_samples", []):
        period = entry.get("period", {})
        start = period.get("start")
        end = period.get("end")
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d").date()
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d").date()

        samples["choppy"].append(
            RegimeSample(
                symbol=entry["symbol"],
                start_date=start,
                end_date=end,
                expected_regime="R1_or_R2",
                threshold=entry.get("max_r0_rate", 0.15),
                threshold_type="max",
                notes=entry.get("notes", ""),
            )
        )

    # Load VIX samples
    for entry in data.get("vix_samples", []):
        period = entry.get("period", {})
        start = period.get("start")
        end = period.get("end")
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d").date()
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d").date()

        samples["vix"].append(
            RegimeSample(
                symbol=entry["symbol"],
                start_date=start,
                end_date=end,
                expected_regime="R2",
                threshold=entry.get("max_r0_rate", 0.05),
                threshold_type="max",
                notes=entry.get("notes", ""),
            )
        )

    # Load downtrend samples
    for entry in data.get("downtrend_samples", []):
        period = entry.get("period", {})
        start = period.get("start")
        end = period.get("end")
        if isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d").date()
        if isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d").date()

        samples["downtrend"].append(
            RegimeSample(
                symbol=entry["symbol"],
                start_date=start,
                end_date=end,
                expected_regime="R2",
                threshold=entry.get("min_r2_rate", 0.70),
                threshold_type="min",
                notes=entry.get("notes", ""),
            )
        )

    return samples


def get_trending_samples() -> List[RegimeSample]:
    """Get trending samples for parametrized tests."""
    samples = load_sensitivity_samples()
    return samples.get("trending", [])


def get_choppy_samples() -> List[RegimeSample]:
    """Get choppy samples for parametrized tests."""
    samples = load_sensitivity_samples()
    return samples.get("choppy", [])


def get_vix_samples() -> List[RegimeSample]:
    """Get VIX samples for parametrized tests."""
    samples = load_sensitivity_samples()
    return samples.get("vix", [])


def get_downtrend_samples() -> List[RegimeSample]:
    """Get downtrend samples for parametrized tests."""
    samples = load_sensitivity_samples()
    return samples.get("downtrend", [])


def load_price_data(
    symbol: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Load historical price data for symbol."""
    try:
        from src.infrastructure.adapters.yahoo.yahoo_adapter import YahooMarketDataAdapter

        adapter = YahooMarketDataAdapter()

        # Need extra history for regime calculation
        extended_start = start_date - timedelta(days=300)
        start_dt = datetime.combine(extended_start, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.max.time())

        bars = adapter.get_historical_bars(
            symbol=symbol,
            start=start_dt,
            end=end_dt,
            timeframe="1d",
        )

        if not bars:
            return None

        df = pd.DataFrame([b.to_dict() for b in bars])
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df = df.set_index("date").sort_index()
        return df

    except Exception as e:
        pytest.skip(f"Failed to load data for {symbol}: {e}")
        return None


def compute_regime_rates(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> Dict[str, float]:
    """
    Compute regime distribution rates for a period.

    Returns dict with R0, R1, R2, R3 rates.
    """
    try:
        from src.domain.signals.indicators.regime.regime_detector import RegimeDetectorIndicator

        detector = RegimeDetectorIndicator()

        # Filter to analysis period
        period_df = df.loc[start_date:end_date]
        if period_df.empty:
            return {"R0": 0, "R1": 0, "R2": 0, "R3": 0, "total_days": 0}

        regime_counts = {"R0": 0, "R1": 0, "R2": 0, "R3": 0}
        total_days = 0

        for idx in range(len(period_df)):
            # Get data up to current date (no look-ahead)
            current_date = period_df.index[idx]
            history = df.loc[:current_date]

            if len(history) < 200:  # Need minimum history for MA200
                continue

            # Run regime detection
            output = detector.calculate(history)
            if output:
                regime = output.regime.value  # "R0", "R1", "R2", or "R3"
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
                total_days += 1

        # Compute rates
        if total_days == 0:
            return {"R0": 0, "R1": 0, "R2": 0, "R3": 0, "total_days": 0}

        return {
            "R0": regime_counts["R0"] / total_days,
            "R1": regime_counts["R1"] / total_days,
            "R2": regime_counts["R2"] / total_days,
            "R3": regime_counts["R3"] / total_days,
            "total_days": total_days,
        }

    except Exception as e:
        pytest.fail(f"Failed to compute regime rates: {e}")
        return {}


@pytest.fixture
def sensitivity_samples():
    """Load all sensitivity samples from fixture."""
    return load_sensitivity_samples()


class TestRegimeSensitivity:
    """
    Regime sensitivity validation tests.

    G10 Gate: Validates regime detector classifies known conditions correctly.
    """

    @pytest.mark.parametrize(
        "sample",
        get_trending_samples(),
        ids=lambda s: f"{s.symbol}_{s.start_date}" if s else "no_sample",
    )
    def test_trending_stock_gets_r0(self, sample: RegimeSample) -> None:
        """
        Trending stocks should be R0 most of the time.

        G10 threshold: R0 >= 85% for trending samples.
        """
        if not sample:
            pytest.skip("No trending samples defined")

        df = load_price_data(sample.symbol, sample.start_date, sample.end_date)
        if df is None:
            pytest.skip(f"No data available for {sample.symbol}")

        rates = compute_regime_rates(df, sample.start_date, sample.end_date)

        assert (
            rates["total_days"] >= 20
        ), f"Insufficient data: only {rates['total_days']} days for {sample.symbol}"

        r0_rate = rates["R0"]
        assert r0_rate >= sample.threshold, (
            f"TRENDING SENSITIVITY FAIL: {sample.symbol} R0 rate {r0_rate:.1%} "
            f"< {sample.threshold:.1%} threshold\n"
            f"Period: {sample.start_date} to {sample.end_date}\n"
            f"Notes: {sample.notes}\n"
            f"Regime distribution: R0={rates['R0']:.1%}, R1={rates['R1']:.1%}, "
            f"R2={rates['R2']:.1%}, R3={rates['R3']:.1%}"
        )

    @pytest.mark.parametrize(
        "sample",
        get_choppy_samples(),
        ids=lambda s: f"{s.symbol}_{s.start_date}" if s else "no_sample",
    )
    def test_choppy_stock_avoids_r0(self, sample: RegimeSample) -> None:
        """
        Choppy stocks should rarely be R0.

        G10 threshold: R0 <= 15% for choppy samples.
        """
        if not sample:
            pytest.skip("No choppy samples defined")

        df = load_price_data(sample.symbol, sample.start_date, sample.end_date)
        if df is None:
            pytest.skip(f"No data available for {sample.symbol}")

        rates = compute_regime_rates(df, sample.start_date, sample.end_date)

        assert (
            rates["total_days"] >= 20
        ), f"Insufficient data: only {rates['total_days']} days for {sample.symbol}"

        r0_rate = rates["R0"]
        assert r0_rate <= sample.threshold, (
            f"CHOPPY SENSITIVITY FAIL: {sample.symbol} R0 rate {r0_rate:.1%} "
            f"> {sample.threshold:.1%} threshold\n"
            f"Period: {sample.start_date} to {sample.end_date}\n"
            f"Notes: {sample.notes}\n"
            f"Regime distribution: R0={rates['R0']:.1%}, R1={rates['R1']:.1%}, "
            f"R2={rates['R2']:.1%}, R3={rates['R3']:.1%}"
        )

    @pytest.mark.parametrize(
        "sample",
        get_vix_samples(),
        ids=lambda s: f"{s.symbol}_{s.start_date}" if s else "no_sample",
    )
    def test_vix_never_trending(self, sample: RegimeSample) -> None:
        """
        VIX should almost never be classified as R0.

        G10 threshold: R0 <= 5% for VIX samples.
        """
        if not sample:
            pytest.skip("No VIX samples defined")

        df = load_price_data(sample.symbol, sample.start_date, sample.end_date)
        if df is None:
            pytest.skip(f"No data available for {sample.symbol}")

        rates = compute_regime_rates(df, sample.start_date, sample.end_date)

        if rates["total_days"] < 20:
            pytest.skip(f"Insufficient data: only {rates['total_days']} days")

        r0_rate = rates["R0"]
        assert r0_rate <= sample.threshold, (
            f"VIX SENSITIVITY FAIL: {sample.symbol} R0 rate {r0_rate:.1%} "
            f"> {sample.threshold:.1%} threshold\n"
            f"Period: {sample.start_date} to {sample.end_date}\n"
            f"Notes: {sample.notes}\n"
            f"VIX should be mean-reverting, never 'trending'"
        )

    @pytest.mark.parametrize(
        "sample",
        get_downtrend_samples(),
        ids=lambda s: f"{s.symbol}_{s.start_date}" if s else "no_sample",
    )
    def test_downtrend_gets_r2(self, sample: RegimeSample) -> None:
        """
        Stocks in clear downtrends should be R2 (Risk-Off) most of the time.

        G10 threshold: R2 >= 70% for downtrend samples.
        """
        if not sample:
            pytest.skip("No downtrend samples defined")

        df = load_price_data(sample.symbol, sample.start_date, sample.end_date)
        if df is None:
            pytest.skip(f"No data available for {sample.symbol}")

        rates = compute_regime_rates(df, sample.start_date, sample.end_date)

        if rates["total_days"] < 20:
            pytest.skip(f"Insufficient data: only {rates['total_days']} days")

        r2_rate = rates["R2"]
        assert r2_rate >= sample.threshold, (
            f"DOWNTREND SENSITIVITY FAIL: {sample.symbol} R2 rate {r2_rate:.1%} "
            f"< {sample.threshold:.1%} threshold\n"
            f"Period: {sample.start_date} to {sample.end_date}\n"
            f"Notes: {sample.notes}\n"
            f"Regime distribution: R0={rates['R0']:.1%}, R1={rates['R1']:.1%}, "
            f"R2={rates['R2']:.1%}, R3={rates['R3']:.1%}"
        )


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
                output = detector.calculate(history)
                if output:
                    regimes.append(output.regime.value)

            # Count regime changes
            changes = sum(1 for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1])
            change_rate = changes / len(regimes) if regimes else 0

            # Should not change more than 20% of the time for stable data
            assert change_rate < 0.20, (
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

            output = detector.calculate(df)
            assert output is None or output.regime is not None

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

            output = detector.calculate(df)
            # Should return None or a safe default
            assert output is None or output.data_quality is not None

        except ImportError:
            pytest.skip("RegimeDetectorIndicator not available")
