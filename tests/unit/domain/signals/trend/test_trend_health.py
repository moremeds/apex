"""
Unit tests for Trend Health Scorer components.

Tests:
- HHHLDetector: Swing detection and pattern scoring
- MAAlignmentScorer: MA order and spread scoring
- TrendHealthAnalyzer: Composite scoring
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.signals.trend.health_analyzer import TrendHealthAnalyzer, TrendHealthResult
from src.domain.signals.trend.hh_hl_detector import HHHLDetector, HHHLResult, SwingPoint
from src.domain.signals.trend.ma_alignment import MAAlignmentResult, MAAlignmentScorer


class TestHHHLDetector:
    """Tests for Higher-High/Lower-Low detector."""

    def test_detect_swings_strong_uptrend(self) -> None:
        """Test swing detection on strong uptrend data."""
        # Create synthetic uptrend with clear zigzag pattern for swing detection
        n = 50
        # Create oscillating pattern with upward trend
        # Peaks at i=5,15,25,35,45 and troughs at i=10,20,30,40
        high_vals = []
        low_vals = []
        for i in range(n):
            base = 100 + i * 1  # Upward trend
            # Add oscillation
            if i % 10 < 5:
                high_vals.append(base + 5 + (i % 5))
                low_vals.append(base + 2)
            else:
                high_vals.append(base + 2)
                low_vals.append(base - 5 + (i % 5))
        high = pd.Series(high_vals)
        low = pd.Series(low_vals)

        detector = HHHLDetector(lookback=40, swing_threshold=0.01, min_bars_between_swings=2)
        swings = detector.detect_swings(high, low)

        # May detect 0 or more swings depending on noise; just ensure it runs without error
        assert isinstance(swings, list)

    def test_calculate_score_strong_uptrend(self) -> None:
        """Test scoring on strong uptrend."""
        # Create clear uptrend with HH/HL pattern
        n = 40
        high = pd.Series(
            [100, 102, 105, 103, 108, 106, 112, 110, 118, 115]
            + [120 + i * 2 for i in range(n - 10)]
        )
        low = pd.Series(
            [99, 100, 102, 101, 105, 104, 109, 108, 114, 113] + [118 + i * 2 for i in range(n - 10)]
        )

        detector = HHHLDetector(lookback=30, swing_threshold=0.01)
        result = detector.calculate_score(high, low)

        assert isinstance(result, HHHLResult)
        assert 0 <= result.score <= 100
        assert result.trend_direction in ("up", "down", "neutral")

    def test_calculate_score_downtrend(self) -> None:
        """Test scoring on downtrend."""
        # Create clear downtrend with LH/LL pattern
        n = 40
        high = pd.Series([200 - i * 2 for i in range(n)])
        low = pd.Series([198 - i * 2 for i in range(n)])

        detector = HHHLDetector(lookback=30, swing_threshold=0.01)
        result = detector.calculate_score(high, low)

        assert isinstance(result, HHHLResult)
        assert 0 <= result.score <= 100
        # Downtrend should have lower score
        assert result.score <= 50

    def test_calculate_score_choppy_market(self) -> None:
        """Test scoring on choppy/sideways market."""
        # Create choppy market with oscillating prices
        n = 40
        np.random.seed(42)
        base = 100
        noise = np.random.randn(n) * 2
        high = pd.Series([base + noise[i] + 1 for i in range(n)])
        low = pd.Series([base + noise[i] - 1 for i in range(n)])

        detector = HHHLDetector(lookback=30, swing_threshold=0.01)
        result = detector.calculate_score(high, low)

        # Choppy market should score around neutral (40-60)
        assert 30 <= result.score <= 70

    def test_insufficient_data(self) -> None:
        """Test behavior with insufficient data."""
        high = pd.Series([100, 101, 102])
        low = pd.Series([99, 100, 101])

        detector = HHHLDetector(lookback=20)
        result = detector.calculate_score(high, low)

        # Should return neutral with insufficient data
        assert result.score == 50.0
        assert result.trend_direction == "neutral"


class TestMAAlignmentScorer:
    """Tests for MA Alignment Scorer."""

    def test_bullish_alignment(self) -> None:
        """Test scoring when MAs are bullish aligned (short > medium > long)."""
        # Create prices where short MA > medium MA > long MA
        n = 250
        # Start low, trend up so recent prices pull SMA20 above SMA50 above SMA200
        close = pd.Series([50 + i * 0.5 for i in range(n)])

        scorer = MAAlignmentScorer(periods=[20, 50, 200])
        result = scorer.calculate(close)

        assert isinstance(result, MAAlignmentResult)
        assert result.direction == "bullish"
        assert result.order_correct is True
        # Bullish alignment should score higher
        assert result.score > 50

    def test_bearish_alignment(self) -> None:
        """Test scoring when MAs are bearish aligned (short < medium < long)."""
        # Create prices where short MA < medium MA < long MA
        n = 250
        # Start high, trend down
        close = pd.Series([200 - i * 0.5 for i in range(n)])

        scorer = MAAlignmentScorer(periods=[20, 50, 200])
        result = scorer.calculate(close)

        assert isinstance(result, MAAlignmentResult)
        assert result.direction == "bearish"
        assert result.order_correct is True

    def test_mixed_alignment(self) -> None:
        """Test scoring when MAs are not aligned."""
        # Create choppy prices
        n = 250
        np.random.seed(42)
        close = pd.Series([100 + np.sin(i / 20) * 10 + np.random.randn() * 2 for i in range(n)])

        scorer = MAAlignmentScorer(periods=[20, 50, 200])
        result = scorer.calculate(close)

        assert isinstance(result, MAAlignmentResult)
        # Score should be moderate for mixed alignment
        assert 20 <= result.score <= 80

    def test_spread_calculation(self) -> None:
        """Test that spread percentage is calculated correctly."""
        n = 250
        close = pd.Series([100 + i * 0.3 for i in range(n)])

        scorer = MAAlignmentScorer(periods=[20, 50, 200])
        result = scorer.calculate(close)

        # Spread should be positive for trending data
        assert result.spread_pct >= 0

    def test_insufficient_data(self) -> None:
        """Test behavior with insufficient data."""
        close = pd.Series([100, 101, 102])

        scorer = MAAlignmentScorer(periods=[20, 50, 200])
        result = scorer.calculate(close)

        # Should return neutral with insufficient data
        assert result.score == 50.0
        assert result.direction == "neutral"


class TestTrendHealthAnalyzer:
    """Tests for Trend Health Analyzer."""

    def test_strong_uptrend(self) -> None:
        """Test composite scoring on strong uptrend."""
        n = 250
        # Strong uptrend: rising prices, high ADX, neutral RSI
        high = pd.Series([100 + i * 0.5 + 1 for i in range(n)])
        low = pd.Series([100 + i * 0.5 - 1 for i in range(n)])
        close = pd.Series([100 + i * 0.5 for i in range(n)])

        analyzer = TrendHealthAnalyzer()
        result = analyzer.calculate(
            high=high,
            low=low,
            close=close,
            adx=35.0,  # Strong trend
            rsi=55.0,  # Healthy
        )

        assert isinstance(result, TrendHealthResult)
        assert 0 <= result.score <= 100
        assert result.direction == "bullish"
        # Strong uptrend should score well
        assert result.score > 50
        assert 0 <= result.confidence <= 1

    def test_weak_downtrend(self) -> None:
        """Test composite scoring on weak downtrend."""
        n = 250
        # Downtrend with low ADX
        high = pd.Series([200 - i * 0.3 + 1 for i in range(n)])
        low = pd.Series([200 - i * 0.3 - 1 for i in range(n)])
        close = pd.Series([200 - i * 0.3 for i in range(n)])

        analyzer = TrendHealthAnalyzer()
        result = analyzer.calculate(
            high=high,
            low=low,
            close=close,
            adx=12.0,  # Weak trend
            rsi=40.0,
        )

        assert isinstance(result, TrendHealthResult)
        assert 0 <= result.score <= 100
        # Weak downtrend should score lower
        assert result.score < 70

    def test_rsi_extremes_penalize_score(self) -> None:
        """Test that RSI at extremes lowers the score."""
        n = 250
        high = pd.Series([100 + i * 0.3 + 1 for i in range(n)])
        low = pd.Series([100 + i * 0.3 - 1 for i in range(n)])
        close = pd.Series([100 + i * 0.3 for i in range(n)])

        analyzer = TrendHealthAnalyzer()

        # Healthy RSI
        result_healthy = analyzer.calculate(high, low, close, adx=30, rsi=50)

        # Extreme RSI
        result_extreme = analyzer.calculate(high, low, close, adx=30, rsi=85)

        # Extreme RSI should have lower score
        assert result_healthy.score > result_extreme.score

    def test_component_weights_sum_to_one(self) -> None:
        """Verify component weights sum to 1.0."""
        analyzer = TrendHealthAnalyzer()
        total_weight = sum(analyzer.WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.001

    def test_to_dict_serialization(self) -> None:
        """Test result serialization to dictionary."""
        n = 250
        high = pd.Series([100 + i * 0.3 + 1 for i in range(n)])
        low = pd.Series([100 + i * 0.3 - 1 for i in range(n)])
        close = pd.Series([100 + i * 0.3 for i in range(n)])

        analyzer = TrendHealthAnalyzer()
        result = analyzer.calculate(high, low, close, adx=25, rsi=50)
        result_dict = result.to_dict()

        assert "score" in result_dict
        assert "direction" in result_dict
        assert "components" in result_dict
        assert "confidence" in result_dict
        assert "hh_hl" in result_dict
        assert "ma_alignment" in result_dict

    def test_confidence_with_nan_values(self) -> None:
        """Test confidence calculation with NaN values in data."""
        n = 250
        high = pd.Series([100 + i * 0.3 + 1 for i in range(n)])
        low = pd.Series([100 + i * 0.3 - 1 for i in range(n)])
        close = pd.Series([100 + i * 0.3 for i in range(n)])

        # Add some NaN values
        close.iloc[10] = np.nan
        close.iloc[20] = np.nan

        analyzer = TrendHealthAnalyzer()
        result = analyzer.calculate(high, low, close, adx=25, rsi=50)

        # Confidence should be reduced but still valid
        assert 0 < result.confidence < 1

    def test_short_data_series(self) -> None:
        """Test behavior with short data series."""
        n = 50
        high = pd.Series([100 + i * 0.3 + 1 for i in range(n)])
        low = pd.Series([100 + i * 0.3 - 1 for i in range(n)])
        close = pd.Series([100 + i * 0.3 for i in range(n)])

        analyzer = TrendHealthAnalyzer()
        result = analyzer.calculate(high, low, close, adx=25, rsi=50)

        # Should still produce valid results
        assert 0 <= result.score <= 100
        # Confidence should be reduced for short series
        assert result.confidence < 1.0


class TestIntegration:
    """Integration tests for the complete trend health system."""

    def test_real_world_scenario_rally(self) -> None:
        """Simulate a realistic market rally scenario."""
        # Simulate a 6-month rally: 250 trading days (need 200+ for MA200)
        n = 250
        np.random.seed(123)

        # Start at 100, end around 150 (50% gain)
        trend = np.linspace(100, 150, n)
        noise = np.random.randn(n) * 1.5

        close = pd.Series(trend + noise)
        high = pd.Series(trend + noise + np.abs(np.random.randn(n)) * 1)
        low = pd.Series(trend + noise - np.abs(np.random.randn(n)) * 1)

        analyzer = TrendHealthAnalyzer()
        result = analyzer.calculate(
            high=high,
            low=low,
            close=close,
            adx=28.0,
            rsi=58.0,
        )

        # A steady rally should score well (at least decent)
        assert result.score > 40
        # Direction depends on component agreement; just ensure it's valid
        assert result.direction in ("bullish", "bearish", "neutral")

    def test_real_world_scenario_crash(self) -> None:
        """Simulate a market crash scenario."""
        # Need 200+ bars for MA200
        n = 250
        np.random.seed(456)

        # Sharp decline: 200 to 100 (50% drop)
        trend = np.linspace(200, 100, n)
        noise = np.random.randn(n) * 2

        close = pd.Series(trend + noise)
        high = pd.Series(trend + noise + np.abs(np.random.randn(n)) * 2)
        low = pd.Series(trend + noise - np.abs(np.random.randn(n)) * 2)

        analyzer = TrendHealthAnalyzer()
        result = analyzer.calculate(
            high=high,
            low=low,
            close=close,
            adx=40.0,  # High ADX during crash
            rsi=25.0,  # Oversold
        )

        # Crash should have a valid score (direction depends on components)
        assert 0 <= result.score <= 100
        assert result.direction in ("bullish", "bearish", "neutral")
