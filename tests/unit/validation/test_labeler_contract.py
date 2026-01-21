"""Tests for Frozen Labeler Contract."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.domain.signals.validation.labeler_contract import (
    LABELER_THRESHOLDS_V1,
    LabeledPeriod,
    RegimeLabel,
    RegimeLabeler,
    RegimeLabelerConfig,
)


class TestRegimeLabelerConfig:
    """Tests for RegimeLabelerConfig dataclass."""

    def test_load_v1_daily(self):
        """Test loading v1.0 config for daily timeframe."""
        cfg = RegimeLabelerConfig.load_v1("1d", horizon_days=20)

        assert cfg.version == "v1.0"
        assert cfg.timeframe == "1d"
        assert cfg.trending_forward_return_min == 0.10
        assert cfg.trending_sharpe_min == 1.0
        assert cfg.choppy_volatility_min == 0.25
        assert cfg.choppy_drawdown_max == -0.10
        assert cfg.label_horizon_bars == 20

    def test_load_v1_4h(self):
        """Test loading v1.0 config for 4h timeframe."""
        cfg = RegimeLabelerConfig.load_v1("4h", horizon_days=20)

        assert cfg.version == "v1.0"
        assert cfg.timeframe == "4h"
        assert cfg.trending_forward_return_min == 0.06
        assert cfg.trending_sharpe_min == 0.8
        assert cfg.label_horizon_bars == 32  # 20 * 1.625

    def test_load_v1_2h(self):
        """Test loading v1.0 config for 2h timeframe."""
        cfg = RegimeLabelerConfig.load_v1("2h", horizon_days=20)

        assert cfg.timeframe == "2h"
        assert cfg.trending_forward_return_min == 0.04
        assert cfg.label_horizon_bars == 65  # 20 * 3.25

    def test_load_v1_unknown_timeframe(self):
        """Test that unknown timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            RegimeLabelerConfig.load_v1("3h")

    def test_config_is_frozen(self):
        """Test that config is immutable."""
        cfg = RegimeLabelerConfig.load_v1("1d")

        with pytest.raises(AttributeError):
            cfg.trending_forward_return_min = 0.5  # type: ignore

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cfg = RegimeLabelerConfig.load_v1("1d")
        d = cfg.to_dict()

        assert d["version"] == "v1.0"
        assert d["timeframe"] == "1d"
        assert d["trending_forward_return_min"] == 0.10
        assert d["label_horizon_bars"] == 20


class TestRegimeLabeler:
    """Tests for RegimeLabeler class."""

    @pytest.fixture
    def labeler_1d(self):
        """Create daily labeler."""
        config = RegimeLabelerConfig.load_v1("1d", horizon_days=10)
        return RegimeLabeler(config)

    @pytest.fixture
    def trending_df(self):
        """Create DataFrame with trending price action."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Strong uptrend: 15% return over 10 bars
        prices = 100 * (1.015 ** np.arange(50))  # ~1.5% daily
        return pd.DataFrame({"close": prices}, index=dates)

    @pytest.fixture
    def choppy_df(self):
        """Create DataFrame with choppy/volatile price action."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # High volatility oscillation
        np.random.seed(42)
        returns = np.random.normal(0, 0.03, 50)  # 3% daily vol
        prices = 100 * np.cumprod(1 + returns)
        return pd.DataFrame({"close": prices}, index=dates)

    @pytest.fixture
    def neutral_df(self):
        """Create DataFrame with neutral/sideways price action."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        # Slight uptrend but low volatility
        prices = 100 + np.arange(50) * 0.1  # Very slight drift
        return pd.DataFrame({"close": prices}, index=dates)

    def test_label_trending_period(self, labeler_1d, trending_df):
        """Test that strong uptrend is labeled as TRENDING."""
        labels = labeler_1d.label_period(trending_df)

        # Should have labels for bars 0 to n-horizon
        assert len(labels) > 0

        # Strong uptrend should have TRENDING labels
        trending_count = sum(1 for lp in labels if lp.label == RegimeLabel.TRENDING)
        assert trending_count > len(labels) * 0.5, "Strong uptrend should be mostly TRENDING"

    def test_label_choppy_period(self, labeler_1d, choppy_df):
        """Test that high volatility is labeled as CHOPPY."""
        labels = labeler_1d.label_period(choppy_df)

        assert len(labels) > 0

        # High vol should have CHOPPY labels
        choppy_count = sum(1 for lp in labels if lp.label == RegimeLabel.CHOPPY)
        assert choppy_count > len(labels) * 0.3, "High vol should have some CHOPPY labels"

    def test_label_neutral_period(self, labeler_1d, neutral_df):
        """Test that sideways market is labeled as NEUTRAL."""
        labels = labeler_1d.label_period(neutral_df)

        assert len(labels) > 0

        # Sideways should be mostly NEUTRAL (neither trending nor choppy)
        neutral_count = sum(1 for lp in labels if lp.label == RegimeLabel.NEUTRAL)
        assert neutral_count > 0, "Sideways market should have NEUTRAL labels"

    def test_max_end_date_boundary(self, labeler_1d):
        """Test that max_end_date prevents label leakage."""
        dates = pd.date_range("2024-01-01", periods=50, freq="D")
        prices = 100 * (1.01 ** np.arange(50))
        df = pd.DataFrame({"close": prices}, index=dates)

        # Set boundary at bar 20
        boundary = date(2024, 1, 15)

        labels_with_boundary = labeler_1d.label_period(df, max_end_date=boundary)
        labels_without = labeler_1d.label_period(df)

        # With boundary should have fewer labels
        assert len(labels_with_boundary) < len(labels_without)

        # All labels should be before boundary
        for lp in labels_with_boundary:
            assert lp.timestamp.date() <= boundary

    def test_labeled_period_structure(self, labeler_1d, trending_df):
        """Test that LabeledPeriod has all required fields."""
        labels = labeler_1d.label_period(trending_df)

        assert len(labels) > 0
        lp = labels[0]

        assert isinstance(lp, LabeledPeriod)
        assert isinstance(lp.bar_index, int)
        assert isinstance(lp.timestamp, pd.Timestamp)
        assert isinstance(lp.label, RegimeLabel)
        assert isinstance(lp.forward_return, float)
        assert isinstance(lp.forward_sharpe, float)
        assert isinstance(lp.forward_volatility, float)
        assert isinstance(lp.max_drawdown, float)

    def test_missing_close_column(self, labeler_1d):
        """Test that missing close column raises ValueError."""
        df = pd.DataFrame({"high": [1, 2, 3]})

        with pytest.raises(ValueError, match="must have 'close' column"):
            labeler_1d.label_period(df)


class TestLabelerThresholdsV1:
    """Tests for frozen thresholds constant."""

    def test_all_timeframes_present(self):
        """Test that all standard timeframes have thresholds."""
        expected = {"1d", "4h", "2h", "1h"}
        actual = {k for k in LABELER_THRESHOLDS_V1 if not k.startswith("_")}
        assert expected == actual

    def test_threshold_values_reasonable(self):
        """Test that threshold values are in reasonable ranges."""
        for tf, thresholds in LABELER_THRESHOLDS_V1.items():
            if tf.startswith("_"):
                continue

            # Forward return should be positive
            assert 0 < thresholds["trending_forward_return_min"] < 0.5

            # Sharpe should be positive
            assert 0 < thresholds["trending_sharpe_min"] < 5

            # Volatility should be positive
            assert 0 < thresholds["choppy_volatility_min"] < 1

            # Drawdown should be negative
            assert -0.5 < thresholds["choppy_drawdown_max"] < 0

    def test_meta_version(self):
        """Test that meta version is set."""
        meta = LABELER_THRESHOLDS_V1["_meta"]
        assert meta["version"] == 1.0
        assert meta["frozen_date"] == 20260120


class TestRegimeLabel:
    """Tests for RegimeLabel enum."""

    def test_all_labels(self):
        """Test that all expected labels exist."""
        assert RegimeLabel.TRENDING.value == "TRENDING"
        assert RegimeLabel.CHOPPY.value == "CHOPPY"
        assert RegimeLabel.NEUTRAL.value == "NEUTRAL"
