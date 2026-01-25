"""Tests for ValidationTimeConfig and time unit conversion."""

import pytest

from src.domain.signals.validation.time_units import (
    BARS_PER_DAY,
    ValidationTimeConfig,
    get_bars_per_day,
    validate_time_config,
)


class TestValidationTimeConfig:
    """Tests for ValidationTimeConfig dataclass."""

    def test_from_days_1d_timeframe(self) -> None:
        """Test config creation for daily timeframe."""
        cfg = ValidationTimeConfig.from_days("1d", 20, 5, 3)

        assert cfg.timeframe == "1d"
        assert cfg.label_horizon_bars == 20  # 20 days * 1 bar/day
        assert cfg.purge_bars == 5
        assert cfg.embargo_bars == 3
        assert cfg.label_horizon_days == 20
        assert cfg.purge_days == 5
        assert cfg.embargo_days == 3

    def test_from_days_4h_timeframe(self) -> None:
        """Test config creation for 4h timeframe."""
        cfg = ValidationTimeConfig.from_days("4h", 20, 5, 3)

        # 4h has 1.625 bars per day
        assert cfg.timeframe == "4h"
        assert cfg.label_horizon_bars == 32  # 20 * 1.625 = 32.5 -> 32
        assert cfg.purge_bars == 8  # 5 * 1.625 = 8.125 -> 8
        assert cfg.embargo_bars == 4  # 3 * 1.625 = 4.875 -> 4

    def test_from_days_2h_timeframe(self) -> None:
        """Test config creation for 2h timeframe."""
        cfg = ValidationTimeConfig.from_days("2h", 20, 5, 3)

        # 2h has 3.25 bars per day
        assert cfg.timeframe == "2h"
        assert cfg.label_horizon_bars == 65  # 20 * 3.25 = 65
        assert cfg.purge_bars == 16  # 5 * 3.25 = 16.25 -> 16
        assert cfg.embargo_bars == 9  # 3 * 3.25 = 9.75 -> 9

    def test_from_days_1h_timeframe(self) -> None:
        """Test config creation for 1h timeframe."""
        cfg = ValidationTimeConfig.from_days("1h", 20, 5, 3)

        # 1h has 6.5 bars per day
        assert cfg.timeframe == "1h"
        assert cfg.label_horizon_bars == 130  # 20 * 6.5 = 130
        assert cfg.purge_bars == 32  # 5 * 6.5 = 32.5 -> 32
        assert cfg.embargo_bars == 19  # 3 * 6.5 = 19.5 -> 19

    def test_from_days_unknown_timeframe(self) -> None:
        """Test that unknown timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            ValidationTimeConfig.from_days("3h", 20, 5, 3)

    def test_horizon_bars_by_tf(self) -> None:
        """Test pre-computed horizon bars for multiple timeframes."""
        cfg = ValidationTimeConfig.from_days("1d", 20, 5, 3, timeframes=("1d", "4h", "2h", "1h"))

        assert cfg.horizon_bars_by_tf["1d"] == 20
        assert cfg.horizon_bars_by_tf["4h"] == 32
        assert cfg.horizon_bars_by_tf["2h"] == 65
        assert cfg.horizon_bars_by_tf["1h"] == 130

    def test_purge_bars_by_tf(self) -> None:
        """Test pre-computed purge bars for multiple timeframes."""
        cfg = ValidationTimeConfig.from_days("1d", 20, 5, 3, timeframes=("1d", "4h", "2h"))

        assert cfg.purge_bars_by_tf["1d"] == 5
        assert cfg.purge_bars_by_tf["4h"] == 8
        assert cfg.purge_bars_by_tf["2h"] == 16

    def test_config_is_frozen(self) -> None:
        """Test that config is immutable (frozen dataclass)."""
        cfg = ValidationTimeConfig.from_days("1d", 20, 5, 3)

        with pytest.raises(AttributeError):
            cfg.label_horizon_bars = 100  # type: ignore

    def test_to_dict(self) -> None:
        """Test serialization to dictionary."""
        cfg = ValidationTimeConfig.from_days("4h", 20, 5, 3)
        d = cfg.to_dict()

        assert d["timeframe"] == "4h"
        assert d["label_horizon_bars"] == 32
        assert d["purge_bars"] == 8
        assert d["embargo_bars"] == 4
        assert d["label_horizon_days"] == 20
        assert "horizon_bars_by_tf" in d


class TestValidateTimeConfig:
    """Tests for validate_time_config function."""

    def test_valid_config(self) -> None:
        """Test that valid config passes validation."""
        cfg = ValidationTimeConfig.from_days("1d", 20, 25, 3)  # purge > horizon
        validate_time_config(cfg)  # Should not raise

    def test_equal_purge_and_horizon(self) -> None:
        """Test that purge == horizon passes validation."""
        cfg = ValidationTimeConfig.from_days("1d", 20, 20, 3)
        validate_time_config(cfg)  # Should not raise

    def test_invalid_config_purge_less_than_horizon(self) -> None:
        """Test that purge < horizon fails validation."""
        cfg = ValidationTimeConfig.from_days("1d", 20, 15, 3)  # purge < horizon

        with pytest.raises(ValueError, match="purge_bars.*must be >= label_horizon_bars"):
            validate_time_config(cfg)

    def test_validates_all_timeframes(self) -> None:
        """Test that validation checks all timeframes."""
        # Create config where 1d passes but 4h fails
        # This requires custom construction since from_days uses same ratio
        cfg = ValidationTimeConfig(
            timeframe="1d",
            label_horizon_bars=20,
            purge_bars=25,  # Valid for 1d
            embargo_bars=3,
            horizon_bars_by_tf={"1d": 20, "4h": 32},
            purge_bars_by_tf={"1d": 25, "4h": 20},  # 4h: purge < horizon
            embargo_bars_by_tf={"1d": 3, "4h": 4},
            label_horizon_days=20,
            purge_days=5,
            embargo_days=3,
        )

        with pytest.raises(ValueError, match="For timeframe 4h"):
            validate_time_config(cfg)


class TestGetBarsPerDay:
    """Tests for get_bars_per_day function."""

    def test_all_known_timeframes(self) -> None:
        """Test bars per day for all known timeframes."""
        assert get_bars_per_day("1d") == 1.0
        assert get_bars_per_day("4h") == 1.625
        assert get_bars_per_day("2h") == 3.25
        assert get_bars_per_day("1h") == 6.5
        assert get_bars_per_day("30m") == 13.0
        assert get_bars_per_day("15m") == 26.0
        assert get_bars_per_day("5m") == 78.0
        assert get_bars_per_day("1m") == 390.0

    def test_unknown_timeframe(self) -> None:
        """Test that unknown timeframe raises ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            get_bars_per_day("3h")


class TestBarsPerDayConstant:
    """Tests for BARS_PER_DAY constant."""

    def test_all_standard_timeframes_present(self) -> None:
        """Test that all standard timeframes are defined."""
        expected = {"1d", "4h", "2h", "1h", "30m", "15m", "5m", "1m"}
        assert set(BARS_PER_DAY.keys()) == expected

    def test_values_are_reasonable(self) -> None:
        """Test that bar counts are reasonable."""
        for tf, bars in BARS_PER_DAY.items():
            assert bars > 0, f"Bars for {tf} should be positive"
            assert bars <= 400, f"Bars for {tf} seems too high"
