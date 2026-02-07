"""
Unit tests for PositionSizer.

Covers: ATR sizing formula, confidence scaling, regime factor,
min/max constraints, stop-based sizing, and edge cases.
"""

from __future__ import annotations

import pytest

from src.domain.strategy.position_sizer import PositionSizer, SizingResult


def _default_sizer(**overrides: object) -> PositionSizer:
    defaults = {
        "portfolio_value": 100_000.0,
        "risk_per_trade_pct": 0.02,
        "max_position_pct": 0.10,
        "min_shares": 1,
    }
    defaults.update(overrides)
    return PositionSizer(**defaults)  # type: ignore[arg-type]


class TestSizingResult:
    def test_dataclass_fields(self) -> None:
        r = SizingResult(
            shares=100,
            dollar_risk=200.0,
            notional=10000.0,
            risk_pct_of_portfolio=0.02,
            stop_distance=2.0,
        )
        assert r.shares == 100
        assert r.dollar_risk == 200.0
        assert r.notional == 10000.0
        assert r.risk_pct_of_portfolio == 0.02
        assert r.is_valid is True

    def test_zero_shares_is_invalid(self) -> None:
        r = SizingResult(
            shares=0,
            dollar_risk=0.0,
            notional=0.0,
            risk_pct_of_portfolio=0.0,
            stop_distance=0.0,
        )
        assert r.is_valid is False


class TestBasicSizing:
    def test_standard_sizing_uncapped(self) -> None:
        """With high max_position_pct, raw sizing formula applies.
        Risk = 100000 * 0.02 / (2.0 * 3.0) = 333 shares.
        """
        sizer = _default_sizer(
            portfolio_value=100_000.0,
            risk_per_trade_pct=0.02,
            max_position_pct=1.0,  # No cap
        )
        result = sizer.calculate(
            symbol="AAPL",
            price=150.0,
            atr=2.0,
            stop_distance_atr_mult=3.0,
        )
        assert result.shares == 333
        assert result.dollar_risk == pytest.approx(333 * 6.0, rel=0.01)
        assert result.notional == pytest.approx(333 * 150.0)

    def test_max_position_pct_limits_shares(self) -> None:
        """With 10% cap: max_notional = 10000, shares = 10000/150 = 66."""
        sizer = _default_sizer(
            portfolio_value=100_000.0,
            risk_per_trade_pct=0.02,
            max_position_pct=0.10,
        )
        result = sizer.calculate(
            symbol="AAPL",
            price=150.0,
            atr=2.0,
            stop_distance_atr_mult=3.0,
        )
        assert result.shares == 66
        assert result.notional <= 100_000.0 * 0.10 + 150.0

    def test_small_atr_gives_more_shares(self) -> None:
        sizer = _default_sizer(max_position_pct=1.0)
        result_small = sizer.calculate(
            symbol="AAPL",
            price=100.0,
            atr=1.0,
            stop_distance_atr_mult=2.0,
        )
        result_large = sizer.calculate(
            symbol="AAPL",
            price=100.0,
            atr=5.0,
            stop_distance_atr_mult=2.0,
        )
        assert result_small.shares > result_large.shares


class TestConfidenceScaling:
    def test_confidence_reduces_size(self) -> None:
        sizer = _default_sizer(max_position_pct=1.0)  # No cap to test pure scaling
        full = sizer.calculate(
            symbol="AAPL",
            price=100.0,
            atr=2.0,
            stop_distance_atr_mult=3.0,
            confidence=1.0,
        )
        half = sizer.calculate(
            symbol="AAPL",
            price=100.0,
            atr=2.0,
            stop_distance_atr_mult=3.0,
            confidence=0.5,
        )
        assert half.shares < full.shares
        # Should be approximately half (floor rounding may differ by 1)
        assert half.shares == pytest.approx(full.shares * 0.5, abs=1)


class TestRegimeSizeFactor:
    def test_regime_factor_reduces_size(self) -> None:
        sizer = _default_sizer(max_position_pct=1.0)
        full = sizer.calculate(
            symbol="AAPL",
            price=100.0,
            atr=2.0,
            stop_distance_atr_mult=3.0,
            regime_size_factor=1.0,
        )
        reduced = sizer.calculate(
            symbol="AAPL",
            price=100.0,
            atr=2.0,
            stop_distance_atr_mult=3.0,
            regime_size_factor=0.3,
        )
        assert reduced.shares < full.shares


class TestConstraints:
    def test_max_position_pct_cap(self) -> None:
        """Even with tiny ATR, position should not exceed max_position_pct."""
        sizer = _default_sizer(max_position_pct=0.10)
        result = sizer.calculate(
            symbol="AAPL",
            price=10.0,
            atr=0.01,
            stop_distance_atr_mult=1.0,
        )
        max_notional = 100_000.0 * 0.10
        assert result.notional <= max_notional + 10.0  # Small tolerance for rounding

    def test_min_shares_floor(self) -> None:
        """Even with large ATR, should get at least min_shares."""
        sizer = _default_sizer(min_shares=1)
        result = sizer.calculate(
            symbol="AAPL",
            price=100.0,
            atr=100.0,
            stop_distance_atr_mult=10.0,
        )
        assert result.shares >= 1

    def test_zero_atr_returns_zero_shares(self) -> None:
        """Zero ATR is invalid input, returns 0 shares."""
        sizer = _default_sizer(min_shares=1)
        result = sizer.calculate(
            symbol="AAPL",
            price=100.0,
            atr=0.0,
            stop_distance_atr_mult=3.0,
        )
        assert result.shares == 0
        assert result.is_valid is False


class TestCalculateFromStop:
    def test_stop_based_sizing(self) -> None:
        sizer = _default_sizer(
            portfolio_value=100_000.0,
            risk_per_trade_pct=0.02,
            max_position_pct=1.0,  # No cap
        )
        result = sizer.calculate_from_stop_price(
            symbol="AAPL",
            entry_price=100.0,
            stop_price=95.0,
        )
        # Risk per share = 100 - 95 = 5
        # Shares = (100000 * 0.02) / 5 = 400
        assert result.shares == 400
        assert result.dollar_risk == pytest.approx(2000.0)

    def test_stop_equal_to_entry_returns_zero(self) -> None:
        sizer = _default_sizer(min_shares=1)
        result = sizer.calculate_from_stop_price(
            symbol="AAPL",
            entry_price=100.0,
            stop_price=100.0,
        )
        assert result.shares == 0


class TestPortfolioValueUpdate:
    def test_portfolio_value_property(self) -> None:
        sizer = _default_sizer(portfolio_value=50_000.0)
        assert sizer.portfolio_value == 50_000.0
        sizer.portfolio_value = 200_000.0
        assert sizer.portfolio_value == 200_000.0

    def test_larger_portfolio_gives_more_shares(self) -> None:
        sizer_small = _default_sizer(portfolio_value=50_000.0)
        sizer_large = _default_sizer(portfolio_value=200_000.0)
        small = sizer_small.calculate(
            symbol="AAPL",
            price=100.0,
            atr=2.0,
            stop_distance_atr_mult=3.0,
        )
        large = sizer_large.calculate(
            symbol="AAPL",
            price=100.0,
            atr=2.0,
            stop_distance_atr_mult=3.0,
        )
        assert large.shares > small.shares
