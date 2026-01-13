"""Unit tests for P&L calculator."""

import pytest
from src.domain.services.risk.calculators.pnl_calculator import (
    calculate_pnl,
    calculate_pnl_delta,
    DataQuality,
    PnLResult,
)


class TestCalculatePnl:
    """Tests for calculate_pnl function."""

    def test_basic_long_position_profit(self):
        """Long position with price increase should show profit."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=153.0,
            quantity=100,
            multiplier=1,
        )

        assert result.unrealized == 500.0  # (155-150) * 100
        assert result.daily == 100.0  # (155-154) * 100
        assert result.intraday == 200.0  # (155-153) * 100
        assert result.is_reliable is True
        assert result.mark_price == 155.0

    def test_basic_long_position_loss(self):
        """Long position with price decrease should show loss."""
        result = calculate_pnl(
            mark=145.0,
            avg_cost=150.0,
            yesterday_close=148.0,
            session_open=147.0,
            quantity=100,
            multiplier=1,
        )

        assert result.unrealized == -500.0  # (145-150) * 100
        assert result.daily == -300.0  # (145-148) * 100
        assert result.intraday == -200.0  # (145-147) * 100
        assert result.is_reliable is True

    def test_short_position_profit(self):
        """Short position with price decrease should show profit."""
        result = calculate_pnl(
            mark=145.0,
            avg_cost=150.0,
            yesterday_close=148.0,
            session_open=147.0,
            quantity=-100,  # Short position
            multiplier=1,
        )

        assert result.unrealized == 500.0  # (145-150) * -100
        assert result.daily == 300.0  # (145-148) * -100
        assert result.intraday == 200.0  # (145-147) * -100

    def test_short_position_loss(self):
        """Short position with price increase should show loss."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=152.0,
            session_open=153.0,
            quantity=-100,  # Short position
            multiplier=1,
        )

        assert result.unrealized == -500.0  # (155-150) * -100
        assert result.daily == -300.0  # (155-152) * -100
        assert result.intraday == -200.0  # (155-153) * -100

    def test_option_with_multiplier(self):
        """Option position should scale by multiplier."""
        result = calculate_pnl(
            mark=5.50,
            avg_cost=5.00,
            yesterday_close=5.25,
            session_open=5.10,
            quantity=10,  # 10 contracts
            multiplier=100,  # Option multiplier
        )

        assert result.unrealized == 500.0  # (5.50-5.00) * 10 * 100 = 0.50 * 1000
        assert result.daily == 250.0  # (5.50-5.25) * 10 * 100 = 0.25 * 1000
        assert pytest.approx(result.intraday, rel=0.01) == 400.0  # (5.50-5.10) * 10 * 100 = 0.40 * 1000

    def test_missing_yesterday_close(self):
        """Daily P&L should be zero when yesterday_close is None."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=None,
            session_open=153.0,
            quantity=100,
            multiplier=1,
        )

        assert result.unrealized == 500.0
        assert result.daily == 0.0  # No yesterday_close
        assert result.intraday == 200.0

    def test_missing_session_open(self):
        """Intraday P&L should be zero when session_open is None."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=None,
            quantity=100,
            multiplier=1,
        )

        assert result.unrealized == 500.0
        assert result.daily == 100.0
        assert result.intraday == 0.0  # No session_open

    def test_zero_yesterday_close(self):
        """Daily P&L should be zero when yesterday_close is zero."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=0.0,
            session_open=153.0,
            quantity=100,
            multiplier=1,
        )

        assert result.daily == 0.0  # Zero yesterday_close treated as unavailable

    def test_zero_session_open(self):
        """Intraday P&L should be zero when session_open is zero."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=0.0,
            quantity=100,
            multiplier=1,
        )

        assert result.intraday == 0.0  # Zero session_open treated as unavailable


class TestDataQuality:
    """Tests for data quality handling."""

    def test_good_quality(self):
        """Good quality should be reliable."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=153.0,
            quantity=100,
            multiplier=1,
            data_quality=DataQuality.GOOD,
        )

        assert result.is_reliable is True

    def test_stale_quality(self):
        """Stale quality should not be reliable."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=153.0,
            quantity=100,
            multiplier=1,
            data_quality=DataQuality.STALE,
        )

        assert result.is_reliable is False

    def test_suspicious_quality(self):
        """Suspicious quality should not be reliable."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=153.0,
            quantity=100,
            multiplier=1,
            data_quality=DataQuality.SUSPICIOUS,
        )

        assert result.is_reliable is False

    def test_zero_quote_quality(self):
        """Zero quote quality should not be reliable."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=153.0,
            quantity=100,
            multiplier=1,
            data_quality=DataQuality.ZERO_QUOTE,
        )

        assert result.is_reliable is False


class TestCalculatePnlDelta:
    """Tests for calculate_pnl_delta function."""

    def test_delta_calculation(self):
        """Delta should be new - old for all fields."""
        old = PnLResult(
            unrealized=500.0,
            daily=100.0,
            intraday=200.0,
            is_reliable=True,
            mark_price=155.0,
        )

        new = PnLResult(
            unrealized=600.0,
            daily=150.0,
            intraday=250.0,
            is_reliable=True,
            mark_price=156.0,
        )

        delta = calculate_pnl_delta(old, new)

        assert delta.unrealized == 100.0  # 600 - 500
        assert delta.daily == 50.0  # 150 - 100
        assert delta.intraday == 50.0  # 250 - 200
        assert delta.mark_price == 156.0  # New mark price
        assert delta.is_reliable is True

    def test_delta_with_negative_change(self):
        """Delta should handle negative changes."""
        old = PnLResult(
            unrealized=500.0,
            daily=100.0,
            intraday=200.0,
            is_reliable=True,
            mark_price=155.0,
        )

        new = PnLResult(
            unrealized=400.0,
            daily=50.0,
            intraday=100.0,
            is_reliable=False,
            mark_price=154.0,
        )

        delta = calculate_pnl_delta(old, new)

        assert delta.unrealized == -100.0
        assert delta.daily == -50.0
        assert delta.intraday == -100.0
        assert delta.is_reliable is False  # Uses new reliability


class TestPnLResultImmutability:
    """Tests for PnLResult frozen dataclass."""

    def test_result_is_frozen(self):
        """PnLResult should be immutable."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=153.0,
            quantity=100,
            multiplier=1,
        )

        with pytest.raises(AttributeError):
            result.unrealized = 1000.0  # type: ignore

    def test_result_is_hashable(self):
        """Frozen dataclass should be hashable."""
        result = calculate_pnl(
            mark=155.0,
            avg_cost=150.0,
            yesterday_close=154.0,
            session_open=153.0,
            quantity=100,
            multiplier=1,
        )

        # Should not raise
        hash(result)
