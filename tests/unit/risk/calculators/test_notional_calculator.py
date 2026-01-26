"""Unit tests for notional calculator."""

import pytest

from src.domain.services.risk.calculators.notional_calculator import (
    calculate_concentration,
    calculate_delta_dollars,
    calculate_notional,
)


class TestCalculateNotional:
    """Tests for calculate_notional function."""

    def test_long_stock_position(self) -> None:
        """Long stock should have positive notional."""
        result = calculate_notional(
            mark_price=155.0,
            quantity=100,
            multiplier=1,
        )

        assert result.notional == 15500.0  # 155 * 100 * 1
        assert result.gross_notional == 15500.0

    def test_short_stock_position(self) -> None:
        """Short stock should have negative notional, positive gross."""
        result = calculate_notional(
            mark_price=155.0,
            quantity=-100,  # Short
            multiplier=1,
        )

        assert result.notional == -15500.0  # 155 * -100 * 1
        assert result.gross_notional == 15500.0  # Absolute value

    def test_long_option_position(self) -> None:
        """Option should scale by multiplier."""
        result = calculate_notional(
            mark_price=5.50,
            quantity=10,  # 10 contracts
            multiplier=100,  # Option multiplier
        )

        assert result.notional == 5500.0  # 5.50 * 10 * 100
        assert result.gross_notional == 5500.0

    def test_short_option_position(self) -> None:
        """Short option should have negative notional."""
        result = calculate_notional(
            mark_price=5.50,
            quantity=-10,  # Short 10 contracts
            multiplier=100,
        )

        assert result.notional == -5500.0
        assert result.gross_notional == 5500.0

    def test_zero_quantity(self) -> None:
        """Zero quantity should give zero notional."""
        result = calculate_notional(
            mark_price=155.0,
            quantity=0,
            multiplier=1,
        )

        assert result.notional == 0.0
        assert result.gross_notional == 0.0

    def test_fractional_quantity(self) -> None:
        """Fractional quantity should work correctly."""
        result = calculate_notional(
            mark_price=100.0,
            quantity=0.5,
            multiplier=1,
        )

        assert result.notional == 50.0
        assert result.gross_notional == 50.0


class TestCalculateDeltaDollars:
    """Tests for calculate_delta_dollars function."""

    def test_basic_delta_dollars(self) -> None:
        """Basic delta dollars calculation."""
        delta_dollars, beta_adj = calculate_delta_dollars(
            delta=0.5,
            underlying_price=175.0,
            quantity=10,
            multiplier=100,
        )

        # Delta dollars: 0.5 * 175 * 10 * 100 = 87500
        assert delta_dollars == 87500.0
        # Beta adjusted: 0.5 * 10 * 100 * 1.0 (default beta) = 500
        assert beta_adj == 500.0

    def test_with_beta(self) -> None:
        """Delta dollars with custom beta."""
        delta_dollars, beta_adj = calculate_delta_dollars(
            delta=0.5,
            underlying_price=175.0,
            quantity=10,
            multiplier=100,
            beta=1.2,  # Higher beta
        )

        assert delta_dollars == 87500.0  # Unchanged
        # Beta adjusted: 0.5 * 10 * 100 * 1.2 = 600
        assert beta_adj == 600.0

    def test_negative_delta(self) -> None:
        """Negative delta (put option) should give negative delta dollars."""
        delta_dollars, beta_adj = calculate_delta_dollars(
            delta=-0.3,  # Put delta
            underlying_price=100.0,
            quantity=5,
            multiplier=100,
        )

        # Delta dollars: -0.3 * 100 * 5 * 100 = -15000
        assert delta_dollars == -15000.0
        assert beta_adj == -150.0

    def test_short_position(self) -> None:
        """Short position should flip delta direction."""
        delta_dollars, beta_adj = calculate_delta_dollars(
            delta=0.5,
            underlying_price=100.0,
            quantity=-10,  # Short
            multiplier=100,
        )

        # Delta dollars: 0.5 * 100 * -10 * 100 = -50000
        assert delta_dollars == -50000.0
        assert beta_adj == -500.0

    def test_zero_beta(self) -> None:
        """Zero beta should work."""
        delta_dollars, beta_adj = calculate_delta_dollars(
            delta=0.5,
            underlying_price=100.0,
            quantity=10,
            multiplier=100,
            beta=0.0,
        )

        assert delta_dollars == 50000.0
        assert beta_adj == 0.0

    def test_none_beta_uses_default(self) -> None:
        """None beta should default to 1.0."""
        delta_dollars, beta_adj = calculate_delta_dollars(
            delta=1.0,
            underlying_price=100.0,
            quantity=10,
            multiplier=1,
            beta=None,
        )

        assert delta_dollars == 1000.0
        assert beta_adj == 10.0  # Uses default beta of 1.0


class TestCalculateConcentration:
    """Tests for calculate_concentration function."""

    def test_basic_concentration(self) -> None:
        """Basic concentration calculation."""
        symbol, notional, pct = calculate_concentration(
            notional_by_underlying={"AAPL": 25000, "SPY": -15000, "TSLA": 10000},
            total_gross_notional=50000,
        )

        assert symbol == "AAPL"
        assert notional == 25000.0
        assert pct == 0.5  # 25000 / 50000

    def test_short_dominant_position(self) -> None:
        """Short position can be most concentrated by absolute value."""
        symbol, notional, pct = calculate_concentration(
            notional_by_underlying={"AAPL": 10000, "SPY": -30000, "TSLA": 10000},
            total_gross_notional=50000,
        )

        assert symbol == "SPY"
        assert notional == 30000.0  # Absolute value
        assert pct == 0.6  # 30000 / 50000

    def test_single_position(self) -> None:
        """Single position should be 100% concentration."""
        symbol, notional, pct = calculate_concentration(
            notional_by_underlying={"AAPL": 50000},
            total_gross_notional=50000,
        )

        assert symbol == "AAPL"
        assert notional == 50000.0
        assert pct == 1.0

    def test_empty_portfolio(self) -> None:
        """Empty portfolio should return empty results."""
        symbol, notional, pct = calculate_concentration(
            notional_by_underlying={},
            total_gross_notional=0,
        )

        assert symbol == ""
        assert notional == 0.0
        assert pct == 0.0

    def test_zero_gross_notional(self) -> None:
        """Zero gross notional should handle division by zero."""
        symbol, notional, pct = calculate_concentration(
            notional_by_underlying={"AAPL": 0},
            total_gross_notional=0,
        )

        assert symbol == ""
        assert notional == 0.0
        assert pct == 0.0

    def test_equal_positions(self) -> None:
        """Equal positions should pick one (deterministic)."""
        symbol, notional, pct = calculate_concentration(
            notional_by_underlying={"AAPL": 25000, "TSLA": 25000},
            total_gross_notional=50000,
        )

        # Should pick one consistently
        assert symbol in ("AAPL", "TSLA")
        assert notional == 25000.0
        assert pct == 0.5


class TestNotionalResultImmutability:
    """Tests for NotionalResult frozen dataclass."""

    def test_result_is_frozen(self) -> None:
        """NotionalResult should be immutable."""
        result = calculate_notional(
            mark_price=155.0,
            quantity=100,
            multiplier=1,
        )

        with pytest.raises(AttributeError):
            result.notional = 1000.0  # type: ignore

    def test_result_is_hashable(self) -> None:
        """Frozen dataclass should be hashable."""
        result = calculate_notional(
            mark_price=155.0,
            quantity=100,
            multiplier=1,
        )

        # Should not raise
        hash(result)
