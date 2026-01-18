"""Unit tests for Greeks calculator."""

import pytest

from src.domain.services.risk.calculators.greeks_calculator import (
    calculate_near_term_greeks,
    calculate_position_greeks,
)
from src.models.position import AssetType


class TestCalculatePositionGreeks:
    """Tests for calculate_position_greeks function."""

    def test_stock_position_delta(self):
        """Stock should have synthetic delta = quantity * multiplier."""
        result = calculate_position_greeks(
            raw_delta=None,  # Ignored for stocks
            raw_gamma=None,
            raw_vega=None,
            raw_theta=None,
            quantity=100,
            multiplier=1,
            asset_type=AssetType.STOCK,
        )

        assert result.delta == 100.0  # 100 shares = 100 delta
        assert result.gamma == 0.0
        assert result.vega == 0.0
        assert result.theta == 0.0
        assert result.has_greeks is False  # Synthetic, not market

    def test_stock_short_position(self):
        """Short stock should have negative delta."""
        result = calculate_position_greeks(
            raw_delta=None,
            raw_gamma=None,
            raw_vega=None,
            raw_theta=None,
            quantity=-100,  # Short position
            multiplier=1,
            asset_type=AssetType.STOCK,
        )

        assert result.delta == -100.0  # Short 100 shares = -100 delta

    def test_option_long_call(self):
        """Long call option should scale Greeks by quantity * multiplier."""
        result = calculate_position_greeks(
            raw_delta=0.5,
            raw_gamma=0.02,
            raw_vega=0.25,
            raw_theta=-0.15,
            quantity=10,  # 10 contracts
            multiplier=100,  # Option multiplier
            asset_type=AssetType.OPTION,
        )

        assert result.delta == 500.0  # 0.5 * 10 * 100
        assert result.gamma == 20.0  # 0.02 * 10 * 100
        assert result.vega == 250.0  # 0.25 * 10 * 100
        assert result.theta == -150.0  # -0.15 * 10 * 100
        assert result.has_greeks is True

    def test_option_short_put(self):
        """Short put option with negative quantity."""
        result = calculate_position_greeks(
            raw_delta=-0.3,  # Put delta
            raw_gamma=0.01,
            raw_vega=0.20,
            raw_theta=-0.10,
            quantity=-5,  # Short 5 contracts
            multiplier=100,
            asset_type=AssetType.OPTION,
        )

        # Negative quantity flips signs
        assert result.delta == 150.0  # -0.3 * -5 * 100
        assert result.gamma == -5.0  # 0.01 * -5 * 100
        assert result.vega == -100.0  # 0.20 * -5 * 100
        assert result.theta == 50.0  # -0.10 * -5 * 100

    def test_option_missing_some_greeks(self):
        """Option with some missing Greeks should use zero for missing."""
        result = calculate_position_greeks(
            raw_delta=0.5,
            raw_gamma=None,  # Missing
            raw_vega=0.25,
            raw_theta=None,  # Missing
            quantity=10,
            multiplier=100,
            asset_type=AssetType.OPTION,
        )

        assert result.delta == 500.0
        assert result.gamma == 0.0  # Missing → zero
        assert result.vega == 250.0
        assert result.theta == 0.0  # Missing → zero
        assert result.has_greeks is True  # Some Greeks were available

    def test_option_all_greeks_missing(self):
        """Option with all missing Greeks should have has_greeks=False."""
        result = calculate_position_greeks(
            raw_delta=None,
            raw_gamma=None,
            raw_vega=None,
            raw_theta=None,
            quantity=10,
            multiplier=100,
            asset_type=AssetType.OPTION,
        )

        assert result.delta == 0.0
        assert result.gamma == 0.0
        assert result.vega == 0.0
        assert result.theta == 0.0
        assert result.has_greeks is False  # All missing

    def test_future_asset_type(self):
        """Future should use market Greeks like options."""
        result = calculate_position_greeks(
            raw_delta=1.0,  # Future delta
            raw_gamma=0.0,
            raw_vega=0.0,
            raw_theta=0.0,
            quantity=5,
            multiplier=50,  # Future multiplier
            asset_type=AssetType.FUTURE,
        )

        assert result.delta == 250.0  # 1.0 * 5 * 50
        assert result.has_greeks is True


class TestCalculateNearTermGreeks:
    """Tests for near-term Greeks concentration metrics."""

    def test_short_dated_gamma_concentration(self):
        """0-7 DTE should show gamma notional."""
        gamma_notional, vega_notional = calculate_near_term_greeks(
            gamma=0.05,
            vega=0.30,
            mark_price=100.0,
            quantity=10,
            multiplier=100,
            days_to_expiry=3,  # Within 7 DTE
        )

        # Gamma notional: |0.05 * 100^2 * 0.01 * 10 * 100|
        # = |0.05 * 10000 * 0.01 * 1000| = 5000
        assert gamma_notional == pytest.approx(5000.0, rel=0.01)
        # Vega notional: |0.30 * 10 * 100| = 300
        assert vega_notional == pytest.approx(300.0, rel=0.01)

    def test_medium_dated_vega_only(self):
        """8-30 DTE should only show vega notional."""
        gamma_notional, vega_notional = calculate_near_term_greeks(
            gamma=0.05,
            vega=0.30,
            mark_price=100.0,
            quantity=10,
            multiplier=100,
            days_to_expiry=15,  # Outside 7 DTE, within 30 DTE
        )

        assert gamma_notional == 0.0  # Outside 7 DTE threshold
        assert vega_notional == pytest.approx(300.0, rel=0.01)

    def test_long_dated_no_concentration(self):
        """31+ DTE should have no concentration metrics."""
        gamma_notional, vega_notional = calculate_near_term_greeks(
            gamma=0.05,
            vega=0.30,
            mark_price=100.0,
            quantity=10,
            multiplier=100,
            days_to_expiry=45,  # Outside both thresholds
        )

        assert gamma_notional == 0.0
        assert vega_notional == 0.0

    def test_no_expiry(self):
        """Stock/no expiry should have no concentration metrics."""
        gamma_notional, vega_notional = calculate_near_term_greeks(
            gamma=0.0,
            vega=0.0,
            mark_price=100.0,
            quantity=100,
            multiplier=1,
            days_to_expiry=None,  # No expiry
        )

        assert gamma_notional == 0.0
        assert vega_notional == 0.0

    def test_zero_dte(self):
        """0DTE should show both gamma and vega notional."""
        gamma_notional, vega_notional = calculate_near_term_greeks(
            gamma=0.10,  # High gamma on 0DTE
            vega=0.10,  # Low vega on 0DTE
            mark_price=150.0,
            quantity=5,
            multiplier=100,
            days_to_expiry=0,
        )

        # High gamma concentration on 0DTE
        # |0.10 * 150^2 * 0.01 * 5 * 100|
        # = |0.10 * 22500 * 0.01 * 500| = 11250
        assert gamma_notional == pytest.approx(11250.0, rel=0.01)
        # |0.10 * 5 * 100| = 50
        assert vega_notional == pytest.approx(50.0, rel=0.01)

    def test_custom_thresholds(self):
        """Custom DTE thresholds should work."""
        gamma_notional, vega_notional = calculate_near_term_greeks(
            gamma=0.05,
            vega=0.30,
            mark_price=100.0,
            quantity=10,
            multiplier=100,
            days_to_expiry=10,
            near_term_gamma_dte=14,  # Extend gamma threshold
            near_term_vega_dte=45,  # Extend vega threshold
        )

        # 10 DTE now within extended gamma threshold
        assert gamma_notional > 0.0
        assert vega_notional > 0.0


class TestGreeksResultImmutability:
    """Tests for GreeksResult frozen dataclass."""

    def test_result_is_frozen(self):
        """GreeksResult should be immutable."""
        result = calculate_position_greeks(
            raw_delta=0.5,
            raw_gamma=0.02,
            raw_vega=0.25,
            raw_theta=-0.15,
            quantity=10,
            multiplier=100,
            asset_type=AssetType.OPTION,
        )

        with pytest.raises(AttributeError):
            result.delta = 1000.0  # type: ignore

    def test_result_is_hashable(self):
        """Frozen dataclass should be hashable."""
        result = calculate_position_greeks(
            raw_delta=0.5,
            raw_gamma=0.02,
            raw_vega=0.25,
            raw_theta=-0.15,
            quantity=10,
            multiplier=100,
            asset_type=AssetType.OPTION,
        )

        # Should not raise
        hash(result)
