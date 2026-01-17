"""
Unit tests for StrategyDetector - Multi-leg strategy detection.
"""

import pytest

from src.domain.services.strategy_detector import StrategyDetector
from src.models.position import AssetType, Position


@pytest.fixture
def detector():
    """Create strategy detector."""
    return StrategyDetector()


def test_vertical_call_spread_detection(detector):
    """Test detection of vertical call spread."""
    positions = [
        # Long 100 call
        Position(
            symbol="TSLA 250131C00100000",
            underlying="TSLA",
            asset_type=AssetType.OPTION,
            quantity=10.0,
            avg_price=5.00,
            multiplier=100,
            expiry="20250131",
            strike=100.0,
            right="C",
        ),
        # Short 105 call (same expiry)
        Position(
            symbol="TSLA 250131C00105000",
            underlying="TSLA",
            asset_type=AssetType.OPTION,
            quantity=-10.0,
            avg_price=3.00,
            multiplier=100,
            expiry="20250131",
            strike=105.0,
            right="C",
        ),
    ]

    strategies = detector.detect(positions)

    assert len(strategies) == 1
    assert "VERTICAL" in strategies[0].strategy_type
    assert "CALL" in strategies[0].strategy_type
    assert strategies[0].underlying == "TSLA"
    assert len(strategies[0].positions) == 2


def test_covered_call_detection(detector):
    """Test detection of covered call strategy."""
    positions = [
        # Long stock
        Position(
            symbol="AAPL",
            underlying="AAPL",
            asset_type=AssetType.STOCK,
            quantity=1000.0,
            avg_price=180.00,
            multiplier=1,
        ),
        # Short call
        Position(
            symbol="AAPL 250207C00185000",
            underlying="AAPL",
            asset_type=AssetType.OPTION,
            quantity=-10.0,
            avg_price=2.50,
            multiplier=100,
            expiry="20250207",
            strike=185.0,
            right="C",
        ),
    ]

    strategies = detector.detect(positions)

    assert len(strategies) == 1
    assert strategies[0].strategy_type == "COVERED_CALL"
    assert strategies[0].underlying == "AAPL"
    assert strategies[0].is_credit is True


def test_diagonal_spread_detection(detector):
    """Test detection of diagonal spread."""
    positions = [
        # Long far-dated call
        Position(
            symbol="NVDA 250228C00500000",
            underlying="NVDA",
            asset_type=AssetType.OPTION,
            quantity=5.0,
            avg_price=15.00,
            multiplier=100,
            expiry="20250228",  # Feb expiry
            strike=500.0,
            right="C",
        ),
        # Short near-dated call (different expiry, different strike)
        Position(
            symbol="NVDA 250131C00520000",
            underlying="NVDA",
            asset_type=AssetType.OPTION,
            quantity=-5.0,
            avg_price=8.00,
            multiplier=100,
            expiry="20250131",  # Jan expiry
            strike=520.0,
            right="C",
        ),
    ]

    strategies = detector.detect(positions)

    assert len(strategies) == 1
    assert strategies[0].strategy_type == "DIAGONAL_SPREAD"
    assert strategies[0].underlying == "NVDA"


def test_calendar_spread_detection(detector):
    """Test detection of calendar spread (same strike, different expiry)."""
    positions = [
        # Long far-dated call
        Position(
            symbol="SPY 250331C00600000",
            underlying="SPY",
            asset_type=AssetType.OPTION,
            quantity=10.0,
            avg_price=10.00,
            multiplier=100,
            expiry="20250331",  # March
            strike=600.0,
            right="C",
        ),
        # Short near-dated call (same strike, different expiry)
        Position(
            symbol="SPY 250228C00600000",
            underlying="SPY",
            asset_type=AssetType.OPTION,
            quantity=-10.0,
            avg_price=6.00,
            multiplier=100,
            expiry="20250228",  # Feb
            strike=600.0,
            right="C",
        ),
    ]

    strategies = detector.detect(positions)

    assert len(strategies) == 1
    assert strategies[0].strategy_type == "CALENDAR_SPREAD"
    assert strategies[0].underlying == "SPY"
    assert strategies[0].metadata["is_calendar"] is True


def test_iron_condor_detection(detector):
    """Test detection of iron condor."""
    positions = [
        # Put spread (lower strikes)
        Position(
            symbol="SPY 250131P00580000",
            underlying="SPY",
            asset_type=AssetType.OPTION,
            quantity=10.0,
            avg_price=1.00,
            multiplier=100,
            expiry="20250131",
            strike=580.0,
            right="P",
        ),
        Position(
            symbol="SPY 250131P00590000",
            underlying="SPY",
            asset_type=AssetType.OPTION,
            quantity=-10.0,
            avg_price=2.00,
            multiplier=100,
            expiry="20250131",
            strike=590.0,
            right="P",
        ),
        # Call spread (higher strikes)
        Position(
            symbol="SPY 250131C00610000",
            underlying="SPY",
            asset_type=AssetType.OPTION,
            quantity=-10.0,
            avg_price=2.50,
            multiplier=100,
            expiry="20250131",
            strike=610.0,
            right="C",
        ),
        Position(
            symbol="SPY 250131C00620000",
            underlying="SPY",
            asset_type=AssetType.OPTION,
            quantity=10.0,
            avg_price=1.20,
            multiplier=100,
            expiry="20250131",
            strike=620.0,
            right="C",
        ),
    ]

    strategies = detector.detect(positions)

    # Should detect iron condor (may also detect individual spreads)
    iron_condors = [s for s in strategies if s.strategy_type == "IRON_CONDOR"]
    assert len(iron_condors) == 1
    assert iron_condors[0].underlying == "SPY"
    assert iron_condors[0].is_credit is True


def test_no_strategy_single_position(detector):
    """Test no strategy detected for single position."""
    positions = [
        Position(
            symbol="TSLA 250131C00300000",
            underlying="TSLA",
            asset_type=AssetType.OPTION,
            quantity=10.0,
            avg_price=5.00,
            multiplier=100,
            expiry="20250131",
            strike=300.0,
            right="C",
        ),
    ]

    strategies = detector.detect(positions)

    assert len(strategies) == 0


def test_multiple_underlyings(detector):
    """Test detection across multiple underlyings."""
    positions = [
        # TSLA vertical spread
        Position(
            symbol="TSLA 250131C00100000",
            underlying="TSLA",
            asset_type=AssetType.OPTION,
            quantity=10.0,
            avg_price=5.00,
            multiplier=100,
            expiry="20250131",
            strike=100.0,
            right="C",
        ),
        Position(
            symbol="TSLA 250131C00105000",
            underlying="TSLA",
            asset_type=AssetType.OPTION,
            quantity=-10.0,
            avg_price=3.00,
            multiplier=100,
            expiry="20250131",
            strike=105.0,
            right="C",
        ),
        # NVDA vertical spread
        Position(
            symbol="NVDA 250131C00500000",
            underlying="NVDA",
            asset_type=AssetType.OPTION,
            quantity=5.0,
            avg_price=10.00,
            multiplier=100,
            expiry="20250131",
            strike=500.0,
            right="C",
        ),
        Position(
            symbol="NVDA 250131C00520000",
            underlying="NVDA",
            asset_type=AssetType.OPTION,
            quantity=-5.0,
            avg_price=6.00,
            multiplier=100,
            expiry="20250131",
            strike=520.0,
            right="C",
        ),
    ]

    strategies = detector.detect(positions)

    assert len(strategies) == 2
    underlyings = {s.underlying for s in strategies}
    assert underlyings == {"TSLA", "NVDA"}
