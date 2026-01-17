"""
Unit tests for PositionRiskAnalyzer - Position-level risk rules.
"""

from datetime import datetime

import pytest

from src.domain.services.position_risk_analyzer import PositionRiskAnalyzer
from src.models.market_data import MarketData
from src.models.position import AssetType, Position
from src.models.risk_signal import SignalSeverity, SuggestedAction


@pytest.fixture
def config():
    """Test configuration."""
    return {
        "risk_signals": {
            "position_rules": {
                "stop_loss_pct": 0.60,
                "take_profit_pct": 1.00,
                "trailing_stop_drawdown": 0.30,
                "dte_exit_ratio": 0.20,
                "short_r_multiple": 1.5,
            }
        }
    }


@pytest.fixture
def analyzer(config):
    """Create position risk analyzer."""
    return PositionRiskAnalyzer(config)


def test_stop_loss_triggered(analyzer):
    """Test stop loss detection at -60%."""
    # Create position with -70% loss
    position = Position(
        symbol="TSLA 250120C00300000",
        underlying="TSLA",
        asset_type=AssetType.OPTION,
        quantity=10.0,
        avg_price=5.00,  # Entry price
        multiplier=100,
        expiry="20250120",
        strike=300.0,
        right="C",
    )

    # Current price: $1.50 (-70% from entry)
    market_data = MarketData(
        symbol="TSLA 250120C00300000",
        timestamp=datetime.now(),
        bid=1.40,
        ask=1.60,
        last=1.50,
    )

    signals = analyzer.check(position, market_data)

    # Should have stop loss signal (may also have DTE signal due to expiry date)
    stop_loss_signals = [s for s in signals if s.trigger_rule == "Stop_Loss_Hit"]
    assert len(stop_loss_signals) == 1
    assert stop_loss_signals[0].severity == SignalSeverity.CRITICAL
    assert stop_loss_signals[0].suggested_action == SuggestedAction.CLOSE
    assert stop_loss_signals[0].symbol == "TSLA 250120C00300000"


def test_take_profit_triggered(analyzer):
    """Test take profit at +100%."""
    position = Position(
        symbol="NVDA 250131C00500000",
        underlying="NVDA",
        asset_type=AssetType.OPTION,
        quantity=5.0,
        avg_price=10.00,
        multiplier=100,
        expiry="20250131",
        strike=500.0,
        right="C",
    )

    # Current price: $25 (+150% from entry)
    market_data = MarketData(
        symbol="NVDA 250131C00500000",
        timestamp=datetime.now(),
        bid=24.50,
        ask=25.50,
        last=25.00,
    )

    signals = analyzer.check(position, market_data)

    # Should have take profit signal (may also have DTE signal)
    take_profit_signals = [s for s in signals if s.trigger_rule == "Take_Profit_Hit"]
    assert len(take_profit_signals) == 1
    assert take_profit_signals[0].severity == SignalSeverity.WARNING
    assert take_profit_signals[0].suggested_action == SuggestedAction.REDUCE


def test_trailing_stop_from_peak(analyzer):
    """Test trailing stop after 30% drawdown from peak."""
    position = Position(
        symbol="AAPL 250207C00180000",
        underlying="AAPL",
        asset_type=AssetType.OPTION,
        quantity=20.0,
        avg_price=5.00,
        multiplier=100,
        expiry="20250207",
        strike=180.0,
        right="C",
        max_profit_reached=1.00,  # Peak was +100%
    )

    # Current price: $8.50 (+70% from entry, but -30% from peak)
    market_data = MarketData(
        symbol="AAPL 250207C00180000",
        timestamp=datetime.now(),
        bid=8.40,
        ask=8.60,
        last=8.50,
    )

    signals = analyzer.check(position, market_data)

    assert len(signals) >= 1
    trailing_signal = next((s for s in signals if s.trigger_rule == "Trailing_Stop_Hit"), None)
    assert trailing_signal is not None
    assert trailing_signal.severity == SignalSeverity.WARNING
    assert trailing_signal.suggested_action == SuggestedAction.CLOSE


def test_low_dte_long_option(analyzer):
    """Test DTE warning for long options."""
    position = Position(
        symbol="SPY 250128C00600000",
        underlying="SPY",
        asset_type=AssetType.OPTION,
        quantity=10.0,
        avg_price=2.50,
        multiplier=100,
        expiry="20250128",  # 1 day to expiry
        strike=600.0,
        right="C",
    )

    market_data = MarketData(
        symbol="SPY 250128C00600000",
        timestamp=datetime.now(),
        bid=3.00,
        ask=3.20,
        last=3.10,
    )

    signals = analyzer.check(position, market_data)

    # Should have low DTE signal
    dte_signals = [s for s in signals if "Low_DTE" in s.trigger_rule]
    assert len(dte_signals) >= 1
    assert dte_signals[0].severity == SignalSeverity.INFO
    assert dte_signals[0].suggested_action == SuggestedAction.ROLL


def test_low_dte_short_option(analyzer):
    """Test DTE warning for short options (assignment risk)."""
    position = Position(
        symbol="TSLA 250128P00250000",
        underlying="TSLA",
        asset_type=AssetType.OPTION,
        quantity=-5.0,  # Short put
        avg_price=3.00,
        multiplier=100,
        expiry="20250128",  # 1 day to expiry
        strike=250.0,
        right="P",
    )

    market_data = MarketData(
        symbol="TSLA 250128P00250000",
        timestamp=datetime.now(),
        bid=2.00,
        ask=2.20,
        last=2.10,
    )

    signals = analyzer.check(position, market_data)

    # Should have assignment risk signal
    dte_signals = [s for s in signals if "Low_DTE" in s.trigger_rule]
    assert len(dte_signals) >= 1
    assert dte_signals[0].severity == SignalSeverity.WARNING
    assert dte_signals[0].suggested_action == SuggestedAction.CLOSE


def test_no_signals_healthy_position(analyzer):
    """Test no signals for healthy position."""
    position = Position(
        symbol="META 250214C00400000",
        underlying="META",
        asset_type=AssetType.OPTION,
        quantity=10.0,
        avg_price=10.00,
        multiplier=100,
        expiry="20250214",  # 18 days to expiry
        strike=400.0,
        right="C",
    )

    # Current price: $12 (+20% - healthy)
    market_data = MarketData(
        symbol="META 250214C00400000",
        timestamp=datetime.now(),
        bid=11.80,
        ask=12.20,
        last=12.00,
    )

    signals = analyzer.check(position, market_data)

    # Should have no critical signals
    critical_signals = [s for s in signals if s.severity == SignalSeverity.CRITICAL]
    assert len(critical_signals) == 0


def test_update_max_profit(analyzer):
    """Test max profit watermark update."""
    position = Position(
        symbol="TEST",
        underlying="TEST",
        asset_type=AssetType.OPTION,
        quantity=10.0,
        avg_price=5.00,
        multiplier=100,
    )

    # Update to +50%
    updated = analyzer.update_max_profit(position, 0.50)
    assert updated is True
    assert position.max_profit_reached == 0.50

    # Update to +70% (higher)
    updated = analyzer.update_max_profit(position, 0.70)
    assert updated is True
    assert position.max_profit_reached == 0.70

    # Update to +60% (lower - should not update)
    updated = analyzer.update_max_profit(position, 0.60)
    assert updated is False
    assert position.max_profit_reached == 0.70
