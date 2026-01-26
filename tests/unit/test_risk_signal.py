"""
Unit tests for RiskSignal model.
"""

from datetime import datetime

from src.models.risk_signal import (
    RiskSignal,
    SignalLevel,
    SignalSeverity,
    SuggestedAction,
)


class TestRiskSignal:
    """Test RiskSignal model."""

    def test_create_basic_signal(self) -> None:
        """Test creating a basic risk signal."""
        signal = RiskSignal(
            signal_id="PORTFOLIO:Delta_Breach:CRITICAL",
            timestamp=datetime.now(),
            level=SignalLevel.PORTFOLIO,
            severity=SignalSeverity.CRITICAL,
            trigger_rule="Delta_Breach",
            current_value=60000,
            threshold=50000,
            breach_pct=20.0,
            suggested_action=SuggestedAction.HEDGE,
            action_details="Portfolio delta exceeded limit by 20%",
        )

        assert signal.signal_id == "PORTFOLIO:Delta_Breach:CRITICAL"
        assert signal.level == SignalLevel.PORTFOLIO
        assert signal.severity == SignalSeverity.CRITICAL
        assert signal.current_value == 60000
        assert signal.threshold == 50000
        assert signal.breach_pct == 20.0
        assert signal.suggested_action == SuggestedAction.HEDGE

    def test_signal_with_symbol(self) -> None:
        """Test signal with symbol context."""
        signal = RiskSignal(
            signal_id="POSITION:TSLA:Stop_Loss",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            symbol="TSLA",
            trigger_rule="Stop_Loss_Hit",
            current_value=-62.5,
            threshold=-60.0,
            suggested_action=SuggestedAction.CLOSE,
            action_details="Long call hit -62.5% stop loss",
        )

        assert signal.symbol == "TSLA"
        assert signal.level == SignalLevel.POSITION
        assert signal.suggested_action == SuggestedAction.CLOSE

    def test_signal_with_strategy(self) -> None:
        """Test signal with strategy context."""
        signal = RiskSignal(
            signal_id="STRATEGY:NVDA:Diagonal_Delta_Flip",
            timestamp=datetime.now(),
            level=SignalLevel.STRATEGY,
            severity=SignalSeverity.CRITICAL,
            symbol="NVDA",
            strategy_type="DIAGONAL",
            trigger_rule="Delta_Flip",
            suggested_action=SuggestedAction.CLOSE,
            action_details="Short leg delta exceeded long leg delta",
            layer=2,
        )

        assert signal.strategy_type == "DIAGONAL"
        assert signal.level == SignalLevel.STRATEGY
        assert signal.layer == 2

    def test_to_dict_serialization(self) -> None:
        """Test serialization to dictionary."""
        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime(2025, 1, 1, 12, 0, 0),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.INFO,
            symbol="SPY",
            trigger_rule="Test_Rule",
            current_value=100.0,
            threshold=90.0,
            breach_pct=11.1,
            suggested_action=SuggestedAction.MONITOR,
            action_details="Test signal",
            layer=1,
        )

        data = signal.to_dict()

        assert data["signal_id"] == "TEST:Signal"
        assert data["level"] == "POSITION"
        assert data["severity"] == "INFO"
        assert data["symbol"] == "SPY"
        assert data["trigger_rule"] == "Test_Rule"
        assert data["current_value"] == 100.0
        assert data["threshold"] == 90.0
        assert data["breach_pct"] == 11.1
        assert data["suggested_action"] == "MONITOR"
        assert data["action_details"] == "Test signal"
        assert data["layer"] == 1

    def test_from_breach_conversion(self) -> None:
        """Test conversion from legacy LimitBreach."""
        from src.domain.services.risk.rule_engine import BreachSeverity, LimitBreach

        breach = LimitBreach(
            limit_name="Portfolio Delta",
            limit_value=50000,
            current_value=60000,
            severity=BreachSeverity.HARD,
        )

        signal = RiskSignal.from_breach(breach, layer=1)

        assert signal.severity == SignalSeverity.CRITICAL
        assert signal.level == SignalLevel.PORTFOLIO
        assert signal.trigger_rule == "Portfolio Delta"
        assert signal.current_value == 60000
        assert signal.threshold == 50000
        assert signal.suggested_action == SuggestedAction.HALT_NEW_TRADES
        assert signal.layer == 1

    def test_from_breach_soft(self) -> None:
        """Test conversion from soft breach."""
        from src.domain.services.risk.rule_engine import BreachSeverity, LimitBreach

        breach = LimitBreach(
            limit_name="Vega Limit",
            limit_value=10000,
            current_value=8500,
            severity=BreachSeverity.SOFT,
        )

        signal = RiskSignal.from_breach(breach)

        assert signal.severity == SignalSeverity.WARNING
        assert signal.suggested_action == SuggestedAction.MONITOR

    def test_signal_str_representation(self) -> None:
        """Test string representation."""
        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            symbol="TSLA",
            trigger_rule="Stop_Loss",
            current_value=-55.0,
            threshold=-50.0,
            suggested_action=SuggestedAction.CLOSE,
        )

        str_repr = str(signal)
        assert "[WARNING]" in str_repr
        assert "Level=POSITION" in str_repr
        assert "Symbol=TSLA" in str_repr
        assert "Rule=Stop_Loss" in str_repr
        assert "Action=CLOSE" in str_repr

    def test_signal_repr(self) -> None:
        """Test debug representation."""
        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.CRITICAL,
            trigger_rule="Test",
        )

        repr_str = repr(signal)
        assert "RiskSignal" in repr_str
        assert "TEST:Signal" in repr_str
        assert "CRITICAL" in repr_str

    def test_signal_with_metadata(self) -> None:
        """Test signal with additional metadata."""
        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.INFO,
            trigger_rule="Test",
            metadata={
                "position_size": 100,
                "entry_price": 50.0,
                "current_price": 45.0,
            },
        )

        assert signal.metadata["position_size"] == 100
        assert signal.metadata["entry_price"] == 50.0

        # Verify metadata is included in dict
        data = signal.to_dict()
        assert "metadata" in data
        assert data["metadata"]["position_size"] == 100

    def test_cooldown_until(self) -> None:
        """Test cooldown_until field."""
        cooldown_time = datetime(2025, 1, 1, 13, 0, 0)
        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
            cooldown_until=cooldown_time,
        )

        assert signal.cooldown_until == cooldown_time

        # Verify it's serialized correctly
        data = signal.to_dict()
        assert data["cooldown_until"] == "2025-01-01T13:00:00"
