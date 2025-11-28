"""
Unit tests for RiskSignalManager.
"""

from datetime import datetime
from src.domain.services.risk.risk_signal_manager import RiskSignalManager
from src.models.risk_signal import (
    RiskSignal,
    SignalLevel,
    SignalSeverity,
)


class TestRiskSignalManager:
    """Test RiskSignalManager debounce and cooldown logic."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = RiskSignalManager(debounce_seconds=10, cooldown_minutes=3)

        assert manager.debounce_seconds == 10
        assert manager.cooldown_minutes == 3
        assert len(manager._pending) == 0
        assert len(manager._cooldowns) == 0

    def test_first_signal_debounced(self):
        """Test first occurrence of signal is debounced."""
        manager = RiskSignalManager(debounce_seconds=15, cooldown_minutes=5)

        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        # First process should return empty (debouncing)
        result = manager.process(signal)
        assert result == []
        assert len(manager._pending) == 1
        assert "TEST:Signal" in manager._pending

        # Check stats
        stats = manager.get_stats()
        assert stats["total_processed"] == 1
        assert stats["debounced"] == 1
        assert stats["fired"] == 0

    def test_signal_fires_after_debounce(self):
        """Test signal fires after debounce period."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        # First call - debounce (even with 0 seconds, needs 2nd call)
        result1 = manager.process(signal)
        assert result1 == []

        # Second call - should fire (0 second debounce elapsed)
        result2 = manager.process(signal)
        assert len(result2) == 1
        assert result2[0].signal_id == "TEST:Signal"

        # Check cooldown was set
        assert "TEST:Signal" in manager._cooldowns
        assert result2[0].cooldown_until is not None

        # Check stats
        stats = manager.get_stats()
        assert stats["fired"] == 1

    def test_cooldown_suppresses_duplicate(self):
        """Test cooldown prevents duplicate signals."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        # Fire signal
        manager.process(signal)
        fired = manager.process(signal)
        assert len(fired) == 1

        # Try to fire again - should be suppressed
        result = manager.process(signal)
        assert result == []

        # Again
        result = manager.process(signal)
        assert result == []

        # Check stats
        stats = manager.get_stats()
        assert stats["cooldown_suppressed"] == 2

    def test_severity_escalation_bypasses_cooldown(self):
        """Test severity escalation allows signal through cooldown."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        # Fire WARNING signal
        warning_signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        manager.process(warning_signal)
        manager.process(warning_signal)  # Fire it

        # Try with same severity - should be suppressed
        result = manager.process(warning_signal)
        assert result == []

        # Escalate to CRITICAL - should fire
        critical_signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.CRITICAL,
            trigger_rule="Test",
        )

        manager.process(critical_signal)  # Debounce (escalation detected here)
        result = manager.process(critical_signal)  # Fire (escalation detected again)
        assert len(result) == 1
        assert result[0].severity == SignalSeverity.CRITICAL

        # Check stats (escalation counted twice - once during debounce, once when fired)
        stats = manager.get_stats()
        assert stats["escalated"] == 2

    def test_lower_severity_still_suppressed(self):
        """Test lower severity doesn't bypass cooldown."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        # Fire CRITICAL signal
        critical_signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.CRITICAL,
            trigger_rule="Test",
        )

        manager.process(critical_signal)
        manager.process(critical_signal)  # Fire

        # Try with lower severity (WARNING) - should still be suppressed
        warning_signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        result = manager.process(warning_signal)
        assert result == []

        # Check it was suppressed, not escalated
        stats = manager.get_stats()
        assert stats["cooldown_suppressed"] == 1
        assert stats["escalated"] == 0

    def test_clear_signal(self):
        """Test clearing a signal."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        # Fire signal
        manager.process(signal)
        manager.process(signal)
        assert "TEST:Signal" in manager._cooldowns

        # Clear it
        manager.clear_signal("TEST:Signal")
        assert "TEST:Signal" not in manager._cooldowns
        assert "TEST:Signal" not in manager._pending

        # Should be able to fire again immediately
        manager.process(signal)
        result = manager.process(signal)
        assert len(result) == 1

    def test_clear_pending_signal(self):
        """Test clearing a pending (debouncing) signal."""
        manager = RiskSignalManager(debounce_seconds=15, cooldown_minutes=5)

        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        # Start debouncing
        manager.process(signal)
        assert "TEST:Signal" in manager._pending

        # Clear while debouncing
        manager.clear_signal("TEST:Signal")
        assert "TEST:Signal" not in manager._pending

    def test_clear_all_for_symbol(self):
        """Test clearing all signals for a symbol."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        # Fire multiple signals for TSLA
        signal1 = RiskSignal(
            signal_id="POSITION:TSLA:Stop_Loss",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Stop_Loss",
            symbol="TSLA",
        )

        signal2 = RiskSignal(
            signal_id="POSITION:TSLA:Take_Profit",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.INFO,
            trigger_rule="Take_Profit",
            symbol="TSLA",
        )

        # Fire both
        manager.process(signal1)
        manager.process(signal1)
        manager.process(signal2)
        manager.process(signal2)

        assert len(manager._cooldowns) == 2

        # Clear all TSLA signals
        manager.clear_all_for_symbol("TSLA")
        assert len(manager._cooldowns) == 0

    def test_different_signals_independent(self):
        """Test different signal IDs are handled independently."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        signal1 = RiskSignal(
            signal_id="TEST:Signal1",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test1",
        )

        signal2 = RiskSignal(
            signal_id="TEST:Signal2",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test2",
        )

        # Fire signal1
        manager.process(signal1)
        result1 = manager.process(signal1)
        assert len(result1) == 1

        # signal2 should still fire (different ID)
        manager.process(signal2)
        result2 = manager.process(signal2)
        assert len(result2) == 1

    def test_cleanup_expired_cooldowns(self):
        """Test cleanup of expired cooldowns."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=0)  # 0 min cooldown

        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        # Fire signal (cooldown expires immediately since it's 0 minutes)
        manager.process(signal)
        manager.process(signal)
        assert len(manager._cooldowns) == 1

        # Cleanup should remove expired cooldown
        manager.cleanup_expired()
        assert len(manager._cooldowns) == 0

    def test_stats(self):
        """Test statistics tracking."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        # Process once (debounce)
        manager.process(signal)
        stats = manager.get_stats()
        assert stats["total_processed"] == 1
        assert stats["debounced"] == 1
        assert stats["fired"] == 0

        # Fire it
        manager.process(signal)
        stats = manager.get_stats()
        assert stats["total_processed"] == 2
        assert stats["fired"] == 1

        # Try again (cooldown)
        manager.process(signal)
        stats = manager.get_stats()
        assert stats["cooldown_suppressed"] == 1

    def test_reset_stats(self):
        """Test resetting statistics."""
        manager = RiskSignalManager(debounce_seconds=0, cooldown_minutes=5)

        signal = RiskSignal(
            signal_id="TEST:Signal",
            timestamp=datetime.now(),
            level=SignalLevel.POSITION,
            severity=SignalSeverity.WARNING,
            trigger_rule="Test",
        )

        # Generate some stats
        manager.process(signal)
        manager.process(signal)
        manager.process(signal)

        stats = manager.get_stats()
        assert stats["total_processed"] > 0

        # Reset
        manager.reset_stats()
        stats = manager.get_stats()
        assert stats["total_processed"] == 0
        assert stats["fired"] == 0

    def test_repr(self):
        """Test string representation."""
        manager = RiskSignalManager(debounce_seconds=10, cooldown_minutes=3)
        repr_str = repr(manager)

        assert "RiskSignalManager" in repr_str
        assert "debounce=10s" in repr_str
        assert "cooldown=3m" in repr_str
