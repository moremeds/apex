"""
Risk Signal Manager - Debounce, Cooldown, and Deduplication for Risk Signals.

Prevents alert fatigue by:
- Debouncing: Require signal to persist for N seconds before firing
- Cooldown: Prevent same signal from repeating within N minutes
- Severity escalation: Allow repeat if severity increases
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from src.models.risk_signal import RiskSignal, SignalSeverity
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class RiskSignalManager:
    """
    Manages signal deduplication, debouncing, and cooldown.

    Key features:
    - Debounce: Require signal to persist for N seconds before firing
    - Cooldown: Prevent same signal from repeating within N minutes
    - Severity escalation: Allow repeat if severity increases
    - Clean up resolved signals
    """

    def __init__(
        self,
        debounce_seconds: int = 15,
        cooldown_minutes: int = 5,
    ):
        """
        Initialize signal manager.

        Args:
            debounce_seconds: How long a signal must persist before firing (default: 15s)
            cooldown_minutes: How long to suppress duplicate signals (default: 5 min)
        """
        self.debounce_seconds = debounce_seconds
        self.cooldown_minutes = cooldown_minutes

        # Track pending signals (for debounce)
        # Format: {signal_id: (signal, first_seen_time)}
        self._pending: Dict[str, Tuple[RiskSignal, datetime]] = {}

        # Track fired signals (for cooldown)
        # Format: {signal_id: (cooldown_expiry_time, severity)}
        self._cooldowns: Dict[str, Tuple[datetime, SignalSeverity]] = {}

        # Statistics
        self._stats = {
            "total_processed": 0,
            "fired": 0,
            "debounced": 0,
            "cooldown_suppressed": 0,
            "escalated": 0,
        }

    def process(self, signal: RiskSignal) -> List[RiskSignal]:
        """
        Process incoming signal through debounce/cooldown logic.

        Args:
            signal: Incoming risk signal to process

        Returns:
            List of signals to fire (empty if suppressed)
        """
        self._stats["total_processed"] += 1
        signal_id = signal.signal_id
        now = datetime.now()

        # Check cooldown first
        if signal_id in self._cooldowns:
            cooldown_time, prev_severity = self._cooldowns[signal_id]

            if now < cooldown_time:
                # Still in cooldown - check severity escalation
                if self._severity_value(signal.severity) <= self._severity_value(prev_severity):
                    # Same or lower severity - suppress
                    logger.debug(
                        f"Signal {signal_id} suppressed (cooldown until "
                        f"{cooldown_time.strftime('%H:%M:%S')})"
                    )
                    self._stats["cooldown_suppressed"] += 1
                    return []
                else:
                    # Severity escalated - allow through
                    logger.info(
                        f"Signal {signal_id} escalated: "
                        f"{prev_severity.value} → {signal.severity.value}"
                    )
                    self._stats["escalated"] += 1
                    # Continue to fire (don't return yet)

        # Debounce logic
        if signal_id not in self._pending:
            # First occurrence - start debounce timer
            self._pending[signal_id] = (signal, now)
            logger.debug(f"Signal {signal_id} pending (debounce for {self.debounce_seconds}s)")
            self._stats["debounced"] += 1
            return []
        else:
            # Signal persisted - check if debounce period elapsed
            pending_signal, first_seen = self._pending[signal_id]
            elapsed = (now - first_seen).total_seconds()

            if elapsed >= self.debounce_seconds:
                # Fire signal
                logger.info(
                    f"Signal {signal_id} fired after {elapsed:.1f}s "
                    f"(severity: {signal.severity.value})"
                )
                self._stats["fired"] += 1

                # Set cooldown
                cooldown_time = now + timedelta(minutes=self.cooldown_minutes)
                self._cooldowns[signal_id] = (cooldown_time, signal.severity)
                signal.cooldown_until = cooldown_time

                # Clear pending
                del self._pending[signal_id]

                return [signal]
            else:
                # Still debouncing - update with latest signal (in case values changed)
                self._pending[signal_id] = (signal, first_seen)
                logger.debug(
                    f"Signal {signal_id} still debouncing "
                    f"({elapsed:.1f}s / {self.debounce_seconds}s)"
                )
                return []

    def clear_signal(self, signal_id: str) -> None:
        """
        Clear signal when condition resolves.

        Args:
            signal_id: Signal ID to clear
        """
        cleared = False

        if signal_id in self._pending:
            del self._pending[signal_id]
            cleared = True
            logger.debug(f"Cleared pending signal: {signal_id}")

        if signal_id in self._cooldowns:
            del self._cooldowns[signal_id]
            cleared = True
            logger.debug(f"Cleared cooldown for signal: {signal_id}")

        if cleared:
            logger.info(f"Signal {signal_id} cleared (condition resolved)")

    def clear_all_for_symbol(self, symbol: str) -> None:
        """
        Clear all signals for a specific symbol (e.g., position closed).

        Args:
            symbol: Symbol to clear signals for
        """
        to_clear = []

        # Find all signal IDs containing this symbol
        for signal_id in list(self._pending.keys()):
            if f":{symbol}:" in signal_id or signal_id.endswith(f":{symbol}"):
                to_clear.append(signal_id)

        for signal_id in list(self._cooldowns.keys()):
            if f":{symbol}:" in signal_id or signal_id.endswith(f":{symbol}"):
                to_clear.append(signal_id)

        # Clear them
        for signal_id in set(to_clear):
            self.clear_signal(signal_id)

        if to_clear:
            logger.info(f"Cleared {len(set(to_clear))} signals for symbol {symbol}")

    def cleanup_expired(self) -> None:
        """
        Clean up expired cooldowns (housekeeping).

        Should be called periodically to prevent memory growth.
        """
        now = datetime.now()
        expired = []

        for signal_id, (cooldown_time, _) in self._cooldowns.items():
            if now >= cooldown_time:
                expired.append(signal_id)

        for signal_id in expired:
            del self._cooldowns[signal_id]

        if expired:
            logger.debug(f"Cleaned up {len(expired)} expired cooldowns")

    def get_stats(self) -> Dict[str, int]:
        """
        Get processing statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            **self._stats,
            "pending": len(self._pending),
            "active_cooldowns": len(self._cooldowns),
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._stats = {
            "total_processed": 0,
            "fired": 0,
            "debounced": 0,
            "cooldown_suppressed": 0,
            "escalated": 0,
        }
        logger.info("Signal manager statistics reset")

    @staticmethod
    def _severity_value(severity: SignalSeverity) -> int:
        """
        Get numeric value for severity comparison.

        Args:
            severity: Signal severity

        Returns:
            Numeric value (higher = more severe)
        """
        severity_map = {
            SignalSeverity.INFO: 1,
            SignalSeverity.WARNING: 2,
            SignalSeverity.CRITICAL: 3,
        }
        return severity_map[severity]

    def __repr__(self) -> str:
        """Debug representation."""
        return (
            f"RiskSignalManager(debounce={self.debounce_seconds}s, "
            f"cooldown={self.cooldown_minutes}m, "
            f"pending={len(self._pending)}, "
            f"cooldowns={len(self._cooldowns)})"
        )
