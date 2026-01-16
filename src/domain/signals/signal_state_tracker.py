"""
Signal State Tracker - Manages signal invalidation lifecycle.

Reuses the deduplication pattern from signals.py:532-550.
When a new signal fires for (symbol, indicator, timeframe),
the previous signal for that key is marked as invalidated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from ...utils.timezone import now_utc
from .models import SignalStatus

if TYPE_CHECKING:
    from .models import TradingSignal


class SignalStateTracker:
    """
    Tracks active signals using same key pattern as _dedupe_to_current_state.

    Key: (symbol, indicator, timeframe)
    Value: Most recent TradingSignal for that key

    When a new signal arrives, the previous signal (if different) is marked
    as INVALIDATED with invalidated_by and invalidated_at populated.
    """

    def __init__(self) -> None:
        # Same key pattern as signals.py deduplication: (symbol, indicator, timeframe)
        self._active: Dict[Tuple[str, str, str], TradingSignal] = {}

    def process_signal(self, signal: TradingSignal) -> Optional[TradingSignal]:
        """
        Process new signal and return invalidated previous signal if exists.

        Args:
            signal: New trading signal to track

        Returns:
            Previous signal that was invalidated, or None if no previous signal
        """
        key = (signal.symbol, signal.indicator, signal.timeframe)
        prev = self._active.get(key)

        if prev and prev.signal_id != signal.signal_id:
            # Mark previous signal as invalidated
            prev.status = SignalStatus.INVALIDATED
            prev.invalidated_by = signal.signal_id
            prev.invalidated_at = now_utc()
            self._active[key] = signal
            return prev

        # Register new signal as active
        self._active[key] = signal
        return None

    def get_active_signals(self) -> Dict[Tuple[str, str, str], TradingSignal]:
        """Return all currently active signals."""
        return dict(self._active)

    def get_signal_status(self, signal_id: str) -> Optional[SignalStatus]:
        """Get status of a signal by ID."""
        for signal in self._active.values():
            if signal.signal_id == signal_id:
                return signal.status
        return None

    def clear(self) -> None:
        """Clear all tracked signals."""
        self._active.clear()
