"""
TradingSignalViewModel - Framework-agnostic trading signal data transformation.

Extracts business logic from TradingSignalsTable:
- Signal formatting (timestamp, direction, strength)
- Timeframe and indicator extraction
- Row key generation for stable incremental updates
- Color coding for direction
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .base import BaseViewModel


class TradingSignalViewModel(BaseViewModel[List[Any]]):
    """
    ViewModel for trading signals table (Tab 6).

    Responsibilities:
    - Transform signals into display rows
    - Generate stable row keys from signal_id
    - Track signal ordering (newest first)
    - Compute cell-level diffs for incremental updates
    """

    def __init__(self, max_signals: int = 100, display_tz: str = "Asia/Hong_Kong") -> None:
        super().__init__()
        self._max_signals = max_signals
        self._signal_map: Dict[str, Any] = {}  # row_key -> signal
        self._display_tz = display_tz

    def compute_display_data(self, signals: List[Any]) -> Dict[str, List[str]]:
        """Transform signal list into display rows."""
        if not signals:
            return {}

        result: Dict[str, List[str]] = {}
        self._signal_map.clear()

        # Limit to max_signals (signals are already ordered newest-first)
        limited = signals[: self._max_signals]

        for signal in limited:
            row_key = self._signal_key(signal)
            self._signal_map[row_key] = signal
            result[row_key] = self._build_row(signal)

        return result

    def get_row_order(self, signals: List[Any]) -> List[str]:
        """Return ordered list of row keys (newest first)."""
        if not signals:
            return []
        limited = signals[: self._max_signals]
        return [self._signal_key(s) for s in limited]

    def get_signal_for_key(self, row_key: str) -> Optional[Any]:
        """Get the signal object for a row key."""
        return self._signal_map.get(row_key)

    # -------------------------------------------------------------------------
    # Row Building
    # -------------------------------------------------------------------------

    def _build_row(self, signal: Any) -> List[str]:
        """Build a display row from a signal object."""
        timestamp = self._format_time(getattr(signal, "timestamp", None))
        symbol = getattr(signal, "symbol", "-") or "-"

        # Direction handling
        direction = self._normalize_direction(getattr(signal, "direction", ""))
        direction_text = self._style_direction(direction)

        # Strength bar
        strength = getattr(signal, "strength", None)
        strength_text = self._format_strength(strength)

        # Extract metadata
        timeframe, indicator = self._extract_signal_metadata(signal)

        # Rule and message
        rule = (
            getattr(signal, "trigger_rule", None)
            or getattr(signal, "strategy_id", "")
            or "-"
        )
        message = (
            getattr(signal, "message", None) or getattr(signal, "reason", "") or "-"
        )

        return [
            timestamp,
            symbol,
            direction_text,
            strength_text,
            timeframe or "-",
            indicator or "-",
            self._truncate(rule, 24),
            self._truncate(message, 50),
        ]

    def _signal_key(self, signal: Any) -> str:
        """Generate a stable row key for a signal."""
        signal_id = getattr(signal, "signal_id", None)
        if signal_id:
            return str(signal_id)
        # Fallback: combine symbol and timestamp
        symbol = getattr(signal, "symbol", "UNKNOWN")
        ts = getattr(signal, "timestamp", None)
        if isinstance(ts, datetime):
            return f"{symbol}:{ts.isoformat()}"
        return f"{symbol}:{id(signal)}"

    # -------------------------------------------------------------------------
    # Metadata Extraction
    # -------------------------------------------------------------------------

    def _extract_signal_metadata(self, signal: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract timeframe and indicator from a signal.

        First checks explicit attributes, then parses from signal_id.
        Signal ID format: "{category}:{indicator}:{symbol}:{timeframe}"
        """
        # Try explicit attributes first
        timeframe = getattr(signal, "timeframe", None)
        indicator = getattr(signal, "indicator", None)

        # Fall back to parsing signal_id
        signal_id = getattr(signal, "signal_id", None)
        if isinstance(signal_id, str):
            parts = signal_id.split(":")
            if len(parts) >= 4:
                if not indicator:
                    indicator = parts[1]
                if not timeframe:
                    timeframe = parts[3]

        return (
            str(timeframe) if timeframe else None,
            str(indicator) if indicator else None,
        )

    # -------------------------------------------------------------------------
    # Formatting Helpers
    # -------------------------------------------------------------------------

    def _format_time(self, value: Optional[Any]) -> str:
        """Format timestamp for display, converting UTC to display timezone."""
        from zoneinfo import ZoneInfo

        dt = None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
            except ValueError:
                return value[:8] if len(value) >= 8 else value

        if dt is None:
            return "-"

        # Convert UTC to display timezone
        try:
            if dt.tzinfo is not None:
                dt = dt.astimezone(ZoneInfo(self._display_tz))
        except Exception:
            pass  # Fall back to original timezone

        return dt.strftime("%H:%M:%S")

    def _format_strength(self, value: Optional[Any]) -> str:
        """Format strength value with visual indicator."""
        if value is None:
            return "-"
        try:
            val = int(float(value))
            # Clamp to 0-100
            val = max(0, min(100, val))
            return str(val)
        except (TypeError, ValueError):
            return "-"

    def _normalize_direction(self, direction: Any) -> str:
        """Normalize direction to BUY/SELL/ALERT."""
        if isinstance(direction, Enum):
            direction = direction.value
        if not direction:
            return "ALERT"

        direction = str(direction).upper()
        direction_map = {
            "LONG": "BUY",
            "BUY": "BUY",
            "SHORT": "SELL",
            "SELL": "SELL",
            "FLAT": "ALERT",
            "ALERT": "ALERT",
        }
        return direction_map.get(direction, "ALERT")

    def _style_direction(self, direction: str) -> str:
        """Apply color styling to direction."""
        styles = {
            "BUY": "[bold #7ee787]BUY[/]",
            "SELL": "[bold #ff6b6b]SELL[/]",
            "ALERT": "[bold #f6d365]ALRT[/]",
        }
        return styles.get(direction, f"[#8b949e]{direction}[/]")

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis if too long."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 1] + "..."
