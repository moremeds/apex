"""
Trading signals table widget with color-coded direction display.

Displays trading signals from the signal engine with:
- Color coding: green (BUY), red (SELL), yellow (ALERT)
- Strength indicator bar
- Timeframe and indicator extraction from signal_id
- FIFO eviction when max_signals exceeded
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Tuple

from textual.reactive import reactive
from textual.widgets import DataTable


def extract_signal_metadata(signal: Any) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract timeframe and indicator from a signal.

    First checks explicit attributes, then parses from signal_id.
    Signal ID format: "{category}:{indicator}:{symbol}:{timeframe}"

    Args:
        signal: TradingSignal or TradingSignalEvent

    Returns:
        Tuple of (timeframe, indicator)
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


class TradingSignalsTable(DataTable):
    """
    Signal feed table with direction color coding.

    Displays trading signals with columns for time, symbol, direction,
    strength, timeframe, indicator, rule, and message.
    """

    COLUMNS = [
        ("Time", 9),
        ("Symbol", 8),
        ("Dir", 5),
        ("Str", 4),
        ("TF", 4),
        ("Indicator", 14),
        ("Rule", 24),
        ("Message", 50),
    ]

    # Reactive signal list
    signals: reactive[List[Any]] = reactive(list, init=False)

    def __init__(self, max_signals: int = 100, **kwargs) -> None:
        """
        Initialize the trading signals table.

        Args:
            max_signals: Maximum signals to display (FIFO eviction)
        """
        super().__init__(cursor_type="row", zebra_stripes=True, **kwargs)
        self._max_signals = max_signals
        self._column_keys: List[str] = []

    def on_mount(self) -> None:
        """Initialize table columns."""
        self._column_keys.clear()
        for idx, (name, width) in enumerate(self.COLUMNS):
            col_key = f"col-{idx}"
            self.add_column(name, width=width, key=col_key)
            self._column_keys.append(col_key)

    def watch_signals(self, signals: List[Any]) -> None:
        """Rebuild table when signals change."""
        self._rebuild(signals)

    def add_signal(self, signal: Any) -> None:
        """
        Add a single signal to the table.

        Maintains max_signals limit with FIFO eviction.
        """
        current = list(self.signals)
        current.insert(0, signal)  # Newest first
        if len(current) > self._max_signals:
            current = current[: self._max_signals]
        self.signals = current

    def clear_signals(self) -> None:
        """Clear all signals from the table."""
        self.signals = []

    def _rebuild(self, signals: List[Any]) -> None:
        """Full table rebuild with signal data."""
        self.clear()

        if not signals:
            self._add_placeholder_row()
            return

        for signal in signals:
            row_key = self._signal_key(signal)
            self.add_row(*self._build_row(signal), key=row_key)

    def _add_placeholder_row(self) -> None:
        """Add a placeholder row when no signals exist."""
        placeholders = [
            "[dim]-[/]",
            "[dim]-[/]",
            "[dim]-[/]",
            "[dim]-[/]",
            "[dim]-[/]",
            "[dim]No signals[/]",
            "[dim]Waiting for events...[/]",
            "[dim]-[/]",
        ]
        self.add_row(*placeholders, key="_placeholder_empty")

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
        timeframe, indicator = extract_signal_metadata(signal)

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

    def _format_time(self, value: Optional[Any]) -> str:
        """Format timestamp for display."""
        if isinstance(value, datetime):
            return value.strftime("%H:%M:%S")
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).strftime("%H:%M:%S")
            except ValueError:
                return value[:8] if len(value) >= 8 else value
        return "-"

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
        return text[: max_len - 1] + "…"
