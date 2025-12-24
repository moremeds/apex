"""
SignalViewModel - Framework-agnostic risk signal data transformation.

Extracts business logic from SignalsTable:
- Signal persistence tracking (first_seen, last_seen, is_active)
- Expired signal retention and cleanup
- Sorting by active status and severity
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseViewModel


@dataclass
class PersistentSignal:
    """Signal with persistence tracking."""

    signal: Any
    first_seen: datetime
    last_seen: datetime
    is_active: bool


class SignalViewModel(BaseViewModel[List[Any]]):
    """
    ViewModel for risk signals table.

    Responsibilities:
    - Track signal persistence (retain cleared signals for display)
    - Sort by active/severity
    - Compute row-level diffs for incremental updates
    """

    def __init__(self, retention_seconds: int = 300) -> None:
        super().__init__()
        self._persistent_signals: Dict[str, PersistentSignal] = {}
        self._retention_seconds = retention_seconds

    def compute_display_data(self, signals: List[Any]) -> Dict[str, List[str]]:
        """Transform signals into display rows with persistence."""
        now = datetime.now()
        display_signals = self._update_persistence(signals, now)

        if not display_signals:
            return {
                "__all_clear__": [
                    "[green][OK][/]",
                    "",
                    "PORTFOLIO",
                    "",
                    "[green]All risk limits within acceptable range[/]",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            }

        # Sort by active first, then severity
        sorted_signals = sorted(
            display_signals,
            key=lambda s: (0 if s.is_active else 1, self._severity_order(s.signal)),
        )

        result: Dict[str, List[str]] = {}
        for idx, ps in enumerate(sorted_signals):
            # Use stable key based on signal identity
            key = self._signal_key(ps.signal)
            result[key] = self._build_signal_row(ps)

        return result

    def get_row_order(self, signals: List[Any]) -> List[str]:
        """Return ordered list of row keys."""
        display_data = self.compute_display_data(signals)
        return list(display_data.keys())

    def _update_persistence(
        self, current_signals: List[Any], now: datetime
    ) -> List[PersistentSignal]:
        """Update persistent signal tracking."""
        current_keys = set()

        for signal in current_signals:
            key = self._signal_key(signal)
            current_keys.add(key)

            if key in self._persistent_signals:
                ps = self._persistent_signals[key]
                ps.signal = signal
                ps.last_seen = now
                ps.is_active = True
            else:
                self._persistent_signals[key] = PersistentSignal(
                    signal=signal,
                    first_seen=now,
                    last_seen=now,
                    is_active=True,
                )

        # Mark inactive
        for key in self._persistent_signals:
            if key not in current_keys:
                self._persistent_signals[key].is_active = False

        # Build display list, removing expired
        display_signals = []
        expired_keys = []

        for key, ps in self._persistent_signals.items():
            age = (now - ps.last_seen).total_seconds()
            if ps.is_active or age <= self._retention_seconds:
                display_signals.append(ps)
            else:
                expired_keys.append(key)

        for key in expired_keys:
            del self._persistent_signals[key]

        return display_signals

    def _signal_key(self, signal: Any) -> str:
        """Generate unique key for signal deduplication."""
        symbol = getattr(signal, "symbol", "PORTFOLIO") or "PORTFOLIO"
        rule = getattr(signal, "trigger_rule", "")
        severity = getattr(signal, "severity", "INFO")
        sev_val = severity.value if hasattr(severity, "value") else str(severity)
        return f"signal-{symbol}_{rule}_{sev_val}"

    def _severity_order(self, signal: Any) -> int:
        """Get sort order for severity."""
        severity = getattr(signal, "severity", "INFO")
        sev_val = severity.value if hasattr(severity, "value") else str(severity)
        return {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(sev_val, 2)

    def _build_signal_row(self, ps: PersistentSignal) -> List[str]:
        """Build a signal row."""
        signal = ps.signal
        is_active = ps.is_active

        severity = getattr(signal, "severity", "INFO")
        sev_val = severity.value if hasattr(severity, "value") else str(severity)

        # Status
        status = "[green]* ACTIVE[/]" if is_active else "[dim]o CLEARED[/]"

        # Severity styling
        if not is_active:
            sev_style = "dim"
            icon = "o"
        elif sev_val == "CRITICAL":
            sev_style = "bold red"
            icon = "[!]"
        elif sev_val == "WARNING":
            sev_style = "bold yellow"
            icon = "[W]"
        else:
            sev_style = "cyan"
            icon = "[i]"

        symbol = getattr(signal, "symbol", None) or "PORTFOLIO"
        layer = getattr(signal, "layer", None)
        layer_str = f"L{layer}" if layer else "-"
        rule = getattr(signal, "trigger_rule", "-")

        current = getattr(signal, "current_value", None)
        current_str = f"{current:,.2f}" if current is not None else "-"
        limit = getattr(signal, "threshold", None)
        limit_str = f"{limit:,.2f}" if limit is not None else "-"
        breach = getattr(signal, "breach_pct", None)
        breach_str = f"{breach:.1f}%" if breach is not None else "-"

        action = getattr(signal, "suggested_action", None)
        action_str = (
            action.value if hasattr(action, "value") else str(action) if action else "-"
        )

        first_str = ps.first_seen.strftime("%H:%M:%S")
        last_str = ps.last_seen.strftime("%H:%M:%S")

        style = sev_style if not is_active else "white"

        return [
            status,
            f"[{sev_style}]{icon} {sev_val}[/]",
            symbol,
            layer_str,
            f"[{style}]{rule}[/]",
            f"[{style}]{current_str}[/]",
            f"[dim]{limit_str}[/]",
            f"[{sev_style}]{breach_str}[/]" if is_active else f"[dim]{breach_str}[/]",
            f"[yellow]{action_str}[/]" if is_active else f"[dim]{action_str}[/]",
            f"[dim]{first_str}[/]",
            f"[dim]{last_str}[/]",
        ]

    def get_signal_for_key(self, key: str) -> Optional[PersistentSignal]:
        """Get the persistent signal for a row key."""
        # Key format: signal-{symbol}_{rule}_{severity}
        return self._persistent_signals.get(key)
