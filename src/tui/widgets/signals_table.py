"""
Risk signals table widget for full-screen display.

Shows all risk signals with detailed information:
- Status, Severity, Symbol, Layer, Rule, Current, Limit, Breach %, Action, Times
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from textual.widgets import DataTable
from textual.reactive import reactive


class SignalsTable(DataTable):
    """Full-screen risk signals display."""

    COLUMNS = [
        ("Status", 10),
        ("Severity", 10),
        ("Symbol", 12),
        ("Layer", 8),
        ("Trigger Rule", 30),
        ("Current", 12),
        ("Limit", 12),
        ("Breach %", 10),
        ("Action", 15),
        ("First Seen", 10),
        ("Last Seen", 10),
    ]

    # Reactive state
    signals: reactive[List[Any]] = reactive([], init=False)
    snapshot: reactive[Optional[Any]] = reactive(None, init=False)

    def __init__(self, **kwargs):
        super().__init__(cursor_type="row", **kwargs)
        self._persistent_signals: Dict[str, Dict] = {}
        self._alert_retention_seconds = 300

    def on_mount(self) -> None:
        """Set up columns when mounted."""
        for name, width in self.COLUMNS:
            self.add_column(name, width=width)

    def watch_signals(self, signals: List[Any]) -> None:
        """Update display when signals change."""
        display_signals = self._update_persistent_signals(signals)
        self._render_signals(display_signals)

    def _update_persistent_signals(self, current_signals: List[Any]) -> List[Dict]:
        """Update persistent signal tracking."""
        now = datetime.now()
        current_keys = set()

        for signal in current_signals:
            signal_key = f"{getattr(signal, 'symbol', 'PORTFOLIO') or 'PORTFOLIO'}_{getattr(signal, 'trigger_rule', '')}_{getattr(signal, 'severity', 'INFO')}"
            current_keys.add(signal_key)

            if signal_key in self._persistent_signals:
                self._persistent_signals[signal_key]["signal"] = signal
                self._persistent_signals[signal_key]["last_seen"] = now
                self._persistent_signals[signal_key]["is_active"] = True
            else:
                self._persistent_signals[signal_key] = {
                    "signal": signal,
                    "first_seen": now,
                    "last_seen": now,
                    "is_active": True,
                }

        for signal_key in self._persistent_signals:
            if signal_key not in current_keys:
                self._persistent_signals[signal_key]["is_active"] = False

        display_signals = []
        expired_keys = []

        for signal_key, signal_info in self._persistent_signals.items():
            age_seconds = (now - signal_info["last_seen"]).total_seconds()

            if signal_info["is_active"]:
                display_signals.append({
                    "signal": signal_info["signal"],
                    "first_seen": signal_info["first_seen"],
                    "last_seen": signal_info["last_seen"],
                    "is_active": True,
                })
            elif age_seconds <= self._alert_retention_seconds:
                display_signals.append({
                    "signal": signal_info["signal"],
                    "first_seen": signal_info["first_seen"],
                    "last_seen": signal_info["last_seen"],
                    "is_active": False,
                })
            else:
                expired_keys.append(signal_key)

        for key in expired_keys:
            del self._persistent_signals[key]

        return display_signals

    def _render_signals(self, display_signals: List[Dict]) -> None:
        """Render signal rows."""
        self.clear()

        if not display_signals:
            self.add_row(
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
                key="__all_clear__",
            )
            return

        # Sort by active first, then severity
        sorted_signals = sorted(
            display_signals,
            key=lambda s: (
                0 if s["is_active"] else 1,
                {"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get(
                    getattr(s["signal"], "severity", "INFO").value
                    if hasattr(getattr(s["signal"], "severity", None), "value")
                    else str(getattr(s["signal"], "severity", "INFO")),
                    2
                )
            )
        )

        for idx, signal_info in enumerate(sorted_signals):
            signal = signal_info["signal"]
            is_active = signal_info["is_active"]
            first_seen = signal_info["first_seen"]
            last_seen = signal_info["last_seen"]

            first_str = first_seen.strftime("%H:%M:%S") if first_seen else ""
            last_str = last_seen.strftime("%H:%M:%S") if last_seen else ""

            severity = getattr(signal, "severity", "INFO")
            severity_val = severity.value if hasattr(severity, "value") else str(severity)

            if is_active:
                status = "[green]* ACTIVE[/]"
            else:
                status = "[dim]o CLEARED[/]"

            if not is_active:
                severity_style = "dim"
                icon = "o"
            elif severity_val == "CRITICAL":
                severity_style = "bold red"
                icon = "[!]"
            elif severity_val == "WARNING":
                severity_style = "bold yellow"
                icon = "[W]"
            else:
                severity_style = "cyan"
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
            action_str = action.value if hasattr(action, "value") else str(action) if action else "-"

            style = severity_style if not is_active else "white"

            self.add_row(
                status,
                f"[{severity_style}]{icon} {severity_val}[/]",
                symbol,
                layer_str,
                f"[{style}]{rule}[/]",
                f"[{style}]{current_str}[/]",
                f"[dim]{limit_str}[/]",
                f"[{severity_style}]{breach_str}[/]" if is_active else f"[dim]{breach_str}[/]",
                f"[yellow]{action_str}[/]" if is_active else f"[dim]{action_str}[/]",
                f"[dim]{first_str}[/]",
                f"[dim]{last_str}[/]",
                key=f"signal-{idx}",
            )
