"""
Risk signals panel rendering for the Terminal Dashboard.
"""

from __future__ import annotations
from typing import Dict, List
from datetime import datetime
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ...models.risk_signal import RiskSignal, SignalSeverity
from ...models.risk_snapshot import RiskSnapshot
from src.domain.services.risk.rule_engine import LimitBreach, BreachSeverity


def update_persistent_risk_signals(
    current_signals: List[RiskSignal],
    persistent_signals: Dict[str, Dict],
    alert_retention_seconds: int = 300,
) -> List[Dict]:
    """
    Update persistent risk signal tracking and return signals to display.

    Active signals are updated with last_seen timestamp.
    Cleared signals are kept for alert_retention_seconds with is_active=False.

    Args:
        current_signals: List of currently active risk signals.
        persistent_signals: Mutable dict to track persistent signals (modified in place).
        alert_retention_seconds: How long to keep cleared signals visible.

    Returns:
        List of signal info dicts to display (active + recently cleared).
    """
    now = datetime.now()

    # Build set of current signal keys
    current_keys = set()
    for signal in current_signals:
        signal_key = f"{signal.symbol or 'PORTFOLIO'}_{signal.trigger_rule}_{signal.severity.value}"
        current_keys.add(signal_key)

        if signal_key in persistent_signals:
            # Update existing signal
            persistent_signals[signal_key]["signal"] = signal
            persistent_signals[signal_key]["last_seen"] = now
            persistent_signals[signal_key]["is_active"] = True
        else:
            # New signal
            persistent_signals[signal_key] = {
                "signal": signal,
                "first_seen": now,
                "last_seen": now,
                "is_active": True,
            }

    # Mark signals not in current set as inactive
    for signal_key in persistent_signals:
        if signal_key not in current_keys:
            persistent_signals[signal_key]["is_active"] = False

    # Build display list and cleanup expired signals
    display_signals = []
    expired_keys = []

    for signal_key, signal_info in persistent_signals.items():
        age_seconds = (now - signal_info["last_seen"]).total_seconds()

        if signal_info["is_active"]:
            # Active signal - always display
            display_signals.append({
                "signal": signal_info["signal"],
                "first_seen": signal_info["first_seen"],
                "last_seen": signal_info["last_seen"],
                "is_active": True,
            })
        elif age_seconds <= alert_retention_seconds:
            # Recently cleared - display with dimmed style
            display_signals.append({
                "signal": signal_info["signal"],
                "first_seen": signal_info["first_seen"],
                "last_seen": signal_info["last_seen"],
                "is_active": False,
            })
        else:
            # Expired - mark for cleanup
            expired_keys.append(signal_key)

    # Cleanup expired signals
    for key in expired_keys:
        del persistent_signals[key]

    return display_signals


def render_breaches(
    breaches: List[LimitBreach] | List[RiskSignal],
    persistent_signals: Dict[str, Dict],
    alert_retention_seconds: int = 300,
) -> Panel:
    """
    Render portfolio risk alerts panel (supports both LimitBreach and RiskSignal).

    Args:
        breaches: List of limit breaches (legacy) or risk signals (new).
        persistent_signals: Mutable dict for persistent signal tracking.
        alert_retention_seconds: How long to keep cleared signals visible.

    Returns:
        Panel containing the risk alerts.
    """
    # Check if we're using RiskSignals or legacy LimitBreaches
    is_risk_signals = breaches and isinstance(breaches[0], RiskSignal)

    if is_risk_signals:
        display_signals = update_persistent_risk_signals(
            breaches, persistent_signals, alert_retention_seconds
        )
        return _render_risk_signals_from_persistent(display_signals)
    else:
        # For legacy breaches, check if empty
        if not breaches:
            # Also check persistent risk signals for cleared items
            display_signals = update_persistent_risk_signals(
                [], persistent_signals, alert_retention_seconds
            )
            if display_signals:
                return _render_risk_signals_from_persistent(display_signals)
            text = Text("All risk limits OK", style="green")
            return Panel(text, title="Portfolio Risk Alert", border_style="green")
        return _render_legacy_breaches(breaches)


def _render_legacy_breaches(breaches: List[LimitBreach]) -> Panel:
    """Render legacy LimitBreach objects."""
    table = Table(show_header=True, box=None)
    table.add_column("Severity", style="bold")
    table.add_column("Risk Metric", style="cyan")
    table.add_column("Status", justify="right")

    for breach in breaches:
        severity_style = "red" if breach.severity == BreachSeverity.HARD else "yellow"
        severity_text = "HARD" if breach.severity == BreachSeverity.HARD else "SOFT"

        table.add_row(
            Text(severity_text, style=severity_style),
            breach.limit_name,
            f"{breach.breach_pct():.1f}%",
        )

    border_style = "red" if any(b.severity == BreachSeverity.HARD for b in breaches) else "yellow"
    return Panel(table, title=f"[W] Portfolio Risk Alert ({len(breaches)})", border_style=border_style)


def render_risk_signals(
    signals: List[RiskSignal],
    persistent_signals: Dict[str, Dict],
    alert_retention_seconds: int = 300,
) -> Panel:
    """
    Render RiskSignal objects with persistent tracking and enhanced display.

    Args:
        signals: List of current risk signals.
        persistent_signals: Mutable dict for persistent signal tracking.
        alert_retention_seconds: How long to keep cleared signals visible.

    Returns:
        Panel containing the risk signals.
    """
    display_signals = update_persistent_risk_signals(
        signals, persistent_signals, alert_retention_seconds
    )
    return _render_risk_signals_from_persistent(display_signals)


def _render_risk_signals_from_persistent(display_signals: List[Dict]) -> Panel:
    """Render risk signals from persistent tracking data."""
    if not display_signals:
        text = Text("All risk limits OK", style="green")
        return Panel(text, title="Portfolio Risk Alert", border_style="green")

    table = Table(show_header=True, box=None)
    table.add_column("Severity", style="bold", no_wrap=True)
    table.add_column("Symbol", style="cyan", no_wrap=True)
    table.add_column("Rule", style="white")
    table.add_column("Action", style="yellow", justify="right")
    table.add_column("Time", style="dim", justify="right")

    # Sort by active status first (active first), then by severity
    sorted_signals = sorted(
        display_signals,
        key=lambda s: (
            0 if s["is_active"] else 1,
            {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[s["signal"].severity.value]
        )
    )

    for signal_info in sorted_signals:
        signal = signal_info["signal"]
        is_active = signal_info["is_active"]
        last_seen = signal_info["last_seen"]

        # Format time display
        time_str = last_seen.strftime("%H:%M:%S") if last_seen else ""

        if not is_active:
            # Cleared signal - dimmed style
            severity_style = "dim"
            icon = "o"
            status_suffix = " [cleared]"
        else:
            severity_style = {
                "CRITICAL": "bold red",
                "WARNING": "bold yellow",
                "INFO": "cyan"
            }[signal.severity.value]

            icon = {
                "CRITICAL": "[!]",
                "WARNING": "[W]",
                "INFO": "[i]"
            }[signal.severity.value]
            status_suffix = ""

        # Format action
        action_text = signal.suggested_action.value
        if signal.breach_pct:
            action_text += f" ({signal.breach_pct:.0f}%)"

        rule_text = signal.trigger_rule + status_suffix

        table.add_row(
            Text(f"{icon} {signal.severity.value}", style=severity_style),
            signal.symbol or "PORTFOLIO",
            Text(rule_text, style=severity_style if not is_active else "white"),
            Text(action_text, style=severity_style if not is_active else "yellow"),
            Text(time_str, style="dim")
        )

    # Set border color based on highest severity of ACTIVE signals
    active_signals = [s for s in display_signals if s["is_active"]]
    has_critical = any(s["signal"].severity == SignalSeverity.CRITICAL for s in active_signals)
    has_warning = any(s["signal"].severity == SignalSeverity.WARNING for s in active_signals)

    active_count = len(active_signals)
    cleared_count = len(display_signals) - active_count

    if has_critical:
        border_style = "red"
        title = f"[!] Portfolio Risk Alert ({active_count} active"
    elif has_warning:
        border_style = "yellow"
        title = f"[W] Portfolio Risk Alert ({active_count} active"
    elif active_count > 0:
        border_style = "cyan"
        title = f"Portfolio Risk Alert ({active_count} active"
    else:
        border_style = "dim"
        title = f"Portfolio Risk Alert (0 active"

    if cleared_count > 0:
        title += f", {cleared_count} cleared)"
    else:
        title += ")"

    return Panel(table, title=title, border_style=border_style)


def render_risk_signals_fullscreen(
    breaches: List[LimitBreach] | List[RiskSignal],
    snapshot: RiskSnapshot,
    persistent_signals: Dict[str, Dict],
    alert_retention_seconds: int = 300,
) -> Panel:
    """
    Render full-screen risk signals view with detailed information.

    Shows:
    - All active signals with full details
    - Signal history (recently cleared)
    - Grouping by severity (CRITICAL -> WARNING -> INFO)
    - More context per signal (current value, limit, breach %)

    Args:
        breaches: List of limit breaches or risk signals.
        snapshot: Current risk snapshot.
        persistent_signals: Mutable dict for persistent signal tracking.
        alert_retention_seconds: How long to keep cleared signals visible.

    Returns:
        Panel containing the full-screen risk signals view.
    """
    # Get display signals (active + recently cleared)
    if breaches and isinstance(breaches[0], RiskSignal):
        display_signals = update_persistent_risk_signals(
            breaches, persistent_signals, alert_retention_seconds
        )
    else:
        # Legacy breaches - convert to display format
        display_signals = update_persistent_risk_signals(
            [], persistent_signals, alert_retention_seconds
        )

    # Create main table with expanded columns
    table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
    table.add_column("Status", style="bold", no_wrap=True, width=10)
    table.add_column("Severity", style="bold", no_wrap=True, width=10)
    table.add_column("Symbol", style="cyan", no_wrap=True, width=12)
    table.add_column("Layer", style="dim", no_wrap=True, width=8)
    table.add_column("Trigger Rule", style="white", width=30)
    table.add_column("Current", justify="right", width=12)
    table.add_column("Limit", justify="right", width=12)
    table.add_column("Breach %", justify="right", width=10)
    table.add_column("Action", style="yellow", width=15)
    table.add_column("First Seen", style="dim", justify="right", width=10)
    table.add_column("Last Seen", style="dim", justify="right", width=10)

    if not display_signals:
        # Show "all clear" message with summary stats
        table.add_row(
            Text("[OK]", style="green"),
            "",
            "PORTFOLIO",
            "",
            Text("All risk limits within acceptable range", style="green"),
            "",
            "",
            "",
            "",
            "",
            "",
        )
    else:
        # Sort by: active first, then by severity, then by symbol
        sorted_signals = sorted(
            display_signals,
            key=lambda s: (
                0 if s["is_active"] else 1,
                {"CRITICAL": 0, "WARNING": 1, "INFO": 2}[s["signal"].severity.value],
                s["signal"].symbol or "ZZZZZ",  # Portfolio signals last within severity
            )
        )

        for signal_info in sorted_signals:
            signal = signal_info["signal"]
            is_active = signal_info["is_active"]
            first_seen = signal_info["first_seen"]
            last_seen = signal_info["last_seen"]

            # Format time displays
            first_seen_str = first_seen.strftime("%H:%M:%S") if first_seen else ""
            last_seen_str = last_seen.strftime("%H:%M:%S") if last_seen else ""

            # Status indicator
            if is_active:
                status_text = Text("* ACTIVE", style="bold green")
            else:
                status_text = Text("o CLEARED", style="dim")

            # Severity styling
            if not is_active:
                severity_style = "dim"
                icon = "o"
            else:
                severity_style = {
                    "CRITICAL": "bold red",
                    "WARNING": "bold yellow",
                    "INFO": "cyan"
                }[signal.severity.value]
                icon = {
                    "CRITICAL": "[!]",
                    "WARNING": "[W]",
                    "INFO": "[i]"
                }[signal.severity.value]

            # Format current value and limit (threshold)
            current_str = f"{signal.current_value:,.2f}" if signal.current_value is not None else "-"
            limit_str = f"{signal.threshold:,.2f}" if signal.threshold is not None else "-"
            breach_str = f"{signal.breach_pct:.1f}%" if signal.breach_pct is not None else "-"

            # Layer info
            layer_str = f"L{signal.layer}" if hasattr(signal, 'layer') and signal.layer else "-"

            # Action
            action_str = signal.suggested_action.value if signal.suggested_action else "-"

            table.add_row(
                status_text,
                Text(f"{icon} {signal.severity.value}", style=severity_style),
                signal.symbol or "PORTFOLIO",
                layer_str,
                Text(signal.trigger_rule, style=severity_style if not is_active else "white"),
                Text(current_str, style=severity_style if not is_active else "white"),
                Text(limit_str, style="dim"),
                Text(breach_str, style=severity_style if not is_active else "yellow"),
                Text(action_str, style=severity_style if not is_active else "yellow"),
                Text(first_seen_str, style="dim"),
                Text(last_seen_str, style="dim"),
            )

    # Calculate summary stats
    active_signals = [s for s in display_signals if s["is_active"]]
    cleared_signals = [s for s in display_signals if not s["is_active"]]

    critical_count = sum(1 for s in active_signals if s["signal"].severity == SignalSeverity.CRITICAL)
    warning_count = sum(1 for s in active_signals if s["signal"].severity == SignalSeverity.WARNING)
    info_count = sum(1 for s in active_signals if s["signal"].severity == SignalSeverity.INFO)

    # Build title with summary
    title_parts = ["Risk Signals"]
    if active_signals:
        title_parts.append(f"({len(active_signals)} active")
        if critical_count:
            title_parts.append(f"[!]{critical_count}")
        if warning_count:
            title_parts.append(f"[W]{warning_count}")
        if info_count:
            title_parts.append(f"[i]{info_count}")
        if cleared_signals:
            title_parts.append(f", {len(cleared_signals)} cleared)")
        else:
            title_parts.append(")")
    elif cleared_signals:
        title_parts.append(f"(0 active, {len(cleared_signals)} cleared)")
    else:
        title_parts.append("([OK] All Clear)")

    title = " ".join(title_parts)

    # Border color based on highest active severity
    if critical_count > 0:
        border_style = "red"
    elif warning_count > 0:
        border_style = "yellow"
    elif info_count > 0:
        border_style = "cyan"
    else:
        border_style = "green"

    return Panel(table, title=title, border_style=border_style)
