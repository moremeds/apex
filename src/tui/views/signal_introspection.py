"""
Signal Introspection view for Tab 3.

Real-time visibility into the signal pipeline:
- Left: Indicator states with warmup progress
- Right top: Rule evaluation history (requires trace_mode)
- Right bottom: Active cooldowns and pipeline stats

Keyboard shortcuts:
- w/s or Up/Down: Navigate indicator list
- t: Cycle timeframe filter
- r: Refresh view
- m: Toggle trace mode on RuleEngine
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import DataTable, Static

if TYPE_CHECKING:
    from ...domain.interfaces.signal_introspection import SignalIntrospectionPort


class SignalIntrospectionView(Container, can_focus=True):
    """
    Signal pipeline introspection view.

    Provides real-time visibility into:
    - Indicator values and warmup status
    - Rule evaluation history (when trace_mode enabled)
    - Active cooldowns
    - Pipeline statistics
    """

    DEFAULT_CSS = ""

    BINDINGS = [
        Binding("w", "move_up", "Up", show=True),
        Binding("s", "move_down", "Down", show=True),
        Binding("up", "move_up", "Up", show=False),
        Binding("down", "move_down", "Down", show=False),
        Binding("t", "cycle_timeframe", "TF", show=True),
        Binding("r", "refresh", "Refresh", show=True),
        Binding("m", "toggle_trace", "Trace", show=True),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._introspection: Optional["SignalIntrospectionPort"] = None
        self._selected_symbol: Optional[str] = None
        self._selected_timeframe: str = "1h"
        self._timeframes = ["1m", "5m", "15m", "1h", "1d"]
        self._timeframe_idx = 3  # Default to 1h

    def on_show(self) -> None:
        """Focus this view when it becomes visible."""
        self.focus()
        if self._introspection is not None:
            self.refresh_view()

    def compose(self) -> ComposeResult:
        """Compose the signal introspection layout."""
        with Horizontal(id="introspection-main"):
            # Left side - Indicators (~40%)
            with Vertical(id="introspection-left"):
                yield Static(
                    "Indicator States",
                    id="intro-indicator-title",
                    classes="panel-title",
                )
                yield DataTable(id="intro-indicator-table")
                yield Static(
                    "Warmup Status",
                    id="intro-warmup-title",
                    classes="panel-title",
                )
                yield DataTable(id="intro-warmup-table")

            # Right side - Evaluations + Stats (~60%)
            with Vertical(id="introspection-right"):
                yield Static(
                    "Rule Evaluations (trace_mode)",
                    id="intro-evaluations-title",
                    classes="panel-title",
                )
                yield DataTable(id="intro-evaluations-table")
                yield Static(
                    "Cooldowns & Stats",
                    id="intro-stats-title",
                    classes="panel-title",
                )
                yield DataTable(id="intro-cooldowns-table")
                yield Static(id="intro-pipeline-stats", classes="stats-panel")

    def on_mount(self) -> None:
        """Set up tables when mounted."""
        self._setup_indicator_table()
        self._setup_warmup_table()
        self._setup_evaluations_table()
        self._setup_cooldowns_table()

    def _setup_indicator_table(self) -> None:
        """Configure the indicator states table."""
        table = self.query_one("#intro-indicator-table", DataTable)
        table.add_columns("Symbol", "TF", "Indicator", "Value", "State")
        table.cursor_type = "row"

    def _setup_warmup_table(self) -> None:
        """Configure the warmup status table."""
        table = self.query_one("#intro-warmup-table", DataTable)
        table.add_columns("Symbol", "TF", "Bars", "Required", "Progress", "Status")
        table.cursor_type = "row"

    def _setup_evaluations_table(self) -> None:
        """Configure the rule evaluations table."""
        table = self.query_one("#intro-evaluations-table", DataTable)
        table.add_columns("Time", "Rule", "Symbol", "TF", "Result", "Reason")
        table.cursor_type = "row"

    def _setup_cooldowns_table(self) -> None:
        """Configure the cooldowns table."""
        table = self.query_one("#intro-cooldowns-table", DataTable)
        table.add_columns("Category", "Indicator", "Symbol", "TF", "Remaining")
        table.cursor_type = "row"

    # -------------------------------------------------------------------------
    # Public API for integration
    # -------------------------------------------------------------------------

    def set_introspection(self, introspection: "SignalIntrospectionPort") -> None:
        """Set the introspection port for data access."""
        self._introspection = introspection
        if self.is_mounted:
            self.refresh_view()

    def refresh_view(self) -> None:
        """Refresh all panels from introspection data."""
        if self._introspection is None:
            return

        self._refresh_indicators()
        self._refresh_warmup()
        self._refresh_evaluations()
        self._refresh_cooldowns()
        self._refresh_stats()

    def _refresh_indicators(self) -> None:
        """Refresh the indicator states table."""
        if self._introspection is None:
            return

        table = self.query_one("#intro-indicator-table", DataTable)
        table.clear()

        states = self._introspection.get_indicator_states(timeframe=self._selected_timeframe)

        for (symbol, tf), indicators in states.items():
            for indicator_name, state in indicators.items():
                value = state.get("value", state.get("direction", "-"))
                state_str = self._format_indicator_state(state)
                table.add_row(symbol, tf, indicator_name, str(value), state_str)

    def _refresh_warmup(self) -> None:
        """Refresh the warmup status table."""
        if self._introspection is None:
            return

        table = self.query_one("#intro-warmup-table", DataTable)
        table.clear()

        warmup_list = self._introspection.get_all_warmup_status()

        for warmup in warmup_list:
            symbol = warmup.get("symbol", "")
            tf = warmup.get("timeframe", "")
            bars = warmup.get("bars_loaded", 0)
            required = warmup.get("bars_required", 0)
            progress = warmup.get("progress_pct", 0.0)
            status = warmup.get("status", "unknown")

            progress_str = f"{progress * 100:.0f}%"
            table.add_row(symbol, tf, str(bars), str(required), progress_str, status)

    def _refresh_evaluations(self) -> None:
        """Refresh the rule evaluations table."""
        if self._introspection is None:
            return

        table = self.query_one("#intro-evaluations-table", DataTable)
        table.clear()

        evaluations = self._introspection.get_rule_evaluations(limit=30)

        for eval_dict in evaluations:
            timestamp = eval_dict.get("timestamp")
            time_str = timestamp.strftime("%H:%M:%S") if timestamp else "-"
            rule = eval_dict.get("rule_name", "")
            symbol = eval_dict.get("symbol", "")
            tf = eval_dict.get("timeframe", "")
            triggered = eval_dict.get("triggered", False)
            reason = eval_dict.get("reason", "")

            result = "[green]TRIGGERED[/]" if triggered else "[dim]no match[/]"
            if eval_dict.get("blocked_by_cooldown"):
                result = "[yellow]cooldown[/]"
            if eval_dict.get("error"):
                result = "[red]error[/]"

            table.add_row(time_str, rule, symbol, tf, result, reason)

    def _refresh_cooldowns(self) -> None:
        """Refresh the cooldowns table."""
        if self._introspection is None:
            return

        table = self.query_one("#intro-cooldowns-table", DataTable)
        table.clear()

        cooldowns = self._introspection.get_all_cooldowns()

        for cd in cooldowns:
            category = cd.get("category", "")
            indicator = cd.get("indicator", "")
            symbol = cd.get("symbol", "")
            tf = cd.get("timeframe", "")
            remaining = cd.get("remaining_seconds", 0)

            remaining_str = f"{remaining}s"
            table.add_row(category, indicator, symbol, tf, remaining_str)

    def _refresh_stats(self) -> None:
        """Refresh the pipeline statistics panel."""
        if self._introspection is None:
            return

        stats = self._introspection.get_pipeline_stats()
        stats_widget = self.query_one("#intro-pipeline-stats", Static)

        running = "[green]Running[/]" if stats.get("running") else "[red]Stopped[/]"
        bars = stats.get("bars_processed", 0)
        rules = stats.get("rules_evaluated", 0)
        signals = stats.get("signals_emitted", 0)
        uptime = stats.get("uptime_seconds", 0)
        cooldowns = stats.get("cooldowns_active", 0)

        uptime_str = f"{uptime / 60:.1f}m" if uptime > 60 else f"{uptime:.0f}s"

        stats_text = (
            f"Status: {running}  |  "
            f"Bars: {bars}  |  "
            f"Rules: {rules}  |  "
            f"Signals: {signals}  |  "
            f"Cooldowns: {cooldowns}  |  "
            f"Uptime: {uptime_str}"
        )
        stats_widget.update(stats_text)

    @staticmethod
    def _format_indicator_state(state: Dict[str, Any]) -> str:
        """Format indicator state for display."""
        # Extract common state fields
        zone = state.get("zone", "")
        direction = state.get("direction", "")
        regime = state.get("regime", "")

        parts = []
        if zone:
            parts.append(zone)
        if direction:
            parts.append(direction)
        if regime:
            parts.append(regime)

        return " | ".join(parts) if parts else "-"

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_move_up(self) -> None:
        """Move up in the indicator table."""
        table = self.query_one("#intro-indicator-table", DataTable)
        table.action_cursor_up()

    def action_move_down(self) -> None:
        """Move down in the indicator table."""
        table = self.query_one("#intro-indicator-table", DataTable)
        table.action_cursor_down()

    def action_cycle_timeframe(self) -> None:
        """Cycle through timeframes."""
        self._timeframe_idx = (self._timeframe_idx + 1) % len(self._timeframes)
        self._selected_timeframe = self._timeframes[self._timeframe_idx]
        self.refresh_view()

    def action_refresh(self) -> None:
        """Manually refresh the view."""
        self.refresh_view()

    def action_toggle_trace(self) -> None:
        """Toggle trace mode on RuleEngine (requires orchestrator access)."""
        # This would need to be wired to the RuleEngine via orchestrator
        self.notify("Trace mode toggle - wire to orchestrator")
