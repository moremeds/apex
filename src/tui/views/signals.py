"""
Risk Signals view for Tab 2.

Full-screen display of all risk signals with detailed information:
- Status, Severity, Symbol, Layer, Rule, Current, Limit, Breach %, Action, Times
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from textual.containers import Container
from textual.app import ComposeResult

from ..widgets.signals_table import SignalsTable

if TYPE_CHECKING:
    from ...models.risk_snapshot import RiskSnapshot
    from ...models.risk_signal import RiskSignal


class SignalsView(Container):
    """Full-screen risk signals view."""

    DEFAULT_CSS = """
    SignalsView {
        height: 1fr;
        width: 1fr;
    }

    SignalsView SignalsTable {
        height: 1fr;
        width: 1fr;
        border: solid red;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the signals view layout."""
        yield SignalsTable(id="signals-table")

    def update_data(
        self,
        signals: Optional[List["RiskSignal"]],
        snapshot: Optional["RiskSnapshot"],
    ) -> None:
        """
        Update the view with new risk signal data.

        Args:
            signals: List of risk signals.
            snapshot: Current risk snapshot for context.
        """
        try:
            signals_table = self.query_one("#signals-table", SignalsTable)
            signals_table.signals = signals or []
            signals_table.snapshot = snapshot
        except Exception:
            pass
