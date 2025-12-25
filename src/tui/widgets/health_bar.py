"""
Component health bar widget with widget pooling.

Horizontal display of component health statuses:
- Status icon ([OK]/[W]/[X]/[?])
- Component name
- Detail message

Uses pre-allocated widget pool to avoid DOM thrashing on updates.
"""

from __future__ import annotations

from typing import Any, List

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import HorizontalScroll
from textual.app import ComposeResult


class HealthBar(Widget):
    """Horizontal component health display with widget pooling."""

    # Maximum number of health components to display
    MAX_COMPONENTS = 10

    # Reactive state - use factory to avoid mutable default
    health: reactive[List[Any]] = reactive(list, init=False)

    def compose(self) -> ComposeResult:
        """Pre-allocate pool of health component widgets."""
        with HorizontalScroll(id="health-content"):
            # Pre-allocate widget pool - hidden by default
            for i in range(self.MAX_COMPONENTS):
                yield Static("", id=f"health-{i}", classes="health-component")

    def on_mount(self) -> None:
        """Render initial state and hide unused widgets."""
        self._render_health(self.health or [])

    def watch_health(self, health: List[Any]) -> None:
        """Update display when health changes."""
        self._render_health(health)

    def _render_health(self, health: List[Any]) -> None:
        """Update pooled widgets instead of DOM manipulation."""
        for i in range(self.MAX_COMPONENTS):
            try:
                component = self.query_one(f"#health-{i}", Static)
                if i < len(health):
                    component.update(self._format_component(health[i]))
                    component.display = True
                else:
                    # Hide unused widgets
                    component.update("")
                    component.display = False
            except Exception as e:
                self.log.error(f"Failed to update health component {i}: {e}")

    def _format_component(self, h: Any) -> str:
        """Format a single health component."""
        status = getattr(h, "status", None)
        name = getattr(h, "component_name", "Unknown")
        message = getattr(h, "message", "")
        metadata = getattr(h, "metadata", {})

        # Determine status styling
        status_val = status.value if hasattr(status, "value") else str(status)

        if status_val == "HEALTHY":
            icon = "[green][OK][/]"
        elif status_val == "DEGRADED":
            icon = "[yellow][W][/]"
        elif status_val == "UNHEALTHY":
            icon = "[red][X][/]"
        else:
            icon = "[dim][?][/]"

        # Build detail string
        detail = message or ""
        if name == "market_data_coverage" and metadata:
            missing = metadata.get("missing_count", 0)
            total = metadata.get("total", 0)
            if missing > 0:
                detail = f"{missing}/{total} missing MD"
            else:
                detail = f"All {total} OK" if total > 0 else "No positions"

        name_display = self._truncate(str(name), 18)
        detail_display = self._truncate(str(detail), 18)

        return f"{icon}\n[cyan]{name_display}[/]\n[dim]{detail_display}[/]"

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate text to fit within the health tile."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
