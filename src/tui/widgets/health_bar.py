"""
Component health bar widget.

Horizontal display of component health statuses:
- Status icon ([OK]/[W]/[X]/[?])
- Component name
- Detail message
"""

from __future__ import annotations

from typing import Any, List

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import HorizontalScroll
from textual.app import ComposeResult


class HealthBar(Widget):
    """Horizontal component health display."""

    # Reactive state
    health: reactive[List[Any]] = reactive([], init=False)

    def compose(self) -> ComposeResult:
        """Compose the health bar layout."""
        yield HorizontalScroll(id="health-content")

    def on_mount(self) -> None:
        """Render initial state on mount."""
        self._render_health(self.health or [])

    def watch_health(self, health: List[Any]) -> None:
        """Update display when health changes."""
        self._render_health(health)

    def _render_health(self, health: List[Any]) -> None:
        """Render health components."""
        try:
            content = self.query_one("#health-content", HorizontalScroll)

            # Clear existing children
            for child in list(content.children):
                child.remove()

            if not health:
                content.mount(Static("[dim]No health data[/]"))
                return

            for h in health:
                status = getattr(h, "status", None)
                name = getattr(h, "component_name", "Unknown")
                message = getattr(h, "message", "")
                metadata = getattr(h, "metadata", {})

                # Determine status styling
                status_val = status.value if hasattr(status, "value") else str(status)

                if status_val == "HEALTHY":
                    icon = "[green][OK][/]"
                    style = "green"
                elif status_val == "DEGRADED":
                    icon = "[yellow][W][/]"
                    style = "yellow"
                elif status_val == "UNHEALTHY":
                    icon = "[red][X][/]"
                    style = "red"
                else:
                    icon = "[dim][?][/]"
                    style = "dim"

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

                # Create component widget
                component_text = f"{icon}\n[cyan]{name_display}[/]\n[dim]{detail_display}[/]"
                component = Static(component_text, classes="health-component")
                content.mount(component)

        except Exception:
            pass

    @staticmethod
    def _truncate(text: str, max_len: int) -> str:
        """Truncate text to fit within the health tile."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."
