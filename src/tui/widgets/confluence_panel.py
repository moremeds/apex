"""
Confluence score panel widget for trading signals view.

Displays multi-indicator confluence analysis:
- Alignment score with visual bar (-100 to +100)
- Bullish/Bearish/Neutral indicator counts
- Divergence pairs between indicators
- MTF (Multi-Timeframe) alignment summary
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class ConfluencePanel(Widget):
    """
    Display confluence score and indicator alignment summary.

    Shows:
    - Visual alignment bar from -100 (bearish) to +100 (bullish)
    - Indicator counts by direction
    - Divergence warnings between conflicting indicators
    """

    # Reactive state
    score: reactive[Optional[Any]] = reactive(None, init=False)
    alignment: reactive[Optional[Any]] = reactive(None, init=False)

    def compose(self) -> ComposeResult:
        """Compose the confluence panel layout."""
        yield Static("[dim]No confluence data[/]", id="confluence-body")

    def watch_score(self, score: Optional[Any]) -> None:
        """Update display when confluence score changes."""
        self._update_display()

    def watch_alignment(self, alignment: Optional[Any]) -> None:
        """Update display when MTF alignment changes."""
        self._update_display()

    def set_confluence(
        self,
        score: Optional[Any] = None,
        alignment: Optional[Any] = None,
    ) -> None:
        """
        Set confluence data.

        Args:
            score: ConfluenceScore object
            alignment: MTFAlignment object
        """
        self.score = score
        self.alignment = alignment

    def clear(self) -> None:
        """Clear confluence data."""
        self.score = None
        self.alignment = None

    def _update_display(self) -> None:
        """Render the confluence summary."""
        try:
            body = self.query_one("#confluence-body", Static)
        except Exception:
            return

        if self.score is None and self.alignment is None:
            body.update("[dim]No confluence data[/]")
            return

        body.update(self._build_content())

    def _build_content(self) -> str:
        """Build the full confluence display content."""
        lines: List[str] = []

        if self.score is not None:
            lines.extend(self._render_score(self.score))

        if self.alignment is not None:
            if lines:
                lines.append("")
            lines.extend(self._render_alignment(self.alignment))

        return "\n".join(lines) if lines else "[dim]No confluence data[/]"

    def _render_score(self, score: Any) -> List[str]:
        """Render ConfluenceScore details."""
        symbol = getattr(score, "symbol", "?")
        timeframe = getattr(score, "timeframe", "?")
        alignment_score = getattr(score, "alignment_score", 0)
        strongest = getattr(score, "strongest_signal", None) or "neutral"
        bullish = getattr(score, "bullish_count", 0)
        bearish = getattr(score, "bearish_count", 0)
        neutral = getattr(score, "neutral_count", 0)
        diverging_pairs: List[Tuple[str, str, str]] = getattr(
            score, "diverging_pairs", []
        )
        timestamp = getattr(score, "timestamp", None)

        # Header
        lines = [
            f"[bold #5fd7ff]{symbol}[/]  [#d66efd]{timeframe}[/]",
        ]

        # Visual alignment bar
        bar = self._alignment_bar(alignment_score)
        score_color = self._score_color(alignment_score)
        lines.append(f"Score: [{score_color}]{alignment_score:+d}[/]  {bar}")

        # Indicator counts
        lines.append(
            f"[#7ee787]▲ {bullish}[/]  "
            f"[#ff6b6b]▼ {bearish}[/]  "
            f"[#f6d365]● {neutral}[/]  "
            f"Strongest: [bold]{strongest}[/]"
        )

        # Divergence warnings
        if diverging_pairs:
            lines.append("")
            lines.append("[#f6d365]Divergences:[/]")
            for pair in diverging_pairs[:3]:  # Show top 3
                if len(pair) >= 3:
                    ind1, ind2, reason = pair[0], pair[1], pair[2]
                    lines.append(f"  • {ind1} ↔ {ind2}: [dim]{reason}[/]")
                elif len(pair) >= 2:
                    lines.append(f"  • {pair[0]} ↔ {pair[1]}")

        # Timestamp
        if timestamp:
            ts_str = self._format_time(timestamp)
            lines.append(f"[dim]Updated: {ts_str}[/]")

        return lines

    def _render_alignment(self, alignment: Any) -> List[str]:
        """Render MTFAlignment summary."""
        symbol = getattr(alignment, "symbol", "?")
        strength = getattr(alignment, "alignment_strength", "unknown")
        direction = getattr(alignment, "dominant_direction", None) or "neutral"
        # MTFAlignment uses tf_scores: Dict[str, ConfluenceScore]
        tf_scores = getattr(alignment, "tf_scores", {})

        # Strength styling
        strength_styles = {
            "strong": "[bold #7ee787]STRONG[/]",
            "moderate": "[#f6d365]MODERATE[/]",
            "weak": "[#ff6b6b]WEAK[/]",
        }
        strength_text = strength_styles.get(
            str(strength).lower(), f"[dim]{strength}[/]"
        )

        lines = [
            f"[bold #5fd7ff]{symbol}[/] MTF Alignment",
            f"Strength: {strength_text}  Direction: [bold]{direction}[/]",
        ]

        # Show timeframe breakdown from tf_scores
        if tf_scores:
            tf_parts = []
            for tf, score in tf_scores.items():
                # Get direction from ConfluenceScore's strongest_signal
                strongest = getattr(score, "strongest_signal", None)
                if strongest:
                    dir_str = str(strongest).lower()
                    if "bull" in dir_str:
                        tf_parts.append(f"[#7ee787]{tf}▲[/]")
                    elif "bear" in dir_str:
                        tf_parts.append(f"[#ff6b6b]{tf}▼[/]")
                    else:
                        tf_parts.append(f"[#8b949e]{tf}●[/]")
                else:
                    tf_parts.append(f"[#8b949e]{tf}●[/]")
            if tf_parts:
                lines.append("TFs: " + " ".join(tf_parts))

        return lines

    def _alignment_bar(self, score: int) -> str:
        """
        Create visual alignment bar from -100 to +100.

        Returns a 21-character bar with center marker.
        Example: ░░░░░░░░░░│█░░░░░░░░░  (score = +10)
        """
        # Normalize score to 0-20 range (21 positions including center)
        # -100 -> 0, 0 -> 10, +100 -> 20
        normalized = int((score + 100) / 10)
        normalized = max(0, min(20, normalized))

        # Build bar
        bar = ["░"] * 10 + ["│"] + ["░"] * 10
        bar[normalized] = "█"

        # Color the bar based on position
        bar_str = "".join(bar)
        if score > 20:
            return f"[#7ee787]{bar_str}[/]"
        elif score < -20:
            return f"[#ff6b6b]{bar_str}[/]"
        return f"[#f6d365]{bar_str}[/]"

    def _score_color(self, score: int) -> str:
        """Get color for alignment score."""
        if score > 20:
            return "#7ee787"  # Green
        elif score < -20:
            return "#ff6b6b"  # Red
        return "#f6d365"  # Yellow

    def _format_time(self, value: Any) -> str:
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

        # Convert UTC to display timezone if configured
        try:
            display_tz = getattr(self.app, "display_tz", "Asia/Hong_Kong")
            if dt.tzinfo is not None:
                dt = dt.astimezone(ZoneInfo(display_tz))
        except Exception:
            pass  # Fall back to original timezone

        return dt.strftime("%H:%M:%S")
