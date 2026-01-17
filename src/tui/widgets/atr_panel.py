"""
ATR analysis panel widget.

Displays ATR-based stop loss and take profit levels:
- Header with symbol, price, ATR info
- Price Levels row (SL-2x to 8R)
- R-Targets and Trailing Stop side-by-side sections
- Exit Plan (if position has shares)
- Summary and keyboard hints

Layout matches original Rich Futu dashboard.
Uses ATRViewModel for all business logic calculations.
"""

from __future__ import annotations

from typing import Any, Optional

from textual import work
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from ..viewmodels.atr_vm import ATRLevels, ATRViewModel


class ATRPanel(Widget):
    """ATR analysis panel matching original Rich layout."""

    # Reactive state
    selected_symbol: reactive[Optional[str]] = reactive(None, init=False)
    atr_data: reactive[Optional[Any]] = reactive(None, init=False)
    position: reactive[Optional[Any]] = reactive(None, init=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._period = 14
        self._timeframe = "Daily"
        self._help_mode = False
        self._loading = False
        self._view_model = ATRViewModel()

    def compose(self) -> ComposeResult:
        """Compose the ATR panel layout."""
        with Vertical(id="atr-content"):
            yield Static("[bold #f59e0b]ATR Analysis[/]", id="atr-title")
            yield Static("[#8b949e]Use w/s to select a position[/]", id="atr-body")

    def watch_selected_symbol(self, symbol: Optional[str]) -> None:
        """Fetch ATR data when symbol changes."""
        if symbol:
            self._loading = True
            self._update_display()
            self._fetch_atr_data(symbol)
        else:
            self.atr_data = None
            self._update_display()

    def watch_atr_data(self, data: Optional[Any]) -> None:
        """Update display when ATR data changes."""
        self._loading = False
        self._update_display()

    @work(exclusive=True)
    async def _fetch_atr_data(self, symbol: str) -> None:
        """Fetch ATR data in background."""
        ta_service = getattr(self.app, "ta_service", None)
        if ta_service is None:
            self.atr_data = None
            self._loading = False
            self._update_display()
            return

        spot_price = self._resolve_spot_price(symbol)
        if spot_price is None or spot_price <= 0:
            self.atr_data = None
            self._loading = False
            self._update_display()
            return

        timeframe = self._timeframe_key()
        try:
            levels = await ta_service.get_atr_levels(
                symbol,
                spot_price=spot_price,
                period=self._period,
                timeframe=timeframe,
            )
            self.atr_data = levels
        except Exception as e:
            self.log.error(f"Failed to fetch ATR data for {symbol}: {e}")
            self.atr_data = None
        finally:
            self._loading = False
            self._update_display()

    def _update_display(self) -> None:
        """Update the panel display based on current state."""
        try:
            body = self.query_one("#atr-body", Static)
            title = self.query_one("#atr-title", Static)

            if self._help_mode:
                body.update(self._render_help())
                title.update("[bold #5fd7ff]ATR Strategy Guide[/]")
                return

            if self._loading:
                body.update("[italic #8b949e]Loading ATR data...[/]")
                title.update("[bold #f6d365]ATR Analysis[/]")
                return

            if self.atr_data is None:
                if self.selected_symbol:
                    body.update(f"[#8b949e]ATR data unavailable for {self.selected_symbol}[/]")
                else:
                    body.update("[#8b949e]Use w/s to select a position[/]")
                title.update("[bold #f59e0b]ATR Analysis[/]")
                return

            # Render full ATR display
            body.update(self._render_atr_display())
            title.update(self._render_title())

        except Exception as e:
            self.log.error(f"Failed to update ATR display: {e}")

    def _render_title(self) -> str:
        """Build ATR panel title."""
        if self.atr_data is None:
            return "[bold #f59e0b]ATR Analysis[/]"

        symbol = getattr(self.atr_data, "symbol", self.selected_symbol or "?")
        atr = getattr(self.atr_data, "atr_value", getattr(self.atr_data, "atr", 0))
        atr_pct = getattr(self.atr_data, "atr_percent", getattr(self.atr_data, "atr_pct", 0))

        return f"[bold #f59e0b]ATR: {symbol} | {self._timeframe} ATR({self._period})=${atr:.2f} ({atr_pct:.1f}%)[/]"

    def _render_atr_display(self) -> str:
        """Render full ATR analysis display matching Rich layout."""
        # Use ViewModel for all calculations
        levels = self._view_model.compute_levels(
            self.atr_data,
            self.position,
            self._timeframe,
            self._period,
        )

        if levels is None:
            if self.atr_data is not None:
                symbol = getattr(self.atr_data, "symbol", "?")
                return f"[#f6d365]Invalid price data for {symbol}[/]"
            return "[#8b949e]No ATR data[/]"

        return self._format_levels(levels)

    def _format_levels(self, lv: ATRLevels) -> str:
        """Format ATRLevels into Rich markup for display."""
        lines = []

        # Row 1: Header line
        qty_str = ""
        if lv.quantity > 0:
            qty_str = (
                f"  [#7ee787]{lv.quantity:,.0f} sh[/]  [#c9d1d9]Cost: ${lv.cost_basis:,.0f}[/]"
            )
        lines.append(
            f"[bold #5fd7ff]{lv.symbol}[/] [bold #c9d1d9]${lv.price:,.2f}[/]  "
            f"[bold #f6d365]ATR ${lv.atr:.2f}[/] [#f6d365]({lv.atr_pct:.1f}%)[/]  "
            f"[#d66efd][{lv.timeframe}][/]{qty_str}"
        )
        lines.append("")

        # Row 2: Price Levels
        lines.append("[bold #c9d1d9]── Price Levels " + "─" * 60 + "[/]")
        lines.append(
            f"[#ff6b6b]SL-2x:[/][#ff6b6b]{lv.sl_2x:>8.2f}[/]  "
            f"[#ff6b6b]SL-1.5x:[/][#ff6b6b]{lv.sl_1_5x:>7.2f}[/]  "
            f"[#5fd7ff]SMA21:[/][#5fd7ff]{lv.sma21:>7.2f}[/]  "
            f"[bold #c9d1d9]Entry:[/][bold #c9d1d9]{lv.price:>7.2f}[/]"
        )
        lines.append(
            f"[#7ee787]1R:[/][#7ee787]{lv.r1:>7.2f}[/]  "
            f"[#7ee787]2R:[/][#7ee787]{lv.r2:>7.2f}[/]  "
            f"[#f6d365]3R:[/][#f6d365]{lv.r3:>7.2f}[/]  "
            f"[#f6d365]4R:[/][#f6d365]{lv.r4:>7.2f}[/]  "
            f"[#5fd7ff]8R:[/][#5fd7ff]{lv.r8:>7.2f}[/]"
        )
        lines.append("")

        # Row 3: Two-column layout for R-TARGETS and TRAILING STOP
        col_width = 38

        left_lines = [
            "[bold #7ee787]R-TARGETS[/]",
            f"[#8b949e]Risk (1R) = ${lv.risk:.0f} (1.5xATR)[/]",
            f"[#ff6b6b]Stop[/]  ${lv.sl_1_5x:>6,.0f}  [#ff6b6b]-${lv.risk:>5,.0f}[/]  Exit all",
            f"[#7ee787]1R[/]    ${lv.r1:>6,.0f}  [#7ee787]+${lv.risk:>5,.0f}[/]  Trail starts",
            f"[#7ee787]2R[/]    ${lv.r2:>6,.0f}  [#7ee787]+${lv.risk*2:>5,.0f}[/]  Sell 33%",
            f"[#f6d365]3R[/]    ${lv.r3:>6,.0f}  [#f6d365]+${lv.risk*3:>5,.0f}[/]  Sell 33%",
            f"[#5fd7ff]8R[/]    ${lv.r8:>6,.0f}  [#5fd7ff]+${lv.risk*8:>5,.0f}[/]  Max target",
        ]

        right_lines = [
            "[bold #f6d365]TRAILING STOP[/]",
            f"[#8b949e]Trail = High - 2xATR (${lv.trail:.0f})[/]",
            f"Activates at 1R: ${lv.r1:>6,.0f}",
            f"${lv.r2:>6,.0f} -> stop ${lv.r2_stop:>6,.0f}",
            f"${lv.r4:>6,.0f} -> stop ${lv.r4_stop:>6,.0f}",
            "",
            "",
        ]

        for left, right in zip(left_lines, right_lines):
            lines.append(f"{left:<{col_width}}{right}")
        lines.append("")

        # Row 4: Exit Plan (if position)
        if lv.quantity > 0:
            lines.append("[bold #d66efd]EXIT PLAN[/]")
            lines.append(
                f"[#7ee787]Sell {lv.q3:,} @ 2R (${lv.r2:,.0f})[/]  "
                f"[#7ee787]+${lv.profit_2r:,.0f}[/]  [#7ee787]+{lv.pct_2r:.1f}%[/]"
            )
            lines.append(
                f"[#f6d365]Sell {lv.q3:,} @ 3R (${lv.r3:,.0f})[/]  "
                f"[#f6d365]+${lv.profit_3r:,.0f}[/]  [#f6d365]+{lv.pct_3r:.1f}%[/]"
            )
            lines.append(f"[#5fd7ff]Trail {lv.runners:,} (2xATR) -> runners 5R-8R[/]")
            lines.append("")

        # Row 5: Summary
        lines.append(
            f"[#8b949e]Risk: [/][#ff6b6b]${lv.risk:.0f}[/] [#8b949e]({lv.risk_pct:.1f}%)[/]  "
            f"[#8b949e]Max @8R: [/][#5fd7ff]${lv.max_gain:.0f}[/] [#8b949e]({lv.max_pct:.0f}%)[/]  "
            f"[#8b949e]R:R @8R = [/][bold #5fd7ff]8.0[/]"
        )
        lines.append("")

        # Row 6: Keyboard hints
        lines.append(
            f"[#5fd7ff][w/s][/] [#8b949e]Sel[/]  "
            f"[#5fd7ff][+/-][/] [#8b949e]Per({self._period})[/]  "
            f"[#5fd7ff][t][/] [#d66efd]{self._timeframe}[/]  "
            f"[#5fd7ff][h][/] [#8b949e]Help[/]  "
            f"[#5fd7ff][r][/] [#8b949e]Reset[/]"
        )

        return "\n".join(lines)

    def _render_help(self) -> str:
        """Render ATR strategy help."""
        lines = [
            "[bold #5fd7ff]R-MULTIPLE TARGETS[/]",
            "",
            "[#8b949e]Fixed exits based on risk:[/]",
            "  [#7ee787]1R[/] = +risk (get risk back)",
            "  [#7ee787]2R[/] = +2x risk",
            "  [#f6d365]3R[/] = +3x risk",
            "  [#5fd7ff]9R[/] = +9x risk",
            "",
            "[#8b949e]Best for: ranging, quick trades[/]",
            "",
            "[bold #f6d365]TRAILING STOP[/]",
            "",
            "[#8b949e]Dynamic exit follows price:[/]",
            "  [#f6d365]Stop = High - 2xATR[/]",
            "  [#8b949e]Activates after 1R[/]",
            "",
            "[#8b949e]Best for: trends, runners[/]",
            "",
            "[bold #d66efd]BEST PRACTICE: Scale Out + Trail[/]",
            "  [#7ee787]Sell 1/3 at 2R -> Lock 2x risk[/]",
            "  [#f6d365]Sell 1/3 at 3R -> Lock 3x risk[/]",
            "  [#5fd7ff]Trail 1/3 -> Catch 5R-11R runners[/]",
            "",
            "[#8b949e][h] to close[/]",
        ]
        return "\n".join(lines)

    def adjust_period(self, delta: int) -> None:
        """Adjust ATR period."""
        new_period = self._period + delta
        if 5 <= new_period <= 50:
            self._period = new_period
            if self.selected_symbol:
                self._fetch_atr_data(self.selected_symbol)

    def cycle_timeframe(self) -> None:
        """Cycle through timeframes."""
        timeframes = ["Daily", "4H", "1H"]
        idx = timeframes.index(self._timeframe) if self._timeframe in timeframes else 0
        self._timeframe = timeframes[(idx + 1) % len(timeframes)]
        if self.selected_symbol:
            self._fetch_atr_data(self.selected_symbol)

    def toggle_help(self) -> None:
        """Toggle help mode."""
        self._help_mode = not self._help_mode
        self._update_display()

    def reset(self) -> None:
        """Reset to defaults."""
        self._period = 14
        self._timeframe = "Daily"
        self._help_mode = False
        if self.selected_symbol:
            self._fetch_atr_data(self.selected_symbol)
        else:
            self._update_display()

    def _timeframe_key(self) -> str:
        """Map UI timeframe label to historical data timeframe."""
        mapping = {
            "Daily": "1d",
            "4H": "4h",
            "1H": "1h",
        }
        return mapping.get(self._timeframe, "1d")

    def _resolve_spot_price(self, symbol: str) -> Optional[float]:
        """Resolve spot price for ATR levels."""
        try:
            snapshot = getattr(self.app, "snapshot", None)
            position_risks = getattr(snapshot, "position_risks", []) if snapshot else []
            for pr in position_risks:
                pr_symbol = getattr(pr, "symbol", None)
                if pr_symbol == symbol and not getattr(pr, "expiry", None):
                    price = getattr(pr, "mark_price", None)
                    if price:
                        return price
        except Exception as e:
            self.log.error(f"Failed to resolve spot price for {symbol}: {e}")

        if self.position:
            price = getattr(self.position, "mark_price", None)
            if price:
                return price

        return None
