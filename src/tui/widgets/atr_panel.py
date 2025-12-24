"""
ATR analysis panel widget.

Displays ATR-based stop loss and take profit levels:
- Header with symbol, price, ATR info
- Horizontal price bar (SL-2x to 8R)
- R-Targets and Trailing Stop sections
- Exit Plan (if position has shares)
- Summary and keyboard hints
"""

from __future__ import annotations

from typing import Any, Optional

from textual.widget import Widget
from textual.reactive import reactive
from textual.widgets import Static
from textual.containers import Vertical, Horizontal, Grid
from textual.app import ComposeResult
from textual import work


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

    def compose(self) -> ComposeResult:
        """Compose the ATR panel layout."""
        with Vertical(id="atr-content"):
            yield Static("[bold]ATR Analysis[/]", id="atr-title")
            yield Static("[dim]Use w/s to select a position[/]", id="atr-body")

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
        except Exception:
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
                title.update("[bold cyan]ATR Strategy Guide[/]")
                return

            if self._loading:
                body.update("[dim italic]Loading ATR data...[/]")
                title.update("[bold yellow]ATR Analysis[/]")
                return

            if self.atr_data is None:
                if self.selected_symbol:
                    body.update(f"[dim]ATR data unavailable for {self.selected_symbol}[/]")
                else:
                    body.update("[dim]Use w/s to select a position[/]")
                title.update("[bold]ATR Analysis[/]")
                return

            # Render full ATR display
            body.update(self._render_atr_display())
            title.update(self._render_title())

        except Exception:
            pass

    def _render_title(self) -> str:
        """Build ATR panel title."""
        if self.atr_data is None:
            return "[bold]ATR Analysis[/]"

        symbol = getattr(self.atr_data, "symbol", self.selected_symbol or "?")
        atr = getattr(self.atr_data, "atr_value", getattr(self.atr_data, "atr", 0))
        atr_pct = getattr(self.atr_data, "atr_percent", getattr(self.atr_data, "atr_pct", 0))

        return f"[bold blue]ATR: {symbol} | {self._timeframe} ATR({self._period})=${atr:.2f} ({atr_pct:.1f}%)[/]"

    def _render_atr_display(self) -> str:
        """Render full ATR analysis display."""
        if self.atr_data is None:
            return "[dim]No ATR data[/]"

        atr = getattr(self.atr_data, "atr_value", getattr(self.atr_data, "atr", 0))
        price = getattr(self.atr_data, "current_price", getattr(self.atr_data, "spot", 0))
        symbol = getattr(self.atr_data, "symbol", "?")

        if price <= 0:
            return f"[yellow]Invalid price data for {symbol}[/]"

        # Calculate values
        risk = atr * 1.5
        sma21 = price * 0.97
        sl_2x = price - (atr * 2)
        sl_1_5x = price - (atr * 1.5)
        trail = atr * 2

        # R-levels
        r_levels = {i: price + (risk * i) for i in range(1, 12)}

        # Position info
        qty = 0
        cost_basis = price
        if self.position:
            qty = getattr(self.position, "quantity", 0)
            if hasattr(self.position, "position"):
                avg_price = getattr(self.position.position, "avg_price", None)
                if avg_price:
                    cost_basis = avg_price

        lines = []

        # Row 1: Header
        lines.append(f"[bold cyan]{symbol}[/]   [bold white]${price:,.2f}[/]  [bold yellow]ATR ${atr:.2f}[/]  [yellow]({getattr(self.atr_data, 'atr_percent', 0):.1f}%)[/]  [magenta][{self._timeframe}][/]")
        if qty:
            lines.append(f"[green]{qty:,.0f} sh[/]  [white]Cost: ${cost_basis:,.0f}[/]")
        lines.append("")

        # Row 2: Price bar (simplified horizontal display)
        lines.append("[bold white]── Price Levels ──[/]")
        lines.append(f"[red]SL-2x: ${sl_2x:.2f}[/]  [red]SL-1.5x: ${sl_1_5x:.2f}[/]  [blue]SMA21: ${sma21:.2f}[/]  [bold white]Entry: ${price:.2f}[/]")
        lines.append(f"[green]1R: ${r_levels[1]:.2f}[/]  [green]2R: ${r_levels[2]:.2f}[/]  [yellow]3R: ${r_levels[3]:.2f}[/]  [yellow]4R: ${r_levels[4]:.2f}[/]  [cyan]8R: ${r_levels[8]:.2f}[/]")
        lines.append("")

        # Row 3: R-Targets
        lines.append("[bold green]R-TARGETS[/]")
        lines.append(f"[dim]Risk (1R) = ${risk:.0f} (1.5xATR)[/]")
        lines.append(f"[red]Stop[/]    ${sl_1_5x:,.0f}   [red]-${risk:.0f}[/]   Exit all")
        lines.append(f"[green]1R[/]      ${r_levels[1]:,.0f}   [green]+${risk:.0f}[/]   Trail starts")
        lines.append(f"[green]2R[/]      ${r_levels[2]:,.0f}   [green]+${risk*2:.0f}[/]   Sell 33%")
        lines.append(f"[yellow]3R[/]      ${r_levels[3]:,.0f}   [yellow]+${risk*3:.0f}[/]   Sell 33%")
        lines.append(f"[cyan]8R[/]      ${r_levels[8]:,.0f}   [cyan]+${risk*8:.0f}[/]   Max target")
        lines.append("")

        # Row 4: Trailing Stop
        lines.append("[bold yellow]TRAILING STOP[/]")
        lines.append(f"[dim]Trail = High - 2xATR (${trail:.0f})[/]")
        lines.append(f"Activates at 1R: ${r_levels[1]:,.0f}")
        lines.append(f"[dim]${r_levels[2]:,.0f} -> stop ${r_levels[2]-trail:,.0f}[/]")
        lines.append(f"[dim]${r_levels[4]:,.0f} -> stop ${r_levels[4]-trail:,.0f}[/]")
        lines.append("")

        # Row 5: Exit Plan (if position)
        if qty > 0:
            lines.append("[bold magenta]EXIT PLAN[/]")
            q3 = int(qty // 3)
            profit2 = (r_levels[2] - cost_basis) * q3
            profit3 = (r_levels[3] - cost_basis) * q3
            pct2 = (r_levels[2] - cost_basis) / cost_basis * 100 if cost_basis else 0
            pct3 = (r_levels[3] - cost_basis) / cost_basis * 100 if cost_basis else 0
            lines.append(f"[green]Sell {q3:,} @ 2R (${r_levels[2]:,.0f})  +${profit2:,.0f}  +{pct2:.1f}%[/]")
            lines.append(f"[yellow]Sell {q3:,} @ 3R (${r_levels[3]:,.0f})  +${profit3:,.0f}  +{pct3:.1f}%[/]")
            lines.append(f"[cyan]Trail {int(qty-2*q3):,} (2xATR) -> runners 5R-8R[/]")
            lines.append("")

        # Row 6: Summary
        max_return = (r_levels[8] - price) / price * 100 if price else 0
        lines.append(f"[dim]Risk: [/][red]${risk:.0f}[/] [dim]({risk/price*100:.1f}%)[/]  [dim]Max @8R: [/][cyan]${r_levels[8]-price:.0f}[/] [dim]({max_return:.0f}%)[/]  [dim]R:R @8R = [/][bold cyan]8.0[/]")
        lines.append("")

        # Row 7: Keyboard hints
        lines.append(f"[cyan][w/s][/][dim] Sel [/][cyan][+/-][/][dim] Per({self._period}) [/][cyan][t][/][magenta] {self._timeframe} [/][cyan][h][/][dim] Help [/][cyan][r][/][dim] Reset[/]")

        return "\n".join(lines)

    def _render_help(self) -> str:
        """Render ATR strategy help."""
        lines = [
            "[bold cyan]R-MULTIPLE TARGETS[/]",
            "",
            "[dim]Fixed exits based on risk:[/]",
            "  [green]1R[/] = +risk (get risk back)",
            "  [green]2R[/] = +2x risk",
            "  [yellow]3R[/] = +3x risk",
            "  [cyan]9R[/] = +9x risk",
            "",
            "[dim]Best for: ranging, quick trades[/]",
            "",
            "[bold yellow]TRAILING STOP[/]",
            "",
            "[dim]Dynamic exit follows price:[/]",
            "  [yellow]Stop = High - 2xATR[/]",
            "  [dim]Activates after 1R[/]",
            "",
            "[dim]Best for: trends, runners[/]",
            "",
            "[bold magenta]BEST PRACTICE: Scale Out + Trail[/]",
            "  [green]Sell 1/3 at 2R -> Lock 2x risk[/]",
            "  [yellow]Sell 1/3 at 3R -> Lock 3x risk[/]",
            "  [cyan]Trail 1/3 -> Catch 5R-11R runners[/]",
            "",
            "[dim cyan][h] to close[/]",
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
        except Exception:
            pass

        if self.position:
            price = getattr(self.position, "mark_price", None)
            if price:
                return price

        return None
