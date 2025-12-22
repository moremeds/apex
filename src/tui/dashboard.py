"""
Terminal Dashboard using rich library.

Real-time terminal UI for risk monitoring with tabbed views:
- Tab 1: Account Summary (consolidated positions by underlying, summary, alerts, health)
- Tab 2: Risk Signals (full screen risk signals)
- Tab 3: IB Positions (detailed Interactive Brokers positions with ATR levels)
- Tab 4: Futu Positions (detailed Futu positions with ATR levels)
- Tab 5: Lab (backtest strategies with parameters and performance results)

Keyboard shortcuts:
- 1: Account Summary view (default)
- 2: Risk Signals view
- 3: IB Positions view
- 4: Futu Positions view
- 5: Lab view
- q: Quit

IB/Futu Positions shortcuts:
- w: Select previous underlying (up)
- s: Select next underlying (down)
- +/-: Adjust ATR period
- r: Reset ATR period to default (14)

Lab view shortcuts:
- w: Select previous strategy (up)
- s: Select next strategy (down)
- Enter: Run backtest for selected strategy
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
import threading
import asyncio
import sys
import os
import select
import re

from rich.console import Console
from rich.live import Live

from .base import DashboardView
from .layouts import (
    create_layout_account_summary,
    create_layout_risk_signals,
    create_layout_broker_positions,
    create_layout_lab,
)
from .panels import (
    render_header,
    render_portfolio_summary,
    render_market_alerts,
    update_persistent_alerts,
    render_risk_signals_fullscreen,
    render_consolidated_positions,
    render_broker_positions,
    render_health,
    render_open_orders,
    render_atr_levels,
    render_atr_loading,
    render_atr_empty,
)
from .panels.positions import get_broker_underlyings
from ..domain.indicators.atr import ATRCache, ATRCalculator, ATRData
from ..services.bar_cache_service import BarPeriod
from ..services.ta_service import TAService
from ..services.historical_data_service import HistoricalDataService
from .panels.strategies import _update_lab_view

from ..models.risk_snapshot import RiskSnapshot
from ..models.risk_signal import RiskSignal
from src.domain.services.risk.rule_engine import LimitBreach
from ..infrastructure.monitoring import ComponentHealth
from src.utils.logging_setup import get_logger

# Import example strategies to register them in the StrategyRegistry
# This triggers the @register_strategy decorators
from src.domain.strategy import examples as _  # noqa: F401
from src.domain.strategy.registry import StrategyRegistry
from src.domain.backtest.backtest_result import BacktestResult

logger = get_logger(__name__)


class BacktestStatus(Enum):
    """Status of a backtest run."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BacktestConfig:
    """Configuration for running a backtest from the Lab."""
    symbols: List[str] = field(default_factory=lambda: ["AAPL"])
    start_date: date = field(default_factory=lambda: date.today() - timedelta(days=365))
    end_date: date = field(default_factory=lambda: date.today() - timedelta(days=1))
    initial_capital: float = 100_000.0
    data_source: str = "historical"  # historical (bar cache) | csv


@dataclass
class BacktestState:
    """State for Lab view backtest functionality."""
    selected_index: int = 0
    status: BacktestStatus = BacktestStatus.IDLE
    running_strategy: Optional[str] = None
    config: BacktestConfig = field(default_factory=BacktestConfig)
    results: Dict[str, BacktestResult] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)


class TerminalDashboard:
    """
    Terminal dashboard using rich library.

    Provides real-time display of:
    - Portfolio metrics
    - Limit breaches
    - Health status
    - Position details (if enabled)
    """

    def __init__(
        self,
        config: dict,
        env: str = "dev",
    ):
        """
        Initialize dashboard.

        Args:
            config: Dashboard configuration dict.
            env: Environment name (dev, demo, prod).
        """
        self.config = config
        self.env = env
        self.show_positions = config.get("show_positions", True)
        self.console = Console()
        self.live: Optional[Live] = None

        # View state
        self._current_view = DashboardView.ACCOUNT_SUMMARY
        self._layout_account_summary = create_layout_account_summary()
        self._layout_risk_signals = create_layout_risk_signals()
        self._layout_broker_positions = create_layout_broker_positions()
        self._layout_lab = create_layout_lab()
        self.layout = self._layout_account_summary

        # Keyboard input handling
        self._input_thread: Optional[threading.Thread] = None
        self._stop_input = threading.Event()
        self._quit_requested = False
        self._input_buffer = ""

        # Persistent alert tracking
        self._persistent_alerts: Dict[str, Dict] = {}
        self._persistent_risk_signals: Dict[str, Dict] = {}
        self._alert_retention_seconds = config.get("alert_retention_seconds", 300)

        # Store latest data for view switching
        self._last_snapshot: Optional[RiskSnapshot] = None
        self._last_breaches: List = []
        self._last_health: List[ComponentHealth] = []
        self._last_market_alerts: List[Dict[str, Any]] = []

        # Lab view backtest state
        self._backtest_state = BacktestState()
        self._backtest_thread: Optional[threading.Thread] = None

        # Broker positions selection state (for ATR display)
        self._ib_selected_index: Optional[int] = None
        self._futu_selected_index: Optional[int] = None
        self._atr_cache = ATRCache()
        self._atr_period: int = 14  # Default ATR period
        self._atr_timeframe: str = "Daily"  # Daily, 4H, 1H
        self._atr_help_mode: bool = False  # Toggle help overlay

        # ATR fetching state
        self._atr_fetch_inflight: set[str] = set()
        self._atr_fetch_lock = threading.Lock()

        # TA Service for ATR calculation (injected after startup via set_ta_service)
        self._ta_service: Optional[TAService] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def set_ta_service(self, ta_service: TAService, event_loop: asyncio.AbstractEventLoop) -> None:
        """
        Inject TA service for ATR calculation.

        Called after startup when historical connection is established.

        Args:
            ta_service: TAService instance for ATR calculation.
            event_loop: Main event loop for scheduling async operations.
        """
        self._ta_service = ta_service
        self._event_loop = event_loop
        logger.info("TAService injected into dashboard")

    def start(self) -> None:
        """Start live dashboard (blocking)."""
        self.live = Live(self.layout, console=self.console, refresh_per_second=2)
        self.live.start()
        self._start_keyboard_listener()
        logger.info("Terminal dashboard started")

    def stop(self) -> None:
        """Stop live dashboard."""
        self._stop_keyboard_listener()
        if self.live:
            self.live.stop()
            logger.info("Terminal dashboard stopped")

    def _start_keyboard_listener(self) -> None:
        """Start background thread for keyboard input."""
        self._stop_input.clear()
        self._input_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._input_thread.start()
        logger.debug("Keyboard listener started")

    def _stop_keyboard_listener(self) -> None:
        """Stop keyboard input thread."""
        self._stop_input.set()
        if self._input_thread and self._input_thread.is_alive():
            self._input_thread.join(timeout=1.0)
        logger.debug("Keyboard listener stopped")

    def _keyboard_listener(self) -> None:
        """Background thread that listens for keyboard input."""
        import termios
        import tty

        try:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
        except (termios.error, ValueError) as e:
            logger.warning(f"Cannot get terminal settings, keyboard shortcuts disabled: {e}")
            return

        try:
            tty.setcbreak(fd)
            logger.debug("Keyboard listener started in cbreak mode")

            while not self._stop_input.is_set():
                try:
                    # Use file descriptor directly with select for reliability
                    if select.select([fd], [], [], 0.1)[0]:
                        # Use os.read() directly to bypass Python's buffering
                        raw = os.read(fd, 32)
                        if raw:
                            data = raw.decode('utf-8', errors='replace')
                            for key in self._decode_keys(data):
                                self._handle_keypress(key)
                except OSError as e:
                    # Handle cases where stdin is closed or unavailable
                    if e.errno == 9:  # Bad file descriptor
                        logger.warning("stdin closed, stopping keyboard listener")
                        break
                    logger.error(f"Keyboard read error: {e}")
                except Exception as e:
                    logger.error(f"Keyboard read error: {e}")
        except Exception as e:
            logger.error(f"Keyboard listener error: {e}")
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                pass

    def _decode_keys(self, data: str) -> List[str]:
        """Decode raw terminal input into logical key tokens.

        Handles:
        - Printable characters (returned as-is)
        - Arrow keys (CSI / SS3): UP/DOWN/LEFT/RIGHT
        """
        if not data:
            return []

        self._input_buffer += data
        keys: List[str] = []

        def map_arrow(code: str) -> Optional[str]:
            return {
                "A": "UP",
                "B": "DOWN",
                "C": "RIGHT",
                "D": "LEFT",
            }.get(code)

        while self._input_buffer:
            if self._input_buffer[0] != "\x1b":
                keys.append(self._input_buffer[0])
                self._input_buffer = self._input_buffer[1:]
                continue

            # Escape sequence. Keep buffering if incomplete.
            if len(self._input_buffer) < 2:
                break

            prefix = self._input_buffer[1]

            # SS3: ESC O A/B/C/D (common in application cursor mode)
            if prefix == "O":
                if len(self._input_buffer) < 3:
                    break
                code = self._input_buffer[2]
                arrow = map_arrow(code)
                if arrow:
                    keys.append(arrow)
                self._input_buffer = self._input_buffer[3:]
                continue

            # CSI: ESC [ ... <final>
            if prefix == "[":
                # Wait until we have the final byte (letter).
                match = re.match(r"^\x1b\[[0-9;]*([A-Za-z])", self._input_buffer)
                if match is None:
                    # Incomplete sequence: keep buffering.
                    if not any(ch.isalpha() for ch in self._input_buffer[2:]):
                        break
                    # Unrecognized CSI with a final byte: consume until final byte and ignore.
                    for idx in range(2, len(self._input_buffer)):
                        if self._input_buffer[idx].isalpha():
                            self._input_buffer = self._input_buffer[idx + 1 :]
                            break
                    continue

                code = match.group(1)
                arrow = map_arrow(code)
                if arrow:
                    keys.append(arrow)
                self._input_buffer = self._input_buffer[match.end() :]
                continue

            # Unknown escape sequence: consume ESC and move on.
            self._input_buffer = self._input_buffer[1:]

        return keys

    def _handle_keypress(self, char: str) -> None:
        """Handle a single keypress."""
        logger.debug(f"Keypress received: {repr(char)} (view={self._current_view.value})")

        # Tab switching (works in all views)
        if char == '1':
            self._switch_view(DashboardView.ACCOUNT_SUMMARY)
        elif char == '2':
            self._switch_view(DashboardView.RISK_SIGNALS)
        elif char == '3':
            self._switch_view(DashboardView.IB_POSITIONS)
        elif char == '4':
            self._switch_view(DashboardView.FUTU_POSITIONS)
        elif char == '5':
            self._switch_view(DashboardView.LAB)
        elif char in ('q', 'Q', '\x03'):
            self._quit_requested = True
            logger.info("Quit requested via keyboard")
        # Broker positions view shortcuts (IB / Futu)
        elif self._current_view in (DashboardView.IB_POSITIONS, DashboardView.FUTU_POSITIONS):
            self._handle_broker_positions_keypress(char)
        # Lab view specific shortcuts
        elif self._current_view == DashboardView.LAB:
            self._handle_lab_keypress(char)

    def _handle_lab_keypress(self, char: str) -> None:
        """Handle keypresses specific to Lab view."""
        strategies = sorted(StrategyRegistry.list_strategies())
        if not strategies:
            logger.warning("No strategies registered, Lab keypresses ignored")
            return
        if self._backtest_state.selected_index >= len(strategies):
            self._backtest_state.selected_index = max(0, len(strategies) - 1)

        logger.info(f"Lab keypress: {repr(char)}, selected={self._backtest_state.selected_index}")

        # s/Down: next strategy
        if char in ('s', 'DOWN'):
            self._backtest_state.selected_index = min(
                self._backtest_state.selected_index + 1,
                len(strategies) - 1
            )
            logger.info(f"Selected strategy index: {self._backtest_state.selected_index}")
            self._refresh_lab_view()
        # w/Up: previous strategy
        elif char in ('w', 'UP'):
            self._backtest_state.selected_index = max(
                self._backtest_state.selected_index - 1,
                0
            )
            logger.info(f"Selected strategy index: {self._backtest_state.selected_index}")
            self._refresh_lab_view()
        # Enter: run backtest (check multiple Enter key representations)
        elif char in ('\r', '\n', '\x0d', '\x0a'):
            logger.info(f"Enter pressed! status={self._backtest_state.status}, strategies={len(strategies)}")
            if self._backtest_state.status != BacktestStatus.RUNNING:
                selected_strategy = strategies[self._backtest_state.selected_index]
                logger.info(f"Starting backtest for: {selected_strategy}")
                self._run_backtest(selected_strategy)
            else:
                logger.warning("Backtest already running, ignoring Enter")

    def _handle_broker_positions_keypress(self, char: str) -> None:
        """Handle keypresses specific to IB/Futu positions views."""
        if not self._last_snapshot:
            return

        # Determine which broker and selection index to use
        if self._current_view == DashboardView.IB_POSITIONS:
            broker = "IB"
            selected_index = self._ib_selected_index
        else:
            broker = "FUTU"
            selected_index = self._futu_selected_index

        # Get list of underlyings for this broker
        underlyings = get_broker_underlyings(self._last_snapshot.position_risks, broker)
        if not underlyings:
            return

        logger.debug(f"Broker keypress: {repr(char)}, broker={broker}, selected={selected_index}, count={len(underlyings)}")

        # Handle first selection - if None, first keypress selects first item
        if selected_index is None:
            if char in ('s', 'DOWN', 'w', 'UP'):
                new_index = 0  # First selection starts at index 0
            else:
                return  # Other keys don't work until something is selected
        else:
            new_index = selected_index
            # s/Down: select next underlying (go down)
            if char in ('s', 'DOWN'):
                new_index = min(selected_index + 1, len(underlyings) - 1)
            # w/Up: select previous underlying (go up)
            elif char in ('w', 'UP'):
                new_index = max(selected_index - 1, 0)
            # +: increase ATR period
            elif char == '+':
                if self._atr_period < 50:
                    self._atr_period += 1
                    self._refresh_broker_positions_view()
                return
            # -: decrease ATR period
            elif char == '-':
                if self._atr_period > 5:
                    self._atr_period -= 1
                    self._refresh_broker_positions_view()
                return
            # t: cycle timeframe (Daily -> 4H -> 1H -> Daily)
            elif char == 't':
                timeframes = ["Daily", "4H", "1H"]
                idx = timeframes.index(self._atr_timeframe)
                self._atr_timeframe = timeframes[(idx + 1) % len(timeframes)]
                self._refresh_broker_positions_view()
                return
            # h: toggle help overlay
            elif char == 'h':
                self._atr_help_mode = not self._atr_help_mode
                self._refresh_broker_positions_view()
                return
            # r: reset ATR period to default
            elif char == 'r':
                self._atr_period = 14
                self._atr_timeframe = "Daily"
                self._refresh_broker_positions_view()
                return

        # Update selection if changed
        if new_index != selected_index:
            if self._current_view == DashboardView.IB_POSITIONS:
                self._ib_selected_index = new_index
            else:
                self._futu_selected_index = new_index
            logger.info(f"Selected {broker} underlying: {underlyings[new_index]} (index {new_index})")
            # Only refresh if selection actually changed
            self._refresh_broker_positions_view()

    def _refresh_broker_positions_view(self) -> None:
        """Refresh broker positions view after state change."""
        if self._current_view not in (DashboardView.IB_POSITIONS, DashboardView.FUTU_POSITIONS):
            return
        if self._last_snapshot:
            broker = "IB" if self._current_view == DashboardView.IB_POSITIONS else "FUTU"
            self._update_broker_positions_view(self._last_snapshot, broker)
            if self.live:
                self.live.refresh()

    def _refresh_lab_view(self) -> None:
        """Refresh Lab view after state change."""
        if self._current_view == DashboardView.LAB:
            _update_lab_view(self._layout_lab, self._backtest_state, self.env, self._current_view, self._last_health)
            if self.live:
                self.live.refresh()

    def _switch_view(self, new_view: DashboardView) -> None:
        """Switch to a different dashboard view."""
        if new_view == self._current_view:
            return

        self._current_view = new_view
        logger.info(f"Switched to {new_view.value} view")

        if new_view == DashboardView.ACCOUNT_SUMMARY:
            self.layout = self._layout_account_summary
        elif new_view == DashboardView.RISK_SIGNALS:
            self.layout = self._layout_risk_signals
        elif new_view in (DashboardView.IB_POSITIONS, DashboardView.FUTU_POSITIONS):
            self.layout = create_layout_broker_positions()
        elif new_view == DashboardView.LAB:
            self.layout = self._layout_lab

        if self.live:
            self.live.update(self.layout)

        # Update the view - Lab view doesn't need snapshot
        if new_view == DashboardView.LAB:
            _update_lab_view(self._layout_lab, self._backtest_state, self.env, self._current_view, self._last_health)
        elif self._last_snapshot:
            self.update(
                self._last_snapshot,
                self._last_breaches,
                self._last_health,
                self._last_market_alerts,
            )

    @property
    def quit_requested(self) -> bool:
        """Check if user requested quit via keyboard."""
        return self._quit_requested

    def update(
        self,
        snapshot: RiskSnapshot,
        breaches: List[LimitBreach] | List[RiskSignal],
        health: List[ComponentHealth],
        market_alerts: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Update dashboard with latest data.

        Args:
            snapshot: Latest risk snapshot (contains pre-calculated position_risks).
            breaches: List of limit breaches (legacy) or risk signals (new).
            health: List of component health statuses.
            market_alerts: List of market-wide alerts (VIX spikes, etc).
        """
        # Cache data for view switching
        self._last_snapshot = snapshot
        self._last_breaches = breaches
        self._last_health = health
        self._last_market_alerts = market_alerts or []

        if self._current_view == DashboardView.ACCOUNT_SUMMARY:
            self._update_account_summary_view(snapshot, breaches, health, market_alerts or [])
        elif self._current_view == DashboardView.RISK_SIGNALS:
            self._update_risk_signals_view(snapshot, breaches)
        elif self._current_view == DashboardView.IB_POSITIONS:
            self._update_broker_positions_view(snapshot, "IB")
        elif self._current_view == DashboardView.FUTU_POSITIONS:
            self._update_broker_positions_view(snapshot, "FUTU")
        elif self._current_view == DashboardView.LAB:
            _update_lab_view(self._layout_lab, self._backtest_state, self.env, self._current_view, self._last_health)

    def _update_account_summary_view(
        self,
        snapshot: RiskSnapshot,
        breaches: List[LimitBreach] | List[RiskSignal],
        health: List[ComponentHealth],
        market_alerts: List[Dict[str, Any]],
    ) -> None:
        """Update the account summary view layout (Tab 1)."""
        layout = self._layout_account_summary

        # Update persistent alerts and get display list
        display_alerts = update_persistent_alerts(
            market_alerts,
            self._persistent_alerts,
            self._alert_retention_seconds,
        )

        layout["header"].update(render_header(self.env, self._current_view))
        layout["body"]["positions"].update(
            render_consolidated_positions(snapshot.position_risks, snapshot)
        )
        layout["body"]["right"]["summary"].update(render_portfolio_summary(snapshot))
        layout["body"]["right"]["alerts"].update(render_market_alerts(display_alerts))
        layout["footer"].update(render_health(health))

    def _update_risk_signals_view(
        self,
        snapshot: RiskSnapshot,
        breaches: List[LimitBreach] | List[RiskSignal],
    ) -> None:
        """Update the risk signals view layout (Tab 2)."""
        layout = self._layout_risk_signals
        layout["header"].update(render_header(self.env, self._current_view))
        layout["signals"].update(
            render_risk_signals_fullscreen(
                breaches,
                snapshot,
                self._persistent_risk_signals,
                self._alert_retention_seconds,
            )
        )

    def _update_broker_positions_view(
        self,
        snapshot: RiskSnapshot,
        broker: str,
    ) -> None:
        """Update the broker positions view layout (Tab 3 & 4)."""
        layout = self.layout

        # Get selection index for this broker
        if broker == "IB":
            selected_index = self._ib_selected_index
        else:
            selected_index = self._futu_selected_index

        layout["header"].update(render_header(self.env, self._current_view))
        layout["body"]["positions"].update(
            render_broker_positions(snapshot.position_risks, broker, selected_index)
        )

        # Render ATR levels for selected position
        atr_panel = self._get_atr_panel_for_selection(snapshot, broker, selected_index)
        layout["body"]["history_panel"]["atr_levels"].update(atr_panel)

        layout["body"]["history_panel"]["open_orders"].update(
            render_open_orders(broker)
        )

    def _get_atr_panel_for_selection(
        self,
        snapshot: RiskSnapshot,
        broker: str,
        selected_index: Optional[int],
    ):
        """Get ATR panel for the currently selected underlying."""
        from ..models.position import PositionSource
        broker_source = PositionSource.IB if broker == "IB" else PositionSource.FUTU

        if selected_index is None:
            return render_atr_empty("Use w/s to select a position")

        underlyings = get_broker_underlyings(snapshot.position_risks, broker)
        if not underlyings or selected_index >= len(underlyings):
            return render_atr_empty("No positions to select")

        symbol = underlyings[selected_index]
        timeframe = {"Daily": "1d", "4H": "4h", "1H": "1h"}.get(self._atr_timeframe, "1d")
        cache_key = f"{symbol}:{timeframe}"

        # Find the STOCK position for spot price (not options)
        # Stock positions have no expiry
        stock_position = None
        any_position = None
        for pr in snapshot.position_risks:
            if pr.underlying != symbol:
                continue
            # Check if this position belongs to the current broker
            pos = pr.position
            is_broker_match = (
                (pos.all_sources and broker_source in pos.all_sources) or
                pos.source == broker_source
            )
            if not is_broker_match:
                continue

            if any_position is None:
                any_position = pr
            # Stock position has no expiry
            if not pr.expiry and pr.mark_price:
                stock_position = pr
                break

        # Use stock position for spot price, fall back to any position for display
        position = stock_position or any_position

        # Check if we have cached ATR data
        cache_entry = self._atr_cache.get_entry(cache_key)
        atr_data = cache_entry.data if cache_entry else None
        optimization = cache_entry.optimization_result if cache_entry else None
        spot_price = stock_position.mark_price if stock_position else None

        if cache_entry and atr_data and atr_data.period != self._atr_period:
            if len(cache_entry.bars) >= self._atr_period + 1:
                atr_data = ATRCalculator.from_bars(
                    symbol=symbol,
                    bars=cache_entry.bars,
                    period=self._atr_period,
                    entry_price=spot_price,
                )
                if atr_data:
                    self._atr_cache.set(cache_key, atr_data, cache_entry.bars, optimization)
            else:
                atr_data = None

        if atr_data is None:
            self._queue_atr_fetch(cache_key, symbol, timeframe, spot_price)
            return render_atr_loading()

        # If help mode is active, show help overlay instead
        if self._atr_help_mode:
            from .panels.atr_levels import render_atr_help
            return render_atr_help(atr_data, position)

        return render_atr_levels(
            atr_data=atr_data,
            position=position,
            optimization=optimization,
            loading=False,
            current_period=self._atr_period,
            timeframe=self._atr_timeframe,
        )

    def _queue_atr_fetch(
        self,
        cache_key: str,
        symbol: str,
        timeframe: str,
        spot_price: Optional[float],
    ) -> None:
        """Queue ATR fetch request using TAService.

        Schedules async ATR fetch on the main event loop.
        Results are stored in ATR cache and UI is refreshed.
        """
        # Check if TAService is available
        if not self._ta_service or not self._event_loop:
            logger.debug(f"ATR fetch skipped for {symbol} - TAService not available yet")
            return

        # Avoid duplicate fetches
        with self._atr_fetch_lock:
            if cache_key in self._atr_fetch_inflight:
                return
            self._atr_fetch_inflight.add(cache_key)

        async def fetch_atr():
            try:
                # Get ATR levels from TAService
                atr_levels = await self._ta_service.get_atr_levels(
                    symbol=symbol,
                    spot_price=spot_price or 0.0,
                    period=self._atr_period,
                    timeframe=timeframe,
                )

                if atr_levels:
                    # Convert to ATRData format for cache
                    atr_data = ATRData(
                        symbol=symbol,
                        current_price=spot_price or atr_levels.spot,
                        atr_value=atr_levels.atr,
                        atr_percent=atr_levels.atr_pct,
                        period=atr_levels.period,
                        # Calculate stop/profit levels from ATR
                        stop_loss_1x=atr_levels.level_1_dn,
                        stop_loss_1_5x=atr_levels.spot - (atr_levels.atr * 1.5),
                        stop_loss_2x=atr_levels.level_2_dn,
                        take_profit_7x=atr_levels.spot + (atr_levels.atr * 7),
                        take_profit_8x=atr_levels.spot + (atr_levels.atr * 8),
                        take_profit_9x=atr_levels.spot + (atr_levels.atr * 9),
                        take_profit_10x=atr_levels.spot + (atr_levels.atr * 10),
                        take_profit_11x=atr_levels.spot + (atr_levels.atr * 11),
                    )
                    # Store in cache
                    self._atr_cache.set(cache_key, atr_data, bars=[], optimization=None)
                    logger.debug(f"ATR fetched for {symbol}: {atr_levels.atr:.2f}")

                    # Trigger UI refresh
                    self._refresh_broker_positions_view()

            except Exception as e:
                logger.warning(f"ATR fetch failed for {symbol}: {e}")
            finally:
                with self._atr_fetch_lock:
                    self._atr_fetch_inflight.discard(cache_key)

        # Schedule async fetch on main event loop
        try:
            asyncio.run_coroutine_threadsafe(fetch_atr(), self._event_loop)
        except Exception as e:
            logger.error(f"Failed to schedule ATR fetch for {symbol}: {e}")
            with self._atr_fetch_lock:
                self._atr_fetch_inflight.discard(cache_key)

    def _run_backtest(self, strategy_name: str) -> None:
        """Run backtest for a strategy in background thread."""
        if self._backtest_state.status == BacktestStatus.RUNNING:
            logger.warning("Backtest already running")
            return

        # Clear previous failure for this strategy.
        self._backtest_state.failures.pop(strategy_name, None)
        self._backtest_state.status = BacktestStatus.RUNNING
        self._backtest_state.running_strategy = strategy_name
        self._refresh_lab_view()

        def run_in_thread():
            try:
                # Use asyncio.run() which properly manages the event loop lifecycle
                # This ensures ib_async creates connections on the correct loop
                result = asyncio.run(self._execute_backtest(strategy_name))
                self._backtest_state.results[strategy_name] = result
                self._backtest_state.status = BacktestStatus.COMPLETED
                logger.info(f"Backtest completed for {strategy_name}: {result.total_return_pct:.2f}%")
            except Exception as e:
                self._backtest_state.status = BacktestStatus.FAILED
                self._backtest_state.failures[strategy_name] = str(e)
                logger.error(f"Backtest failed for {strategy_name}: {e}")
            finally:
                self._backtest_state.running_strategy = None
                self._refresh_lab_view()

        self._backtest_thread = threading.Thread(target=run_in_thread, daemon=True)
        self._backtest_thread.start()
        logger.info(f"Started backtest for {strategy_name}")

    async def _execute_backtest(self, strategy_name: str) -> BacktestResult:
        """Execute the actual backtest using custom Apex engine with ib_async."""
        from src.runners.backtest_runner import BacktestRunner

        config = self._backtest_state.config
        runner = BacktestRunner(
            strategy_name=strategy_name,
            symbols=config.symbols,
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            data_source=config.data_source,
        )
        return await runner.run()
