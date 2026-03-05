"""WebBridge — bridges domain events to WebSocket hub, DuckDB, and Advisor.

Thin adapter layer: subscribes to BAR_CLOSE, INDICATOR_UPDATE, TRADING_SIGNAL
on the shared event bus and forwards to server-layer concerns.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Any, Coroutine, Dict, List, Optional, Union

from src.domain.events.domain_events import (
    BarCloseEvent,
    IndicatorUpdateEvent,
    TradingSignalEvent,
)
from src.domain.events.event_types import EventType
from src.domain.events.priority_event_bus import PriorityEventBus
from src.domain.interfaces.event_bus import EventBus
from src.domain.signals.signal_engine import SignalEngine
from src.server.routes.advisor import _serialize as _serialize_advice
from src.server.signal_helpers import (
    DIRECTION_MAP,
    STRATEGY_INDICATORS,
    map_regime_to_flex,
    tick_to_dict,
)

logger = logging.getLogger(__name__)


class WebBridge:
    """Bridges domain events → WebSocket hub + DuckDB persistence + Advisor.

    Subscribes to the shared event bus and handles:
    - BAR_CLOSE → WS broadcast, DuckDB score snapshots, advisor trigger
    - INDICATOR_UPDATE → WS broadcast (indicator + strategy state)
    - TRADING_SIGNAL → WS broadcast, DuckDB persistence, signal buffer
    - Portfolio events (SNAPSHOT_READY, POSITION_DELTA, ACCOUNT_UPDATED) → WS
    """

    def __init__(
        self,
        event_bus: Union[EventBus, PriorityEventBus],
        signal_engine: SignalEngine,
        hub: Any,
        persistence: Any,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self._event_bus = event_bus
        self._signal_engine = signal_engine
        self._hub = hub
        self._persistence = persistence
        self._loop = loop
        self._started = False

        # Signal buffer for advisor (maps symbol -> list of recent signal dicts)
        self._signal_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._signal_buffer_lock = RLock()

        # Portfolio event handlers — stored for unsubscribe on stop()
        self._portfolio_handlers: list[tuple[EventType, Any, Any]] = []

        # Advisor service (set after bootstrap)
        self._advisor_service: Any = None
        self._advisor_executor = ThreadPoolExecutor(max_workers=1)
        self._advisor_last_compute: float = 0.0
        self._advisor_debounce_sec: float = 10.0

    def start(self) -> None:
        """Subscribe to events on the shared bus."""
        if self._started:
            return

        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None

        self._event_bus.subscribe(EventType.BAR_CLOSE, self._on_bar_close)
        self._event_bus.subscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)
        self._event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)

        self._started = True
        logger.info("WebBridge started")

    def stop(self) -> None:
        """Unsubscribe and clean up."""
        if not self._started:
            return

        self._started = False

        for event_type, handler in [
            (EventType.BAR_CLOSE, self._on_bar_close),
            (EventType.INDICATOR_UPDATE, self._on_indicator_update),
            (EventType.TRADING_SIGNAL, self._on_trading_signal),
        ]:
            try:
                self._event_bus.unsubscribe(event_type, handler)  # type: ignore[arg-type]
            except Exception as e:
                logger.warning("Error unsubscribing %s: %s", event_type, e)

        # Unsubscribe portfolio event handlers
        for event_type, handler, bus in self._portfolio_handlers:
            try:
                bus.unsubscribe(event_type, handler)
            except Exception as e:
                logger.warning("Error unsubscribing portfolio %s: %s", event_type, e)
        self._portfolio_handlers.clear()

        self._advisor_executor.shutdown(wait=True, cancel_futures=True)
        logger.info("WebBridge stopped")

    def set_advisor_service(self, svc: Any) -> None:
        """Set the advisor service (called after bootstrap)."""
        self._advisor_service = svc

    def get_recent_signals(self, symbol: Optional[str] = None) -> list[dict]:
        """Get recent signals for advisor consumption."""
        with self._signal_buffer_lock:
            if symbol:
                return list(self._signal_buffer.get(symbol, []))
            return [sig for sigs in self._signal_buffer.values() for sig in sigs]

    # ── Event handlers ─────────────────────────────────────────

    def _on_bar_close(self, event: BarCloseEvent) -> None:
        """Forward bar close to WS hub. Trigger advisor on daily close."""
        bar_dict = {
            "t": event.timestamp.isoformat() if event.timestamp else None,
            "o": event.open,
            "h": event.high,
            "l": event.low,
            "c": event.close,
            "v": event.volume,
        }
        self._schedule_async(self._hub.broadcast_bar(event.symbol, event.timeframe, bar_dict))

        # Trigger advisor on daily bar close (debounced)
        if event.timeframe == "1d" and self._advisor_service:
            now = time.monotonic()
            if now - self._advisor_last_compute < self._advisor_debounce_sec:
                return
            self._advisor_last_compute = now

            def _compute_and_broadcast() -> None:
                try:
                    advice = self._advisor_service.compute_all()
                    serialized = _serialize_advice(advice)
                    self._schedule_async(self._hub.broadcast_advisor(serialized))
                except Exception:
                    logger.exception("Advisor broadcast failed")

            self._advisor_executor.submit(_compute_and_broadcast)

        # Save score snapshot to DuckDB on daily bar close
        if event.timeframe == "1d" and self._persistence:
            try:
                ie = self._signal_engine.indicator_engine
                regime_states = ie.get_all_indicator_states(timeframe="1d")
                for (sym, tf, ind), state in regime_states.items():
                    if ind == "regime_detector" and sym == event.symbol and state:
                        self._persistence.save_score_snapshot(
                            event.symbol,
                            event.timestamp,
                            state.get("composite_score", 0.0),
                            state.get("trend_state", "unknown"),
                            state.get("regime", "R0"),
                        )
                        break
            except Exception:
                logger.debug("Score snapshot save failed for %s", event.symbol, exc_info=True)

    def _on_indicator_update(self, event: IndicatorUpdateEvent) -> None:
        """Forward indicator update to WS hub."""
        self._schedule_async(
            self._hub.broadcast_indicator(
                event.symbol, event.timeframe, event.indicator, event.value
            )
        )
        # Strategy indicators broadcast full state dict
        if event.indicator in STRATEGY_INDICATORS and event.state:
            state = dict(event.state)
            state["date"] = event.timestamp.isoformat() if event.timestamp else ""
            if event.indicator == "regime_detector":
                state = map_regime_to_flex(state)
            self._schedule_async(
                self._hub.broadcast_strategy_state(
                    event.symbol, event.timeframe, event.indicator, state
                )
            )

    def _on_trading_signal(self, event: TradingSignalEvent) -> None:
        """Forward trading signal to WS hub + buffer for advisor."""
        ws_direction = DIRECTION_MAP.get(event.direction, "neutral")

        signal_dict = {
            "rule": event.indicator,
            "direction": ws_direction,
            "strength": event.strength,
            "timeframe": event.timeframe,
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        }
        self._schedule_async(self._hub.broadcast_signal(event.symbol, signal_dict))

        # Persist signal to DuckDB
        if self._persistence and event.timestamp:
            try:
                self._persistence.insert_signal(
                    symbol=event.symbol,
                    rule=(
                        event.trigger_rule
                        if hasattr(event, "trigger_rule") and event.trigger_rule
                        else event.indicator
                    ),
                    direction=ws_direction,
                    strength=event.strength,
                    timeframe=event.timeframe,
                    indicator=event.indicator,
                    ts=event.timestamp,
                )
            except Exception:
                logger.warning("Signal persistence failed for %s", event.symbol, exc_info=True)

        # Buffer for advisor
        advisor_sig = {
            "rule": event.indicator,
            "direction": ws_direction,
            "strength": event.strength,
        }
        with self._signal_buffer_lock:
            if event.symbol not in self._signal_buffer:
                self._signal_buffer[event.symbol] = []
            self._signal_buffer[event.symbol].append(advisor_sig)
            if len(self._signal_buffer[event.symbol]) > 50:
                self._signal_buffer[event.symbol] = self._signal_buffer[event.symbol][-50:]

    # ── Portfolio event wiring ─────────────────────────────────

    def wire_portfolio_events(self, container: Any) -> None:
        """Subscribe to AppContainer events and bridge to WS broadcasts."""
        from datetime import datetime, timezone

        from src.domain.events.domain_events import QuoteTick

        event_bus = container.event_bus
        hub = self._hub

        def on_snapshot_ready(event: Any) -> None:
            snapshot = None
            if container.orchestrator:
                snapshot = container.orchestrator.get_latest_snapshot()
            account = None
            if container.account_store:
                account = container.account_store.get()
            coro = hub.broadcast_portfolio_snapshot(
                snapshot=snapshot,
                account=account,
                broker_manager=container.broker_manager,
            )
            self._schedule_async(coro)

        event_bus.subscribe(EventType.SNAPSHOT_READY, on_snapshot_ready)
        self._portfolio_handlers.append((EventType.SNAPSHOT_READY, on_snapshot_ready, event_bus))

        def on_position_delta(event: Any) -> None:
            delta_dict = {
                "symbol": event.symbol,
                "underlying": event.underlying,
                "new_mark_price": event.new_mark_price,
                "pnl_change": event.pnl_change,
                "daily_pnl_change": event.daily_pnl_change,
                "delta_change": event.delta_change,
                "gamma_change": event.gamma_change,
                "vega_change": event.vega_change,
                "theta_change": event.theta_change,
                "notional_change": event.notional_change,
                "delta_dollars_change": getattr(event, "delta_dollars_change", 0.0),
                "underlying_price": getattr(event, "underlying_price", 0.0),
                "is_reliable": getattr(event, "is_reliable", True),
            }
            self._schedule_async(hub.broadcast_portfolio_delta(delta_dict))

        event_bus.subscribe(EventType.POSITION_DELTA, on_position_delta)
        self._portfolio_handlers.append((EventType.POSITION_DELTA, on_position_delta, event_bus))

        def on_account_updated(event: Any) -> None:
            account_dict = {
                "account_id": getattr(event, "account_id", ""),
                "net_liquidation": event.net_liquidation,
                "total_cash": event.total_cash,
                "buying_power": event.buying_power,
                "margin_used": event.margin_used,
                "margin_available": event.margin_available,
                "unrealized_pnl": event.unrealized_pnl,
                "realized_pnl": event.realized_pnl,
                "daily_pnl": getattr(event, "daily_pnl", 0.0),
                "position_count": getattr(event, "position_count", 0),
            }
            self._schedule_async(hub.broadcast_account_update(account_dict))

        event_bus.subscribe(EventType.ACCOUNT_UPDATED, on_account_updated)
        self._portfolio_handlers.append((EventType.ACCOUNT_UPDATED, on_account_updated, event_bus))

        def on_market_tick(event: Any) -> None:
            tick = QuoteTick(
                symbol=event.symbol,
                last=event.last,
                bid=event.bid,
                ask=event.ask,
                volume=None,
                source=getattr(event, "source", "ib"),
                timestamp=getattr(event, "timestamp", None) or datetime.now(timezone.utc),
            )
            # signal_engine already receives ticks via its own bus subscription —
            # only broadcast quote to WebSocket clients here
            self._schedule_async(hub.broadcast_quote(tick.symbol, tick_to_dict(tick)))

        event_bus.subscribe(EventType.MARKET_DATA_TICK, on_market_tick)
        self._portfolio_handlers.append((EventType.MARKET_DATA_TICK, on_market_tick, event_bus))
        logger.info("Portfolio events wired to WebSocket hub")

    # ── Internal ───────────────────────────────────────────────

    def _schedule_async(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule a coroutine on the event loop (thread-safe)."""
        if self._loop and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(coro, self._loop)
            except RuntimeError:
                coro.close()
                logger.debug("Event loop closed — discarding coroutine")
        else:
            coro.close()
            logger.debug("No event loop — discarding coroutine")
