"""Server Pipeline — wires tick → bar → indicator → signal → WebSocket hub."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from typing import Any, Coroutine, Dict, List, Optional

from src.domain.events.domain_events import (
    BarCloseEvent,
    IndicatorUpdateEvent,
    QuoteTick,
    TradingSignalEvent,
)
from src.server.routes.advisor import _serialize as _serialize_advice
from src.domain.events.event_types import EventType
from src.domain.events.priority_event_bus import PriorityEventBus
from src.domain.signals.data.bar_aggregator import BarAggregator
from src.domain.signals.indicator_engine import IndicatorEngine
from src.domain.signals.rule_engine import RuleEngine, RuleRegistry

logger = logging.getLogger(__name__)

# Strategy indicators whose full state dict should be broadcast via WS
STRATEGY_INDICATORS = {"dual_macd", "trend_pulse", "regime_detector"}

# Regime short-code → target exposure for RegimeFlex mapping
_REGIME_EXPOSURE = {"R0": 1.0, "R1": 0.5, "R2": 0.0, "R3": 0.25}


def _map_regime_to_flex(state: dict) -> dict:
    """Map regime_detector state → RegimeFlexRow format for frontend."""
    regime_full = state.get("regime", "R1_CHOPPY_EXTENDED")
    regime_short = regime_full.split("_")[0] if "_" in str(regime_full) else str(regime_full)
    signal = "NONE"
    if state.get("regime_changed"):
        prev = (state.get("previous_regime") or "").split("_")[0]
        signal = f"{prev}→{regime_short}"
    return {
        "date": state.get("date", ""),
        "regime": regime_short,
        "target_exposure": _REGIME_EXPOSURE.get(regime_short, 0.5),
        "signal": signal,
    }


class ServerPipeline:
    """
    Wires the domain signal pipeline for the live dashboard server.

    Tick → BarAggregator (per timeframe) → IndicatorEngine → RuleEngine
    Each stage publishes events that are also broadcast to WebSocket clients.
    """

    def __init__(
        self,
        hub: Any,
        timeframes: List[str],
        config: Any = None,
    ) -> None:
        self._hub = hub
        self._timeframes = timeframes
        self._started = False

        # Create event bus
        self._event_bus = PriorityEventBus()

        # Create one BarAggregator per timeframe
        self._aggregators: Dict[str, BarAggregator] = {}
        for tf in timeframes:
            self._aggregators[tf] = BarAggregator(
                timeframe=tf,
                event_bus=self._event_bus,
            )

        # Create IndicatorEngine
        self._indicator_engine = IndicatorEngine(
            event_bus=self._event_bus,
            max_workers=2,
        )

        # Create RuleEngine with default registry
        self._rule_registry = RuleRegistry()
        self._rule_engine = RuleEngine(
            event_bus=self._event_bus,
            registry=self._rule_registry,
        )

        # Async event loop reference (set on start)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Signal buffer for advisor (maps symbol -> list of recent signal dicts)
        # Replaces broken get_evaluation_history() approach
        self._signal_buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._signal_buffer_lock = RLock()
        self._advisor_service: Any = None

        # Persistence reference (set from main.py after construction)
        self._persistence: Any = None

        # Shared executor for advisor (avoids creating a new one per event)
        self._advisor_executor = ThreadPoolExecutor(max_workers=1)
        # Debounce: only recompute advisor once per daily cycle, not per-symbol
        self._advisor_last_compute: float = 0.0
        _ADVISOR_DEBOUNCE_SEC = 10.0  # seconds
        self._advisor_debounce_sec = _ADVISOR_DEBOUNCE_SEC

    async def start(self) -> None:
        """Start the pipeline — subscribe to events and start engines."""
        if self._started:
            return

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

        # Subscribe to events for WS broadcasting
        self._event_bus.subscribe(EventType.BAR_CLOSE, self._on_bar_close)
        self._event_bus.subscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)
        self._event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)

        # Start engines (they subscribe to their upstream events internally)
        self._indicator_engine.start()
        self._rule_engine.start()

        # Start event bus processing (async)
        await self._event_bus.start()

        self._started = True
        logger.info("ServerPipeline started (timeframes=%s)", self._timeframes)

    async def stop(self) -> None:
        """Stop the pipeline."""
        self._indicator_engine.stop()
        self._rule_engine.stop()
        await self._event_bus.stop()
        self._advisor_executor.shutdown(wait=False)
        self._started = False
        logger.info("ServerPipeline stopped")

    def on_tick(self, tick: QuoteTick) -> None:
        """Feed a tick into all BarAggregators."""
        for agg in self._aggregators.values():
            agg.on_tick(tick)

    def set_advisor_service(self, svc: Any) -> None:
        """Set the advisor service (called after bootstrap)."""
        self._advisor_service = svc

    def get_recent_signals(self, symbol: str | None = None) -> list[dict]:
        """Get recent signals for advisor consumption.

        Signals are collected from TradingSignalEvent with direction mapped
        from SignalDirection (buy/sell/alert) to advisor labels (bullish/bearish/neutral).
        """
        with self._signal_buffer_lock:
            if symbol:
                return list(self._signal_buffer.get(symbol, []))
            return [sig for sigs in self._signal_buffer.values() for sig in sigs]

    def inject_history(self, symbol: str, tf: str, bars: list) -> None:
        """Inject historical bars for indicator warmup."""
        bar_dicts = []
        for b in bars:
            bar_dicts.append(
                {
                    "symbol": b.symbol if hasattr(b, "symbol") else symbol,
                    "timeframe": b.timeframe if hasattr(b, "timeframe") else tf,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "timestamp": b.timestamp,
                }
            )
        if bar_dicts:
            self._indicator_engine.inject_historical_bars(symbol, tf, bar_dicts)
            logger.info("Injected %d historical bars for %s/%s", len(bar_dicts), symbol, tf)

    def get_regime_states(self, timeframe: str = "1d") -> Dict[str, Dict[str, Any]]:
        """Get regime state for all symbols from indicator engine cache.

        Returns dict mapping symbol -> {regime, regime_name, confidence, composite_score}.
        """
        results: Dict[str, Dict[str, Any]] = {}
        states = self._indicator_engine.get_all_indicator_states(timeframe=timeframe)
        for (sym, tf, ind), state in states.items():
            if ind == "regime_detector" and state:
                regime = state.get("regime", "R1")
                entry: Dict[str, Any] = {
                    "regime": regime,
                    "regime_name": state.get("regime_name", "Unknown"),
                    "confidence": state.get("confidence", 50),
                }
                cs = state.get("composite_score")
                if cs is not None:
                    entry["composite_score"] = cs
                results[sym] = entry
        return results

    # ── Event handlers (bridge events → WS hub) ────────────

    def _on_bar_close(self, event: BarCloseEvent) -> None:
        """Forward bar close to WebSocket hub. Trigger advisor on daily close."""
        # Fields match frontend OHLCV type: {t, o, h, l, c, v}
        bar_dict = {
            "t": event.timestamp.isoformat() if event.timestamp else None,
            "o": event.open,
            "h": event.high,
            "l": event.low,
            "c": event.close,
            "v": event.volume,
        }
        self._schedule_async(self._hub.broadcast_bar(event.symbol, event.timeframe, bar_dict))

        # Trigger advisor recomputation on daily bar close, debounced + shared executor.
        # Each symbol fires a daily close event; debounce prevents N redundant compute_all() calls.
        if event.timeframe == "1d" and self._advisor_service:
            now = time.monotonic()
            if now - self._advisor_last_compute < self._advisor_debounce_sec:
                return  # Already computed recently, skip
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
                regime_states = self._indicator_engine.get_all_indicator_states(timeframe="1d")
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
        """Forward indicator update to WebSocket hub."""
        self._schedule_async(
            self._hub.broadcast_indicator(
                event.symbol, event.timeframe, event.indicator, event.value
            )
        )
        # For strategy indicators, also broadcast the full state dict
        if event.indicator in STRATEGY_INDICATORS and event.state:
            state = dict(event.state)
            state["date"] = event.timestamp.isoformat() if event.timestamp else ""
            if event.indicator == "regime_detector":
                state = _map_regime_to_flex(state)
            self._schedule_async(
                self._hub.broadcast_strategy_state(
                    event.symbol, event.timeframe, event.indicator, state
                )
            )

    def _on_trading_signal(self, event: TradingSignalEvent) -> None:
        """Forward trading signal to WebSocket hub + buffer for advisor."""
        # Map upstream directions to frontend-friendly labels.
        # Upstream SignalDirection normalizes to LONG/SHORT/FLAT; rule engine
        # uses buy/sell/alert. Map all to bullish/bearish/neutral.
        direction_map = {
            "buy": "bullish",
            "sell": "bearish",
            "alert": "neutral",
            "LONG": "bullish",
            "SHORT": "bearish",
            "FLAT": "neutral",
            "long": "bullish",
            "short": "bearish",
            "flat": "neutral",
        }
        ws_direction = direction_map.get(event.direction, "neutral")

        # Signal dict matches frontend SignalData: {symbol, rule, direction, strength, timeframe, timestamp}
        signal_dict = {
            "rule": event.indicator,
            "direction": ws_direction,
            "strength": event.strength,
            "timeframe": event.timeframe if hasattr(event, "timeframe") else "",
            "timestamp": event.timestamp.isoformat() if event.timestamp else None,
        }
        self._schedule_async(self._hub.broadcast_signal(event.symbol, signal_dict))

        # Persist signal to DuckDB
        if self._persistence and event.timestamp:
            try:
                self._persistence.insert_signal(
                    symbol=event.symbol,
                    rule=event.trigger_rule if hasattr(event, "trigger_rule") and event.trigger_rule else event.indicator,
                    direction=ws_direction,
                    strength=event.strength,
                    timeframe=event.timeframe if hasattr(event, "timeframe") else "",
                    indicator=event.indicator,
                    ts=event.timestamp,
                )
            except Exception:
                logger.debug("Signal persistence failed for %s", event.symbol, exc_info=True)

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
            # Keep only latest 50 signals per symbol
            if len(self._signal_buffer[event.symbol]) > 50:
                self._signal_buffer[event.symbol] = self._signal_buffer[event.symbol][-50:]

    def _schedule_async(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Schedule a coroutine on the event loop (thread-safe)."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            # No loop available — close the coroutine to avoid warnings
            coro.close()
