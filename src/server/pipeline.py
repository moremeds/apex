"""Server Pipeline — wires tick → bar → indicator → signal → WebSocket hub."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from src.domain.events.domain_events import (
    BarCloseEvent,
    IndicatorUpdateEvent,
    QuoteTick,
    TradingSignalEvent,
)
from src.domain.events.event_types import EventType
from src.domain.events.priority_event_bus import PriorityEventBus
from src.domain.signals.data.bar_aggregator import BarAggregator
from src.domain.signals.indicator_engine import IndicatorEngine
from src.domain.signals.rule_engine import RuleEngine, RuleRegistry

logger = logging.getLogger(__name__)


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
        self._started = False
        logger.info("ServerPipeline stopped")

    def on_tick(self, tick: QuoteTick) -> None:
        """Feed a tick into all BarAggregators."""
        for agg in self._aggregators.values():
            agg.on_tick(tick)

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

    # ── Event handlers (bridge events → WS hub) ────────────

    def _on_bar_close(self, event: BarCloseEvent) -> None:
        """Forward bar close to WebSocket hub."""
        bar_dict = {
            "o": event.open,
            "h": event.high,
            "l": event.low,
            "c": event.close,
            "v": event.volume,
            "ts": event.timestamp.isoformat() if event.timestamp else None,
        }
        self._schedule_async(self._hub.broadcast_bar(event.symbol, event.timeframe, bar_dict))

    def _on_indicator_update(self, event: IndicatorUpdateEvent) -> None:
        """Forward indicator update to WebSocket hub."""
        self._schedule_async(
            self._hub.broadcast_indicator(
                event.symbol, event.timeframe, event.indicator, event.value
            )
        )

    def _on_trading_signal(self, event: TradingSignalEvent) -> None:
        """Forward trading signal to WebSocket hub."""
        signal_dict = {
            "rule": event.indicator,
            "direction": event.direction,
            "strength": event.strength,
            "ts": event.timestamp.isoformat() if event.timestamp else None,
        }
        self._schedule_async(self._hub.broadcast_signal(event.symbol, signal_dict))

    def _schedule_async(self, coro) -> None:
        """Schedule a coroutine on the event loop (thread-safe)."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            # No loop available — close the coroutine to avoid warnings
            coro.close()
