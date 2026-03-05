"""
SignalCoordinator - Wires the signal pipeline into the event bus.

Manages the complete signal generation pipeline:
- Tick aggregation into bars (MARKET_DATA_TICK → BAR_CLOSE)
- Indicator computation (BAR_CLOSE → INDICATOR_UPDATE)
- Signal rule evaluation (INDICATOR_UPDATE → TRADING_SIGNAL)

Wraps a SignalEngine (shared domain service) and adds:
- ConfluenceCalculator for cross-indicator analysis
- PostgreSQL persistence for TUI updates
- Bar preloader for historical warmup

When a SignalEngine is injected (server mode), this coordinator reuses it.
When none is provided (standalone TUI mode), it creates its own internally.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...domain.events.event_types import EventType
from ...domain.events.priority_event_bus import PriorityEventBus
from ...domain.interfaces.event_bus import EventBus
from ...domain.signals.confluence_calculator import ConfluenceCalculator
from ...domain.signals.signal_engine import SignalEngine
from ...infrastructure.observability import SignalMetrics
from ...utils.logging_setup import get_logger
from ...utils.timezone import now_utc
from .signal_pipeline import BarPreloader

if TYPE_CHECKING:
    from ...domain.interfaces.signal_persistence import SignalPersistencePort

logger = get_logger(__name__)


class SignalCoordinator:
    """
    Coordinates the signal pipeline (ticks → bars → indicators → signals).

    Wraps a SignalEngine and adds confluence calculation, persistence,
    and bar preloading on top.

    Example:
        coordinator = SignalCoordinator(
            event_bus=event_bus,
            timeframes=["1d"],
        )
        coordinator.start()  # Pipeline now active
    """

    def __init__(
        self,
        event_bus: Union[EventBus, PriorityEventBus],
        timeframes: Optional[List[str]] = None,
        max_workers: int = 4,
        enabled: bool = True,
        signal_metrics: Optional[SignalMetrics] = None,
        historical_data_manager: Optional[Any] = None,
        preload_config: Optional[Dict[str, Any]] = None,
        persistence: Optional["SignalPersistencePort"] = None,
        exclude_options: bool = True,
        signal_engine: Optional[SignalEngine] = None,
    ) -> None:
        self._event_bus = event_bus
        self._timeframes = list(dict.fromkeys(timeframes or ["1d"]))
        self._max_workers = max_workers
        self._enabled = enabled
        self._metrics = signal_metrics
        self._persistence = persistence
        self._exclude_options = exclude_options

        # Historical data manager for preloading and cache refresh
        self._historical_data_manager = historical_data_manager
        self._preload_config = preload_config or {}

        # Injected or internally-created SignalEngine
        self._signal_engine = signal_engine
        self._owns_engine = signal_engine is None  # True if we create it ourselves

        # Confluence calculator (extracted for single responsibility)
        self._confluence_calculator: Optional[ConfluenceCalculator] = None

        # Bar preloader (extracted for single responsibility)
        self._bar_preloader: Optional[BarPreloader] = None

        # Persistence statistics
        self._signals_persisted = 0
        self._indicators_persisted = 0
        self._confluence_persisted = 0

        self._started = False

    @property
    def is_started(self) -> bool:
        """Whether the coordinator is running."""
        return self._started

    @property
    def timeframes(self) -> List[str]:
        """Configured timeframes."""
        return list(self._timeframes)

    @property
    def _indicator_engine(self) -> Any:
        """Backward-compatible access to indicator engine via signal engine."""
        if self._signal_engine and self._signal_engine.is_started:
            return self._signal_engine.indicator_engine
        return None

    @property
    def _rule_engine(self) -> Any:
        """Backward-compatible access to rule engine via signal engine."""
        if self._signal_engine and self._signal_engine.is_started:
            return self._signal_engine._rule_engine
        return None

    @property
    def _bar_aggregators(self) -> dict:
        """Backward-compatible access to bar aggregators via signal engine."""
        if self._signal_engine:
            return self._signal_engine._aggregators
        return {}

    def start(self) -> None:
        """
        Start the signal pipeline.

        If no SignalEngine was injected, creates one internally.
        Then adds confluence calculator, preloader, and persistence.
        """
        if self._started:
            logger.warning("SignalCoordinator already started")
            return

        if not self._enabled:
            logger.info("SignalCoordinator disabled, skipping start")
            return

        # Create SignalEngine if not injected
        if self._signal_engine is None:
            self._signal_engine = SignalEngine(
                event_bus=self._event_bus,
                timeframes=self._timeframes,
                max_workers=self._max_workers,
                signal_metrics=self._metrics,
                exclude_options=self._exclude_options,
            )
            self._owns_engine = True

        # Start engine if we own it (not started externally)
        if self._owns_engine and not self._signal_engine.is_started:
            self._signal_engine.start()

        # Create and start confluence calculator
        self._confluence_calculator = ConfluenceCalculator(
            event_bus=self._event_bus,
            metrics=self._metrics,
        )
        if self._persistence:
            self._confluence_calculator.set_persistence_callback(self._persist_confluence)
        self._confluence_calculator.start()

        # Create bar preloader (if historical data manager configured)
        if self._historical_data_manager is not None and self._signal_engine.is_started:
            self._bar_preloader = BarPreloader(
                historical_data_manager=self._historical_data_manager,
                indicator_engine=self._signal_engine.indicator_engine,
                timeframes=self._timeframes,
                preload_config=self._preload_config,
            )

        # Subscribe to indicator updates for confluence + persistence
        self._event_bus.subscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)

        # Subscribe to trading signals for persistence (if enabled)
        if self._persistence:
            self._event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)

        self._started = True

        # Structured startup log
        ie = self._signal_engine.indicator_engine
        re = self._signal_engine._rule_engine
        logger.info(
            "Signal pipeline started",
            extra={
                "timeframes": self._timeframes,
                "indicators": ie.indicator_count if ie else 0,
                "rules": len(re._registry) if re and hasattr(re, "_registry") else 0,
                "max_workers": self._max_workers,
                "persistence_enabled": self._persistence is not None,
                "shared_engine": not self._owns_engine,
            },
        )

    def stop(self) -> None:
        """Stop the signal pipeline and release resources."""
        if not self._started:
            return

        self._started = False

        # Unsubscribe from indicator updates
        try:
            self._event_bus.unsubscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)
        except Exception as e:
            logger.warning("Error unsubscribing from indicator events: %s", e)

        # Unsubscribe from trading signals (if persistence was enabled)
        if self._persistence:
            try:
                self._event_bus.unsubscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)
            except Exception as e:
                logger.warning("Error unsubscribing from signal events: %s", e)

        # Stop confluence calculator
        if self._confluence_calculator:
            self._confluence_calculator.stop()

        # Stop engine only if we own it
        if self._owns_engine and self._signal_engine:
            self._signal_engine.stop()
            self._signal_engine = None

        self._confluence_calculator = None
        self._bar_preloader = None

        logger.info("SignalCoordinator stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats: Dict[str, Any] = {
            "started": self._started,
            "timeframes": self._timeframes,
            "enabled": self._enabled,
        }

        if self._started and self._signal_engine:
            stats.update(self._signal_engine.get_stats())

        return stats

    def clear_cooldowns(self) -> int:
        """Clear expired signal cooldowns."""
        if self._signal_engine:
            return self._signal_engine.clear_cooldowns()
        return 0

    # ── Historical Bar Preloading ──────────────────────────────

    async def preload_bar_history(self, symbols: List[str]) -> Dict[str, int]:
        """Preload historical bars from Parquet cache on STARTUP."""
        if not self._bar_preloader:
            logger.warning("Cannot preload bars: bar preloader not configured")
            return {}
        return await self._bar_preloader.preload_startup(symbols)

    async def refresh_disk_cache(self, symbols: List[str]) -> bool:
        """Refresh Parquet cache (PERIODIC, disk-only)."""
        if not self._bar_preloader:
            logger.debug("Cache refresh skipped: bar preloader not configured")
            return False
        return await self._bar_preloader.refresh_disk_cache(symbols)

    # ── Indicator Update Handler ───────────────────────────────

    def _on_indicator_update(self, payload: Any) -> None:
        """Handle INDICATOR_UPDATE events for confluence + persistence."""
        if not self._started:
            return

        symbol = getattr(payload, "symbol", None)
        timeframe = getattr(payload, "timeframe", None)
        indicator = getattr(payload, "indicator", None)
        state = getattr(payload, "state", None) or {}

        if not symbol or not timeframe or not indicator:
            return

        # Delegate confluence calculation to ConfluenceCalculator
        if self._confluence_calculator:
            self._confluence_calculator.on_indicator_update(
                symbol=symbol,
                timeframe=timeframe,
                indicator=indicator,
                state=state,
            )

        # Persist indicator (coordinator responsibility)
        if self._persistence:
            previous_state = getattr(payload, "previous_state", None)
            timestamp = getattr(payload, "timestamp", None) or now_utc()
            asyncio.create_task(
                self._persist_indicator(
                    symbol=symbol,
                    timeframe=timeframe,
                    indicator=indicator,
                    timestamp=timestamp,
                    state=state,
                    previous_state=previous_state,
                )
            )

    # ── Persistence Handlers ───────────────────────────────────

    def _on_trading_signal(self, payload: Any) -> None:
        """Handle TRADING_SIGNAL event for persistence."""
        if not self._started or not self._persistence:
            return
        asyncio.create_task(self._persist_signal(payload))

    async def _persist_signal(self, signal: Any) -> None:
        """Persist trading signal to database."""
        if not self._persistence:
            return
        try:
            if hasattr(signal, "signal"):
                signal = signal.signal
            await self._persistence.save_signal(signal)
            self._signals_persisted += 1
        except Exception as e:
            logger.error("Failed to persist signal: %s", e)

    async def _persist_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
        timestamp: datetime,
        state: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist indicator value to database."""
        if not self._persistence:
            return
        try:
            await self._persistence.save_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator=indicator,
                timestamp=timestamp,
                state=state,
                previous_state=previous_state,
            )
            self._indicators_persisted += 1
        except Exception as e:
            logger.error("Failed to persist indicator %s for %s: %s", indicator, symbol, e)

    async def _persist_confluence(
        self,
        symbol: str,
        timeframe: str,
        alignment_score: float,
        bullish_count: int,
        bearish_count: int,
        neutral_count: int,
        total_indicators: int,
        dominant_direction: Optional[str] = None,
    ) -> None:
        """Persist confluence score to database."""
        if not self._persistence:
            return
        try:
            await self._persistence.save_confluence(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=now_utc(),
                alignment_score=alignment_score,
                bullish_count=bullish_count,
                bearish_count=bearish_count,
                neutral_count=neutral_count,
                total_indicators=total_indicators,
                dominant_direction=dominant_direction,
            )
            self._confluence_persisted += 1
        except Exception as e:
            logger.error("Failed to persist confluence for %s/%s: %s", symbol, timeframe, e)
