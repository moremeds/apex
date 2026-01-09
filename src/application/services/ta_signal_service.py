"""
Technical Analysis Signal Service - Decoupled Signal Pipeline.

Orchestrates the full signal pipeline with optional persistence:
- Can run standalone via signal_runner.py (for validation)
- Can be wired into Orchestrator (production mode)
- Can be split into microservices later (future optionality)

Pipeline: TICK → BAR → INDICATOR → RULE → SIGNAL → PERSISTENCE → NOTIFY
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...utils.logging_setup import get_logger
from ...utils.timezone import now_utc
from ...domain.events.event_types import EventType

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus
    from ...domain.interfaces.signal_persistence import SignalPersistencePort
    from ...domain.signals import BarAggregator, IndicatorEngine, RuleEngine
    from ...domain.signals.models import TradingSignal
    from ...infrastructure.observability import SignalMetrics

logger = get_logger(__name__)


class TASignalService:
    """
    Technical Analysis Signal Service - Orchestrates the full signal pipeline.

    Can run:
    1. Standalone via signal_runner.py (for validation)
    2. Wired into Orchestrator (production mode)
    3. As separate microservice (future)

    Pipeline: TICK → BAR → INDICATOR → RULE → SIGNAL → PERSISTENCE → NOTIFY

    Usage:
        # Standalone (validation)
        service = TASignalService(
            event_bus=PriorityEventBus(),
            persistence=None,  # No DB for validation
            timeframes=["1d"],
        )
        await service.start()

        # With persistence (production)
        repo = TASignalRepository(db)
        service = TASignalService(
            event_bus=event_bus,
            persistence=repo,
            timeframes=config.signals.timeframes,
        )
        await service.start()
    """

    def __init__(
        self,
        event_bus: "EventBus",
        persistence: Optional["SignalPersistencePort"] = None,
        timeframes: Optional[List[str]] = None,
        max_workers: int = 4,
        enabled: bool = True,
        signal_metrics: Optional["SignalMetrics"] = None,
    ) -> None:
        """
        Initialize the TA signal service.

        Args:
            event_bus: Event bus for subscriptions and publishing.
            persistence: Optional persistence port for saving signals/indicators.
            timeframes: Bar timeframes to aggregate (default: ["1d"]).
            max_workers: ThreadPool size for indicator calculations.
            enabled: Whether to enable the signal pipeline.
            signal_metrics: Metrics collector for pipeline instrumentation.
        """
        self._event_bus = event_bus
        self._persistence = persistence
        self._timeframes = list(dict.fromkeys(timeframes or ["1d"]))
        self._max_workers = max_workers
        self._enabled = enabled
        self._metrics = signal_metrics

        # Pipeline components (lazy init)
        self._bar_aggregators: Dict[str, "BarAggregator"] = {}
        self._indicator_engine: Optional["IndicatorEngine"] = None
        self._rule_engine: Optional["RuleEngine"] = None

        # Confluence analyzers
        self._cross_analyzer = None
        self._mtf_analyzer = None

        # State tracking
        self._running = False
        self._indicator_states: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        self._last_confluence_calc: Dict[Tuple[str, str], float] = {}
        self._confluence_debounce_ms: float = 500.0

        # Statistics
        self._bars_processed = 0
        self._indicators_computed = 0
        self._signals_emitted = 0
        self._signals_persisted = 0
        self._start_time: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """
        Start the signal pipeline.

        Creates and wires all components:
        - BarAggregators for each timeframe
        - IndicatorEngine for indicator calculations
        - RuleEngine with pre-built rules
        - Persistence hooks for signals and indicators
        """
        if self._running:
            logger.warning("TASignalService already running")
            return

        if not self._enabled:
            logger.info("TASignalService disabled, skipping start")
            return

        self._init_pipeline()
        self._subscribe_events()
        self._running = True
        self._start_time = now_utc()

        logger.info(
            f"TASignalService started",
            extra={
                "timeframes": self._timeframes,
                "indicators": self._indicator_engine.indicator_count if self._indicator_engine else 0,
                "persistence_enabled": self._persistence is not None,
                "max_workers": self._max_workers,
            },
        )

    async def stop(self) -> None:
        """
        Stop the signal pipeline gracefully.

        Ordering:
        1. Unsubscribe from events
        2. Flush remaining bars
        3. Stop engines
        """
        if not self._running:
            return

        self._running = False

        # Unsubscribe from events
        self._unsubscribe_events()

        # Clear state caches
        self._indicator_states.clear()
        self._last_confluence_calc.clear()

        # Flush remaining bars
        for aggregator in list(self._bar_aggregators.values()):
            try:
                aggregator.flush()
            except Exception as e:
                logger.warning(f"Error flushing aggregator {aggregator.timeframe}: {e}")

        # Stop engines
        if self._indicator_engine:
            self._indicator_engine.stop()
        if self._rule_engine:
            self._rule_engine.stop()

        # Cleanup
        self._bar_aggregators.clear()
        self._indicator_engine = None
        self._rule_engine = None
        self._cross_analyzer = None
        self._mtf_analyzer = None

        logger.info(
            f"TASignalService stopped",
            extra={
                "bars_processed": self._bars_processed,
                "indicators_computed": self._indicators_computed,
                "signals_emitted": self._signals_emitted,
                "signals_persisted": self._signals_persisted,
            },
        )

    @property
    def is_running(self) -> bool:
        """Whether the service is running."""
        return self._running

    @property
    def timeframes(self) -> List[str]:
        """Configured timeframes."""
        return list(self._timeframes)

    @property
    def stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics for validation/monitoring.

        Returns:
            Dict with running status, counts, and timing info.
        """
        uptime_seconds = 0
        if self._start_time:
            uptime_seconds = (now_utc() - self._start_time).total_seconds()

        return {
            "running": self._running,
            "timeframes": self._timeframes,
            "persistence_enabled": self._persistence is not None,
            "bars_processed": self._bars_processed,
            "indicators_computed": self._indicators_computed,
            "signals_emitted": self._signals_emitted,
            "signals_persisted": self._signals_persisted,
            "indicator_count": self._indicator_engine.indicator_count if self._indicator_engine else 0,
            "uptime_seconds": round(uptime_seconds, 1),
        }

    # -------------------------------------------------------------------------
    # Pipeline Initialization
    # -------------------------------------------------------------------------

    def _init_pipeline(self) -> None:
        """Initialize pipeline components."""
        # Import here to avoid circular imports
        from ...domain.signals import BarAggregator, IndicatorEngine, RuleEngine, RuleRegistry
        from ...domain.signals.rules import ALL_RULES
        from ...domain.signals.divergence import CrossIndicatorAnalyzer, MTFDivergenceAnalyzer

        # Create bar aggregators for each timeframe
        self._bar_aggregators = {
            tf: BarAggregator(tf, self._event_bus, signal_metrics=self._metrics)
            for tf in self._timeframes
        }

        # Create rule registry with pre-built rules
        registry = RuleRegistry()
        registry.add_rules(ALL_RULES)

        # Create engines
        self._indicator_engine = IndicatorEngine(
            self._event_bus,
            max_workers=self._max_workers,
            signal_metrics=self._metrics,
        )

        # RuleEngine with persistence callback
        self._rule_engine = RuleEngine(
            self._event_bus,
            registry,
            signal_metrics=self._metrics,
        )

        # Start engines
        self._indicator_engine.start()
        self._rule_engine.start()

        # Create confluence analyzers
        self._cross_analyzer = CrossIndicatorAnalyzer()
        self._mtf_analyzer = MTFDivergenceAnalyzer(self._cross_analyzer)

    def _subscribe_events(self) -> None:
        """Subscribe to relevant events."""
        self._event_bus.subscribe(EventType.MARKET_DATA_TICK, self._on_market_data_tick)
        self._event_bus.subscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)
        self._event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)

    def _unsubscribe_events(self) -> None:
        """Unsubscribe from events."""
        try:
            self._event_bus.unsubscribe(EventType.MARKET_DATA_TICK, self._on_market_data_tick)
        except Exception as e:
            logger.warning(f"Error unsubscribing from tick events: {e}")

        try:
            self._event_bus.unsubscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)
        except Exception as e:
            logger.warning(f"Error unsubscribing from indicator events: {e}")

        try:
            self._event_bus.unsubscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)
        except Exception as e:
            logger.warning(f"Error unsubscribing from signal events: {e}")

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _on_market_data_tick(self, payload: Any) -> None:
        """Handle MARKET_DATA_TICK by fanning out to all aggregators."""
        if not self._running:
            return

        for aggregator in self._bar_aggregators.values():
            try:
                aggregator.on_tick(payload)
            except Exception as e:
                logger.error(f"Error processing tick in aggregator: {e}")

        self._bars_processed += 1

    def _on_indicator_update(self, payload: Any) -> None:
        """
        Handle INDICATOR_UPDATE - cache state and persist if enabled.

        Args:
            payload: IndicatorUpdateEvent with symbol, timeframe, indicator, state.
        """
        if not self._running:
            return

        self._indicators_computed += 1

        # Extract event data
        symbol = getattr(payload, "symbol", None)
        timeframe = getattr(payload, "timeframe", None)
        indicator = getattr(payload, "indicator", None)
        state = getattr(payload, "state", None)
        previous_state = getattr(payload, "previous_state", None)
        timestamp = getattr(payload, "timestamp", None) or now_utc()

        if not all([symbol, timeframe, indicator, state]):
            return

        # Cache indicator state for confluence calculation
        key = (symbol, timeframe)
        if key not in self._indicator_states:
            self._indicator_states[key] = {}
        self._indicator_states[key][indicator] = state

        # Persist indicator value if persistence enabled
        if self._persistence:
            asyncio.create_task(self._persist_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator=indicator,
                timestamp=timestamp,
                state=state,
                previous_state=previous_state,
            ))

        # Maybe calculate confluence (debounced)
        self._maybe_calculate_confluence(symbol, timeframe)

    def _on_trading_signal(self, payload: Any) -> None:
        """
        Handle TRADING_SIGNAL - persist if enabled.

        This is called AFTER RuleEngine emits the signal.
        We persist here to keep RuleEngine decoupled from persistence.

        Args:
            payload: TradingSignalEvent or TradingSignal.
        """
        if not self._running:
            return

        self._signals_emitted += 1

        # Persist signal if persistence enabled
        if self._persistence:
            asyncio.create_task(self._persist_signal(payload))

    # -------------------------------------------------------------------------
    # Persistence Helpers
    # -------------------------------------------------------------------------

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
        try:
            await self._persistence.save_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator=indicator,
                timestamp=timestamp,
                state=state,
                previous_state=previous_state,
            )
        except Exception as e:
            logger.error(f"Failed to persist indicator {indicator} for {symbol}: {e}")

    async def _persist_signal(self, signal: Any) -> None:
        """Persist trading signal to database."""
        try:
            # Handle both TradingSignalEvent and TradingSignal
            if hasattr(signal, "signal"):
                # It's a TradingSignalEvent wrapper
                signal = signal.signal
            await self._persistence.save_signal(signal)
            self._signals_persisted += 1
        except Exception as e:
            logger.error(f"Failed to persist signal: {e}")

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
        except Exception as e:
            logger.error(f"Failed to persist confluence for {symbol}/{timeframe}: {e}")

    # -------------------------------------------------------------------------
    # Confluence Calculation (copied from SignalCoordinator)
    # -------------------------------------------------------------------------

    def _maybe_calculate_confluence(self, symbol: str, timeframe: str) -> None:
        """
        Calculate confluence if enough time has passed (debounced).

        Prevents excessive calculations when multiple indicators update in quick succession.
        """
        key = (symbol, timeframe)
        now_ms = time.time() * 1000

        last_calc = self._last_confluence_calc.get(key, 0)
        if now_ms - last_calc < self._confluence_debounce_ms:
            return

        self._last_confluence_calc[key] = now_ms

        # Get cached indicator states
        states = self._indicator_states.get(key, {})
        if len(states) < 2:
            return  # Need at least 2 indicators for confluence

        # Calculate confluence
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        for indicator_name, state in states.items():
            direction = state.get("direction") or state.get("trend") or state.get("zone")
            if direction in ("bullish", "buy", "oversold"):
                bullish_count += 1
            elif direction in ("bearish", "sell", "overbought"):
                bearish_count += 1
            else:
                neutral_count += 1

        total = bullish_count + bearish_count + neutral_count
        if total == 0:
            return

        # Calculate alignment score: -1.0 (all bearish) to +1.0 (all bullish)
        alignment_score = (bullish_count - bearish_count) / total

        # Determine dominant direction
        if bullish_count > bearish_count:
            dominant_direction = "bullish"
        elif bearish_count > bullish_count:
            dominant_direction = "bearish"
        else:
            dominant_direction = "neutral"

        # Publish confluence update event
        from ...domain.events.domain_events import ConfluenceUpdateEvent
        event = ConfluenceUpdateEvent(
            symbol=symbol,
            timeframe=timeframe,
            alignment_score=alignment_score,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            total_indicators=total,
            dominant_direction=dominant_direction,
        )
        self._event_bus.publish(EventType.CONFLUENCE_UPDATE, event)

        # Persist confluence if enabled
        if self._persistence:
            asyncio.create_task(self._persist_confluence(
                symbol=symbol,
                timeframe=timeframe,
                alignment_score=alignment_score,
                bullish_count=bullish_count,
                bearish_count=bearish_count,
                neutral_count=neutral_count,
                total_indicators=total,
                dominant_direction=dominant_direction,
            ))

    # -------------------------------------------------------------------------
    # Historical Data Integration
    # -------------------------------------------------------------------------

    async def inject_historical_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: List[Dict[str, Any]],
    ) -> int:
        """
        Inject historical bars for indicator warmup.

        Delegates to IndicatorEngine.inject_historical_bars().

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            bars: List of bar dicts with open, high, low, close, volume, timestamp.

        Returns:
            Number of bars injected.
        """
        if not self._indicator_engine:
            logger.warning("Cannot inject bars - IndicatorEngine not initialized")
            return 0

        return self._indicator_engine.inject_historical_bars(symbol, timeframe, bars)
