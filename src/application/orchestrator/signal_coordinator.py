"""
SignalCoordinator - Wires the signal pipeline into the event bus.

Manages the complete signal generation pipeline:
- Tick aggregation into bars (MARKET_DATA_TICK → BAR_CLOSE)
- Indicator computation (BAR_CLOSE → INDICATOR_UPDATE)
- Signal rule evaluation (INDICATOR_UPDATE → TRADING_SIGNAL)

Optional persistence support:
- When SignalPersistencePort is provided, signals/indicators/confluence
  are persisted to database with PostgreSQL NOTIFY for TUI updates

This coordinator follows the same pattern as DataCoordinator and
SnapshotCoordinator, keeping the Orchestrator thin.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...domain.events.event_types import EventType
from ...domain.signals.confluence_calculator import ConfluenceCalculator
from ...infrastructure.observability import SignalMetrics
from ...utils.logging_setup import get_logger
from ...utils.timezone import now_utc
from .signal_pipeline import BarPreloader

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus
    from ...domain.interfaces.signal_persistence import SignalPersistencePort
    from ...domain.signals import BarAggregator, IndicatorEngine, RuleEngine

logger = get_logger(__name__)


class SignalCoordinator:
    """
    Coordinates the signal pipeline (ticks → bars → indicators → signals).

    Pipeline flow:
    1. MARKET_DATA_TICK events trigger bar aggregation
    2. Bar close triggers indicator calculations (ThreadPool)
    3. Indicator updates trigger rule evaluation
    4. Triggered rules emit TRADING_SIGNAL events

    Example:
        coordinator = SignalCoordinator(
            event_bus=event_bus,
            timeframes=["1d"],  # Daily bars only
        )
        coordinator.start()  # Pipeline now active
    """

    def __init__(
        self,
        event_bus: "EventBus",
        timeframes: Optional[List[str]] = None,
        max_workers: int = 4,
        enabled: bool = True,
        signal_metrics: Optional[SignalMetrics] = None,
        historical_data_manager: Optional[Any] = None,
        preload_config: Optional[Dict[str, Any]] = None,
        persistence: Optional["SignalPersistencePort"] = None,
        exclude_options: bool = True,
    ) -> None:
        """
        Initialize the signal coordinator.

        Args:
            event_bus: Event bus for subscriptions and publishing
            timeframes: Bar timeframes to aggregate (default: ["1d"])
            max_workers: ThreadPool size for indicator calculations
            enabled: Whether to enable the signal pipeline (can be disabled for testing)
            signal_metrics: Metrics collector for pipeline instrumentation
            historical_data_manager: Manager for historical bar data (Parquet cache)
            preload_config: Configuration for bar preloading:
                            - lookback_days: Number of days to load (default: 365)
                            - cache_refresh_hours: Refresh interval (default: 24)
                            - slow_preload_warn_sec: Warn if preload takes longer (default: 30)
            persistence: Optional persistence port for saving signals/indicators/confluence
                        to database. When provided, enables PostgreSQL NOTIFY for TUI updates.
            exclude_options: If True, filter out options symbols from bar aggregation (default: True).
                            Options symbols are detected by pattern matching (length > 10, contains
                            digits in date/strike format).
        """
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

        # Lazy initialization - components created on start()
        self._bar_aggregators: Dict[str, "BarAggregator"] = {}
        self._indicator_engine: Optional["IndicatorEngine"] = None
        self._rule_engine: Optional["RuleEngine"] = None

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

    def start(self) -> None:
        """
        Start the signal pipeline.

        Creates and wires all components:
        - BarAggregators for each timeframe
        - IndicatorEngine for indicator calculations
        - RuleEngine with pre-built rules
        """
        if self._started:
            logger.warning("SignalCoordinator already started")
            return

        if not self._enabled:
            logger.info("SignalCoordinator disabled, skipping start")
            return

        # Import here to avoid circular imports and allow lazy loading
        from ...domain.signals import BarAggregator, IndicatorEngine, RuleEngine, RuleRegistry
        from ...domain.signals.rules import ALL_RULES

        # Create bar aggregators for each timeframe (pass metrics for instrumentation)
        self._bar_aggregators = {
            tf: BarAggregator(tf, self._event_bus, signal_metrics=self._metrics)
            for tf in self._timeframes
        }

        # Create rule registry with pre-built rules
        registry = RuleRegistry()
        registry.add_rules(ALL_RULES)

        # Create engines (pass metrics for instrumentation)
        self._indicator_engine = IndicatorEngine(
            self._event_bus,
            max_workers=self._max_workers,
            signal_metrics=self._metrics,
        )
        self._rule_engine = RuleEngine(self._event_bus, registry, signal_metrics=self._metrics)

        # Start engines (they subscribe to their respective events)
        self._indicator_engine.start()
        self._rule_engine.start()

        # Create and start confluence calculator
        self._confluence_calculator = ConfluenceCalculator(
            event_bus=self._event_bus,
            metrics=self._metrics,
        )
        if self._persistence:
            self._confluence_calculator.set_persistence_callback(self._persist_confluence)
        self._confluence_calculator.start()

        # Create bar preloader (if historical data manager configured)
        if self._historical_data_manager is not None:
            self._bar_preloader = BarPreloader(
                historical_data_manager=self._historical_data_manager,
                indicator_engine=self._indicator_engine,
                timeframes=self._timeframes,
                preload_config=self._preload_config,
            )

        # Subscribe to tick events for bar aggregation
        self._event_bus.subscribe(EventType.MARKET_DATA_TICK, self._on_market_data_tick)

        # Subscribe to indicator updates for confluence + persistence
        self._event_bus.subscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)

        # Subscribe to trading signals for persistence (if enabled)
        if self._persistence:
            self._event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)

        self._started = True

        # Structured startup log
        logger.info(
            "Signal pipeline started",
            extra={
                "timeframes": self._timeframes,
                "indicators": self._indicator_engine.indicator_count,
                "rules": len(registry),
                "max_workers": self._max_workers,
                "persistence_enabled": self._persistence is not None,
            },
        )

    def stop(self) -> None:
        """
        Stop the signal pipeline and release resources.

        Ordering is important:
        1. Unsubscribe from tick events first (stops new data flow)
        2. Flush remaining bars (while engines are still running)
        3. Stop engines last
        """
        if not self._started:
            return

        # Mark as stopped first to prevent processing in callbacks
        self._started = False

        # Unsubscribe from tick events to stop new data flow
        try:
            self._event_bus.unsubscribe(EventType.MARKET_DATA_TICK, self._on_market_data_tick)
        except Exception as e:
            logger.warning(f"Error unsubscribing from tick events: {e}")

        # Unsubscribe from indicator updates
        try:
            self._event_bus.unsubscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)
        except Exception as e:
            logger.warning(f"Error unsubscribing from indicator events: {e}")

        # Unsubscribe from trading signals (if persistence was enabled)
        if self._persistence:
            try:
                self._event_bus.unsubscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)
            except Exception as e:
                logger.warning(f"Error unsubscribing from signal events: {e}")

        # Stop confluence calculator
        if self._confluence_calculator:
            self._confluence_calculator.stop()

        # Flush remaining bars BEFORE stopping engines
        # This ensures final BAR_CLOSE events are processed through the pipeline
        for aggregator in list(self._bar_aggregators.values()):
            try:
                aggregator.flush()
            except Exception as e:
                logger.warning(f"Error flushing aggregator {aggregator.timeframe}: {e}")

        # Stop engines after flush is complete
        if self._indicator_engine:
            self._indicator_engine.stop()
        if self._rule_engine:
            self._rule_engine.stop()

        self._bar_aggregators.clear()
        self._indicator_engine = None
        self._rule_engine = None
        self._confluence_calculator = None
        self._bar_preloader = None

        logger.info("SignalCoordinator stopped")

    def _is_options_symbol(self, symbol: str) -> bool:
        """
        Detect if a symbol is an options symbol based on pattern matching.

        Options symbols typically have formats like:
        - IB: "AAPL  250117C00250000" (with spaces)
        - OCC: "AAPL250117C00250000" (compact)
        - Custom: "AAPL:OPT:250117:250:C"

        Simple heuristics:
        - Length > 10 characters (stock symbols are typically 1-5 chars)
        - Contains digits mixed with letters in option-like patterns
        - Contains "C" or "P" followed by digits (strike price pattern)

        Args:
            symbol: The symbol string to check

        Returns:
            True if the symbol appears to be an options symbol
        """
        if not symbol:
            return False

        # Stock symbols are typically short (1-5 chars, occasionally 6)
        if len(symbol) <= 6:
            return False

        # Options symbols are typically longer
        if len(symbol) > 10:
            return True

        # Check for common option patterns: contains both letters and many digits
        # Pattern: 6+ digits (date YYMMDD + strike) anywhere in symbol
        if re.search(r"\d{6,}", symbol):
            return True

        # Pattern: C or P followed by digits (call/put + strike)
        if re.search(r"[CP]\d{5,}", symbol, re.IGNORECASE):
            return True

        # Pattern: colon-separated format (AAPL:OPT:...)
        if ":OPT:" in symbol.upper():
            return True

        return False

    def _on_market_data_tick(self, payload: Any) -> None:
        """
        Handle MARKET_DATA_TICK event by fanning out to all aggregators.

        This is the entry point for the signal pipeline. Each aggregator
        independently tracks bar state for its timeframe.
        """
        # Guard against processing after stop
        if not self._started:
            return

        # Filter out options symbols if configured
        if self._exclude_options:
            symbol = getattr(payload, "symbol", None)
            if symbol and self._is_options_symbol(symbol):
                return  # Skip options - don't aggregate their ticks

        # Iterate over a copy to avoid race with stop() clearing the dict
        for aggregator in list(self._bar_aggregators.values()):
            try:
                aggregator.on_tick(payload)
            except Exception as e:
                logger.error(f"Bar aggregation error for {aggregator.timeframe}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with bars_emitted, bars_processed, signals_emitted, etc.
        """
        stats: Dict[str, Any] = {
            "started": self._started,
            "timeframes": self._timeframes,
            "enabled": self._enabled,
        }

        if self._started:
            stats["bars_emitted"] = {
                tf: agg.bars_emitted for tf, agg in self._bar_aggregators.items()
            }
            if self._indicator_engine:
                stats["bars_processed"] = self._indicator_engine.bars_processed
                stats["indicator_count"] = self._indicator_engine.indicator_count
            if self._rule_engine:
                stats["rules_evaluated"] = self._rule_engine.rules_evaluated
                stats["signals_emitted"] = self._rule_engine.signals_emitted

        return stats

    def clear_cooldowns(self) -> int:
        """
        Clear expired signal cooldowns.

        Should be called periodically to prevent memory growth.

        Returns:
            Number of cooldowns cleared
        """
        if self._rule_engine:
            return self._rule_engine.clear_cooldowns()
        return 0

    # -------------------------------------------------------------------------
    # Historical Bar Preloading (Startup + Periodic Refresh)
    # -------------------------------------------------------------------------

    async def preload_bar_history(self, symbols: List[str]) -> Dict[str, int]:
        """
        Preload historical bars from Parquet cache on STARTUP.

        Delegates to BarPreloader which handles:
        - Gap detection via DuckDB
        - Download missing data from IB/Yahoo
        - Store to Parquet
        - Inject into IndicatorEngine for warmup

        Args:
            symbols: List of symbols to preload (e.g., ["AAPL", "TSLA"])

        Returns:
            Dict mapping symbol -> number of bars injected
        """
        if not self._bar_preloader:
            logger.warning("Cannot preload bars: bar preloader not configured")
            return {}
        return await self._bar_preloader.preload_startup(symbols)

    async def refresh_disk_cache(self, symbols: List[str]) -> bool:
        """
        Refresh Parquet cache (PERIODIC, disk-only).

        Delegates to BarPreloader which handles:
        - Time-based gating (respects cache_refresh_hours)
        - Updates Parquet files via HistoricalDataManager
        - Does NOT inject into IndicatorEngine (live bars handle that)

        Args:
            symbols: List of symbols to refresh

        Returns:
            True if refresh was performed, False if skipped (not due yet)
        """
        if not self._bar_preloader:
            logger.debug("Cache refresh skipped: bar preloader not configured")
            return False
        return await self._bar_preloader.refresh_disk_cache(symbols)

    # -------------------------------------------------------------------------
    # Indicator Update Handler (Delegates confluence to ConfluenceCalculator)
    # -------------------------------------------------------------------------

    def _on_indicator_update(self, payload: Any) -> None:
        """
        Handle INDICATOR_UPDATE events.

        Delegates confluence calculation to ConfluenceCalculator and handles
        indicator persistence separately.

        Args:
            payload: IndicatorUpdateEvent with symbol, timeframe, indicator, state
        """
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

    # -------------------------------------------------------------------------
    # Persistence Handlers
    # -------------------------------------------------------------------------

    def _on_trading_signal(self, payload: Any) -> None:
        """
        Handle TRADING_SIGNAL event for persistence.

        Persists signals to database, which triggers PostgreSQL NOTIFY
        for real-time TUI updates.

        Args:
            payload: TradingSignalEvent or TradingSignal.
        """
        if not self._started or not self._persistence:
            return

        asyncio.create_task(self._persist_signal(payload))

    async def _persist_signal(self, signal: Any) -> None:
        """Persist trading signal to database."""
        try:
            # Handle both TradingSignalEvent wrapper and TradingSignal directly
            if hasattr(signal, "signal"):
                signal = signal.signal
            await self._persistence.save_signal(signal)
            self._signals_persisted += 1
        except Exception as e:
            logger.error(f"Failed to persist signal: {e}")

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
            self._indicators_persisted += 1
        except Exception as e:
            logger.error(f"Failed to persist indicator {indicator} for {symbol}: {e}")

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
            self._confluence_persisted += 1
        except Exception as e:
            logger.error(f"Failed to persist confluence for {symbol}/{timeframe}: {e}")
