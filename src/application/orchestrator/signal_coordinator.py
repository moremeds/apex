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
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...utils.logging_setup import get_logger
from ...utils.timezone import now_utc
from ...domain.events.event_types import EventType
from ...infrastructure.observability import (
    SignalMetrics,
    time_confluence_calculation,
    time_alignment_calculation,
)

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus
    from ...domain.interfaces.signal_persistence import SignalPersistencePort
    from ...domain.signals import BarAggregator, IndicatorEngine, RuleEngine
    from ...domain.signals.divergence import CrossIndicatorAnalyzer, MTFDivergenceAnalyzer

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
        self._last_cache_refresh: Optional[datetime] = None

        # Lazy initialization - components created on start()
        self._bar_aggregators: Dict[str, "BarAggregator"] = {}
        self._indicator_engine: Optional["IndicatorEngine"] = None
        self._rule_engine: Optional["RuleEngine"] = None

        # Confluence calculation components
        self._cross_analyzer: Optional["CrossIndicatorAnalyzer"] = None
        self._mtf_analyzer: Optional["MTFDivergenceAnalyzer"] = None

        # Indicator state cache: (symbol, timeframe) -> {indicator_name: state_dict}
        self._indicator_states: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}

        # Debounce tracking: (symbol, timeframe) -> last_calc_time_ms
        self._last_confluence_calc: Dict[Tuple[str, str], float] = {}
        self._confluence_debounce_ms: float = 500.0

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
        from ...domain.signals.divergence import CrossIndicatorAnalyzer, MTFDivergenceAnalyzer

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
        self._rule_engine = RuleEngine(
            self._event_bus, registry, signal_metrics=self._metrics
        )

        # Start engines (they subscribe to their respective events)
        self._indicator_engine.start()
        self._rule_engine.start()

        # Create confluence analyzers
        self._cross_analyzer = CrossIndicatorAnalyzer()
        self._mtf_analyzer = MTFDivergenceAnalyzer(self._cross_analyzer)

        # Subscribe to tick events for bar aggregation
        self._event_bus.subscribe(EventType.MARKET_DATA_TICK, self._on_market_data_tick)

        # Subscribe to indicator updates for confluence calculation
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

        # Clear state caches (prevents stale data on restart)
        self._indicator_states.clear()
        self._last_confluence_calc.clear()

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
        self._cross_analyzer = None
        self._mtf_analyzer = None

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
        if re.search(r'\d{6,}', symbol):
            return True

        # Pattern: C or P followed by digits (call/put + strike)
        if re.search(r'[CP]\d{5,}', symbol, re.IGNORECASE):
            return True

        # Pattern: colon-separated format (AAPL:OPT:...)
        if ':OPT:' in symbol.upper():
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
            symbol = getattr(payload, 'symbol', None)
            if symbol and self._is_options_symbol(symbol):
                return  # Skip options - don't aggregate their ticks

        # Iterate over a copy to avoid race with stop() clearing the dict
        for aggregator in list(self._bar_aggregators.values()):
            try:
                aggregator.on_tick(payload)
            except Exception as e:
                logger.error(
                    f"Bar aggregation error for {aggregator.timeframe}: {e}"
                )

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
                tf: agg.bars_emitted
                for tf, agg in self._bar_aggregators.items()
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

        Performs gap detection via DuckDB, downloads missing data from IB/Yahoo,
        stores to Parquet, and injects into IndicatorEngine for warmup.

        Args:
            symbols: List of symbols to preload (e.g., ["AAPL", "TSLA"])

        Returns:
            Dict mapping symbol -> number of bars injected
        """
        if not self._indicator_engine:
            logger.warning("Cannot preload bars: indicator engine not initialized")
            return {}

        if not self._historical_data_manager:
            logger.warning("Cannot preload bars: historical data manager not configured")
            return {}

        if not symbols:
            logger.debug("No symbols to preload")
            return {}

        lookback_days = self._preload_config.get("lookback_days", 365)
        slow_warn_sec = self._preload_config.get("slow_preload_warn_sec", 30)
        end_dt = now_utc()
        start_dt = end_dt - timedelta(days=lookback_days)

        results: Dict[str, int] = {}
        start_time = time.monotonic()

        logger.info(
            "Starting bar cache preload (startup)",
            extra={
                "symbols_count": len(symbols),
                "symbols": symbols[:10] if len(symbols) > 10 else symbols,
                "timeframes": self._timeframes,
                "lookback_days": lookback_days,
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat(),
            },
        )

        for timeframe in self._timeframes:
            for symbol in symbols:
                try:
                    # ensure_data: gap check → download → store → return bars
                    bars = await self._historical_data_manager.ensure_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_dt,
                        end=end_dt,
                    )

                    if bars:
                        bar_dicts = [
                            {
                                "timestamp": b.bar_start,
                                "open": b.open,
                                "high": b.high,
                                "low": b.low,
                                "close": b.close,
                                "volume": b.volume or 0,
                            }
                            for b in bars
                        ]
                        # INJECT into indicator engine for warmup
                        count = self._indicator_engine.inject_historical_bars(
                            symbol, timeframe, bar_dicts
                        )
                        results[symbol] = results.get(symbol, 0) + count

                        # Compute indicators immediately to generate signals
                        indicators_computed = await self._indicator_engine.compute_on_history(
                            symbol, timeframe
                        )

                        logger.debug(
                            "Preloaded bars for symbol",
                            extra={
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "bars_injected": count,
                                "indicators_computed": indicators_computed,
                                "date_range": f"{bars[0].bar_start} to {bars[-1].bar_start}",
                            },
                        )
                    else:
                        logger.warning(
                            "No bars returned for symbol",
                            extra={"symbol": symbol, "timeframe": timeframe},
                        )

                except Exception as e:
                    # Don't crash startup on single symbol failure
                    logger.error(
                        "Failed to preload bars for symbol (continuing)",
                        extra={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                    )

        elapsed_sec = time.monotonic() - start_time
        self._last_cache_refresh = now_utc()

        log_level = logging.WARNING if elapsed_sec > slow_warn_sec else logging.INFO
        logger.log(
            log_level,
            "Bar cache preload complete",
            extra={
                "symbols_loaded": len(results),
                "total_bars_injected": sum(results.values()),
                "elapsed_sec": round(elapsed_sec, 2),
                "slow_threshold_sec": slow_warn_sec,
                "cache_refreshed_at": self._last_cache_refresh.isoformat(),
            },
        )

        return results

    async def refresh_disk_cache(self, symbols: List[str]) -> bool:
        """
        Refresh Parquet cache (PERIODIC, disk-only).

        Updates disk cache with new bars from broker.
        Does NOT inject into IndicatorEngine - live BAR_CLOSE events handle that.

        This keeps the Parquet files up-to-date for the next restart.

        Args:
            symbols: List of symbols to refresh

        Returns:
            True if refresh was performed, False if skipped (not due yet)
        """
        refresh_hours = self._preload_config.get("cache_refresh_hours", 24)

        if self._last_cache_refresh:
            elapsed = now_utc() - self._last_cache_refresh
            if elapsed.total_seconds() < refresh_hours * 3600:
                logger.debug(
                    "Cache refresh skipped (not due yet)",
                    extra={
                        "last_refresh": self._last_cache_refresh.isoformat(),
                        "next_refresh_hours": round(
                            refresh_hours - elapsed.total_seconds() / 3600, 2
                        ),
                    },
                )
                return False

        if not self._historical_data_manager:
            logger.debug("Cache refresh skipped: no historical data manager")
            return False

        if not symbols:
            logger.debug("Cache refresh skipped: no symbols")
            return False

        logger.info(
            "Triggering scheduled disk cache refresh (no injection)",
            extra={
                "symbols_count": len(symbols),
                "refresh_interval_hours": refresh_hours,
            },
        )

        # Only refresh disk - call ensure_data but DON'T inject
        lookback_days = self._preload_config.get("lookback_days", 365)
        end_dt = now_utc()
        start_dt = end_dt - timedelta(days=lookback_days)

        for timeframe in self._timeframes:
            for symbol in symbols:
                try:
                    # This updates Parquet files (gap fill + new bars)
                    # We intentionally DON'T inject - live BAR_CLOSE handles new bars
                    await self._historical_data_manager.ensure_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        start=start_dt,
                        end=end_dt,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to refresh cache for symbol",
                        extra={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "error": str(e),
                        },
                    )

        self._last_cache_refresh = now_utc()

        logger.info(
            "Disk cache refresh complete",
            extra={"cache_refreshed_at": self._last_cache_refresh.isoformat()},
        )
        return True

    # -------------------------------------------------------------------------
    # Confluence Calculation (Event-Driven)
    # -------------------------------------------------------------------------

    def _on_indicator_update(self, payload: Any) -> None:
        """
        Aggregate INDICATOR_UPDATE events for confluence calculation.

        Caches indicator states per (symbol, timeframe) and triggers
        debounced confluence calculation when sufficient data arrives.

        Args:
            payload: IndicatorUpdateEvent with symbol, timeframe, indicator, state
        """
        # Log every indicator update received (for debugging signal flow)
        indicator = getattr(payload, "indicator", "?")
        symbol = getattr(payload, "symbol", "?")
        logger.debug(
            "SignalCoordinator received INDICATOR_UPDATE",
            extra={"indicator": indicator, "symbol": symbol, "started": self._started},
        )

        if not self._started:
            logger.warning("SignalCoordinator not started, ignoring INDICATOR_UPDATE")
            return

        symbol = getattr(payload, "symbol", None)
        timeframe = getattr(payload, "timeframe", None)
        indicator = getattr(payload, "indicator", None)
        state = getattr(payload, "state", None)

        # Require symbol, timeframe, indicator - state can be empty dict
        if not symbol or not timeframe or not indicator:
            return

        # state can be None or empty dict - both are valid (indicator computed but no signal)
        if state is None:
            state = {}

        # Cache the indicator state
        key = (symbol, timeframe)
        if key not in self._indicator_states:
            self._indicator_states[key] = {}
        self._indicator_states[key][indicator] = state

        # Persist indicator if persistence enabled
        if self._persistence:
            previous_state = getattr(payload, "previous_state", None)
            timestamp = getattr(payload, "timestamp", None) or now_utc()
            asyncio.create_task(self._persist_indicator(
                symbol=symbol,
                timeframe=timeframe,
                indicator=indicator,
                timestamp=timestamp,
                state=state,
                previous_state=previous_state,
            ))

        # Log indicator cache update
        cached_count = len(self._indicator_states[key])
        logger.debug(
            "Indicator state cached for confluence",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator": indicator,
                "cached_indicators": cached_count,
                "state_keys": list(state.keys()) if state else [],
            },
        )

        # Update cache size metric
        if self._metrics:
            total_entries = sum(
                len(indicators) for indicators in self._indicator_states.values()
            )
            self._metrics.set_indicator_state_cache_size(total_entries)

        # Debounced confluence calculation
        self._maybe_calculate_confluence(symbol, timeframe)

    def _maybe_calculate_confluence(self, symbol: str, timeframe: str) -> None:
        """
        Calculate confluence if debounce period has passed.

        Implements 500ms debounce per (symbol, timeframe) to prevent
        excessive calculations when multiple indicators update in bursts.

        Publishes CONFLUENCE_UPDATE and ALIGNMENT_UPDATE events to the event bus,
        enabling decoupled consumption by TUI and other subscribers.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
        """
        from ...domain.events.domain_events import ConfluenceUpdateEvent, AlignmentUpdateEvent

        key = (symbol, timeframe)
        now_ms = time.time() * 1000

        last_calc = self._last_confluence_calc.get(key, 0)
        if now_ms - last_calc < self._confluence_debounce_ms:
            return

        # Get cached indicator states for this symbol/timeframe
        indicator_states = self._indicator_states.get(key, {})
        if len(indicator_states) < 2:
            # Need at least 2 indicators for meaningful confluence
            # DON'T update debounce time - we want to try again when more indicators arrive
            logger.debug(
                "Confluence skipped: insufficient indicators",
                extra={"symbol": symbol, "timeframe": timeframe, "count": len(indicator_states)},
            )
            return

        # Only update debounce time when we actually have enough indicators to calculate
        self._last_confluence_calc[key] = now_ms

        logger.info(
            "Calculating confluence",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator_count": len(indicator_states),
                "has_cross_analyzer": self._cross_analyzer is not None,
                "has_mtf_analyzer": self._mtf_analyzer is not None,
            },
        )

        # Calculate single-timeframe confluence score and publish event
        if self._cross_analyzer:
            start_time = time.perf_counter()
            try:
                with time_confluence_calculation(self._metrics):
                    score = self._cross_analyzer.analyze(symbol, timeframe, indicator_states)

                # Publish event instead of calling callback
                event = ConfluenceUpdateEvent.from_score(score)
                self._event_bus.publish(EventType.CONFLUENCE_UPDATE, event)

                # Persist confluence if enabled
                if self._persistence:
                    asyncio.create_task(self._persist_confluence(
                        symbol=symbol,
                        timeframe=timeframe,
                        alignment_score=getattr(score, "alignment_score", 0.0),
                        bullish_count=getattr(score, "bullish_count", 0),
                        bearish_count=getattr(score, "bearish_count", 0),
                        neutral_count=getattr(score, "neutral_count", 0),
                        total_indicators=getattr(score, "total_indicators", 0),
                        dominant_direction=getattr(score, "dominant_direction", None),
                    ))

                # Structured debug log with confluence results
                duration_ms = (time.perf_counter() - start_time) * 1000
                alignment_score = getattr(score, "alignment_score", None)
                bullish = getattr(score, "bullish_count", 0)
                bearish = getattr(score, "bearish_count", 0)
                neutral = getattr(score, "neutral_count", 0)

                logger.debug(
                    "Confluence calculated and published",
                    extra={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "alignment_score": alignment_score,
                        "bullish": bullish,
                        "bearish": bearish,
                        "neutral": neutral,
                        "duration_ms": round(duration_ms, 2),
                    },
                )

                # Performance warning for slow calculations
                if duration_ms > 100:
                    logger.warning(
                        "Slow confluence calculation",
                        extra={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "duration_ms": round(duration_ms, 2),
                        },
                    )

            except Exception as e:
                if self._metrics:
                    self._metrics.record_error("confluence", "calculate")
                logger.error(
                    "Confluence calculation failed",
                    extra={"symbol": symbol, "timeframe": timeframe, "error": str(e)},
                )

        # Calculate multi-timeframe alignment if we have data for multiple TFs
        if self._mtf_analyzer:
            states_by_tf: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for (sym, tf), indicators in self._indicator_states.items():
                if sym == symbol:
                    states_by_tf[tf] = indicators

            if len(states_by_tf) >= 2:
                start_time = time.perf_counter()
                try:
                    with time_alignment_calculation(self._metrics):
                        alignment = self._mtf_analyzer.analyze(
                            symbol, list(states_by_tf.keys()), states_by_tf
                        )

                    # Publish event instead of calling callback
                    event = AlignmentUpdateEvent.from_alignment(alignment)
                    self._event_bus.publish(EventType.ALIGNMENT_UPDATE, event)

                    # Structured debug log with alignment results
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    strength = getattr(alignment, "strength", None)
                    direction = getattr(alignment, "direction", None)
                    aligned_tfs = getattr(alignment, "aligned_timeframes", [])

                    logger.debug(
                        "MTF alignment calculated and published",
                        extra={
                            "symbol": symbol,
                            "timeframes": list(states_by_tf.keys()),
                            "strength": strength,
                            "direction": direction,
                            "aligned_count": len(aligned_tfs),
                            "duration_ms": round(duration_ms, 2),
                        },
                    )

                    # Log strong alignments at INFO level
                    if strength == "strong":
                        logger.info(
                            "Strong MTF alignment detected",
                            extra={
                                "symbol": symbol,
                                "strength": strength,
                                "direction": direction,
                                "timeframes": aligned_tfs,
                            },
                        )

                except Exception as e:
                    if self._metrics:
                        self._metrics.record_error("alignment", "calculate")
                    logger.error(
                        "MTF alignment calculation failed",
                        extra={"symbol": symbol, "error": str(e)},
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
