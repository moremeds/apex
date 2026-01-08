"""
SignalCoordinator - Wires the signal pipeline into the event bus.

Manages the complete signal generation pipeline:
- Tick aggregation into bars (MARKET_DATA_TICK → BAR_CLOSE)
- Indicator computation (BAR_CLOSE → INDICATOR_UPDATE)
- Signal rule evaluation (INDICATOR_UPDATE → TRADING_SIGNAL)

This coordinator follows the same pattern as DataCoordinator and
SnapshotCoordinator, keeping the Orchestrator thin.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...utils.logging_setup import get_logger
from ...domain.events.event_types import EventType
from ...infrastructure.observability import (
    SignalMetrics,
    time_confluence_calculation,
    time_alignment_calculation,
)

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus
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
            timeframes=["1m", "5m", "1h"],
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
    ) -> None:
        """
        Initialize the signal coordinator.

        Args:
            event_bus: Event bus for subscriptions and publishing
            timeframes: Bar timeframes to aggregate (default: ["1m", "5m", "1h"])
            max_workers: ThreadPool size for indicator calculations
            enabled: Whether to enable the signal pipeline (can be disabled for testing)
            signal_metrics: Metrics collector for pipeline instrumentation
        """
        self._event_bus = event_bus
        self._timeframes = list(dict.fromkeys(timeframes or ["1m", "5m", "1h"]))
        self._max_workers = max_workers
        self._enabled = enabled
        self._metrics = signal_metrics

        # Lazy initialization - components created on start()
        self._bar_aggregators: Dict[str, "BarAggregator"] = {}
        self._indicator_engine: Optional["IndicatorEngine"] = None
        self._rule_engine: Optional["RuleEngine"] = None

        # Confluence calculation components
        self._cross_analyzer: Optional["CrossIndicatorAnalyzer"] = None
        self._mtf_analyzer: Optional["MTFDivergenceAnalyzer"] = None

        # Indicator state cache: (symbol, timeframe) -> {indicator_name: state_dict}
        self._indicator_states: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}

        # Confluence callbacks
        self._confluence_callback: Optional[Callable[[Any], None]] = None
        self._alignment_callback: Optional[Callable[[Any], None]] = None

        # Debounce tracking: (symbol, timeframe) -> last_calc_time_ms
        self._last_confluence_calc: Dict[Tuple[str, str], float] = {}
        self._confluence_debounce_ms: float = 500.0

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

        self._started = True

        # Structured startup log
        logger.info(
            "Signal pipeline started",
            extra={
                "timeframes": self._timeframes,
                "indicators": self._indicator_engine.indicator_count,
                "rules": len(registry),
                "max_workers": self._max_workers,
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

    def _on_market_data_tick(self, payload: Any) -> None:
        """
        Handle MARKET_DATA_TICK event by fanning out to all aggregators.

        This is the entry point for the signal pipeline. Each aggregator
        independently tracks bar state for its timeframe.
        """
        # Guard against processing after stop
        if not self._started:
            return

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
    # Confluence Calculation
    # -------------------------------------------------------------------------

    def set_confluence_callback(
        self, callback: Optional[Callable[[Any], None]]
    ) -> None:
        """
        Register callback to receive ConfluenceScore updates.

        Args:
            callback: Function to call with ConfluenceScore objects
        """
        self._confluence_callback = callback

    def set_alignment_callback(
        self, callback: Optional[Callable[[Any], None]]
    ) -> None:
        """
        Register callback to receive MTFAlignment updates.

        Args:
            callback: Function to call with MTFAlignment objects
        """
        self._alignment_callback = callback

    def _on_indicator_update(self, payload: Any) -> None:
        """
        Aggregate INDICATOR_UPDATE events for confluence calculation.

        Caches indicator states per (symbol, timeframe) and triggers
        debounced confluence calculation when sufficient data arrives.

        Args:
            payload: IndicatorUpdateEvent with symbol, timeframe, indicator, state
        """
        if not self._started:
            return

        symbol = getattr(payload, "symbol", None)
        timeframe = getattr(payload, "timeframe", None)
        indicator = getattr(payload, "indicator", None)
        state = getattr(payload, "state", None)

        if not all([symbol, timeframe, indicator, state]):
            return

        # Cache the indicator state
        key = (symbol, timeframe)
        if key not in self._indicator_states:
            self._indicator_states[key] = {}
        self._indicator_states[key][indicator] = state

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

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
        """
        key = (symbol, timeframe)
        now_ms = time.time() * 1000

        last_calc = self._last_confluence_calc.get(key, 0)
        if now_ms - last_calc < self._confluence_debounce_ms:
            return

        self._last_confluence_calc[key] = now_ms

        # Get cached indicator states for this symbol/timeframe
        indicator_states = self._indicator_states.get(key, {})
        if len(indicator_states) < 2:
            # Need at least 2 indicators for meaningful confluence
            return

        # Calculate single-timeframe confluence score
        if self._cross_analyzer and self._confluence_callback:
            start_time = time.perf_counter()
            try:
                with time_confluence_calculation(self._metrics):
                    score = self._cross_analyzer.analyze(symbol, timeframe, indicator_states)

                self._confluence_callback(score)

                # Structured debug log with confluence results
                duration_ms = (time.perf_counter() - start_time) * 1000
                alignment_score = getattr(score, "alignment_score", None)
                bullish = getattr(score, "bullish_count", 0)
                bearish = getattr(score, "bearish_count", 0)
                neutral = getattr(score, "neutral_count", 0)

                logger.debug(
                    "Confluence calculated",
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
        if self._mtf_analyzer and self._alignment_callback:
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

                    self._alignment_callback(alignment)

                    # Structured debug log with alignment results
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    strength = getattr(alignment, "strength", None)
                    direction = getattr(alignment, "direction", None)
                    aligned_tfs = getattr(alignment, "aligned_timeframes", [])

                    logger.debug(
                        "MTF alignment calculated",
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
