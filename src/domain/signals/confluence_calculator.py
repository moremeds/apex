"""
ConfluenceCalculator - Calculates indicator confluence and multi-timeframe alignment.

Extracted from SignalCoordinator for single responsibility.
This class handles:
- Indicator state caching per (symbol, timeframe)
- Debounced confluence calculation (prevents burst recalculation)
- Single-timeframe confluence scoring via CrossIndicatorAnalyzer
- Multi-timeframe alignment via MTFDivergenceAnalyzer
- Event publishing for downstream consumers (TUI, persistence)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from ...domain.events.event_types import EventType
from ...infrastructure.observability import (
    SignalMetrics,
    time_alignment_calculation,
    time_confluence_calculation,
)
from ...utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus
    from ...domain.signals.divergence import CrossIndicatorAnalyzer, MTFDivergenceAnalyzer

logger = get_logger(__name__)


class ConfluenceCalculator:
    """
    Calculates indicator confluence and multi-timeframe alignment.

    Responsibilities:
    - Cache indicator states as they arrive from IndicatorEngine
    - Debounce calculations to avoid burst recalculation
    - Compute single-timeframe confluence scores
    - Compute multi-timeframe alignment
    - Publish events for downstream consumers

    Usage:
        calculator = ConfluenceCalculator(
            event_bus=event_bus,
            metrics=signal_metrics,
        )
        calculator.start()

        # On each indicator update:
        calculator.on_indicator_update(symbol, timeframe, indicator, state)
    """

    def __init__(
        self,
        event_bus: "EventBus",
        metrics: Optional[SignalMetrics] = None,
        debounce_ms: float = 500.0,
        min_indicators: int = 2,
    ) -> None:
        """
        Initialize confluence calculator.

        Args:
            event_bus: Event bus for publishing confluence events.
            metrics: Optional metrics collector for instrumentation.
            debounce_ms: Minimum interval between calculations (default: 500ms).
            min_indicators: Minimum indicators required for calculation (default: 2).
        """
        self._event_bus = event_bus
        self._metrics = metrics
        self._debounce_ms = debounce_ms
        self._min_indicators = min_indicators

        # Confluence analyzers (lazy init on start)
        self._cross_analyzer: Optional["CrossIndicatorAnalyzer"] = None
        self._mtf_analyzer: Optional["MTFDivergenceAnalyzer"] = None

        # Indicator state cache: (symbol, timeframe) -> {indicator_name: state_dict}
        self._indicator_states: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}

        # Debounce tracking: (symbol, timeframe) -> last_calc_time_ms
        self._last_calc_time: Dict[Tuple[str, str], float] = {}

        # Optional callback for persistence (injected by coordinator)
        self._persistence_callback: Optional[Callable] = None

        self._started = False

    @property
    def indicator_state_count(self) -> int:
        """Total number of cached indicator entries."""
        return sum(len(indicators) for indicators in self._indicator_states.values())

    def start(self) -> None:
        """Start the calculator and create analyzers."""
        if self._started:
            logger.warning("ConfluenceCalculator already started")
            return

        from ...domain.signals.divergence import CrossIndicatorAnalyzer, MTFDivergenceAnalyzer

        self._cross_analyzer = CrossIndicatorAnalyzer()
        self._mtf_analyzer = MTFDivergenceAnalyzer(self._cross_analyzer)
        self._started = True

        logger.info(
            "ConfluenceCalculator started",
            extra={
                "debounce_ms": self._debounce_ms,
                "min_indicators": self._min_indicators,
            },
        )

    def stop(self) -> None:
        """Stop the calculator and release resources."""
        if not self._started:
            return

        self._started = False
        self._indicator_states.clear()
        self._last_calc_time.clear()
        self._cross_analyzer = None
        self._mtf_analyzer = None

        logger.info("ConfluenceCalculator stopped")

    def set_persistence_callback(self, callback: Callable) -> None:
        """
        Set optional persistence callback for confluence data.

        Args:
            callback: Async function (symbol, timeframe, score) -> None
        """
        self._persistence_callback = callback

    def on_indicator_update(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
        state: Dict[str, Any],
    ) -> None:
        """
        Process an indicator update for confluence calculation.

        Caches the indicator state and triggers debounced confluence
        calculation when sufficient data has accumulated.

        Args:
            symbol: Trading symbol (e.g., "AAPL")
            timeframe: Bar timeframe (e.g., "5m", "1h")
            indicator: Indicator name (e.g., "rsi", "macd")
            state: Indicator state dictionary with current values
        """
        if not self._started:
            return

        # Normalize empty state
        if state is None:
            state = {}

        # Cache the indicator state
        key = (symbol, timeframe)
        if key not in self._indicator_states:
            self._indicator_states[key] = {}
        self._indicator_states[key][indicator] = state

        # Update cache size metric
        if self._metrics:
            self._metrics.set_indicator_state_cache_size(self.indicator_state_count)

        logger.debug(
            "Indicator state cached for confluence",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator": indicator,
                "cached_count": len(self._indicator_states[key]),
            },
        )

        # Trigger debounced confluence calculation
        self._maybe_calculate(symbol, timeframe)

    def _maybe_calculate(self, symbol: str, timeframe: str) -> None:
        """
        Calculate confluence if debounce period has passed.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
        """
        key = (symbol, timeframe)
        now_ms = time.time() * 1000

        # Check debounce
        last_calc = self._last_calc_time.get(key, 0)
        if now_ms - last_calc < self._debounce_ms:
            return

        # Check minimum indicators
        indicator_states = self._indicator_states.get(key, {})
        if len(indicator_states) < self._min_indicators:
            logger.debug(
                "Confluence skipped: insufficient indicators",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "count": len(indicator_states),
                    "required": self._min_indicators,
                },
            )
            return

        # Update debounce time
        self._last_calc_time[key] = now_ms

        logger.info(
            "Calculating confluence",
            extra={
                "symbol": symbol,
                "timeframe": timeframe,
                "indicator_count": len(indicator_states),
            },
        )

        # Calculate single-timeframe confluence
        self._calculate_single_timeframe(symbol, timeframe, indicator_states)

        # Calculate multi-timeframe alignment
        self._calculate_multi_timeframe(symbol)

    def _calculate_single_timeframe(
        self,
        symbol: str,
        timeframe: str,
        indicator_states: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Calculate single-timeframe confluence and publish event.

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            indicator_states: Cached indicator states for this symbol/timeframe
        """
        if not self._cross_analyzer:
            return

        from ...domain.events.domain_events import ConfluenceUpdateEvent

        start_time = time.perf_counter()
        try:
            with time_confluence_calculation(self._metrics):
                score = self._cross_analyzer.analyze(symbol, timeframe, indicator_states)

            # Publish event
            event = ConfluenceUpdateEvent.from_score(score)
            self._event_bus.publish(EventType.CONFLUENCE_UPDATE, event)

            # Optional persistence callback
            if self._persistence_callback:
                import asyncio

                asyncio.create_task(
                    self._persistence_callback(
                        symbol=symbol,
                        timeframe=timeframe,
                        alignment_score=getattr(score, "alignment_score", 0.0),
                        bullish_count=getattr(score, "bullish_count", 0),
                        bearish_count=getattr(score, "bearish_count", 0),
                        neutral_count=getattr(score, "neutral_count", 0),
                        total_indicators=getattr(score, "total_indicators", 0),
                        dominant_direction=getattr(score, "dominant_direction", None),
                    )
                )

            # Log results
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                "Confluence calculated and published",
                extra={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "alignment_score": getattr(score, "alignment_score", None),
                    "bullish": getattr(score, "bullish_count", 0),
                    "bearish": getattr(score, "bearish_count", 0),
                    "neutral": getattr(score, "neutral_count", 0),
                    "duration_ms": round(duration_ms, 2),
                },
            )

            # Performance warning
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

    def _calculate_multi_timeframe(self, symbol: str) -> None:
        """
        Calculate multi-timeframe alignment and publish event.

        Args:
            symbol: Trading symbol
        """
        if not self._mtf_analyzer:
            return

        # Gather states across all timeframes for this symbol
        states_by_tf: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for (sym, tf), indicators in self._indicator_states.items():
            if sym == symbol:
                states_by_tf[tf] = indicators

        if len(states_by_tf) < 2:
            return

        from ...domain.events.domain_events import AlignmentUpdateEvent

        start_time = time.perf_counter()
        try:
            with time_alignment_calculation(self._metrics):
                alignment = self._mtf_analyzer.analyze(
                    symbol, list(states_by_tf.keys()), states_by_tf
                )

            # Publish event
            event = AlignmentUpdateEvent.from_alignment(alignment)
            self._event_bus.publish(EventType.ALIGNMENT_UPDATE, event)

            # Log results
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

    def get_cached_states(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get cached indicator states for a symbol.

        Args:
            symbol: Trading symbol
            timeframe: Optional specific timeframe (if None, returns all TFs)

        Returns:
            Dictionary of indicator states, either for specific TF or all TFs combined
        """
        if timeframe:
            return dict(self._indicator_states.get((symbol, timeframe), {}))

        # Combine states across all timeframes
        combined: Dict[str, Dict[str, Any]] = {}
        for (sym, tf), states in self._indicator_states.items():
            if sym == symbol:
                for indicator, state in states.items():
                    combined[f"{indicator}_{tf}"] = state
        return combined

    def clear_cache(self, symbol: Optional[str] = None) -> int:
        """
        Clear cached indicator states.

        Args:
            symbol: Optional specific symbol to clear (if None, clears all)

        Returns:
            Number of entries cleared
        """
        if symbol:
            count = 0
            keys_to_remove = [k for k in self._indicator_states if k[0] == symbol]
            for key in keys_to_remove:
                count += len(self._indicator_states.pop(key, {}))
            return count

        count = self.indicator_state_count
        self._indicator_states.clear()
        self._last_calc_time.clear()
        return count
