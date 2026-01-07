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

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ...utils.logging_setup import get_logger
from ...domain.events.event_types import EventType

if TYPE_CHECKING:
    from ...domain.interfaces.event_bus import EventBus
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
    ) -> None:
        """
        Initialize the signal coordinator.

        Args:
            event_bus: Event bus for subscriptions and publishing
            timeframes: Bar timeframes to aggregate (default: ["1m", "5m", "1h"])
            max_workers: ThreadPool size for indicator calculations
            enabled: Whether to enable the signal pipeline (can be disabled for testing)
        """
        self._event_bus = event_bus
        self._timeframes = list(dict.fromkeys(timeframes or ["1m", "5m", "1h"]))
        self._max_workers = max_workers
        self._enabled = enabled

        # Lazy initialization - components created on start()
        self._bar_aggregators: Dict[str, "BarAggregator"] = {}
        self._indicator_engine: Optional["IndicatorEngine"] = None
        self._rule_engine: Optional["RuleEngine"] = None

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

        # Create bar aggregators for each timeframe
        self._bar_aggregators = {
            tf: BarAggregator(tf, self._event_bus)
            for tf in self._timeframes
        }

        # Create rule registry with pre-built rules
        registry = RuleRegistry()
        registry.add_rules(ALL_RULES)

        # Create engines
        self._indicator_engine = IndicatorEngine(
            self._event_bus,
            max_workers=self._max_workers,
        )
        self._rule_engine = RuleEngine(self._event_bus, registry)

        # Start engines (they subscribe to their respective events)
        self._indicator_engine.start()
        self._rule_engine.start()

        # Subscribe to tick events for bar aggregation
        self._event_bus.subscribe(EventType.MARKET_DATA_TICK, self._on_market_data_tick)

        self._started = True
        logger.info(
            f"SignalCoordinator started: timeframes={self._timeframes}, "
            f"indicators={self._indicator_engine.indicator_count}, "
            f"rules={len(registry)}, max_workers={self._max_workers}"
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
