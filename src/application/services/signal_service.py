"""
SignalService - Standalone pluggable signal generation service.

A unified signal generation service that:
- Runs independently (no TUI or event bus dependency)
- Logs all pipeline events with tree-style formatting
- Loads indicators and rules from YAML configuration
- Can be validated via a standalone runner

Pipeline flow:
    Tick → BarAggregator → BAR_CLOSE
    BAR_CLOSE → IndicatorEngine → INDICATOR_COMPUTED
    INDICATOR_COMPUTED → RuleEngine → SIGNAL_GENERATED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from src.domain.signals.data.bar_aggregator import BarAggregator
from src.domain.signals.indicator_engine import IndicatorEngine
from src.domain.signals.models import TradingSignal
from src.domain.signals.rule_engine import RuleEngine, RuleRegistry
from src.utils.logging_setup import get_logger
from src.utils.timezone import now_utc

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


# Swing trading timeframes (no intraday)
DEFAULT_TIMEFRAMES = ["30m", "1h", "4h", "1d", "1w"]


@dataclass
class SignalServiceStats:
    """Statistics for signal service operation."""

    ticks_received: int = 0
    bars_emitted: Dict[str, int] = field(default_factory=dict)
    indicators_computed: Dict[str, int] = field(default_factory=dict)
    rules_evaluated: int = 0
    signals_generated: int = 0
    signals_by_direction: Dict[str, int] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def log_summary(self) -> None:
        """Log a summary of statistics with tree-style formatting."""
        runtime = "N/A"
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            runtime = f"{delta.total_seconds() / 60:.1f} minutes"

        logger.info("═" * 55)
        logger.info("SIGNAL SERVICE SUMMARY")
        logger.info("═" * 55)
        logger.info(f"Runtime: {runtime}")
        logger.info("")
        logger.info("BARS AGGREGATED:")
        for tf, count in sorted(self.bars_emitted.items()):
            logger.info(f"├─ {tf}: {count} bars")
        logger.info("")
        logger.info("INDICATORS COMPUTED:")
        for name, count in sorted(self.indicators_computed.items()):
            logger.info(f"├─ {name}: {count} computations")
        logger.info("")
        logger.info("SIGNALS GENERATED:")
        logger.info(f"├─ Total: {self.signals_generated} signals")
        for direction, count in sorted(self.signals_by_direction.items()):
            logger.info(f"├─ {direction}: {count}")
        logger.info("═" * 55)


class InternalEventBus:
    """
    Lightweight internal event bus for SignalService.

    Replaces PriorityEventBus for standalone operation.
    Simply routes events to registered callbacks.
    """

    def __init__(self) -> None:
        self._callbacks: Dict[str, List[Callable[[Any], None]]] = {}

    def subscribe(self, event_type: Any, callback: Callable[[Any], None]) -> None:
        """Subscribe to an event type."""
        key = str(event_type)
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def publish(self, event_type: Any, payload: Any, priority: Optional[int] = None) -> None:
        """Publish an event to all subscribers."""
        key = str(event_type)
        for callback in self._callbacks.get(key, []):
            try:
                callback(payload)
            except Exception as e:
                logger.error(f"Event callback error: {e}", exc_info=True)


class SignalService:
    """
    Unified signal generation service.

    Lifecycle:
        service = SignalService.from_config(config)
        service.start()
        # Feed ticks via on_tick() or inject_bars()
        service.stop()
        service.stats.log_summary()
    """

    def __init__(
        self,
        timeframes: Optional[List[str]] = None,
        indicator_config: Optional[Dict[str, Any]] = None,
        rule_config: Optional[Dict[str, Any]] = None,
        max_workers: int = 4,
        log_signals: bool = True,
    ) -> None:
        """
        Initialize the signal service.

        Args:
            timeframes: Bar timeframes to aggregate (default: swing trading TFs)
            indicator_config: Indicator configuration dict (name -> settings)
            rule_config: Rule configuration dict (name -> settings)
            max_workers: ThreadPool size for indicator calculations
            log_signals: Whether to log signals when generated
        """
        self._timeframes = timeframes or DEFAULT_TIMEFRAMES
        self._indicator_config = indicator_config or {}
        self._rule_config = rule_config or {}
        self._max_workers = max_workers
        self._log_signals = log_signals

        # Internal event bus (replaces PriorityEventBus)
        self._event_bus = InternalEventBus()

        # Pipeline components (created on start)
        self._bar_aggregators: Dict[str, BarAggregator] = {}
        self._indicator_engine: Optional[IndicatorEngine] = None
        self._rule_engine: Optional[RuleEngine] = None

        # Signal callback (for external consumers)
        self._signal_callback: Optional[Callable[[TradingSignal], None]] = None

        # Statistics
        self._stats = SignalServiceStats()
        self._started = False

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SignalService":
        """
        Create service from configuration dict.

        Expected config structure:
            timeframes: [30m, 1h, 4h, 1d, 1w]
            max_workers: 4
            log_signals: true
            indicators:
              rsi:
                enabled: true
                params: {period: 14}
            rules:
              rsi_oversold_exit:
                enabled: true
                indicator: rsi
                ...
        """
        return cls(
            timeframes=config.get("timeframes", DEFAULT_TIMEFRAMES),
            indicator_config=config.get("indicators", {}),
            rule_config=config.get("rules", {}),
            max_workers=config.get("max_workers", 4),
            log_signals=config.get("log_signals", True),
        )

    @property
    def stats(self) -> SignalServiceStats:
        """Access service statistics."""
        return self._stats

    @property
    def is_started(self) -> bool:
        """Whether the service is running."""
        return self._started

    def set_signal_callback(self, callback: Callable[[TradingSignal], None]) -> None:
        """Set callback for signal generation (for external consumers)."""
        self._signal_callback = callback

    def start(self) -> None:
        """
        Initialize and start the pipeline.

        Creates:
        - BarAggregators for each timeframe
        - IndicatorEngine with configured indicators
        - RuleEngine with configured rules
        """
        if self._started:
            logger.warning("SignalService already started")
            return

        self._stats.start_time = now_utc()

        # Create bar aggregators for each timeframe
        for tf in self._timeframes:
            self._bar_aggregators[tf] = BarAggregator(
                timeframe=tf,
                event_bus=self._event_bus,
            )
            self._stats.bars_emitted[tf] = 0

        # Create indicator engine
        self._indicator_engine = IndicatorEngine(
            event_bus=self._event_bus,
            max_workers=self._max_workers,
        )
        # TODO: Load indicators from config (Milestone 3)

        # Create rule registry from config and engine
        registry = RuleRegistry.from_config({"rules": self._rule_config})

        self._rule_engine = RuleEngine(
            event_bus=self._event_bus,
            registry=registry,
            trace_mode=True,  # Enable verbose logging
        )

        # Wire internal event handlers
        self._wire_event_handlers()

        # Start engines
        self._indicator_engine.start()
        self._rule_engine.start()

        self._started = True

        # Log initialization with tree-style format
        indicator_names = [ind.name for ind in self._indicator_engine._indicators]
        rule_count = len(registry)

        logger.info("SignalService initialized")
        logger.info(f"├─ timeframes: {self._timeframes}")
        logger.info(
            f"├─ indicators: {len(indicator_names)} loaded ({', '.join(indicator_names[:5])}{'...' if len(indicator_names) > 5 else ''})"
        )
        logger.info(f"└─ rules: {rule_count} loaded")

    def stop(self) -> None:
        """Stop the service and clean up resources."""
        if not self._started:
            return

        self._stats.end_time = now_utc()

        # Flush remaining bars
        for tf, aggregator in self._bar_aggregators.items():
            events = aggregator.flush()
            self._stats.bars_emitted[tf] += len(events)

        # Stop engines
        if self._indicator_engine:
            self._indicator_engine.stop()
        if self._rule_engine:
            self._rule_engine.stop()

        self._started = False
        logger.info("SignalService stopped")

    def on_tick(self, tick: Any) -> None:
        """
        Process a single market data tick.

        The tick flows through:
        1. BarAggregators (one per timeframe)
        2. When bar closes → IndicatorEngine
        3. When indicator updates → RuleEngine
        4. When rule triggers → Signal logged/callback

        Args:
            tick: Tick data with symbol, price, timestamp, etc.
        """
        if not self._started:
            return

        self._stats.ticks_received += 1

        # Fan out to all aggregators
        for aggregator in self._bar_aggregators.values():
            aggregator.on_tick(tick)

    async def inject_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: List[Dict[str, Any]],
    ) -> int:
        """
        Inject historical bars directly (bypasses tick aggregation).

        Used for:
        - Warmup with historical data
        - Backtesting
        - Testing

        Args:
            symbol: Trading symbol
            timeframe: Bar timeframe
            bars: List of bar dicts with OHLCV data

        Returns:
            Number of bars injected
        """
        if not self._started or not self._indicator_engine:
            return 0

        injected = self._indicator_engine.inject_historical_bars(
            symbol=symbol,
            timeframe=timeframe,
            bar_dicts=bars,
        )

        # Trigger indicator computation on injected history
        await self._indicator_engine.compute_on_history(symbol, timeframe)

        return injected

    def _wire_event_handlers(self) -> None:
        """Wire internal event handlers for logging and stats."""
        from src.domain.events.event_types import EventType

        # Bar close logging
        self._event_bus.subscribe(EventType.BAR_CLOSE, self._on_bar_close)

        # Indicator update logging
        self._event_bus.subscribe(EventType.INDICATOR_UPDATE, self._on_indicator_update)

        # Signal generation logging
        self._event_bus.subscribe(EventType.TRADING_SIGNAL, self._on_trading_signal)

    def _on_bar_close(self, event: Any) -> None:
        """Handle BAR_CLOSE event - log and update stats."""
        tf = getattr(event, "timeframe", "?")
        symbol = getattr(event, "symbol", "?")

        self._stats.bars_emitted[tf] = self._stats.bars_emitted.get(tf, 0) + 1

        logger.info(f"BAR_CLOSE symbol={symbol} tf={tf}")
        logger.info(
            f"├─ open={getattr(event, 'open', 0):.2f} "
            f"high={getattr(event, 'high', 0):.2f} "
            f"low={getattr(event, 'low', 0):.2f} "
            f"close={getattr(event, 'close', 0):.2f}"
        )
        bar_end = getattr(event, "bar_end", None)
        logger.info(f"└─ volume={getattr(event, 'volume', 0)} bar_end={bar_end}")

    def _on_indicator_update(self, event: Any) -> None:
        """Handle INDICATOR_UPDATE event - log and update stats."""
        indicator = getattr(event, "indicator", "?")
        symbol = getattr(event, "symbol", "?")
        tf = getattr(event, "timeframe", "?")
        state = getattr(event, "state", {})

        self._stats.indicators_computed[indicator] = (
            self._stats.indicators_computed.get(indicator, 0) + 1
        )

        # Extract key state values for logging
        value = state.get("value", "N/A")
        zone = state.get("zone", state.get("direction", state.get("trend", "")))

        logger.info(f"INDICATOR_COMPUTED symbol={symbol} tf={tf} indicator={indicator}")
        logger.info(f"├─ value={value} {f'zone={zone}' if zone else ''}")

        # Show warmup status
        if self._indicator_engine:
            warmup = self._indicator_engine.get_warmup_status(symbol, tf)
            bars_loaded = warmup.get("bars_loaded", 0)
            bars_required = warmup.get("bars_required", 0)
            status = warmup.get("status", "?")
            logger.info(f"└─ warmup={bars_loaded}/{bars_required} ({status})")

    def _on_trading_signal(self, event: Any) -> None:
        """Handle TRADING_SIGNAL event - log prominently and update stats."""
        # Extract signal (may be wrapped in TradingSignalEvent)
        signal = getattr(event, "signal", event)

        self._stats.signals_generated += 1
        direction = getattr(signal, "direction", None)
        if direction:
            dir_str = direction.value if hasattr(direction, "value") else str(direction)
            self._stats.signals_by_direction[dir_str] = (
                self._stats.signals_by_direction.get(dir_str, 0) + 1
            )

        if self._log_signals:
            self._log_signal(signal)

        # Call external callback if set
        if self._signal_callback:
            self._signal_callback(signal)

    def _log_signal(self, signal: TradingSignal) -> None:
        """Log a signal with prominent tree-style formatting."""
        logger.info("═" * 55)
        logger.info("SIGNAL GENERATED")
        logger.info(f"├─ signal_id: {signal.signal_id}")
        logger.info(f"├─ symbol: {signal.symbol}")
        logger.info(f"├─ indicator: {signal.indicator}")

        direction = (
            signal.direction.value if hasattr(signal.direction, "value") else signal.direction
        )
        logger.info(f"├─ direction: {direction}")

        strength = signal.strength
        priority = signal.priority.value if hasattr(signal.priority, "value") else signal.priority
        logger.info(f"├─ strength: {strength} ({priority})")

        logger.info(f"├─ timeframe: {signal.timeframe}")
        logger.info(f"├─ value: {signal.current_value} → threshold: {signal.threshold}")
        logger.info(f"└─ rule: {signal.trigger_rule}")
        logger.info("═" * 55)
