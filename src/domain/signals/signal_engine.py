"""SignalEngine — domain service: tick → bar → indicator → signal pipeline.

Creates and wires BarAggregators, IndicatorEngine, and RuleEngine.
Shared across server (WebBridge) and TUI (SignalCoordinator).

This is the single source of truth for indicator state — ONE instance
is shared by all consumers (web dashboard, TUI, advisor).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ...utils.logging_setup import get_logger
from ..events.event_types import EventType
from ..events.priority_event_bus import PriorityEventBus
from ..interfaces.event_bus import EventBus

if TYPE_CHECKING:
    from ...infrastructure.observability import SignalMetrics
    from .data.bar_aggregator import BarAggregator
    from .indicator_engine import IndicatorEngine
    from .rule_engine import RuleEngine

logger = get_logger(__name__)


class SignalEngine:
    """Domain service: tick → bar → indicator → signal pipeline.

    Creates and wires BarAggregators, IndicatorEngine, and RuleEngine.
    Shared across server (WebBridge) and TUI (SignalCoordinator).
    """

    def __init__(
        self,
        event_bus: Union[EventBus, PriorityEventBus],
        timeframes: Optional[List[str]] = None,
        max_workers: int = 4,
        signal_metrics: Optional["SignalMetrics"] = None,
        exclude_options: bool = True,
    ) -> None:
        self._event_bus = event_bus
        self._timeframes = list(dict.fromkeys(timeframes or ["1d"]))
        self._max_workers = max_workers
        self._metrics = signal_metrics
        self._exclude_options = exclude_options

        # Lazy initialization — components created on start()
        self._aggregators: Dict[str, "BarAggregator"] = {}
        self._indicator_engine: Optional["IndicatorEngine"] = None
        self._rule_engine: Optional["RuleEngine"] = None
        self._started = False

    @property
    def indicator_engine(self) -> "IndicatorEngine":
        """Access the indicator engine (must be started)."""
        assert self._indicator_engine is not None, "SignalEngine not started"
        return self._indicator_engine

    @property
    def timeframes(self) -> List[str]:
        """Configured timeframes."""
        return list(self._timeframes)

    @property
    def is_started(self) -> bool:
        """Whether the engine is running."""
        return self._started

    def start(self) -> None:
        """Start engines + subscribe to MARKET_DATA_TICK."""
        if self._started:
            logger.warning("SignalEngine already started")
            return

        from . import BarAggregator, IndicatorEngine, RuleEngine, RuleRegistry
        from .rules import ALL_RULES

        # Create bar aggregators per timeframe
        self._aggregators = {
            tf: BarAggregator(tf, self._event_bus, signal_metrics=self._metrics)
            for tf in self._timeframes
        }

        # Create rule registry with ALL rules
        registry = RuleRegistry()
        registry.add_rules(ALL_RULES)

        # Create engines
        self._indicator_engine = IndicatorEngine(
            self._event_bus,
            max_workers=self._max_workers,
            signal_metrics=self._metrics,
        )
        self._rule_engine = RuleEngine(self._event_bus, registry, signal_metrics=self._metrics)

        # Start engines (they subscribe to their upstream events internally)
        self._indicator_engine.start()
        self._rule_engine.start()

        # Subscribe to tick events for bar aggregation
        self._event_bus.subscribe(EventType.MARKET_DATA_TICK, self._on_tick)

        self._started = True
        logger.info(
            "SignalEngine started",
            extra={
                "timeframes": self._timeframes,
                "indicators": self._indicator_engine.indicator_count,
                "rules": len(registry),
                "max_workers": self._max_workers,
            },
        )

    def stop(self) -> None:
        """Flush bars, stop engines, unsubscribe."""
        if not self._started:
            return

        self._started = False

        # Unsubscribe from tick events
        try:
            self._event_bus.unsubscribe(EventType.MARKET_DATA_TICK, self._on_tick)
        except Exception as e:
            logger.warning("Error unsubscribing from tick events: %s", e)

        # Flush remaining bars BEFORE stopping engines
        for aggregator in list(self._aggregators.values()):
            try:
                aggregator.flush()
            except Exception as e:
                logger.warning("Error flushing aggregator %s: %s", aggregator.timeframe, e)

        # Stop engines
        if self._indicator_engine:
            self._indicator_engine.stop()
        if self._rule_engine:
            self._rule_engine.stop()

        self._aggregators.clear()
        self._indicator_engine = None
        self._rule_engine = None
        logger.info("SignalEngine stopped")

    # ── Tick dispatch ──────────────────────────────────────────

    def _on_tick(self, payload: Any) -> None:
        """Handle MARKET_DATA_TICK by fanning out to all aggregators."""
        if not self._started:
            return

        # Filter out options symbols
        if self._exclude_options:
            symbol = getattr(payload, "symbol", None)
            if symbol and self._is_options_symbol(symbol):
                return

        for aggregator in list(self._aggregators.values()):
            try:
                aggregator.on_tick(payload)
            except Exception as e:
                logger.error("Bar aggregation error for %s: %s", aggregator.timeframe, e)

    def on_tick(self, tick: Any) -> None:
        """Direct tick injection (for Longbridge callback bypass)."""
        if not self._started:
            return
        if self._exclude_options:
            symbol = getattr(tick, "symbol", None)
            if symbol and self._is_options_symbol(symbol):
                return
        for agg in list(self._aggregators.values()):
            try:
                agg.on_tick(tick)
            except Exception as e:
                logger.error("Bar aggregation error for %s: %s", agg.timeframe, e)

    # ── Query API ──────────────────────────────────────────────

    def inject_history(self, symbol: str, tf: str, bars: list) -> int:
        """Inject historical bars for indicator warmup.

        Accepts either bar dicts or objects with open/high/low/close/volume/timestamp.
        Returns number of bars injected.
        """
        if not self._indicator_engine:
            return 0

        bar_dicts = []
        for b in bars:
            if isinstance(b, dict):
                bar_dicts.append(b)
            else:
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
            n = self._indicator_engine.inject_historical_bars(symbol, tf, bar_dicts)
            logger.info("Injected %d historical bars for %s/%s", n, symbol, tf)
            return n
        return 0

    def get_regime_states(self, timeframe: str = "1d") -> Dict[str, Dict[str, Any]]:
        """Get regime state for all symbols from indicator engine cache.

        Returns dict mapping symbol -> {regime, regime_name, confidence, composite_score}.
        """
        if not self._indicator_engine:
            return {}

        results: Dict[str, Dict[str, Any]] = {}
        states = self._indicator_engine.get_all_indicator_states(timeframe=timeframe)
        for (sym, tf, ind), state in states.items():
            if ind == "regime_detector" and state:
                entry: Dict[str, Any] = {
                    "regime": state.get("regime", "R1"),
                    "regime_name": state.get("regime_name", "Unknown"),
                    "confidence": state.get("confidence", 50),
                }
                cs = state.get("composite_score")
                if cs is not None:
                    entry["composite_score"] = cs
                results[sym] = entry
        return results

    def get_indicator_states(self, timeframe: Optional[str] = None) -> Dict[Any, Dict[str, Any]]:
        """Get all indicator states from engine cache."""
        if not self._indicator_engine:
            return {}
        return self._indicator_engine.get_all_indicator_states(timeframe=timeframe)

    def get_history(self, symbol: str, tf: str) -> Any:
        """Get bar history for a symbol/timeframe from engine."""
        if not self._indicator_engine:
            return None
        return self._indicator_engine.get_history(symbol, tf)

    async def compute_on_history(self, symbol: str, tf: str) -> int:
        """Compute indicators on injected history."""
        if not self._indicator_engine:
            return 0
        return await self._indicator_engine.compute_on_history(symbol, tf)

    def get_recent_signals(self, symbol: Optional[str] = None) -> list[dict]:
        """Stub for signal buffer — WebBridge owns this."""
        return []

    def clear_cooldowns(self) -> int:
        """Clear expired signal cooldowns."""
        if self._rule_engine:
            return self._rule_engine.clear_cooldowns()
        return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats: Dict[str, Any] = {
            "started": self._started,
            "timeframes": self._timeframes,
        }
        if self._started and self._indicator_engine and self._rule_engine:
            stats["bars_emitted"] = {tf: agg.bars_emitted for tf, agg in self._aggregators.items()}
            stats["bars_processed"] = self._indicator_engine.bars_processed
            stats["indicator_count"] = self._indicator_engine.indicator_count
            stats["rules_evaluated"] = self._rule_engine.rules_evaluated
            stats["signals_emitted"] = self._rule_engine.signals_emitted
        return stats

    # ── Options symbol filter ──────────────────────────────────

    @staticmethod
    def _is_options_symbol(symbol: str) -> bool:
        """Detect if a symbol is an options symbol based on pattern matching."""
        if not symbol:
            return False
        if len(symbol) <= 6:
            return False
        if len(symbol) > 10:
            return True
        if re.search(r"\d{6,}", symbol):
            return True
        if re.search(r"[CP]\d{5,}", symbol, re.IGNORECASE):
            return True
        if ":OPT:" in symbol.upper():
            return True
        return False
