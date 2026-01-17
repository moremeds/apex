"""
SignalIntrospectionAdapter - Facade for signal pipeline introspection.

Aggregates data from production components to implement SignalIntrospectionPort:
- ConfluenceCalculator._indicator_states (via SignalCoordinator)
- IndicatorEngine (warmup status, indicator states)
- RuleEngine (evaluation history, cooldowns)

This adapter provides READ-ONLY access to live signal pipeline state.
It does NOT duplicate buffers - queries are delegated to existing components.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from threading import Lock
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

from ...domain.interfaces.signal_introspection import SignalIntrospectionPort
from ...utils.logging_setup import get_logger

if TYPE_CHECKING:
    from ...application.orchestrator.signal_coordinator import SignalCoordinator
    from ...domain.signals.indicator_engine import IndicatorEngine
    from ...domain.signals.rule_engine import RuleEngine

logger = get_logger(__name__)


class SignalIntrospectionAdapter(SignalIntrospectionPort):
    """
    Facade aggregating data from production signal pipeline components.

    Does NOT duplicate state - queries live from:
    - SignalCoordinator → ConfluenceCalculator._indicator_states
    - IndicatorEngine._history, _previous_states
    - RuleEngine._last_triggered, _evaluation_history

    Thread-safe: All underlying components use locks for concurrent access.

    Usage:
        # Wire into orchestrator after SignalCoordinator is created
        introspection = SignalIntrospectionAdapter(
            signal_coordinator=signal_coordinator,
            indicator_engine=signal_coordinator._indicator_engine,
            rule_engine=signal_coordinator._rule_engine,
        )

        # Query indicator states
        states = introspection.get_indicator_states(symbol="AAPL")

        # Query warmup status
        warmup = introspection.get_all_warmup_status()

        # Query rule evaluations (requires trace_mode on RuleEngine)
        evals = introspection.get_rule_evaluations(limit=20)
    """

    def __init__(
        self,
        signal_coordinator: "SignalCoordinator",
        indicator_engine: "IndicatorEngine",
        rule_engine: "RuleEngine",
    ) -> None:
        """
        Initialize the introspection adapter.

        Args:
            signal_coordinator: Production SignalCoordinator with ConfluenceCalculator.
            indicator_engine: IndicatorEngine for warmup and indicator state queries.
            rule_engine: RuleEngine for cooldown and evaluation history queries.
        """
        self._signal_coordinator = signal_coordinator
        self._indicator_engine = indicator_engine
        self._rule_engine = rule_engine

        # Recent signals buffer (ring buffer for in-memory TUI display)
        # Separate from persistence - this is for real-time dashboard only
        self._recent_signals: Deque[Dict[str, Any]] = deque(maxlen=100)
        self._signals_lock = Lock()  # Protects _recent_signals

        # Pipeline start time for uptime calculation
        self._start_time: Optional[datetime] = None

        logger.info("SignalIntrospectionAdapter initialized")

    def start(self) -> None:
        """Record pipeline start time for uptime tracking."""
        from ...utils.timezone import now_utc

        self._start_time = now_utc()

    # -------------------------------------------------------------------------
    # Indicator State (from ConfluenceCalculator + IndicatorEngine)
    # -------------------------------------------------------------------------

    def get_indicator_state(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
    ) -> Optional[Dict[str, Any]]:
        """Get current state of a specific indicator."""
        return self._indicator_engine.get_indicator_state(symbol, timeframe, indicator)

    def get_indicator_states(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
        """
        Get all cached indicator states, optionally filtered.

        Queries from ConfluenceCalculator (primary) which has per-(symbol, timeframe)
        caching with indicator state dictionaries.
        """
        confluence_calc = self._signal_coordinator._confluence_calculator
        if confluence_calc is None:
            return {}

        # ConfluenceCalculator stores: (symbol, tf) -> {indicator_name: state_dict}
        all_states = confluence_calc._indicator_states

        if symbol is None and timeframe is None:
            return dict(all_states)

        # Filter by symbol and/or timeframe
        result: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = {}
        for (sym, tf), indicators in all_states.items():
            if symbol is not None and sym != symbol:
                continue
            if timeframe is not None and tf != timeframe:
                continue
            result[(sym, tf)] = dict(indicators)

        return result

    # -------------------------------------------------------------------------
    # Warmup Status (from IndicatorEngine)
    # -------------------------------------------------------------------------

    def get_warmup_status(
        self,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, Any]:
        """Get warmup progress for a symbol/timeframe."""
        return self._indicator_engine.get_warmup_status(symbol, timeframe)

    def get_all_warmup_status(self) -> List[Dict[str, Any]]:
        """Get warmup status for all symbol/timeframe combinations."""
        return self._indicator_engine.get_all_warmup_status()

    # -------------------------------------------------------------------------
    # Rule Evaluations (from RuleEngine - gated behind trace_mode)
    # -------------------------------------------------------------------------

    def get_rule_evaluations(
        self,
        limit: int = 50,
        triggered_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get recent rule evaluation history.

        NOTE: Returns empty list if trace_mode was not enabled on RuleEngine.
        """
        return self._rule_engine.get_evaluation_history(
            limit=limit,
            triggered_only=triggered_only,
        )

    # -------------------------------------------------------------------------
    # Cooldown Status (from RuleEngine)
    # -------------------------------------------------------------------------

    def get_cooldown_status(
        self,
        category: str,
        indicator: str,
        symbol: str,
        timeframe: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cooldown status for a specific category:indicator:symbol:timeframe."""
        return self._rule_engine.get_cooldown_status(
            category=category,
            indicator=indicator,
            symbol=symbol,
            timeframe=timeframe,
        )

    def get_all_cooldowns(self) -> List[Dict[str, Any]]:
        """Get all active cooldowns."""
        return self._rule_engine.get_all_cooldowns()

    # -------------------------------------------------------------------------
    # Recent Signals (in-memory buffer)
    # -------------------------------------------------------------------------

    def record_signal(self, signal: Dict[str, Any]) -> None:
        """
        Record a signal for introspection.

        Called by SignalCoordinator when TRADING_SIGNAL events are published.
        This enables real-time TUI display without database queries.
        """
        with self._signals_lock:
            self._recent_signals.append(signal)

    def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently emitted signals."""
        with self._signals_lock:
            signals = list(self._recent_signals)
        # Most recent first
        return signals[-limit:][::-1]

    # -------------------------------------------------------------------------
    # Pipeline Statistics
    # -------------------------------------------------------------------------

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics for monitoring."""
        from ...utils.timezone import now_utc

        # Calculate uptime
        uptime_seconds = 0.0
        if self._start_time:
            uptime_seconds = (now_utc() - self._start_time).total_seconds()

        # Get indicator engine stats
        indicator_count = len(self._indicator_engine._indicators) if self._indicator_engine else 0

        # Get recent signals count thread-safely
        with self._signals_lock:
            recent_signals_count = len(self._recent_signals)

        return {
            "running": self._signal_coordinator.is_started,
            "bars_processed": (
                self._indicator_engine.bars_processed if self._indicator_engine else 0
            ),
            "rules_evaluated": self._rule_engine.rules_evaluated if self._rule_engine else 0,
            "signals_emitted": self._rule_engine.signals_emitted if self._rule_engine else 0,
            "uptime_seconds": uptime_seconds,
            "indicator_count": indicator_count,
            "timeframes": self._signal_coordinator.timeframes,
            "recent_signals_count": recent_signals_count,
            "cooldowns_active": (
                len(self._rule_engine.get_all_cooldowns()) if self._rule_engine else 0
            ),
        }
