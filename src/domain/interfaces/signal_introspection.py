"""
Signal introspection port for real-time visibility into the signal pipeline.

This port provides READ-ONLY access to live signal pipeline state:
- Indicator values and warmup status
- Rule evaluation history (when trace_mode enabled)
- Cooldown status
- Recent signals

This is ADDITIVE to SignalPersistencePort:
- SignalPersistencePort: Write to DB, read historical
- SignalIntrospectionPort: Read-only, real-time from live caches

Implementations:
- SignalIntrospectionAdapter (infrastructure) - aggregates from live components
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SignalIntrospectionPort(ABC):
    """
    Port for signal pipeline introspection.

    Domain defines the contract; Infrastructure implements via facade.
    TUI/debugging tools query through this interface for real-time visibility.

    Design Notes:
    - Read-only: Does not modify any state
    - Zero-copy where possible: Returns references to existing caches
    - Thread-safe: Implementations must handle concurrent reads
    - Non-blocking: No async methods - all data is in-memory

    Usage:
        # In TUI ViewModel
        class SignalIntrospectionViewModel:
            def __init__(self, introspection: SignalIntrospectionPort):
                self._introspection = introspection

            def refresh(self):
                self.indicators = self._introspection.get_indicator_states()
                self.warmup = self._introspection.get_all_warmup_status()
    """

    # -------------------------------------------------------------------------
    # Indicator State (reuses existing caches)
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_indicator_state(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get current state of a specific indicator.

        Args:
            symbol: Trading symbol (e.g., "AAPL").
            timeframe: Bar timeframe (e.g., "1h", "1d").
            indicator: Indicator name (e.g., "rsi", "macd").

        Returns:
            Indicator state dict or None if not cached.
            Example: {"value": 45.2, "zone": "neutral", "direction": "bullish"}
        """

    @abstractmethod
    def get_indicator_states(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> Dict[tuple, Dict[str, Dict[str, Any]]]:
        """
        Get all cached indicator states, optionally filtered.

        Args:
            symbol: Optional filter by symbol.
            timeframe: Optional filter by timeframe.

        Returns:
            Dict mapping (symbol, timeframe) -> {indicator_name: state_dict}
        """

    # -------------------------------------------------------------------------
    # Warmup Status (from IndicatorEngine._history)
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_warmup_status(
        self,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, Any]:
        """
        Get warmup progress for a symbol/timeframe.

        Per-symbol/timeframe granularity (cheap), not per-indicator.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.

        Returns:
            Dict with warmup info:
            {
                "symbol": str,
                "timeframe": str,
                "bars_loaded": int,
                "bars_required": int,
                "progress_pct": float,  # 0.0 to 1.0
                "status": str,  # "ready" or "warming_up"
            }
        """

    @abstractmethod
    def get_all_warmup_status(self) -> List[Dict[str, Any]]:
        """
        Get warmup status for all symbol/timeframe combinations.

        Returns:
            List of warmup status dicts.
        """

    # -------------------------------------------------------------------------
    # Rule Evaluations (gated behind trace_mode)
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_rule_evaluations(
        self,
        limit: int = 50,
        triggered_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get recent rule evaluation history.

        NOTE: Returns empty list if trace_mode was not enabled.
        Enable trace_mode on RuleEngine for evaluation recording.

        Args:
            limit: Maximum number of evaluations to return.
            triggered_only: If True, only return evaluations that triggered.

        Returns:
            List of evaluation dicts, most recent first:
            {
                "rule_name": str,
                "indicator": str,
                "symbol": str,
                "timeframe": str,
                "triggered": bool,
                "blocked_by_cooldown": bool,
                "condition_met": bool,
                "reason": str,
                "timestamp": datetime,
            }
        """

    # -------------------------------------------------------------------------
    # Cooldown Status (from RuleEngine._last_triggered)
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_cooldown_status(
        self,
        category: str,
        indicator: str,
        symbol: str,
        timeframe: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cooldown status for a specific category:indicator:symbol:tf.

        NOTE: Cooldowns are keyed by category:indicator:symbol:timeframe,
        not by rule name.

        Args:
            category: Signal category (e.g., "momentum", "trend").
            indicator: Indicator name (e.g., "rsi").
            symbol: Trading symbol.
            timeframe: Bar timeframe.

        Returns:
            Cooldown info or None if not in cooldown:
            {
                "category": str,
                "indicator": str,
                "symbol": str,
                "timeframe": str,
                "last_triggered": datetime,
                "cooldown_seconds": int,
                "remaining_seconds": int,
                "active": bool,
            }
        """

    @abstractmethod
    def get_all_cooldowns(self) -> List[Dict[str, Any]]:
        """
        Get all active cooldowns.

        Returns:
            List of active cooldown dicts.
        """

    # -------------------------------------------------------------------------
    # Recent Signals
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_recent_signals(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get recently emitted signals.

        For historical signals, use SignalPersistencePort.get_recent_signals().

        Args:
            limit: Maximum number of signals to return.

        Returns:
            List of signal dicts, most recent first.
        """

    # -------------------------------------------------------------------------
    # Pipeline Statistics
    # -------------------------------------------------------------------------

    @abstractmethod
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics for monitoring.

        Returns:
            Dict with stats:
            {
                "running": bool,
                "bars_processed": int,
                "indicators_computed": int,
                "signals_emitted": int,
                "uptime_seconds": float,
                "indicator_count": int,
                "timeframes": List[str],
            }
        """
