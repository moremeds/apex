"""
Signal persistence port for technical analysis signals and indicators.

This port defines the contract for persisting TA signals and indicator values.
Domain components (RuleEngine, SignalCoordinator) depend on this abstract port,
not on concrete SQL implementations - enabling testing and backend swaps.

Implementations:
- TASignalRepository (PostgreSQL/TimescaleDB) - production
- InMemorySignalStore - testing/validation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..signals.models import TradingSignal


class SignalPersistencePort(ABC):
    """
    Port for signal and indicator persistence operations.

    Domain defines the contract; Infrastructure implements it.
    This allows RuleEngine to save signals without knowing about SQL/TimescaleDB.

    Usage:
        # In RuleEngine (domain layer)
        class RuleEngine:
            def __init__(self, persistence: Optional[SignalPersistencePort] = None):
                self._persistence = persistence

            async def _emit_signal(self, signal: TradingSignal) -> None:
                if self._persistence:
                    await self._persistence.save_signal(signal)
                # ... publish to event bus

        # In production (main.py)
        repo = TASignalRepository(db)  # Infrastructure adapter
        rule_engine = RuleEngine(persistence=repo)
    """

    # -------------------------------------------------------------------------
    # Signal Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def save_signal(self, signal: "TradingSignal") -> None:
        """
        Persist a trading signal.

        Called by RuleEngine immediately after signal generation.
        Implementation should use UPSERT for idempotency.

        Args:
            signal: TradingSignal domain object to persist.
        """
        pass

    @abstractmethod
    async def get_recent_signals(
        self,
        limit: int = 100,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List["TradingSignal"]:
        """
        Retrieve recent signals for TUI startup.

        Used by ApexApp.on_mount() to load historical signals
        before subscribing to live events.

        Args:
            limit: Maximum number of signals to return.
            symbol: Optional filter by symbol.
            timeframe: Optional filter by timeframe.
            category: Optional filter by category (momentum, trend, etc.).

        Returns:
            List of TradingSignal objects, ordered by time DESC.
        """
        pass

    @abstractmethod
    async def get_signals_since(
        self,
        since: datetime,
        symbol: Optional[str] = None,
        indicator: Optional[str] = None,
        limit: int = 1000,
    ) -> List["TradingSignal"]:
        """
        Retrieve signals since a given timestamp.

        Useful for TUI reconnection or backfill scenarios.

        Args:
            since: Start timestamp (exclusive).
            symbol: Optional filter by symbol.
            indicator: Optional filter by indicator.
            limit: Maximum number of signals to return.

        Returns:
            List of TradingSignal objects, ordered by time ASC.
        """
        pass

    # -------------------------------------------------------------------------
    # Indicator Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def save_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
        timestamp: datetime,
        state: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]] = None,
        bar_close: Optional[float] = None,
    ) -> None:
        """
        Persist indicator state.

        Called by SignalCoordinator/TASignalService on each INDICATOR_UPDATE.
        Stores time-series data for charting and analysis.

        Args:
            symbol: Trading symbol (e.g., "AAPL").
            timeframe: Bar timeframe (e.g., "1d", "4h").
            indicator: Indicator name (e.g., "rsi", "macd").
            timestamp: Bar close timestamp.
            state: Current indicator state dict (e.g., {"value": 45.2, "zone": "oversold"}).
            previous_state: Previous indicator state for transition detection.
            bar_close: Reference bar close price.
        """
        pass

    @abstractmethod
    async def get_indicator_history(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve indicator history for charting.

        Returns time-series data for indicator visualization in Lab view.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            indicator: Indicator name.
            start: Start timestamp (inclusive). If None, returns most recent.
            end: End timestamp (inclusive). If None, up to now.
            limit: Maximum number of records to return.

        Returns:
            List of dicts with keys: time, state, previous_state, bar_close.
            Ordered by time ASC for charting.
        """
        pass

    @abstractmethod
    async def get_latest_indicator(
        self,
        symbol: str,
        timeframe: str,
        indicator: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest indicator value for a symbol/timeframe/indicator combo.

        Useful for quick state checks without loading full history.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            indicator: Indicator name.

        Returns:
            Dict with time, state, previous_state, bar_close or None if not found.
        """
        pass

    # -------------------------------------------------------------------------
    # Confluence Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def save_confluence(
        self,
        symbol: str,
        timeframe: str,
        timestamp: datetime,
        alignment_score: float,
        bullish_count: int,
        bearish_count: int,
        neutral_count: int,
        total_indicators: int,
        dominant_direction: Optional[str] = None,
    ) -> None:
        """
        Persist confluence score.

        Called by SignalCoordinator after multi-indicator analysis.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            timestamp: Calculation timestamp.
            alignment_score: -1.0 to +1.0 alignment score.
            bullish_count: Number of bullish indicators.
            bearish_count: Number of bearish indicators.
            neutral_count: Number of neutral indicators.
            total_indicators: Total indicators evaluated.
            dominant_direction: "bullish", "bearish", or "neutral".
        """
        pass

    @abstractmethod
    async def get_confluence_history(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve confluence score history.

        Args:
            symbol: Trading symbol.
            timeframe: Bar timeframe.
            start: Start timestamp (inclusive).
            end: End timestamp (inclusive).
            limit: Maximum number of records.

        Returns:
            List of confluence score dicts, ordered by time DESC.
        """
        pass

    # -------------------------------------------------------------------------
    # TUI Tab 7 Support Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    async def get_indicator_summary(self) -> List[Dict[str, Any]]:
        """
        Get high-level summary per indicator type for Tab 7 display.

        Used by ApexApp._poll_data_view() to show indicator status.

        Returns:
            List of dicts with indicator, symbol_count, last_update, oldest_update.
        """
        pass

    @abstractmethod
    async def get_indicator_details(self, indicator: str) -> List[Dict[str, Any]]:
        """
        Get detailed per-symbol info for drill-down in Tab 7.

        Used when user expands an indicator row to see per-symbol details.

        Args:
            indicator: Indicator name to get details for.

        Returns:
            List of dicts with symbol, timeframe, last_update, state.
        """
        pass
