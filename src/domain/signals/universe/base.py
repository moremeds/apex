"""
Universe Provider Protocol and Base Classes.

Defines the interface for ticker universe providers that determine
which symbols and timeframes to monitor for signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set


@dataclass
class SymbolConfig:
    """
    Configuration for an individual symbol in the universe.

    Contains symbol-specific settings for signal generation.
    """

    symbol: str
    timeframes: List[str]  # e.g., ["1h", "4h", "1d"]
    min_volume_usd: Optional[float] = None
    enabled: bool = True
    group: Optional[str] = None  # e.g., "tech_megacap"
    custom_rules: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_timeframe(self, timeframe: str) -> bool:
        """Check if this symbol is configured for a timeframe."""
        return timeframe in self.timeframes


class UniverseProvider(Protocol):
    """
    Protocol for universe providers.

    Universe providers determine which symbols and timeframes
    should be monitored for signal generation. Implementations
    can load from YAML, database, or dynamic screeners.
    """

    def get_symbols(self) -> List[str]:
        """
        Get all symbols in the universe.

        Returns:
            List of symbol strings (e.g., ["AAPL", "MSFT", "GOOGL"])
        """
        ...

    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """
        Get configuration for a specific symbol.

        Args:
            symbol: Symbol to get config for

        Returns:
            SymbolConfig or None if symbol not in universe
        """
        ...

    def get_symbols_for_timeframe(self, timeframe: str) -> List[str]:
        """
        Get symbols configured for a specific timeframe.

        Args:
            timeframe: Timeframe string (e.g., "1h")

        Returns:
            List of symbols that should be monitored at this timeframe
        """
        ...

    def get_all_timeframes(self) -> List[str]:
        """
        Get all unique timeframes in the universe.

        Returns:
            Sorted list of unique timeframes
        """
        ...

    async def refresh(self) -> None:
        """
        Refresh the universe (for dynamic providers).

        May reload from source, update from screener, etc.
        """
        ...


class UniverseProviderBase:
    """
    Base class for universe providers with common functionality.

    Provides default implementations and helper methods.
    Subclasses must implement _load() to populate _symbols.
    """

    def __init__(self) -> None:
        """Initialize with empty symbol map."""
        self._symbols: Dict[str, SymbolConfig] = {}
        self._timeframe_index: Dict[str, Set[str]] = {}  # timeframe -> symbols

    def get_symbols(self) -> List[str]:
        """Get all symbols in the universe."""
        return list(self._symbols.keys())

    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """Get configuration for a specific symbol."""
        return self._symbols.get(symbol)

    def get_symbols_for_timeframe(self, timeframe: str) -> List[str]:
        """Get symbols configured for a specific timeframe."""
        return list(self._timeframe_index.get(timeframe, set()))

    def get_all_timeframes(self) -> List[str]:
        """Get all unique timeframes, sorted by duration."""
        timeframes = list(self._timeframe_index.keys())
        return sorted(timeframes, key=self._timeframe_to_minutes)

    async def refresh(self) -> None:
        """Refresh the universe - default is no-op."""
        pass

    def _rebuild_timeframe_index(self) -> None:
        """Rebuild the timeframe -> symbols index."""
        self._timeframe_index.clear()
        for symbol, config in self._symbols.items():
            if not config.enabled:
                continue
            for tf in config.timeframes:
                if tf not in self._timeframe_index:
                    self._timeframe_index[tf] = set()
                self._timeframe_index[tf].add(symbol)

    @staticmethod
    def _timeframe_to_minutes(tf: str) -> int:
        """
        Convert timeframe string to minutes for sorting.

        Args:
            tf: Timeframe string (e.g., "1m", "5m", "1h", "1d")

        Returns:
            Minutes equivalent
        """
        multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080}
        if not tf:
            return 0

        unit = tf[-1].lower()
        try:
            value = int(tf[:-1])
        except ValueError:
            return 0

        return value * multipliers.get(unit, 1)

    def __len__(self) -> int:
        """Return number of symbols in universe."""
        return len(self._symbols)

    def __contains__(self, symbol: str) -> bool:
        """Check if symbol is in universe."""
        return symbol in self._symbols
