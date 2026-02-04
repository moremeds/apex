"""
Abstract base class for data feeds.

All data feeds inherit from DataFeed and implement:
- load(): Load data from source
- stream_bars(): Async iterator yielding BarData
- get_symbols(): List of symbols in feed
- bar_count: Total bars loaded
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, AsyncIterator, List

if TYPE_CHECKING:
    from ....domain.events.domain_events import BarData


class DataFeed(ABC):
    """Abstract base class for data feeds."""

    @abstractmethod
    async def load(self) -> None:
        """Load data from source."""
        ...

    @abstractmethod
    def stream_bars(self) -> AsyncIterator[BarData]:
        """Stream bars in chronological order."""
        ...

    @abstractmethod
    def get_symbols(self) -> List[str]:
        """Get list of symbols in feed."""
        ...

    @property
    @abstractmethod
    def bar_count(self) -> int:
        """Get total number of bars loaded."""
        ...
