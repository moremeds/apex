"""
Base ViewModel interface - framework agnostic.

Provides diff computation for incremental UI updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


@dataclass
class CellUpdate:
    """Represents a single cell update for incremental rendering."""

    row_key: str
    column_index: int
    value: str


@dataclass
class RowUpdate:
    """Represents a row operation."""

    row_key: str
    action: str  # "add", "remove"
    values: Optional[List[str]] = None


class BaseViewModel(ABC, Generic[T]):
    """
    Base class for framework-agnostic ViewModels.

    ViewModels are responsible for:
    - Transforming domain data into display-ready rows
    - Caching previous state
    - Computing diffs between old and new data
    - Returning only changed cells for incremental updates

    ViewModels MUST NOT:
    - Import any Textual modules
    - Contain display/rendering logic
    - Hold references to widgets
    """

    def __init__(self) -> None:
        self._previous_data: Optional[T] = None
        self._row_cache: Dict[str, List[str]] = {}
        self._row_order: List[str] = []

    @abstractmethod
    def compute_display_data(self, data: T) -> Dict[str, List[str]]:
        """
        Transform domain data into display-ready rows.

        Args:
            data: Domain model (List[PositionRisk], List[RiskSignal], etc.)

        Returns:
            Dict mapping row_key -> list of formatted cell values
        """
        pass

    @abstractmethod
    def get_row_order(self, data: T) -> List[str]:
        """
        Return the desired order of row keys for display.

        Args:
            data: Domain model

        Returns:
            Ordered list of row keys
        """
        pass

    def compute_updates(
        self, data: T
    ) -> Tuple[List[RowUpdate], List[CellUpdate], List[str]]:
        """
        Compare new data with cached state and return minimal updates.

        Returns:
            Tuple of (row_operations, cell_updates, new_row_order)
        """
        new_display = self.compute_display_data(data)
        new_order = self.get_row_order(data)

        row_ops: List[RowUpdate] = []
        cell_updates: List[CellUpdate] = []

        old_keys = set(self._row_cache.keys())
        new_keys = set(new_display.keys())

        # Rows to remove
        for key in old_keys - new_keys:
            row_ops.append(RowUpdate(row_key=key, action="remove"))

        # Rows to add
        for key in new_keys - old_keys:
            row_ops.append(
                RowUpdate(row_key=key, action="add", values=new_display[key])
            )

        # Rows to diff for cell updates
        for key in old_keys & new_keys:
            old_row = self._row_cache[key]
            new_row = new_display[key]
            for col_idx, (old_val, new_val) in enumerate(zip(old_row, new_row)):
                if old_val != new_val:
                    cell_updates.append(
                        CellUpdate(row_key=key, column_index=col_idx, value=new_val)
                    )

        # Update cache
        self._row_cache = new_display
        self._row_order = new_order
        self._previous_data = data

        return row_ops, cell_updates, new_order

    def full_refresh_needed(self, data: T) -> bool:
        """
        Check if a full refresh is needed (e.g., first load or structural change).

        Override in subclasses for custom logic.
        """
        if self._previous_data is None:
            return True

        # Check if row keys changed (structural change)
        new_display = self.compute_display_data(data)
        old_keys = set(self._row_cache.keys())
        new_keys = set(new_display.keys())

        # Structural change if keys differ
        return old_keys != new_keys

    def invalidate(self) -> None:
        """Clear cache, forcing full refresh on next update."""
        self._previous_data = None
        self._row_cache.clear()
        self._row_order.clear()

    def get_cached_row_order(self) -> List[str]:
        """Get the cached row order."""
        return self._row_order.copy()
