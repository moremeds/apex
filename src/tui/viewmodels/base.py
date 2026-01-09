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
        # Pending computation cache to avoid double compute
        self._pending_display: Optional[Dict[str, List[str]]] = None
        self._pending_order: Optional[List[str]] = None
        self._pending_data_id: Optional[int] = None

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

    def _get_or_compute(self, data: T) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Get cached computation or compute fresh.

        Uses object id to detect if data changed since last computation.
        This avoids calling compute_display_data twice (once for full_refresh_needed,
        once for compute_updates).

        Returns:
            Tuple of (display_data, row_order)
        """
        data_id = id(data)
        if self._pending_data_id == data_id and self._pending_display is not None:
            # Reuse cached computation
            return self._pending_display, self._pending_order or []

        # Fresh computation
        display = self.compute_display_data(data)
        order = self.get_row_order(data)

        # Cache for potential reuse
        self._pending_display = display
        self._pending_order = order
        self._pending_data_id = data_id

        return display, order

    def _clear_pending(self) -> None:
        """Clear pending computation cache after use."""
        self._pending_display = None
        self._pending_order = None
        self._pending_data_id = None

    def compute_updates(
        self, data: T
    ) -> Tuple[List[RowUpdate], List[CellUpdate], List[str]]:
        """
        Compare new data with cached state and return minimal updates.

        Returns:
            Tuple of (row_operations, cell_updates, new_row_order)
        """
        # Use cached computation if available (from full_refresh_needed check)
        new_display, new_order = self._get_or_compute(data)
        self._clear_pending()  # Clear after use

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
        Check if a full refresh is needed (first load only).

        OPT-PERF: Changed to only return True on first load.
        Row additions/removals are now handled by compute_updates() via RowUpdate
        operations, enabling incremental updates instead of full table rebuilds.

        Override in subclasses for custom logic (e.g., schema changes).
        """
        # Only full refresh on first load - let compute_updates handle row add/remove
        return self._previous_data is None

    def invalidate(self) -> None:
        """Clear cache, forcing full refresh on next update."""
        self._previous_data = None
        self._row_cache.clear()
        self._row_order.clear()
        self._clear_pending()

    def get_cached_row_order(self) -> List[str]:
        """Get the cached row order."""
        return self._row_order.copy()
