"""
IndicatorStatusViewModel - Data transformation for indicator hierarchy.

Extracts business logic from IndicatorStatusPanel:
- Groups indicators by category
- Computes staleness per category
- Builds flat list of renderable rows

Framework-agnostic: returns raw data, widget handles Rich markup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class RowType(Enum):
    """Type of row in the indicator hierarchy."""
    EMPTY = auto()
    CATEGORY = auto()
    INDICATOR = auto()
    DESCRIPTION = auto()
    DETAIL = auto()
    LOADING = auto()


@dataclass(slots=True)
class IndicatorRow:
    """A single renderable row in the indicator hierarchy."""
    row_type: RowType
    key: str
    # Category/indicator rows
    label: str = ""  # Category label or indicator short name
    full_name: str = ""  # Indicator full name (e.g., "Relative Strength Index")
    is_expanded: bool = False
    indicator_count: int = 0  # For category rows
    symbol_count: int = 0  # For indicator rows
    # Timestamp (raw, widget formats)
    timestamp: Optional[datetime] = None
    # Detail rows
    symbol: str = ""
    timeframe: str = ""
    # Description rows
    description: str = ""


# Category display order
CATEGORY_ORDER = ("momentum", "trend", "volatility", "volume", "moving_avg", "pattern", "other")

CATEGORY_LABELS = {
    "momentum": "📈 Momentum",
    "trend": "📊 Trend",
    "volatility": "📉 Volatility",
    "volume": "📦 Volume",
    "moving_avg": "〰️ Moving Avg",
    "pattern": "🔷 Patterns",
    "other": "📋 Other",
}


class IndicatorStatusViewModel:
    """
    Transforms indicator summary data into renderable rows.

    Usage:
        vm = IndicatorStatusViewModel()
        rows = vm.build_rows(summary, expanded_cats, expanded_inds, details)
        for row in rows:
            # render row based on row.row_type
    """

    __slots__ = ()

    def build_rows(
        self,
        summary: List[Dict[str, Any]],
        expanded_categories: Set[str],
        expanded_indicators: Set[str],
        details: Dict[str, List[Dict[str, Any]]],
    ) -> List[IndicatorRow]:
        """
        Build flat list of renderable rows from indicator data.

        Args:
            summary: List of indicator summaries with 'category', 'indicator', etc.
            expanded_categories: Set of expanded category names
            expanded_indicators: Set of expanded indicator names
            details: Dict mapping indicator name to list of symbol details

        Returns:
            Flat list of IndicatorRow objects ready for rendering
        """
        if not summary:
            return [IndicatorRow(row_type=RowType.EMPTY, key="empty", label="No data")]

        # Group by category
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for record in summary:
            cat = record.get("category", "other")
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(record)

        rows: List[IndicatorRow] = []

        # Build rows in category order
        for category in CATEGORY_ORDER:
            if category not in by_category:
                continue

            indicators = by_category[category]
            is_expanded = category in expanded_categories
            cat_label = CATEGORY_LABELS.get(category, category.title())

            # Compute oldest (most stale) update in category
            oldest = self._find_oldest_timestamp(indicators)

            rows.append(IndicatorRow(
                row_type=RowType.CATEGORY,
                key=f"cat:{category}",
                label=cat_label,
                is_expanded=is_expanded,
                indicator_count=len(indicators),
                timestamp=oldest,
            ))

            # Add indicator rows if category expanded
            if is_expanded:
                for record in sorted(indicators, key=lambda r: r.get("indicator", "")):
                    self._add_indicator_rows(
                        rows, record, expanded_indicators, details
                    )

        return rows

    def _add_indicator_rows(
        self,
        rows: List[IndicatorRow],
        record: Dict[str, Any],
        expanded_indicators: Set[str],
        details: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """Add indicator row and its details if expanded."""
        indicator = record.get("indicator", "?")
        full_name = record.get("full_name", indicator)
        description = record.get("description", "")
        symbol_count = record.get("symbol_count", 0)
        oldest_update = record.get("oldest_update")
        is_expanded = indicator in expanded_indicators

        # Indicator row
        rows.append(IndicatorRow(
            row_type=RowType.INDICATOR,
            key=f"ind:{indicator}",
            label=indicator,
            full_name=full_name,
            is_expanded=is_expanded,
            symbol_count=symbol_count,
            timestamp=oldest_update,
        ))

        # Detail rows if expanded
        if is_expanded:
            # Description row
            if description:
                rows.append(IndicatorRow(
                    row_type=RowType.DESCRIPTION,
                    key=f"desc:{indicator}",
                    description=description,
                ))

            # Symbol detail rows
            indicator_details = details.get(indicator, [])
            if not indicator_details:
                rows.append(IndicatorRow(
                    row_type=RowType.LOADING,
                    key=f"det:{indicator}/loading",
                    label="Loading symbols...",
                ))
            else:
                for detail in sorted(indicator_details, key=lambda d: d.get("symbol", "")):
                    rows.append(IndicatorRow(
                        row_type=RowType.DETAIL,
                        key=f"det:{indicator}/{detail.get('symbol', '?')}/{detail.get('timeframe', '?')}",
                        symbol=detail.get("symbol", "?"),
                        timeframe=detail.get("timeframe", "?"),
                        timestamp=detail.get("last_update"),
                    ))

    @staticmethod
    def _find_oldest_timestamp(indicators: List[Dict[str, Any]]) -> Optional[datetime]:
        """Find the oldest (most stale) timestamp across indicators."""
        oldest = None
        for ind in indicators:
            ts = ind.get("oldest_update")
            if ts is not None:
                if oldest is None or ts < oldest:
                    oldest = ts
        return oldest
