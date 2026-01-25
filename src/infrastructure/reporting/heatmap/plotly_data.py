"""
Plotly Data - Treemap data structure builder for Plotly.

Builds the data structure required by Plotly's treemap chart.
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.utils.logging_setup import get_logger

from .model import HeatmapModel

logger = get_logger(__name__)


def build_plotly_data(model: HeatmapModel) -> Dict[str, Any]:
    """
    Build Plotly treemap data structure.

    Plotly treemap requires:
    - One root node with parent=""
    - All other nodes must connect back to root through parent chain
    - branchvalues="total" means parent values = sum of children

    CRITICAL: With branchvalues="total", parent nodes MUST have values equal
    to the sum of their children, otherwise the treemap renders empty.

    Args:
        model: HeatmapModel with sectors populated

    Returns:
        Dict with Plotly treemap data:
        - ids: List of unique identifiers
        - labels: List of display labels
        - parents: List of parent identifiers
        - values: List of size values
        - colors: List of colors
        - customdata: List of additional metadata for tooltips
    """
    ids: List[str] = []
    labels: List[str] = []
    parents: List[str] = []
    values: List[float] = []
    colors: List[str] = []
    customdata: List[Dict[str, Any]] = []

    # Track parent-child relationships for value aggregation
    parent_children: Dict[str, List[int]] = {}

    def add_node(
        node_id: str,
        label: str,
        parent: str,
        value: float,
        color: str,
        data: Dict[str, Any],
    ) -> int:
        """Add a node and return its index."""
        idx = len(ids)
        ids.append(node_id)
        labels.append(label)
        parents.append(parent)
        values.append(value)
        colors.append(color)
        customdata.append(data)

        # Track parent-child relationship
        if parent:
            if parent not in parent_children:
                parent_children[parent] = []
            parent_children[parent].append(idx)

        return idx

    # Single root node - ALL other nodes must trace back to this
    # NOTE: Treemap is stocks-only; ETFs are in the dashboard above
    add_node("root", "Stock Universe", "", 0, "#0c0f14", {"type": "root"})

    # Sector groups with stocks (under root) - NO ETFs in treemap
    for sector in model.sectors:
        # Sector container - directly under root
        sector_id = f"sector_{sector.sector_id}"
        add_node(
            sector_id,
            sector.sector_name,
            "root",
            0,  # Will be calculated from children
            "#1c2230",  # Match CSS --bg-tertiary
            {"type": "sector", "sector_id": sector.sector_id},
        )

        # Stocks in sector
        for stock in sector.stocks:
            add_node(
                f"stock_{stock.symbol}",
                stock.label,
                sector_id,
                stock.value if stock.value > 0 else 1.0,  # Ensure positive value
                stock.color,
                stock.to_dict(),
            )

    # === CRITICAL: Calculate parent values from children ===
    # With branchvalues="total", parent values must equal sum of children.
    # We need to aggregate bottom-up: sectors -> categories -> root

    def calculate_parent_value(parent_id: str) -> float:
        """Recursively calculate parent value as sum of children."""
        if parent_id not in parent_children:
            return 0.0

        total = 0.0
        for child_idx in parent_children[parent_id]:
            child_id = ids[child_idx]
            child_value = values[child_idx]

            # If child has its own children, recursively calculate
            if child_id in parent_children:
                child_value = calculate_parent_value(child_id)
                values[child_idx] = child_value

            total += child_value

        return total

    # Calculate for root (which triggers all descendants)
    root_value = calculate_parent_value("root")
    values[0] = root_value  # Set root value

    logger.debug(f"Heatmap data built: {len(ids)} nodes, root_value={root_value:.2f}")

    return {
        "ids": ids,
        "labels": labels,
        "parents": parents,
        "values": values,
        "colors": colors,
        "customdata": customdata,
    }
