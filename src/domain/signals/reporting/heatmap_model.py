"""
Heatmap Data Model for Signal Reports.

PR-C Deliverable: Defines the data structures for treemap/heatmap visualization.

This module contains pure data structures (no rendering logic) that can be:
1. Tested independently with JSON schema validation
2. Serialized for frontend consumption
3. Transformed by the builder without coupling to HTML

Architecture:
    HeatmapModel (pure data)
        ↓
    heatmap_builder.py (transforms)
        ↓
    heatmap.js (renders via Plotly)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class RegimeColor(Enum):
    """Color mapping for regime states (R0-R3)."""

    R0 = "#22c55e"  # Green - Healthy uptrend
    R1 = "#eab308"  # Yellow - Choppy/Extended
    R2 = "#ef4444"  # Red - Risk-off
    R3 = "#3b82f6"  # Blue - Rebound window
    UNKNOWN = "#9ca3af"  # Gray - No data


class SizeMetric(Enum):
    """Metric used to size treemap rectangles."""

    MARKET_CAP = "market_cap"
    VOLUME = "volume"
    EQUAL = "equal"  # Equal weighting fallback


class ColorMetric(Enum):
    """Metric used to color treemap rectangles."""

    REGIME = "regime"
    DAILY_CHANGE = "daily_change"
    ALIGNMENT_SCORE = "alignment_score"


@dataclass
class TreemapNode:
    """
    Single node in the treemap hierarchy.

    Can represent:
    - A stock (leaf node)
    - A sector ETF (group header)
    - A market ETF (top-level)
    """

    # Identity
    symbol: str
    label: str  # Display name (e.g., "AAPL" or "Apple Inc.")

    # Treemap positioning
    parent: str  # Parent sector/category ID, empty for root
    value: float  # Size value (market cap, volume, or 1.0 for equal)

    # Coloring
    color: str  # Hex color code
    regime: Optional[str] = None  # R0/R1/R2/R3
    daily_change_pct: Optional[float] = None

    # Metadata for tooltip
    market_cap: Optional[float] = None
    market_cap_missing: bool = False
    close_price: Optional[float] = None
    alignment_score: Optional[float] = None
    volume: Optional[float] = None

    # Navigation
    report_url: Optional[str] = None  # Link to individual symbol report

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON/frontend consumption."""
        return {
            "symbol": self.symbol,
            "label": self.label,
            "parent": self.parent,
            "value": self.value,
            "color": self.color,
            "regime": self.regime,
            "daily_change_pct": self.daily_change_pct,
            "market_cap": self.market_cap,
            "market_cap_missing": self.market_cap_missing,
            "close_price": self.close_price,
            "alignment_score": self.alignment_score,
            "volume": self.volume,
            "report_url": self.report_url,
        }


@dataclass
class SectorGroup:
    """Grouping of stocks by sector."""

    sector_id: str  # Sector ETF symbol (e.g., "XLK")
    sector_name: str  # Display name (e.g., "Technology")
    gics_sector: Optional[str] = None  # GICS sector code (e.g., "45")
    stocks: List[TreemapNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON."""
        return {
            "sector_id": self.sector_id,
            "sector_name": self.sector_name,
            "gics_sector": self.gics_sector,
            "stocks": [s.to_dict() for s in self.stocks],
        }


@dataclass
class HeatmapModel:
    """
    Complete heatmap data model.

    This is the output of heatmap_builder.build_heatmap_model() and
    the input to render_heatmap_html().

    Hierarchy:
        market_etfs (SPY, QQQ, IWM, DIA)
        sector_etfs (XLK, XLF, XLV, ...)
        stocks_by_sector {
            "XLK": [AAPL, MSFT, NVDA, ...],
            "XLF": [JPM, BAC, GS, ...],
            ...
        }
    """

    # Top-level market ETFs
    market_etfs: List[TreemapNode] = field(default_factory=list)

    # Sector ETFs (11 sectors)
    sector_etfs: List[TreemapNode] = field(default_factory=list)

    # Stocks grouped by sector
    sectors: List[SectorGroup] = field(default_factory=list)

    # Metadata
    generated_at: Optional[datetime] = None
    size_metric: SizeMetric = SizeMetric.MARKET_CAP
    color_metric: ColorMetric = ColorMetric.REGIME
    symbol_count: int = 0
    cap_missing_count: int = 0

    # Summary statistics
    regime_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON."""
        return {
            "market_etfs": [e.to_dict() for e in self.market_etfs],
            "sector_etfs": [e.to_dict() for e in self.sector_etfs],
            "sectors": [s.to_dict() for s in self.sectors],
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "size_metric": self.size_metric.value,
            "color_metric": self.color_metric.value,
            "symbol_count": self.symbol_count,
            "cap_missing_count": self.cap_missing_count,
            "regime_distribution": self.regime_distribution,
        }

    def get_all_nodes(self) -> List[TreemapNode]:
        """Get all nodes for Plotly treemap rendering."""
        nodes = []
        nodes.extend(self.market_etfs)
        nodes.extend(self.sector_etfs)
        for sector in self.sectors:
            nodes.extend(sector.stocks)
        return nodes


# Sector mapping: ETF symbol -> (Display Name, GICS Code)
SECTOR_MAPPING: Dict[str, tuple] = {
    "XLK": ("Technology", "45"),
    "XLF": ("Financials", "40"),
    "XLV": ("Health Care", "35"),
    "XLC": ("Communication Services", "50"),
    "XLY": ("Consumer Discretionary", "25"),
    "XLP": ("Consumer Staples", "30"),
    "XLE": ("Energy", "10"),
    "XLI": ("Industrials", "20"),
    "XLB": ("Materials", "15"),
    "XLRE": ("Real Estate", "60"),
    "XLU": ("Utilities", "55"),
}

# Market ETF list
MARKET_ETFS = ["SPY", "QQQ", "IWM", "DIA"]


def get_regime_color(regime: Optional[str]) -> str:
    """Map regime string to color hex code."""
    if regime is None:
        return RegimeColor.UNKNOWN.value

    regime_upper = regime.upper()
    if regime_upper in ("R0", "HEALTHY", "BULLISH"):
        return RegimeColor.R0.value
    elif regime_upper in ("R1", "CHOPPY", "EXTENDED"):
        return RegimeColor.R1.value
    elif regime_upper in ("R2", "RISK_OFF", "BEARISH"):
        return RegimeColor.R2.value
    elif regime_upper in ("R3", "REBOUND", "RECOVERY"):
        return RegimeColor.R3.value
    else:
        return RegimeColor.UNKNOWN.value


def get_daily_change_color(change_pct: Optional[float]) -> str:
    """Map daily change percentage to color (red-green gradient)."""
    if change_pct is None:
        return RegimeColor.UNKNOWN.value

    # Clamp to [-5%, +5%] for color mapping
    clamped = max(-5.0, min(5.0, change_pct))

    if clamped >= 0:
        # Green gradient: 0% = light green, 5% = dark green
        intensity = int(200 - (clamped / 5.0) * 100)
        return f"#{intensity:02x}c55e"
    else:
        # Red gradient: 0% = light red, -5% = dark red
        intensity = int(200 + (clamped / 5.0) * 100)
        return f"#f4{intensity:02x}{intensity:02x}"


def get_alignment_color(score: Optional[float]) -> str:
    """Map alignment score (-100 to +100) to color."""
    if score is None:
        return RegimeColor.UNKNOWN.value

    # Clamp to [-100, +100]
    clamped = max(-100.0, min(100.0, score))

    if clamped >= 0:
        # Green gradient for positive alignment
        intensity = int(255 - (clamped / 100.0) * 100)
        return f"#{intensity:02x}{220:02x}5e"
    else:
        # Red gradient for negative alignment
        intensity = int(255 + (clamped / 100.0) * 100)
        return f"#{240:02x}{intensity:02x}{intensity:02x}"
