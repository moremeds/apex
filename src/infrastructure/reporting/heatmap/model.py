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
from typing import Any, Dict, List, Optional, Set

# =============================================================================
# ETF Configuration - Single Source of Truth
# =============================================================================
# All ETF lists are derived from this config to prevent drift across modules.

ETF_CONFIG: Dict[str, Dict[str, Any]] = {
    "market_indices": {
        "name": "Market Indices",
        "symbols": ["SPY", "QQQ", "IWM", "DIA"],
        "card_style": "large",  # 4 columns grid
    },
    "commodities": {
        "name": "Commodities & Safe Haven",
        "symbols": ["GLD", "SLV"],
        "card_style": "compact",  # Inline display
    },
    "fixed_income": {
        "name": "Fixed Income",
        "symbols": ["TLT"],
        "card_style": "compact",
    },
    "volatility": {
        "name": "Volatility",
        "symbols": ["UVXY"],
        "card_style": "compact",
    },
    "sectors": {
        "name": "Sector ETFs",
        "symbols": [
            "XLK",
            "XLF",
            "XLV",
            "XLP",
            "XLE",
            "XLI",
            "XLB",
            "XLRE",
            "XLU",
            "XLC",
            "XLY",
            "SMH",
        ],
        "card_style": "mini",  # 6+ columns grid
    },
}

# Derived sets for quick lookup (computed once at module load)
ALL_DASHBOARD_ETFS: Set[str] = {s for cat in ETF_CONFIG.values() for s in cat["symbols"]}

# Sector ETF display names (for dashboard cards)
SECTOR_ETF_NAMES: Dict[str, str] = {
    "XLK": "Technology",
    "XLC": "Communication",
    "XLY": "Cons. Disc.",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLP": "Cons. Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "SMH": "Semiconductors",
}

# Market ETF display names
MARKET_ETF_NAMES: Dict[str, str] = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "IWM": "Russell 2000",
    "DIA": "Dow Jones",
}

# Non-sector ETF display names
OTHER_ETF_NAMES: Dict[str, str] = {
    "GLD": "Gold",
    "SLV": "Silver",
    "TLT": "Long Treasury",
    "UVXY": "VIX Short",
}


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
    RULE_FREQUENCY = "rule_frequency"  # Phase 3: Trending mode (activity)
    RULE_FREQUENCY_DIRECTION = "rule_frequency_direction"  # Phase 3: Trending mode (direction)


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

    # Phase 3: Rule frequency for trending mode
    signal_count: int = 0
    buy_signal_count: int = 0
    sell_signal_count: int = 0
    rule_frequency_color: Optional[str] = None
    rule_frequency_direction_color: Optional[str] = None

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
            "signal_count": self.signal_count,
            "buy_signal_count": self.buy_signal_count,
            "sell_signal_count": self.sell_signal_count,
            "rule_frequency_color": self.rule_frequency_color,
            "rule_frequency_direction_color": self.rule_frequency_direction_color,
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
class ETFCardData:
    """
    Data for a single ETF card in the dashboard.

    Used by the ETF dashboard to render cards with regime, price, and daily change.
    """

    symbol: str
    display_name: str  # From MarketCapService.short_name or fallback
    category: str  # Category key from ETF_CONFIG (e.g., "market_indices", "sectors")

    # Regime info
    regime: Optional[str] = None  # R0/R1/R2/R3
    regime_name: Optional[str] = None  # "Healthy Uptrend", etc.

    # Price info
    close_price: Optional[float] = None
    daily_change_pct: Optional[float] = None

    # Navigation
    report_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON/frontend."""
        return {
            "symbol": self.symbol,
            "display_name": self.display_name,
            "category": self.category,
            "regime": self.regime,
            "regime_name": self.regime_name,
            "close_price": self.close_price,
            "daily_change_pct": self.daily_change_pct,
            "report_url": self.report_url,
        }


@dataclass
class HeatmapModel:
    """
    Complete heatmap data model.

    This is the output of heatmap_builder.build_heatmap_model() and
    the input to render_heatmap_html().

    Hierarchy:
        etf_dashboard: Card grid showing ALL ETFs (above treemap)
        stocks-only treemap: Only individual stocks by sector
    """

    # ETF Dashboard: Card data organized by category
    # Keys match ETF_CONFIG categories: "market_indices", "commodities", etc.
    etf_dashboard: Dict[str, List[ETFCardData]] = field(default_factory=dict)

    # Legacy: Top-level market ETFs (kept for backward compat with treemap)
    market_etfs: List[TreemapNode] = field(default_factory=list)

    # Legacy: Sector ETFs (kept for backward compat)
    sector_etfs: List[TreemapNode] = field(default_factory=list)

    # Stocks grouped by sector (no ETFs in treemap)
    sectors: List[SectorGroup] = field(default_factory=list)

    # Metadata
    generated_at: Optional[datetime] = None
    size_metric: SizeMetric = SizeMetric.MARKET_CAP
    color_metric: ColorMetric = ColorMetric.REGIME
    symbol_count: int = 0
    cap_missing_count: int = 0

    # Summary statistics
    regime_distribution: Dict[str, int] = field(default_factory=dict)

    # Phase 3: Rule frequency summary for trending mode
    total_signals: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON."""
        return {
            "etf_dashboard": {
                cat: [c.to_dict() for c in cards] for cat, cards in self.etf_dashboard.items()
            },
            "market_etfs": [e.to_dict() for e in self.market_etfs],
            "sector_etfs": [e.to_dict() for e in self.sector_etfs],
            "sectors": [s.to_dict() for s in self.sectors],
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "size_metric": self.size_metric.value,
            "color_metric": self.color_metric.value,
            "symbol_count": self.symbol_count,
            "cap_missing_count": self.cap_missing_count,
            "regime_distribution": self.regime_distribution,
            "total_signals": self.total_signals,
        }

    def get_all_nodes(self) -> List[TreemapNode]:
        """Get all nodes for Plotly treemap rendering (stocks only, no ETFs)."""
        nodes = []
        # NOTE: Removed market_etfs and sector_etfs - treemap is stocks-only now
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

# Derived from ETF_CONFIG for backward compatibility
MARKET_ETFS: List[str] = ETF_CONFIG["market_indices"]["symbols"]
SECTOR_ETFS: Set[str] = set(ETF_CONFIG["sectors"]["symbols"])


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
    """Map daily change percentage to color (red-green gradient).

    Bug 3 fix: Proper RGB interpolation instead of single-channel manipulation.
    - Positive changes: interpolate from neutral gray toward #22c55e (green-500)
    - Negative changes: interpolate from neutral gray toward #ef4444 (red-500)
    """
    if change_pct is None:
        return RegimeColor.UNKNOWN.value

    # Clamp to [-5%, +5%] for color mapping
    clamped = max(-5.0, min(5.0, change_pct))

    if clamped >= 0:
        # Green gradient: interpolate toward #22c55e (green-500)
        t = clamped / 5.0
        r = int(200 - t * (200 - 34))  # 200 -> 34
        g = int(220 - t * (220 - 197))  # 220 -> 197
        b = int(180 - t * (180 - 94))  # 180 -> 94
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        # Red gradient: interpolate toward #ef4444 (red-500)
        t = abs(clamped) / 5.0
        r = int(200 + t * (239 - 200))  # 200 -> 239
        g = int(200 - t * (200 - 68))  # 200 -> 68
        b = int(200 - t * (200 - 68))  # 200 -> 68
        return f"#{r:02x}{g:02x}{b:02x}"


def get_alignment_color(score: Optional[float]) -> str:
    """Map alignment score (-100 to +100) to color.

    Bug 4 fix: Align with JavaScript logic in heatmap_builder.py.
    - Positive alignment: vary green channel (180 -> 220) like JS
    - Negative alignment: vary red channel (200 -> 244)
    """
    if score is None:
        return RegimeColor.UNKNOWN.value

    # Clamp to [-100, +100]
    clamped = max(-100.0, min(100.0, score))

    if clamped >= 0:
        # Green gradient: match JS (vary green channel 180->220)
        g = int(180 + (clamped / 100.0) * 40)  # 180 -> 220
        return f"#22{g:02x}5e"
    else:
        # Red gradient: interpolate toward red
        r = int(200 - (clamped / 100.0) * 44)  # 200 -> 244 (clamped is negative)
        return f"#{r:02x}4444"


def get_rule_frequency_color(signal_count: int, max_count: int) -> str:
    """
    Map signal count to color intensity for trending mode (Phase 3).

    Color gradient from cold (gray/green) to hot (orange/red):
    - 0 signals: Gray (#444444)
    - 1-2 signals: Cool green (#88cc88)
    - 3-4 signals: Yellow (#ffcc44)
    - 5-7 signals: Warm orange (#ff8844)
    - 8+ signals: Hot red (#ff4444)

    Args:
        signal_count: Number of signals for this symbol
        max_count: Maximum signal count across all symbols (for normalization)

    Returns:
        Hex color code
    """
    if signal_count == 0:
        return "#444444"  # Gray - no signals

    if max_count == 0:
        return "#444444"

    # Use absolute thresholds for intuitive color mapping
    if signal_count >= 8:
        return "#ff4444"  # Hot red
    elif signal_count >= 5:
        return "#ff8844"  # Warm orange
    elif signal_count >= 3:
        return "#ffcc44"  # Yellow
    elif signal_count >= 1:
        return "#88cc88"  # Cool green
    else:
        return "#444444"  # Gray


def get_rule_frequency_direction_color(buy_count: int, sell_count: int) -> str:
    """
    Map net signal direction to color for trending direction mode (Phase 3).

    Color shows direction (green=bullish, red=bearish), intensity shows magnitude.
    - Net bullish (buy > sell): Green gradient based on net count
    - Net bearish (sell > buy): Red gradient based on net count
    - Neutral (buy == sell): Gray
    - No signals: Dark gray

    Args:
        buy_count: Number of buy signals for this symbol
        sell_count: Number of sell signals for this symbol

    Returns:
        Hex color code
    """
    total = buy_count + sell_count
    if total == 0:
        return "#444444"  # Dark gray - no signals

    net = buy_count - sell_count

    if net == 0:
        return "#6b7280"  # Neutral gray - balanced

    # Calculate intensity based on net magnitude (capped at 8 for consistency)
    magnitude = min(abs(net), 8)
    intensity = magnitude / 8.0  # 0.0 to 1.0

    if net > 0:
        # Bullish: interpolate from muted green to bright green
        # Base: #4ade80 (green-400), Bright: #22c55e (green-500)
        r = int(74 - intensity * 52)  # 74 -> 22
        g = int(222 - intensity * 25)  # 222 -> 197
        b = int(128 - intensity * 34)  # 128 -> 94
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        # Bearish: interpolate from muted red to bright red
        # Base: #f87171 (red-400), Bright: #ef4444 (red-500)
        r = int(248 - intensity * 9)  # 248 -> 239
        g = int(113 - intensity * 45)  # 113 -> 68
        b = int(113 - intensity * 45)  # 113 -> 68
        return f"#{r:02x}{g:02x}{b:02x}"
