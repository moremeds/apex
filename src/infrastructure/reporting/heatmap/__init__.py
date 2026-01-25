"""
Heatmap Sub-Package.

Treemap/heatmap visualization for signal landing page (PR-C).
"""

from __future__ import annotations

from .builder import HeatmapBuilder
from .css import HEATMAP_CSS
from .etf_dashboard import build_etf_dashboard, render_etf_dashboard_html
from .extractors import extract_alignment_score, extract_regime
from .html_template import render_heatmap_template
from .model import (
    ALL_DASHBOARD_ETFS,
    ETF_CONFIG,
    MARKET_ETF_NAMES,
    OTHER_ETF_NAMES,
    SECTOR_ETF_NAMES,
    ColorMetric,
    ETFCardData,
    HeatmapModel,
    SectorGroup,
    SizeMetric,
    TreemapNode,
    get_alignment_color,
    get_daily_change_color,
    get_regime_color,
)
from .plotly_data import build_plotly_data

__all__ = [
    # Main builder
    "HeatmapBuilder",
    # CSS
    "HEATMAP_CSS",
    # Model classes
    "HeatmapModel",
    "TreemapNode",
    "SectorGroup",
    "SizeMetric",
    "ColorMetric",
    "ETFCardData",
    # Constants
    "ETF_CONFIG",
    "ALL_DASHBOARD_ETFS",
    "MARKET_ETF_NAMES",
    "SECTOR_ETF_NAMES",
    "OTHER_ETF_NAMES",
    # Color functions
    "get_regime_color",
    "get_daily_change_color",
    "get_alignment_color",
    # Module functions
    "build_etf_dashboard",
    "render_etf_dashboard_html",
    "build_plotly_data",
    "render_heatmap_template",
    "extract_regime",
    "extract_alignment_score",
]
