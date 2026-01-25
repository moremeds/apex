"""
Heatmap Builder for Signal Reports.

PR-C Deliverable: Builds treemap/heatmap visualization for the signal landing page.

Three-Layer Architecture:
1. build_heatmap_model() - Transforms summary + caps into HeatmapModel (pure data)
2. render_heatmap_html() - Renders HeatmapModel to HTML with embedded data
3. heatmap.js - Frontend Plotly rendering with interactive toggles

ETF Dashboard + Stocks-Only Treemap Architecture:
- ETF Dashboard: Card grid showing ALL ETFs (market indices, commodities, sectors)
- Treemap: Shows ONLY individual stocks grouped by sector

Usage:
    from src.infrastructure.reporting.heatmap_builder import HeatmapBuilder

    builder = HeatmapBuilder(market_cap_service)
    model = builder.build_heatmap_model(summary_data, universe_config)
    html = builder.render_heatmap_html(model, output_dir)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.services.market_cap_service import MarketCapService
from src.utils.logging_setup import get_logger

from .heatmap_model import (  # ETF configuration from single source of truth; Model classes
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

logger = get_logger(__name__)

# Mapping from yfinance sector names to sector ETFs
# yfinance returns sector names like "Technology", "Financial Services", etc.
YFINANCE_SECTOR_TO_ETF: Dict[str, str] = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",  # yfinance uses "Consumer Cyclical"
    "Consumer Discretionary": "XLY",  # Alternative name
    "Financial Services": "XLF",
    "Financials": "XLF",  # Alternative name
    "Healthcare": "XLV",
    "Health Care": "XLV",  # Alternative name
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",  # Alternative name
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",  # Alternative name
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}

# Regime name mapping
REGIME_NAMES: Dict[str, str] = {
    "R0": "Healthy Uptrend",
    "R1": "Choppy/Extended",
    "R2": "Risk-Off",
    "R3": "Rebound Window",
}

# =============================================================================
# Heatmap CSS - Professional Dark Trading Theme
# =============================================================================
# All CSS is scoped under .apex-hm to prevent global bleed
HEATMAP_CSS = """
/* APEX Heatmap Theme - Professional Dark Trading Dashboard */
/* All styles scoped under .apex-hm to prevent global bleed */

.apex-hm {
    /* Base Colors */
    --bg-primary: #0c0f14;
    --bg-secondary: #151921;
    --bg-tertiary: #1c2230;
    --bg-hover: #252d3d;

    /* Borders */
    --border-subtle: #2d3748;
    --border-default: #3d4a5c;
    --border-focus: #4a6fa5;

    /* Text */
    --text-primary: #f0f4f8;
    --text-secondary: #a0aec0;
    --text-muted: #718096;

    /* Regime Colors */
    --regime-r0: #22c55e;
    --regime-r1: #f59e0b;
    --regime-r2: #ef4444;
    --regime-r3: #3b82f6;

    /* Performance */
    --positive: #10b981;
    --negative: #f43f5e;

    /* Accent (Claude-inspired) */
    --accent-primary: #e07a3b;

    /* Base styles */
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Inter', sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    margin: 0;
    padding: 0;
}

/* Header */
.apex-hm .hm-header {
    padding: 20px 24px;
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.apex-hm .hm-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0;
    color: var(--text-primary);
}

.apex-hm .hm-header .meta {
    font-size: 0.875rem;
    color: var(--text-muted);
}

/* ETF Dashboard Container */
.apex-hm .hm-dashboard {
    padding: 20px 24px;
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border-subtle);
}

.apex-hm .hm-dashboard-title {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 16px;
}

/* ETF Category Section */
.apex-hm .hm-etf-category {
    margin-bottom: 20px;
}

.apex-hm .hm-etf-category:last-child {
    margin-bottom: 0;
}

.apex-hm .hm-category-label {
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    color: var(--text-muted);
    margin-bottom: 10px;
}

/* ETF Card Grid */
.apex-hm .hm-cards-large {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
}

.apex-hm .hm-cards-compact {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.apex-hm .hm-cards-mini {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(90px, 1fr));
    gap: 8px;
}

/* ETF Card Base */
.apex-hm .hm-card {
    background: var(--bg-tertiary);
    border: 1px solid var(--border-subtle);
    border-radius: 8px;
    padding: 12px;
    text-decoration: none;
    color: inherit;
    transition: all 0.15s ease;
    cursor: pointer;
    display: flex;
    flex-direction: column;
}

.apex-hm .hm-card:hover {
    background: var(--bg-hover);
    border-color: var(--border-default);
    transform: translateY(-1px);
}

/* Large Card (Market Indices) */
.apex-hm .hm-card-large {
    padding: 14px 16px;
}

.apex-hm .hm-card-large .hm-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}

.apex-hm .hm-card-large .hm-card-symbol {
    font-size: 1.1rem;
    font-weight: 700;
    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
}

.apex-hm .hm-card-large .hm-card-name {
    font-size: 0.75rem;
    color: var(--text-secondary);
    flex: 1;
}

.apex-hm .hm-card-large .hm-card-price {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 4px;
}

.apex-hm .hm-card-large .hm-card-change {
    font-size: 0.875rem;
    font-weight: 500;
}

/* Compact Card (Commodities, Fixed Income, Volatility) */
.apex-hm .hm-card-compact {
    flex-direction: row;
    align-items: center;
    gap: 12px;
    padding: 10px 14px;
    min-width: 140px;
}

.apex-hm .hm-card-compact .hm-card-symbol {
    font-size: 0.9rem;
    font-weight: 600;
    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
}

.apex-hm .hm-card-compact .hm-card-price {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.apex-hm .hm-card-compact .hm-card-change {
    font-size: 0.8rem;
    font-weight: 500;
    margin-left: auto;
}

/* Mini Card (Sector ETFs) */
.apex-hm .hm-card-mini {
    padding: 8px 10px;
    text-align: center;
}

.apex-hm .hm-card-mini .hm-card-symbol {
    font-size: 0.85rem;
    font-weight: 600;
    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
    margin-bottom: 2px;
}

.apex-hm .hm-card-mini .hm-card-name {
    font-size: 0.65rem;
    color: var(--text-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* Regime Badge */
.apex-hm .hm-regime {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 700;
    font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
}

.apex-hm .hm-regime-r0 {
    background: rgba(34, 197, 94, 0.15);
    color: var(--regime-r0);
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.apex-hm .hm-regime-r1 {
    background: rgba(245, 158, 11, 0.15);
    color: var(--regime-r1);
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.apex-hm .hm-regime-r2 {
    background: rgba(239, 68, 68, 0.15);
    color: var(--regime-r2);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.apex-hm .hm-regime-r3 {
    background: rgba(59, 130, 246, 0.15);
    color: var(--regime-r3);
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.apex-hm .hm-regime-unknown {
    background: rgba(113, 128, 150, 0.15);
    color: var(--text-muted);
    border: 1px solid rgba(113, 128, 150, 0.3);
}

/* Change Color */
.apex-hm .hm-positive { color: var(--positive); }
.apex-hm .hm-negative { color: var(--negative); }
.apex-hm .hm-neutral { color: var(--text-muted); }

/* Controls Bar */
.apex-hm .hm-controls {
    padding: 12px 24px;
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border-subtle);
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
}

.apex-hm .hm-control-group {
    display: flex;
    align-items: center;
    gap: 8px;
}

.apex-hm .hm-control-group label {
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.apex-hm .hm-control-group select {
    padding: 6px 12px;
    border-radius: 6px;
    border: 1px solid var(--border-default);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-size: 0.8rem;
    cursor: pointer;
    outline: none;
}

.apex-hm .hm-control-group select:focus {
    border-color: var(--border-focus);
}

/* Legend */
.apex-hm .hm-legend {
    display: flex;
    gap: 16px;
    margin-left: auto;
}

.apex-hm .hm-legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.7rem;
    color: var(--text-secondary);
}

.apex-hm .hm-legend-color {
    width: 14px;
    height: 14px;
    border-radius: 3px;
}

/* Stats Bar */
.apex-hm .hm-stats {
    padding: 8px 24px;
    background: var(--bg-primary);
    display: flex;
    gap: 24px;
    font-size: 0.7rem;
    color: var(--text-muted);
    border-bottom: 1px solid var(--border-subtle);
}

.apex-hm .hm-stat {
    display: flex;
    gap: 6px;
}

.apex-hm .hm-stat-value {
    color: var(--text-primary);
    font-weight: 500;
}

/* Treemap Container */
.apex-hm .hm-treemap-container {
    flex: 1;
    min-height: 500px;
    padding: 0;
}

.apex-hm #heatmap {
    width: 100%;
    height: 100%;
    min-height: 500px;
}

/* Responsive */
@media (max-width: 1024px) {
    .apex-hm .hm-cards-large {
        grid-template-columns: repeat(2, 1fr);
    }
    .apex-hm .hm-cards-mini {
        grid-template-columns: repeat(auto-fill, minmax(80px, 1fr));
    }
}

@media (max-width: 640px) {
    .apex-hm .hm-cards-large {
        grid-template-columns: 1fr;
    }
    .apex-hm .hm-controls {
        flex-direction: column;
        align-items: flex-start;
    }
    .apex-hm .hm-legend {
        margin-left: 0;
        margin-top: 10px;
    }
}
"""


class HeatmapBuilder:
    """
    Builds heatmap visualization from summary data and market caps.

    This builder follows the three-layer architecture:
    1. Model building (pure data transformation)
    2. HTML rendering (template-based)
    3. Frontend rendering (deferred to heatmap.js)
    """

    def __init__(
        self,
        market_cap_service: Optional[MarketCapService] = None,
        size_metric: SizeMetric = SizeMetric.MARKET_CAP,
        color_metric: ColorMetric = ColorMetric.REGIME,
    ) -> None:
        """
        Initialize heatmap builder.

        Args:
            market_cap_service: Service for market cap lookups (creates default if None)
            size_metric: How to size rectangles (market_cap, volume, or equal)
            color_metric: How to color rectangles (regime, daily_change, alignment_score)
        """
        self._cap_service = market_cap_service or MarketCapService()
        self._size_metric = size_metric
        self._color_metric = color_metric

    def build_heatmap_model(
        self,
        summary_data: Dict[str, Any],
        manifest: Optional[Dict[str, Any]] = None,
    ) -> HeatmapModel:
        """
        Build HeatmapModel from summary.json data.

        ETF Dashboard + Stocks-Only Treemap:
        - ETF Dashboard: Shows ALL ETFs in card grid (excluded from treemap)
        - Treemap: Shows ONLY individual stocks by sector

        Args:
            summary_data: Parsed summary.json content
            manifest: Optional manifest.json for report URLs

        Returns:
            HeatmapModel ready for rendering
        """
        tickers = summary_data.get("tickers", [])
        if not tickers:
            logger.warning("No tickers in summary data")
            return HeatmapModel(generated_at=datetime.now())

        # Extract all symbols
        symbols = [t.get("symbol") for t in tickers if t.get("symbol")]

        # Get market caps (auto-fetches missing ones from yfinance)
        cap_results = self._cap_service.ensure_market_caps(symbols)

        # Build report URL map from manifest
        report_urls: Dict[str, str] = {}
        if manifest and "symbol_reports" in manifest:
            report_urls = manifest["symbol_reports"]

        # Build ETF dashboard (cards for all dashboard ETFs)
        etf_dashboard = self._build_etf_dashboard(tickers, cap_results, report_urls)

        # Stocks-only classification (exclude all dashboard ETFs from treemap)
        stocks_by_sector: Dict[str, List[TreemapNode]] = {}  # sector_etf -> stocks
        other_stocks: List[TreemapNode] = []

        # Track statistics
        regime_counts: Dict[str, int] = {}
        cap_missing_count = 0

        for ticker in tickers:
            symbol = ticker.get("symbol")
            if not symbol:
                continue

            # Skip ALL dashboard ETFs - they go in the ETF dashboard, not treemap
            if symbol in ALL_DASHBOARD_ETFS:
                continue

            # Extract regime from ticker data
            regime = self._extract_regime(ticker)
            if regime:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Get market cap and sector info from cache
            cap_result = cap_results.get(symbol)
            market_cap = cap_result.market_cap if cap_result else 0.0
            cap_missing = cap_result.cap_missing if cap_result else True
            sector = cap_result.sector if cap_result else None
            quote_type = cap_result.quote_type if cap_result else "EQUITY"

            if cap_missing and quote_type != "ETF":
                cap_missing_count += 1

            # Determine color based on metric
            color = self._get_node_color(ticker, regime)

            # Calculate size value
            size_value = self._get_size_value(ticker, market_cap)

            # Build node for STOCKS ONLY (ETFs excluded above)
            node = TreemapNode(
                symbol=symbol,
                label=symbol,
                parent="",  # Will be set based on classification
                value=size_value,
                color=color,
                regime=regime,
                daily_change_pct=ticker.get("daily_change_pct"),
                market_cap=market_cap if not cap_missing else None,
                market_cap_missing=cap_missing,
                close_price=ticker.get("close"),
                alignment_score=self._extract_alignment_score(ticker),
                volume=ticker.get("volume"),
                report_url=report_urls.get(symbol),
            )

            # Classify stock by sector
            if sector:
                # Use yfinance sector to determine sector ETF
                sector_etf = YFINANCE_SECTOR_TO_ETF.get(sector)
                if sector_etf:
                    node.parent = sector_etf
                    if sector_etf not in stocks_by_sector:
                        stocks_by_sector[sector_etf] = []
                    stocks_by_sector[sector_etf].append(node)
                else:
                    # Unknown sector mapping
                    node.parent = "Other"
                    other_stocks.append(node)
            else:
                # No sector info - put in "Other"
                node.parent = "Other"
                other_stocks.append(node)

        # Build sector groups (stocks only)
        sectors: List[SectorGroup] = []
        for sector_etf in sorted(stocks_by_sector.keys()):
            stocks = stocks_by_sector[sector_etf]
            if stocks:
                sector_name = SECTOR_ETF_NAMES.get(sector_etf, sector_etf)
                sectors.append(
                    SectorGroup(
                        sector_id=sector_etf,
                        sector_name=f"{sector_name} ({sector_etf})",
                        gics_sector=None,
                        stocks=stocks,
                    )
                )

        # Add "Other" sector if there are unclassified stocks
        if other_stocks:
            sectors.append(
                SectorGroup(
                    sector_id="Other",
                    sector_name="Other",
                    gics_sector=None,
                    stocks=other_stocks,
                )
            )

        return HeatmapModel(
            etf_dashboard=etf_dashboard,
            market_etfs=[],  # Legacy - kept for backward compat
            sector_etfs=[],  # Legacy - kept for backward compat
            sectors=sectors,
            generated_at=datetime.now(),
            size_metric=self._size_metric,
            color_metric=self._color_metric,
            symbol_count=len(symbols),
            cap_missing_count=cap_missing_count,
            regime_distribution=regime_counts,
        )

    def _build_etf_dashboard(
        self,
        tickers: List[Dict[str, Any]],
        cap_results: Dict[str, Any],
        report_urls: Dict[str, str],
    ) -> Dict[str, List[ETFCardData]]:
        """
        Build ETF dashboard card data organized by category.

        Args:
            tickers: List of ticker summary dicts from summary.json
            cap_results: Market cap results from MarketCapService
            report_urls: Symbol -> report URL mapping

        Returns:
            Dict mapping category key to list of ETFCardData
        """
        # Build ticker lookup for quick access
        ticker_lookup = {t["symbol"]: t for t in tickers if t.get("symbol")}

        dashboard: Dict[str, List[ETFCardData]] = {}

        for category_key, category_config in ETF_CONFIG.items():
            cards: List[ETFCardData] = []

            for symbol in category_config["symbols"]:
                ticker = ticker_lookup.get(symbol, {})
                cap_result = cap_results.get(symbol)

                # Get display name - prefer short_name from MarketCapService
                display_name = self._get_etf_display_name(symbol, cap_result)

                # Extract regime info
                regime = self._extract_regime(ticker)
                regime_name = REGIME_NAMES.get(regime, None) if regime else None

                card = ETFCardData(
                    symbol=symbol,
                    display_name=display_name,
                    category=category_key,
                    regime=regime,
                    regime_name=regime_name,
                    close_price=ticker.get("close"),
                    daily_change_pct=ticker.get("daily_change_pct"),
                    report_url=report_urls.get(symbol, f"report.html?symbol={symbol}"),
                )
                cards.append(card)

            if cards:
                dashboard[category_key] = cards

        return dashboard

    def _get_etf_display_name(
        self,
        symbol: str,
        cap_result: Optional[Any],
    ) -> str:
        """
        Get display name for ETF from MarketCapService or fallback.

        Priority:
        1. MarketCapService.short_name (from yfinance)
        2. Predefined name constants (MARKET_ETF_NAMES, SECTOR_ETF_NAMES, etc.)
        3. Symbol itself
        """
        # Try short_name from MarketCapService
        if cap_result and hasattr(cap_result, "short_name") and cap_result.short_name:
            return cap_result.short_name

        # Try predefined names
        if symbol in MARKET_ETF_NAMES:
            return MARKET_ETF_NAMES[symbol]
        if symbol in SECTOR_ETF_NAMES:
            return SECTOR_ETF_NAMES[symbol]
        if symbol in OTHER_ETF_NAMES:
            return OTHER_ETF_NAMES[symbol]

        return symbol

    def _ensure_css_asset(self, output_dir: Path) -> str:
        """
        Write CSS to assets/ directory and return relative path.

        Args:
            output_dir: Package output directory

        Returns:
            Relative path to CSS file (e.g., "assets/heatmap-theme.css")
        """
        assets_dir = output_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        css_path = assets_dir / "heatmap-theme.css"
        css_path.write_text(HEATMAP_CSS, encoding="utf-8")

        logger.debug(f"Wrote heatmap CSS to {css_path}")
        return "assets/heatmap-theme.css"

    def _render_etf_dashboard_html(self, model: HeatmapModel) -> str:
        """
        Render ETF dashboard HTML from model data.

        Args:
            model: HeatmapModel with etf_dashboard populated

        Returns:
            HTML string for ETF dashboard section
        """
        if not model.etf_dashboard:
            return ""

        sections: List[str] = []

        # Category display order
        category_order = [
            "market_indices",
            "commodities",
            "fixed_income",
            "volatility",
            "sectors",
        ]

        for category_key in category_order:
            if category_key not in model.etf_dashboard:
                continue

            cards = model.etf_dashboard[category_key]
            if not cards:
                continue

            # Get category config
            config = ETF_CONFIG.get(category_key, {})
            category_name = config.get("name", category_key.replace("_", " ").title())
            card_style = config.get("card_style", "compact")

            # Determine grid class
            if card_style == "large":
                grid_class = "hm-cards-large"
            elif card_style == "mini":
                grid_class = "hm-cards-mini"
            else:
                grid_class = "hm-cards-compact"

            # Build cards HTML
            cards_html = []
            for card in cards:
                card_html = self._render_etf_card_html(card, card_style)
                cards_html.append(card_html)

            section_html = f"""
            <div class="hm-etf-category">
                <div class="hm-category-label">{category_name}</div>
                <div class="{grid_class}">
                    {"".join(cards_html)}
                </div>
            </div>"""
            sections.append(section_html)

        return f"""
        <div class="hm-dashboard">
            <div class="hm-dashboard-title">Market Overview</div>
            {"".join(sections)}
        </div>"""

    def _render_etf_card_html(self, card: ETFCardData, style: str) -> str:
        """
        Render a single ETF card HTML.

        Args:
            card: ETFCardData to render
            style: Card style ("large", "compact", or "mini")

        Returns:
            HTML string for the card
        """
        # Regime class
        regime_class = f"hm-regime-{card.regime.lower()}" if card.regime else "hm-regime-unknown"
        regime_text = card.regime if card.regime else "—"

        # Price formatting
        price_str = f"${card.close_price:.2f}" if card.close_price else "—"

        # Change formatting
        if card.daily_change_pct is not None:
            change_sign = "+" if card.daily_change_pct >= 0 else ""
            change_class = "hm-positive" if card.daily_change_pct >= 0 else "hm-negative"
            change_str = f"{change_sign}{card.daily_change_pct:.2f}%"
        else:
            change_class = "hm-neutral"
            change_str = "—"

        url = card.report_url or f"report.html?symbol={card.symbol}"

        if style == "large":
            return f"""
            <a href="{url}" class="hm-card hm-card-large">
                <div class="hm-card-header">
                    <span class="hm-card-symbol">{card.symbol}</span>
                    <span class="hm-card-name">{card.display_name}</span>
                    <span class="hm-regime {regime_class}">{regime_text}</span>
                </div>
                <div class="hm-card-price">{price_str}</div>
                <div class="hm-card-change {change_class}">{change_str}</div>
            </a>"""
        elif style == "mini":
            return f"""
            <a href="{url}" class="hm-card hm-card-mini">
                <div class="hm-card-symbol">{card.symbol}</div>
                <span class="hm-regime {regime_class}">{regime_text}</span>
                <div class="hm-card-name">{card.display_name}</div>
            </a>"""
        else:  # compact
            return f"""
            <a href="{url}" class="hm-card hm-card-compact">
                <span class="hm-regime {regime_class}">{regime_text}</span>
                <span class="hm-card-symbol">{card.symbol}</span>
                <span class="hm-card-price">{price_str}</span>
                <span class="hm-card-change {change_class}">{change_str}</span>
            </a>"""

    def render_heatmap_html(
        self,
        model: HeatmapModel,
        output_dir: Path,
    ) -> str:
        """
        Render HeatmapModel to HTML page with ETF dashboard and stocks-only treemap.

        Args:
            model: HeatmapModel to render
            output_dir: Directory where HTML will be written

        Returns:
            HTML content as string
        """
        # Ensure CSS asset is written to assets/
        css_path = self._ensure_css_asset(output_dir)

        # Build Plotly-compatible data structure (stocks only)
        plotly_data = self._build_plotly_data(model)

        # Build ETF dashboard HTML
        dashboard_html = self._render_etf_dashboard_html(model)

        # Generate HTML with embedded data
        html = self._render_template(model, plotly_data, output_dir, css_path, dashboard_html)

        return html

    def _extract_regime(self, ticker: Dict[str, Any]) -> Optional[str]:
        """Extract regime string from ticker data."""
        # Try different possible locations for regime
        regime = ticker.get("regime")
        if regime:
            return regime

        # Check nested regime_output
        regime_output = ticker.get("regime_output", {})
        if isinstance(regime_output, dict):
            return regime_output.get("regime")

        return None

    def _extract_alignment_score(self, ticker: Dict[str, Any]) -> Optional[float]:
        """Extract alignment score from ticker data."""
        # Try different possible locations
        score = ticker.get("alignment_score")
        if score is not None:
            return float(score)

        confluence = ticker.get("confluence", {})
        if isinstance(confluence, dict):
            score = confluence.get("alignment_score")
            if score is not None:
                return float(score)

        return None

    def _render_etf_chips(self, sector_etfs: List[TreemapNode]) -> str:
        """Render horizontal ETF chips HTML."""
        if not sector_etfs:
            return '<span style="color: #64748b; font-size: 0.75rem;">No sector ETFs in universe</span>'

        # Sort by symbol for consistent ordering
        sorted_etfs = sorted(sector_etfs, key=lambda e: e.symbol)

        chips = []
        for etf in sorted_etfs:
            # Get regime color
            regime_colors = {
                "R0": "#22c55e",
                "R1": "#eab308",
                "R2": "#ef4444",
                "R3": "#3b82f6",
            }
            color = regime_colors.get(etf.regime or "R0", "#6b7280")

            # Get sector name
            sector_name = SECTOR_ETF_NAMES.get(etf.symbol, "")

            # Build URL
            url = etf.report_url or f"report.html?symbol={etf.symbol}"

            chip = f"""<a href="{url}" class="etf-chip">
                <span class="regime-dot" style="background: {color};"></span>
                <span>{etf.symbol}</span>
                <span class="etf-name">{sector_name}</span>
            </a>"""
            chips.append(chip)

        return "\n        ".join(chips)

    def _get_node_color(self, ticker: Dict[str, Any], regime: Optional[str]) -> str:
        """Determine node color based on configured metric."""
        if self._color_metric == ColorMetric.REGIME:
            return get_regime_color(regime)
        elif self._color_metric == ColorMetric.DAILY_CHANGE:
            return get_daily_change_color(ticker.get("daily_change_pct"))
        elif self._color_metric == ColorMetric.ALIGNMENT_SCORE:
            return get_alignment_color(self._extract_alignment_score(ticker))
        else:
            return get_regime_color(regime)

    def _get_size_value(self, ticker: Dict[str, Any], market_cap: float) -> float:
        """Determine size value based on configured metric."""
        if self._size_metric == SizeMetric.MARKET_CAP:
            return market_cap if market_cap > 0 else 1.0  # Fallback to 1.0
        elif self._size_metric == SizeMetric.VOLUME:
            volume = ticker.get("volume", 0)
            return float(volume) if volume > 0 else 1.0
        else:  # EQUAL
            return 1.0

    def _build_plotly_data(self, model: HeatmapModel) -> Dict[str, Any]:
        """
        Build Plotly treemap data structure.

        Plotly treemap requires:
        - One root node with parent=""
        - All other nodes must connect back to root through parent chain
        - branchvalues="total" means parent values = sum of children

        CRITICAL: With branchvalues="total", parent nodes MUST have values equal
        to the sum of their children, otherwise the treemap renders empty.

        Returns dict with:
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
        # Map: parent_id -> list of (child_index, child_value)
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

        # Calculate parent values bottom-up
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

    def _render_template(
        self,
        model: HeatmapModel,
        plotly_data: Dict[str, Any],
        output_dir: Path,
        css_path: str,
        dashboard_html: str,
    ) -> str:
        """
        Render the HTML template with external CSS and ETF dashboard.

        Args:
            model: HeatmapModel with etf_dashboard populated
            plotly_data: Plotly treemap data structure
            output_dir: Output directory (for potential asset paths)
            css_path: Relative path to external CSS file
            dashboard_html: Pre-rendered ETF dashboard HTML

        Returns:
            Complete HTML page content
        """
        # Embedded model data for frontend
        model_json = json.dumps(model.to_dict(), indent=2)
        plotly_json = json.dumps(plotly_data)

        # Build regime distribution stats
        regime_stats = "".join(
            f'<div class="hm-stat"><span>{r}:</span><span class="hm-stat-value">{c}</span></div>'
            for r, c in sorted(model.regime_distribution.items())
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Signal Heatmap - {model.generated_at.strftime('%Y-%m-%d') if model.generated_at else 'N/A'}</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <link rel="stylesheet" href="{css_path}">
</head>
<body class="apex-hm">
    <div class="hm-header">
        <h1>Signal Heatmap</h1>
        <div class="meta">
            Generated: {model.generated_at.strftime('%Y-%m-%d %H:%M') if model.generated_at else 'N/A'}
        </div>
    </div>

    <!-- ETF Dashboard -->
    {dashboard_html}

    <!-- Controls -->
    <div class="hm-controls">
        <div class="hm-control-group">
            <label>Color by:</label>
            <select id="colorMetric">
                <option value="regime" {'selected' if model.color_metric == ColorMetric.REGIME else ''}>Regime (R0-R3)</option>
                <option value="daily_change" {'selected' if model.color_metric == ColorMetric.DAILY_CHANGE else ''}>Daily Change</option>
                <option value="alignment" {'selected' if model.color_metric == ColorMetric.ALIGNMENT_SCORE else ''}>Alignment Score</option>
            </select>
        </div>
        <div class="hm-control-group">
            <label>Size by:</label>
            <select id="sizeMetric">
                <option value="market_cap" {'selected' if model.size_metric == SizeMetric.MARKET_CAP else ''}>Market Cap</option>
                <option value="volume" {'selected' if model.size_metric == SizeMetric.VOLUME else ''}>Volume</option>
                <option value="equal" {'selected' if model.size_metric == SizeMetric.EQUAL else ''}>Equal Weight</option>
            </select>
        </div>
        <div class="hm-legend">
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #22c55e;"></div>
                <span>R0 Healthy</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #f59e0b;"></div>
                <span>R1 Choppy</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #ef4444;"></div>
                <span>R2 Risk-Off</span>
            </div>
            <div class="hm-legend-item">
                <div class="hm-legend-color" style="background: #3b82f6;"></div>
                <span>R3 Rebound</span>
            </div>
        </div>
    </div>

    <!-- Stats -->
    <div class="hm-stats">
        <div class="hm-stat">
            <span>Symbols:</span>
            <span class="hm-stat-value">{model.symbol_count}</span>
        </div>
        <div class="hm-stat">
            <span>Missing Caps:</span>
            <span class="hm-stat-value">{model.cap_missing_count}</span>
        </div>
        {regime_stats}
    </div>

    <!-- Treemap (Stocks Only) -->
    <div class="hm-treemap-container">
        <div id="heatmap"></div>
    </div>

    <script>
        // Embedded model data
        const modelData = {model_json};
        const plotlyData = {plotly_json};

        // Report URL mapping for click navigation (from etf_dashboard and sectors)
        const reportUrls = {{}};

        // Build URLs from ETF dashboard (new structure)
        if (modelData.etf_dashboard) {{
            Object.values(modelData.etf_dashboard).forEach(cards => {{
                cards.forEach(card => {{
                    if (card.report_url) reportUrls[card.symbol] = card.report_url;
                }});
            }});
        }}

        // Build URLs from sectors (stocks in treemap)
        modelData.sectors.forEach(s => {{
            s.stocks.forEach(stock => {{
                if (stock.report_url) reportUrls[stock.symbol] = stock.report_url;
            }});
        }});

        // Debug: Check if Plotly loaded
        console.log('Plotly loaded:', typeof Plotly !== 'undefined');

        // Determine scale based on size metric (Bug 1 fix: metric-aware normalization)
        const sizeMetric = modelData.size_metric;
        const getScale = (metric) => metric === 'volume' ? 1e6 : metric === 'equal' ? 1 : 1e9;
        let currentScale = getScale(sizeMetric);
        const normalizedValues = plotlyData.values.map(v => v / currentScale);
        console.log('Data loaded - ids:', plotlyData.ids.length, 'values:', normalizedValues, 'scale:', currentScale);

        // Build parent-children map once for parent recalculation (Bug 2 fix)
        function buildChildrenMap(ids, parents) {{
            const children = {{}};
            parents.forEach((parent, i) => {{
                if (parent) {{
                    if (!children[parent]) children[parent] = [];
                    children[parent].push(i);
                }}
            }});
            return children;
        }}

        const childrenMap = buildChildrenMap(plotlyData.ids, plotlyData.parents);

        // Recalculate parent values bottom-up (Bug 2 fix)
        function recalculateParents(values, children, ids) {{
            function sumChildren(nodeId) {{
                const idx = ids.indexOf(nodeId);
                if (idx === -1) return 0;
                if (!children[nodeId]) return values[idx];

                let sum = 0;
                for (const childIdx of children[nodeId]) {{
                    const childId = ids[childIdx];
                    sum += sumChildren(childId);
                }}
                values[idx] = sum;
                return sum;
            }}
            sumChildren('root');
        }}

        // Debug: Check container dimensions
        const container = document.getElementById('heatmap');
        console.log('Container dimensions:', container.offsetWidth, 'x', container.offsetHeight);

        // Color schemes
        function getRegimeColor(regime) {{
            const colors = {{'R0': '#22c55e', 'R1': '#eab308', 'R2': '#ef4444', 'R3': '#3b82f6'}};
            return colors[regime] || '#9ca3af';
        }}

        function getDailyChangeColor(pct) {{
            if (pct === null || pct === undefined) return '#9ca3af';
            const clamped = Math.max(-5, Math.min(5, pct));
            if (clamped >= 0) {{
                const intensity = Math.floor(200 - (clamped / 5) * 100);
                return `rgb(${{intensity}}, 197, 94)`;
            }} else {{
                const intensity = Math.floor(200 + (clamped / 5) * 100);
                return `rgb(244, ${{intensity}}, ${{intensity}})`;
            }}
        }}

        function getAlignmentColor(score) {{
            if (score === null || score === undefined) return '#9ca3af';
            const clamped = Math.max(-100, Math.min(100, score));
            if (clamped >= 0) {{
                const g = Math.floor(180 + (clamped / 100) * 40);
                return `rgb(34, ${{g}}, 94)`;
            }} else {{
                const r = Math.floor(200 + (clamped / 100) * 44);
                return `rgb(${{r}}, 68, 68)`;
            }}
        }}

        function updateColors(metric) {{
            const newColors = plotlyData.customdata.map((d, i) => {{
                if (!d || !d.symbol) return plotlyData.colors[i];
                if (metric === 'regime') return getRegimeColor(d.regime);
                if (metric === 'daily_change') return getDailyChangeColor(d.daily_change_pct);
                if (metric === 'alignment') return getAlignmentColor(d.alignment_score);
                return plotlyData.colors[i];
            }});
            Plotly.restyle('heatmap', {{'marker.colors': [newColors]}});
        }}

        function updateSizes(metric) {{
            const scale = getScale(metric);
            currentScale = scale;

            // Copy original values to avoid mutating
            const newValues = [...plotlyData.values];

            // Update leaf node values based on metric
            plotlyData.customdata.forEach((d, i) => {{
                if (d && d.symbol) {{
                    // Only update leaf nodes (nodes with a symbol)
                    if (metric === 'market_cap') {{
                        newValues[i] = (d.market_cap || 1) / scale;
                    }} else if (metric === 'volume') {{
                        newValues[i] = (d.volume || 1) / scale;
                    }} else {{
                        newValues[i] = 1;  // equal weight
                    }}
                }}
            }});

            // Bug 2 fix: Recalculate parent values bottom-up
            recalculateParents(newValues, childrenMap, plotlyData.ids);

            // Update hovertemplate unit suffix
            const unitSuffix = metric === 'volume' ? 'M' : metric === 'equal' ? '' : 'B';
            const newTemplate = '<b>%{{label}}</b><br>' +
                'Regime: %{{customdata.regime}}<br>' +
                'Close: $%{{customdata.close_price}}<br>' +
                'Daily: %{{customdata.daily_change_pct}}%<br>' +
                'Value: %{{value:,.2f}}' + unitSuffix + '<extra></extra>';

            Plotly.restyle('heatmap', {{'values': [newValues], 'hovertemplate': newTemplate}});
        }}

        // Initial render - stocks-only treemap
        const trace = {{
            type: 'treemap',
            ids: plotlyData.ids,
            labels: plotlyData.labels,
            parents: plotlyData.parents,
            values: normalizedValues,
            customdata: plotlyData.customdata,
            marker: {{
                colors: plotlyData.colors,
                line: {{ width: 1, color: '#0c0f14' }}  // Match CSS --bg-primary
            }},
            textinfo: 'label',
            textfont: {{ size: 14, color: '#f0f4f8' }},  // Match CSS --text-primary
            // Hover template - customdata fields accessed via customdata.field
            // Note: Plotly shows empty string for null values
            // Bug 1 fix: Dynamic unit suffix based on size metric
            hovertemplate: '<b>%{{label}}</b><br>' +
                'Regime: %{{customdata.regime}}<br>' +
                'Close: $%{{customdata.close_price}}<br>' +
                'Daily: %{{customdata.daily_change_pct}}%<br>' +
                'Value: %{{value:,.2f}}' + (sizeMetric === 'volume' ? 'M' : sizeMetric === 'equal' ? '' : 'B') + '<extra></extra>',
            pathbar: {{ visible: true, textfont: {{ size: 12 }} }},
            branchvalues: 'total',
            maxdepth: 3  // Show root -> category -> items
        }};

        // Calculate treemap height - account for ETF dashboard
        function calculateTreemapHeight() {{
            const dashboard = document.querySelector('.hm-dashboard');
            const dashboardHeight = dashboard ? dashboard.offsetHeight : 0;
            // Header + dashboard + controls + stats + some padding
            const reservedHeight = 60 + dashboardHeight + 50 + 40 + 20;
            return Math.max(500, window.innerHeight - reservedHeight);
        }}

        const layout = {{
            margin: {{ t: 30, l: 10, r: 10, b: 10 }},
            paper_bgcolor: '#0c0f14',  // Match CSS --bg-primary
            font: {{ color: '#f0f4f8' }},  // Match CSS --text-primary
            autosize: true,
            height: calculateTreemapHeight()
        }};

        const config = {{
            displayModeBar: true,
            modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
            responsive: true
        }};

        console.log('Calling Plotly.newPlot with', plotlyData.ids.length, 'nodes, root_value:', normalizedValues[0]);
        Plotly.newPlot('heatmap', [trace], layout, config).then(function(gd) {{
            console.log('Plotly render complete!');

            // Click handler for navigation
            gd.on('plotly_click', function(data) {{
                if (data && data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    const symbol = point.label;
                    const url = reportUrls[symbol];
                    console.log('Clicked:', symbol, 'URL:', url);
                    if (url) {{
                        window.location.href = url;
                    }}
                }}
            }});

        }}).catch(function(err) {{
            console.error('Plotly render error:', err);
        }});

        // Control handlers
        document.getElementById('colorMetric').addEventListener('change', function(e) {{
            updateColors(e.target.value);
        }});

        document.getElementById('sizeMetric').addEventListener('change', function(e) {{
            updateSizes(e.target.value);
        }});

        // Handle window resize
        window.addEventListener('resize', function() {{
            Plotly.relayout('heatmap', {{
                height: calculateTreemapHeight()
            }});
        }});
    </script>
</body>
</html>"""

        return html

    def save_heatmap(
        self,
        model: HeatmapModel,
        output_dir: Path,
        filename: str = "heatmap.html",
    ) -> Path:
        """
        Build and save heatmap HTML to file.

        Args:
            model: HeatmapModel to render
            output_dir: Output directory
            filename: Output filename

        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / filename

        html = self.render_heatmap_html(model, output_dir)

        with open(output_path, "w") as f:
            f.write(html)

        logger.info(f"Saved heatmap to {output_path}")
        return output_path
