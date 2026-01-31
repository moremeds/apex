"""
Heatmap Builder - Orchestrator for heatmap visualization.

PR-C Deliverable: Builds treemap/heatmap visualization for the signal landing page.

Three-Layer Architecture:
1. build_heatmap_model() - Transforms summary + caps into HeatmapModel (pure data)
2. render_heatmap_html() - Renders HeatmapModel to HTML with embedded data
3. heatmap.js - Frontend Plotly rendering with interactive toggles

ETF Dashboard + Stocks-Only Treemap Architecture:
- ETF Dashboard: Card grid showing ALL ETFs (market indices, commodities, sectors)
- Treemap: Shows ONLY individual stocks grouped by sector
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.services.market_cap_service import MarketCapService
from src.utils.logging_setup import get_logger
from src.utils.timezone import now_utc

from .css import HEATMAP_CSS
from .etf_dashboard import build_etf_dashboard, render_etf_dashboard_html
from .extractors import extract_alignment_score, extract_composite_score, extract_regime
from .html_template import render_heatmap_template
from .model import (
    ALL_DASHBOARD_ETFS,
    SECTOR_ETF_NAMES,
    ColorMetric,
    HeatmapModel,
    SectorGroup,
    SizeMetric,
    TreemapNode,
    get_alignment_color,
    get_daily_change_color,
    get_regime_color,
    get_rule_frequency_color,
    get_rule_frequency_direction_color,
)
from .plotly_data import build_plotly_data

logger = get_logger(__name__)

# Mapping from yfinance sector names to sector ETFs
YFINANCE_SECTOR_TO_ETF: Dict[str, str] = {
    "Technology": "XLK",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}


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
        display_timezone: str = "US/Eastern",
    ) -> None:
        """
        Initialize heatmap builder.

        Args:
            market_cap_service: Service for market cap lookups (creates default if None)
            size_metric: How to size rectangles (market_cap, volume, or equal)
            color_metric: How to color rectangles (regime, daily_change, alignment_score)
            display_timezone: IANA timezone for display timestamps
        """
        self._cap_service = market_cap_service or MarketCapService()
        self._size_metric = size_metric
        self._color_metric = color_metric
        self._display_timezone = display_timezone

    def build_heatmap_model(
        self,
        summary_data: Dict[str, Any],
        manifest: Optional[Dict[str, Any]] = None,
        score_sparklines: Optional[Dict[str, List[float]]] = None,
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
            return HeatmapModel(generated_at=now_utc())

        # Extract all symbols
        symbols = [t.get("symbol") for t in tickers if t.get("symbol")]

        # Get market caps (auto-fetches missing ones from yfinance)
        cap_results = self._cap_service.ensure_market_caps(symbols)

        # Build report URL map from manifest
        report_urls: Dict[str, str] = {}
        if manifest and "symbol_reports" in manifest:
            report_urls = manifest["symbol_reports"]

        # Phase 3: Extract rule frequency data for trending mode
        rule_frequency = summary_data.get("rule_frequency", {})
        signal_counts_by_symbol = rule_frequency.get("by_symbol", {})
        buy_counts_by_symbol = rule_frequency.get("buy_by_symbol", {})
        sell_counts_by_symbol = rule_frequency.get("sell_by_symbol", {})
        max_signal_count = max(signal_counts_by_symbol.values(), default=1)

        # Build ETF dashboard (cards for all dashboard ETFs) with signal counts
        etf_dashboard = build_etf_dashboard(
            tickers,
            cap_results,
            report_urls,
            buy_counts_by_symbol,
            sell_counts_by_symbol,
            score_sparklines,
        )

        # Stocks-only classification (exclude all dashboard ETFs from treemap)
        stocks_by_sector: Dict[str, List[TreemapNode]] = {}
        other_stocks: List[TreemapNode] = []

        # Track statistics
        regime_counts: Dict[str, int] = {}
        cap_missing_count = 0
        total_signals = rule_frequency.get("total_signals", 0)

        for ticker in tickers:
            symbol = ticker.get("symbol")
            if not symbol:
                continue

            # Skip ALL dashboard ETFs - they go in the ETF dashboard, not treemap
            if symbol in ALL_DASHBOARD_ETFS:
                continue

            # Extract regime from ticker data
            regime = extract_regime(ticker)
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

            # Phase 3: Get signal count for trending mode
            signal_count = signal_counts_by_symbol.get(symbol, 0)
            buy_signal_count = buy_counts_by_symbol.get(symbol, 0)
            sell_signal_count = sell_counts_by_symbol.get(symbol, 0)
            freq_color = get_rule_frequency_color(signal_count, max_signal_count)
            freq_dir_color = get_rule_frequency_direction_color(buy_signal_count, sell_signal_count)

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
                alignment_score=extract_alignment_score(ticker),
                volume=ticker.get("volume"),
                signal_count=signal_count,
                buy_signal_count=buy_signal_count,
                sell_signal_count=sell_signal_count,
                rule_frequency_color=freq_color,
                rule_frequency_direction_color=freq_dir_color,
                composite_score=extract_composite_score(ticker),
                report_url=report_urls.get(symbol),
            )

            # Classify stock by sector
            if sector:
                sector_etf = YFINANCE_SECTOR_TO_ETF.get(sector)
                if sector_etf:
                    node.parent = sector_etf
                    if sector_etf not in stocks_by_sector:
                        stocks_by_sector[sector_etf] = []
                    stocks_by_sector[sector_etf].append(node)
                else:
                    node.parent = "Other"
                    other_stocks.append(node)
            else:
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

        from src.utils.timezone import DisplayTimezone

        gen_time = now_utc()
        _tz = DisplayTimezone(self._display_timezone)
        gen_str = _tz.format_with_tz(gen_time, "%Y-%m-%d %H:%M %Z")

        return HeatmapModel(
            etf_dashboard=etf_dashboard,
            market_etfs=[],  # Legacy - kept for backward compat
            sector_etfs=[],  # Legacy - kept for backward compat
            sectors=sectors,
            generated_at=gen_time,
            generated_at_str=gen_str,
            size_metric=self._size_metric,
            color_metric=self._color_metric,
            symbol_count=len(symbols),
            cap_missing_count=cap_missing_count,
            regime_distribution=regime_counts,
            total_signals=total_signals,
        )

    def _get_node_color(self, ticker: Dict[str, Any], regime: Optional[str]) -> str:
        """Determine node color based on configured metric."""
        if self._color_metric == ColorMetric.REGIME:
            return get_regime_color(regime)
        elif self._color_metric == ColorMetric.DAILY_CHANGE:
            return get_daily_change_color(ticker.get("daily_change_pct"))
        elif self._color_metric == ColorMetric.ALIGNMENT_SCORE:
            return get_alignment_color(extract_alignment_score(ticker))
        else:
            return get_regime_color(regime)

    def _get_size_value(self, ticker: Dict[str, Any], market_cap: float) -> float:
        """Determine size value based on configured metric."""
        if self._size_metric == SizeMetric.MARKET_CAP:
            return market_cap if market_cap > 0 else 1.0
        elif self._size_metric == SizeMetric.VOLUME:
            volume = ticker.get("volume", 0)
            return float(volume) if volume > 0 else 1.0
        else:  # EQUAL
            return 1.0

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
        plotly_data = build_plotly_data(model)

        # Build ETF dashboard HTML
        dashboard_html = render_etf_dashboard_html(model)

        # Generate HTML with embedded data
        html = render_heatmap_template(model, plotly_data, output_dir, css_path, dashboard_html)

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
