"""
ETF Dashboard - Card grid rendering for ETF dashboard section.

Renders ETF cards organized by category (market indices, commodities, sectors, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .extractors import extract_composite_score, extract_regime
from .model import (
    ETF_CONFIG,
    MARKET_ETF_NAMES,
    OTHER_ETF_NAMES,
    SECTOR_ETF_NAMES,
    ETFCardData,
    HeatmapModel,
)

# Regime name mapping
REGIME_NAMES: Dict[str, str] = {
    "R0": "Healthy Uptrend",
    "R1": "Choppy/Extended",
    "R2": "Risk-Off",
    "R3": "Rebound Window",
}


def get_etf_display_name(symbol: str, cap_result: Optional[Any]) -> str:
    """
    Get display name for ETF from MarketCapService or fallback.

    Priority:
    1. MarketCapService.short_name (from yfinance)
    2. Predefined name constants (MARKET_ETF_NAMES, SECTOR_ETF_NAMES, etc.)
    3. Symbol itself

    Args:
        symbol: ETF symbol
        cap_result: MarketCapResult from MarketCapService (may be None)

    Returns:
        Display name string
    """
    # Try short_name from MarketCapService
    if cap_result and hasattr(cap_result, "short_name") and cap_result.short_name:
        return str(cap_result.short_name)

    # Try predefined names
    if symbol in MARKET_ETF_NAMES:
        return MARKET_ETF_NAMES[symbol]
    if symbol in SECTOR_ETF_NAMES:
        return SECTOR_ETF_NAMES[symbol]
    if symbol in OTHER_ETF_NAMES:
        return OTHER_ETF_NAMES[symbol]

    return symbol


def _render_sparkline_svg(points: List[float], width: int = 60, height: int = 20) -> str:
    """
    Render an inline SVG sparkline from score history points.

    Args:
        points: Score values (0-100), oldest first
        width: SVG width in pixels
        height: SVG height in pixels

    Returns:
        Inline SVG string, or empty string if insufficient data
    """
    if len(points) < 2:
        return ""

    # Determine trend color from first to last point
    delta = points[-1] - points[0]
    if delta > 3:
        color = "#10b981"  # green
    elif delta < -3:
        color = "#ef4444"  # red
    else:
        color = "#94a3b8"  # gray

    # Scale points to SVG coordinates (y-axis: 0=top, height=bottom)
    min_val = max(0.0, min(points) - 5)
    max_val = min(100.0, max(points) + 5)
    val_range = max_val - min_val or 1.0

    n = len(points)
    coords = []
    for i, v in enumerate(points):
        x = round(i / (n - 1) * width, 1)
        y = round(height - ((v - min_val) / val_range) * height, 1)
        coords.append(f"{x},{y}")

    polyline = " ".join(coords)

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="display:inline-block;vertical-align:middle;margin-left:4px;" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{coords[-1].split(",")[0]}" cy="{coords[-1].split(",")[1]}" '
        f'r="2" fill="{color}"/>'
        f"</svg>"
    )


def build_etf_dashboard(
    tickers: List[Dict[str, Any]],
    cap_results: Dict[str, Any],
    report_urls: Dict[str, str],
    buy_counts_by_symbol: Optional[Dict[str, int]] = None,
    sell_counts_by_symbol: Optional[Dict[str, int]] = None,
    score_sparklines: Optional[Dict[str, List[float]]] = None,
) -> Dict[str, List[ETFCardData]]:
    """
    Build ETF dashboard card data organized by category.

    Args:
        tickers: List of ticker summary dicts from summary.json
        cap_results: Market cap results from MarketCapService
        report_urls: Symbol -> report URL mapping
        buy_counts_by_symbol: Optional map of symbol -> buy signal count
        sell_counts_by_symbol: Optional map of symbol -> sell signal count

    Returns:
        Dict mapping category key to list of ETFCardData
    """
    # Build ticker lookup for quick access
    ticker_lookup = {t["symbol"]: t for t in tickers if t.get("symbol")}

    # Default to empty dicts if not provided
    buy_counts = buy_counts_by_symbol or {}
    sell_counts = sell_counts_by_symbol or {}
    sparklines = score_sparklines or {}

    dashboard: Dict[str, List[ETFCardData]] = {}

    for category_key, category_config in ETF_CONFIG.items():
        cards: List[ETFCardData] = []

        for symbol in category_config["symbols"]:
            ticker = ticker_lookup.get(symbol, {})
            cap_result = cap_results.get(symbol)

            # Get display name
            display_name = get_etf_display_name(symbol, cap_result)

            # Extract composite score (preferred) or fall back to old regime
            composite_score = extract_composite_score(ticker)
            regime: Optional[str] = None
            regime_name: Optional[str] = None
            if composite_score is not None:
                if composite_score >= 70:
                    regime = "R0"
                    regime_name = "Healthy Uptrend"
                elif composite_score >= 30:
                    regime = "R1"
                    regime_name = "Choppy/Extended"
                else:
                    regime = "R2"
                    regime_name = "Risk-Off"
            else:
                regime = extract_regime(ticker)
                regime_name = REGIME_NAMES.get(regime, None) if regime else None

            card = ETFCardData(
                symbol=symbol,
                display_name=display_name,
                category=category_key,
                regime=regime,
                regime_name=regime_name,
                composite_score=composite_score,
                score_sparkline=sparklines.get(symbol, []),
                close_price=ticker.get("close"),
                daily_change_pct=ticker.get("daily_change_pct"),
                report_url=report_urls.get(symbol, f"report.html?symbol={symbol}"),
                buy_signal_count=buy_counts.get(symbol, 0),
                sell_signal_count=sell_counts.get(symbol, 0),
            )
            cards.append(card)

        if cards:
            dashboard[category_key] = cards

    return dashboard


def render_etf_dashboard_html(model: HeatmapModel) -> str:
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
        cards_html = [render_etf_card_html(card, card_style) for card in cards]

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


def render_etf_card_html(card: ETFCardData, style: str) -> str:
    """
    Render a single ETF card HTML.

    Args:
        card: ETFCardData to render
        style: Card style ("large", "compact", or "mini")

    Returns:
        HTML string for the card
    """
    # Regime class and text - prefer composite score display
    regime_class = f"hm-regime-{card.regime.lower()}" if card.regime else "hm-regime-unknown"
    if card.composite_score is not None:
        regime_text = f"{card.composite_score:.0f}"
    else:
        regime_text = card.regime if card.regime else "—"

    # Sparkline SVG from score history
    sparkline_svg = _render_sparkline_svg(card.score_sparkline)

    # Direction class for symbol coloring (based on buy/sell signals)
    direction_class = card.direction_class

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
        <a href="{url}" class="hm-card hm-card-large {direction_class}">
            <div class="hm-card-header">
                <span class="hm-card-symbol">{card.symbol}</span>
                <span class="hm-card-name">{card.display_name}</span>
                <span class="hm-regime {regime_class}">{regime_text}</span>{sparkline_svg}
            </div>
            <div class="hm-card-price">{price_str}</div>
            <div class="hm-card-change {change_class}">{change_str}</div>
        </a>"""
    elif style == "mini":
        return f"""
        <a href="{url}" class="hm-card hm-card-mini {direction_class}">
            <div class="hm-card-symbol">{card.symbol}</div>
            <span class="hm-regime {regime_class}">{regime_text}</span>{sparkline_svg}
            <div class="hm-card-name">{card.display_name}</div>
        </a>"""
    else:  # compact
        return f"""
        <a href="{url}" class="hm-card hm-card-compact {direction_class}">
            <span class="hm-regime {regime_class}">{regime_text}</span>{sparkline_svg}
            <span class="hm-card-symbol">{card.symbol}</span>
            <span class="hm-card-price">{price_str}</span>
            <span class="hm-card-change {change_class}">{change_str}</span>
        </a>"""
