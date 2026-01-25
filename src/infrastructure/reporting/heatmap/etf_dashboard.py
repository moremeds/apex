"""
ETF Dashboard - Card grid rendering for ETF dashboard section.

Renders ETF cards organized by category (market indices, commodities, sectors, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .extractors import extract_regime
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


def build_etf_dashboard(
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

            # Get display name
            display_name = get_etf_display_name(symbol, cap_result)

            # Extract regime info
            regime = extract_regime(ticker)
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
