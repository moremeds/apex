"""
Regime Dashboard - Legacy dashboard functions for regime visualization.

Provides multi-symbol regime cards, timelines, and action summaries.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from src.domain.services.regime import HierarchicalRegime, get_hierarchy_level
from src.domain.signals.indicators.regime import RegimeOutput

from ..value_card import escape_html, format_currency, format_percentage
from .utils import (
    ACTION_COLORS,
    REGIME_COLORS,
    get_theme_colors,
    render_component_card,
    render_regime_level,
)


def generate_regime_dashboard_html(
    hierarchical_regimes: Dict[str, HierarchicalRegime],
    theme: str = "dark",
) -> str:
    """
    Generate HTML for regime dashboard section.

    Shows color-coded regime cards for all symbols organized by level.
    """
    colors = get_theme_colors(theme)

    # Organize by level
    market_symbols = []
    sector_symbols = []
    stock_symbols = []

    for symbol, regime in hierarchical_regimes.items():
        level = get_hierarchy_level(symbol)
        if level == "market":
            market_symbols.append((symbol, regime))
        elif level == "sector":
            sector_symbols.append((symbol, regime))
        else:
            stock_symbols.append((symbol, regime))

    html = f"""
    <div class="regime-dashboard">
        <h2 class="section-header">Regime Dashboard</h2>
        <div class="regime-timestamp">
            Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    """

    # Market Level
    if market_symbols:
        html += render_regime_level("Market", market_symbols, colors)

    # Sector Level
    if sector_symbols:
        html += render_regime_level("Sector", sector_symbols, colors)

    # Stock Level
    if stock_symbols:
        html += render_regime_level("Stock", stock_symbols, colors)

    html += "</div>"
    return html


def generate_regime_timeline_html(
    regime_history: Dict[str, List[Dict[str, Any]]],
    theme: str = "dark",
) -> str:
    """
    Generate HTML for regime timeline section.

    Shows historical regime changes with duration.
    """
    get_theme_colors(theme)

    html = """
    <div class="regime-timeline">
        <h2 class="section-header">Regime Timeline</h2>
    """

    for symbol, history in regime_history.items():
        if not history:
            continue

        html += f"""
        <div class="timeline-symbol">
            <h3>{escape_html(symbol)}</h3>
            <div class="timeline-events">
        """

        for event in history[-10:]:  # Show last 10 changes
            regime = event.get("regime", "R1")
            rc = REGIME_COLORS.get(regime, REGIME_COLORS["R1"])
            timestamp = event.get("timestamp", "")
            duration = event.get("bars_in_regime", 0)

            html += f"""
            <div class="timeline-event">
                <div class="event-regime" style="background: {rc['bg']}; color: {rc['text']}">
                    {escape_html(regime)}
                </div>
                <div class="event-details">
                    <div class="event-time">{escape_html(str(timestamp))}</div>
                    <div class="event-duration">{duration} bars</div>
                </div>
            </div>
            """

        html += "</div></div>"

    html += "</div>"
    return html


def generate_component_breakdown_html(
    regime_output: RegimeOutput,
    theme: str = "dark",
) -> str:
    """
    Generate HTML for component breakdown section.

    Shows detailed values for all regime components.
    """
    colors = get_theme_colors(theme)
    components = regime_output.component_values
    states = regime_output.component_states

    html = f"""
    <div class="component-breakdown">
        <h2 class="section-header">Component Breakdown - {escape_html(regime_output.symbol)}</h2>
        <div class="component-grid">
    """

    # Trend Component
    html += render_component_card(
        "Trend",
        states.trend_state.value,
        [
            ("Close", format_currency(components.close)),
            ("MA20", format_currency(components.ma20)),
            ("MA50", format_currency(components.ma50)),
            ("MA200", format_currency(components.ma200)),
            ("MA50 Slope", format_percentage(components.ma50_slope, multiply=True)),
        ],
        colors,
    )

    # Volatility Component
    html += render_component_card(
        "Volatility",
        states.vol_state.value,
        [
            ("ATR20", format_currency(components.atr20)),
            ("ATR %", format_percentage(components.atr_pct, multiply=True)),
            ("ATR Pct 63d", f"{components.atr_pct_63:.1f}"),
            ("ATR Pct 252d", f"{components.atr_pct_252:.1f}"),
        ],
        colors,
    )

    # Choppiness Component
    html += render_component_card(
        "Choppiness",
        states.chop_state.value,
        [
            ("CHOP", f"{components.chop:.1f}"),
            ("CHOP Pct 252d", f"{components.chop_pct_252:.1f}"),
            ("MA20 Crosses", f"{components.ma20_crosses}"),
        ],
        colors,
    )

    # Extension Component
    html += render_component_card(
        "Extension",
        states.ext_state.value,
        [
            ("Extension", f"{components.ext:.2f} ATR"),
            ("Distance from MA20", format_currency(components.close - components.ma20)),
        ],
        colors,
    )

    # IV Component (if available)
    if states.iv_state.value != "na":
        html += render_component_card(
            "Implied Volatility",
            states.iv_state.value,
            [
                (
                    "IV Value",
                    f"{components.iv_value:.2f}" if components.iv_value else "N/A",
                ),
                (
                    "IV Pct 63d",
                    f"{components.iv_pct_63:.1f}" if components.iv_pct_63 else "N/A",
                ),
            ],
            colors,
        )

    html += "</div></div>"
    return html


def generate_action_summary_html(
    hierarchical_regimes: Dict[str, HierarchicalRegime],
    theme: str = "dark",
) -> str:
    """
    Generate HTML for action summary section.

    Shows synthesized trading decisions per symbol.
    """
    get_theme_colors(theme)

    html = """
    <div class="action-summary">
        <h2 class="section-header">Action Summary</h2>
        <table class="action-table">
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Market</th>
                    <th>Sector</th>
                    <th>Stock</th>
                    <th>Action</th>
                    <th>Recommendation</th>
                </tr>
            </thead>
            <tbody>
    """

    for symbol, regime in hierarchical_regimes.items():
        market_rc = REGIME_COLORS.get(regime.market_regime.value, REGIME_COLORS["R1"])
        sector_rc = (
            REGIME_COLORS.get(regime.sector_regime.value, REGIME_COLORS["R1"])
            if regime.sector_regime is not None
            else None
        )
        stock_rc = (
            REGIME_COLORS.get(regime.stock_regime.value, REGIME_COLORS["R1"])
            if regime.stock_regime is not None
            else None
        )
        action_color = ACTION_COLORS.get(regime.action.display_name, ACTION_COLORS["No Go"])

        sector_cell = (
            f'<span class="regime-badge" style="background: {sector_rc["bg"]}; color: {sector_rc["text"]}">{escape_html(regime.sector_regime.value)}</span>'
            if sector_rc is not None and regime.sector_regime is not None
            else "-"
        )
        stock_cell = (
            f'<span class="regime-badge" style="background: {stock_rc["bg"]}; color: {stock_rc["text"]}">{escape_html(regime.stock_regime.value)}</span>'
            if stock_rc is not None and regime.stock_regime is not None
            else "-"
        )

        # Build recommendation
        ctx = regime.action_context
        recommendation = []
        if ctx.dte_min and ctx.dte_max:
            recommendation.append(f"DTE: {ctx.dte_min}-{ctx.dte_max}")
        if ctx.delta_min and ctx.delta_max:
            recommendation.append(f"Delta: {ctx.delta_min:.2f}-{ctx.delta_max:.2f}")
        if ctx.position_type != "any":
            recommendation.append(ctx.position_type.replace("_", " ").title())
        if ctx.size_factor < 1.0:
            recommendation.append(f"Size: {int(ctx.size_factor * 100)}%")
        recommendation_str = ", ".join(recommendation) if recommendation else ctx.rationale

        html += f"""
            <tr>
                <td><strong>{escape_html(symbol)}</strong></td>
                <td><span class="regime-badge" style="background: {market_rc['bg']}; color: {market_rc['text']}">{escape_html(regime.market_regime.value)}</span></td>
                <td>{sector_cell}</td>
                <td>{stock_cell}</td>
                <td><span class="action-badge" style="background: {action_color['bg']}; color: {action_color['text']}">{escape_html(regime.action.display_name)}</span></td>
                <td>{escape_html(recommendation_str)}</td>
            </tr>
        """

    html += "</tbody></table></div>"
    return html


def generate_alerts_html(
    hierarchical_regimes: Dict[str, HierarchicalRegime],
    theme: str = "dark",
) -> str:
    """
    Generate HTML for alerts section.

    Shows active 4H early warnings.
    """
    get_theme_colors(theme)

    # Collect all alerts
    all_alerts: List[Dict[str, str]] = []
    for symbol, regime in hierarchical_regimes.items():
        for alert_msg in regime.alerts:
            all_alerts.append({"symbol": symbol, "message": alert_msg})

    if not all_alerts:
        return """
        <div class="alerts-section">
            <h2 class="section-header">Alerts</h2>
            <div class="no-alerts">No active alerts</div>
        </div>
        """

    html = f"""
    <div class="alerts-section">
        <h2 class="section-header">Alerts ({len(all_alerts)})</h2>
        <div class="alert-list">
    """

    for alert in all_alerts:
        html += f"""
        <div class="alert-item">
            <span class="alert-symbol">{escape_html(alert['symbol'])}</span>
            <span class="alert-message">{escape_html(alert['message'])}</span>
        </div>
        """

    html += "</div></div>"
    return html
