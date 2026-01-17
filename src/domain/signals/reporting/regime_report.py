"""
Regime Report Generator - HTML sections for regime analysis.

Generates HTML sections for:
- Regime Dashboard: Color-coded regime cards
- Regime Timeline: Historical regime changes
- Component Breakdown: Detailed indicator values
- Action Summary: Trading decisions
- 4H Alerts: Early warning signals
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.domain.services.regime import (
    MARKET_BENCHMARKS,
    SECTOR_NAMES,
    AccountType,
    HierarchicalRegime,
    TradingAction,
    get_hierarchy_level,
)
from src.domain.signals.indicators.regime import MarketRegime, RegimeOutput


# Regime color mapping
REGIME_COLORS = {
    "R0": {"bg": "#166534", "text": "#ffffff", "name": "Healthy"},  # Green
    "R1": {"bg": "#ca8a04", "text": "#ffffff", "name": "Choppy"},  # Yellow/Amber
    "R2": {"bg": "#dc2626", "text": "#ffffff", "name": "Risk-Off"},  # Red
    "R3": {"bg": "#2563eb", "text": "#ffffff", "name": "Rebound"},  # Blue
}

ACTION_COLORS = {
    "Go": {"bg": "#166534", "text": "#ffffff"},
    "Go Small": {"bg": "#2563eb", "text": "#ffffff"},
    "No Go": {"bg": "#ca8a04", "text": "#ffffff"},
    "Hard No": {"bg": "#dc2626", "text": "#ffffff"},
}


def generate_regime_dashboard_html(
    hierarchical_regimes: Dict[str, HierarchicalRegime],
    theme: str = "dark",
) -> str:
    """
    Generate HTML for regime dashboard section.

    Shows color-coded regime cards for all symbols organized by level.
    """
    colors = _get_theme_colors(theme)

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
        html += _render_regime_level("Market", market_symbols, colors)

    # Sector Level
    if sector_symbols:
        html += _render_regime_level("Sector", sector_symbols, colors)

    # Stock Level
    if stock_symbols:
        html += _render_regime_level("Stock", stock_symbols, colors)

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
    colors = _get_theme_colors(theme)

    html = f"""
    <div class="regime-timeline">
        <h2 class="section-header">Regime Timeline</h2>
    """

    for symbol, history in regime_history.items():
        if not history:
            continue

        html += f"""
        <div class="timeline-symbol">
            <h3>{symbol}</h3>
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
                    {regime}
                </div>
                <div class="event-details">
                    <div class="event-time">{timestamp}</div>
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
    colors = _get_theme_colors(theme)
    components = regime_output.component_values
    states = regime_output.component_states

    html = f"""
    <div class="component-breakdown">
        <h2 class="section-header">Component Breakdown - {regime_output.symbol}</h2>
        <div class="component-grid">
    """

    # Trend Component
    html += _render_component_card(
        "Trend",
        states.trend_state.value,
        [
            ("Close", f"${components.close:.2f}"),
            ("MA20", f"${components.ma20:.2f}"),
            ("MA50", f"${components.ma50:.2f}"),
            ("MA200", f"${components.ma200:.2f}"),
            ("MA50 Slope", f"{components.ma50_slope * 100:.2f}%"),
        ],
        colors,
    )

    # Volatility Component
    html += _render_component_card(
        "Volatility",
        states.vol_state.value,
        [
            ("ATR20", f"${components.atr20:.2f}"),
            ("ATR %", f"{components.atr_pct * 100:.2f}%"),
            ("ATR Pct 63d", f"{components.atr_pct_63:.1f}"),
            ("ATR Pct 252d", f"{components.atr_pct_252:.1f}"),
        ],
        colors,
    )

    # Choppiness Component
    html += _render_component_card(
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
    html += _render_component_card(
        "Extension",
        states.ext_state.value,
        [
            ("Extension", f"{components.ext:.2f} ATR"),
            ("Distance from MA20", f"${components.close - components.ma20:.2f}"),
        ],
        colors,
    )

    # IV Component (if available)
    if states.iv_state.value != "na":
        html += _render_component_card(
            "Implied Volatility",
            states.iv_state.value,
            [
                ("IV Value", f"{components.iv_value:.2f}" if components.iv_value else "N/A"),
                ("IV Pct 63d", f"{components.iv_pct_63:.1f}" if components.iv_pct_63 else "N/A"),
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
    colors = _get_theme_colors(theme)

    html = f"""
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
        sector_rc = REGIME_COLORS.get(regime.sector_regime.value, REGIME_COLORS["R1"]) if regime.sector_regime else None
        stock_rc = REGIME_COLORS.get(regime.stock_regime.value, REGIME_COLORS["R1"]) if regime.stock_regime else None
        action_color = ACTION_COLORS.get(regime.action.display_name, ACTION_COLORS["No Go"])

        sector_cell = f'<span class="regime-badge" style="background: {sector_rc["bg"]}; color: {sector_rc["text"]}">{regime.sector_regime.value}</span>' if sector_rc else "-"
        stock_cell = f'<span class="regime-badge" style="background: {stock_rc["bg"]}; color: {stock_rc["text"]}">{regime.stock_regime.value}</span>' if stock_rc else "-"

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
                <td><strong>{symbol}</strong></td>
                <td><span class="regime-badge" style="background: {market_rc['bg']}; color: {market_rc['text']}">{regime.market_regime.value}</span></td>
                <td>{sector_cell}</td>
                <td>{stock_cell}</td>
                <td><span class="action-badge" style="background: {action_color['bg']}; color: {action_color['text']}">{regime.action.display_name}</span></td>
                <td>{recommendation_str}</td>
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
    colors = _get_theme_colors(theme)

    # Collect all alerts
    all_alerts = []
    for symbol, regime in hierarchical_regimes.items():
        for alert in regime.alerts:
            all_alerts.append({"symbol": symbol, "message": alert})

    if not all_alerts:
        return f"""
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
            <span class="alert-symbol">{alert['symbol']}</span>
            <span class="alert-message">{alert['message']}</span>
        </div>
        """

    html += "</div></div>"
    return html


def generate_regime_styles() -> str:
    """Generate CSS styles for regime report sections."""
    return """
    .regime-dashboard {
        margin-bottom: 24px;
    }

    .regime-timestamp {
        font-size: 12px;
        color: var(--text-muted);
        margin-bottom: 16px;
    }

    .regime-level {
        margin-bottom: 24px;
    }

    .regime-level h3 {
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 12px;
    }

    .regime-cards {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
        gap: 12px;
    }

    .regime-card {
        padding: 16px;
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .regime-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .regime-card-symbol {
        font-size: 18px;
        font-weight: 600;
    }

    .regime-badge {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }

    .regime-card-details {
        font-size: 12px;
        color: var(--text-muted);
    }

    .action-badge {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .action-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .action-table th {
        text-align: left;
        padding: 12px 8px;
        border-bottom: 2px solid var(--border);
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        font-size: 11px;
    }

    .action-table td {
        padding: 10px 8px;
        border-bottom: 1px solid var(--border);
    }

    .component-breakdown {
        margin-bottom: 24px;
    }

    .component-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
        gap: 16px;
    }

    .component-card {
        padding: 16px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .component-card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    .component-card-title {
        font-weight: 600;
    }

    .component-state {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .component-metrics {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }

    .component-metric {
        display: flex;
        justify-content: space-between;
        font-size: 13px;
    }

    .metric-label {
        color: var(--text-muted);
    }

    .metric-value {
        font-weight: 500;
    }

    .alerts-section {
        margin-bottom: 24px;
    }

    .no-alerts {
        text-align: center;
        color: var(--text-muted);
        padding: 24px;
        font-style: italic;
    }

    .alert-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .alert-item {
        padding: 12px;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        display: flex;
        gap: 12px;
        align-items: center;
    }

    .alert-symbol {
        font-weight: 600;
        color: #ef4444;
    }

    .alert-message {
        color: var(--text);
    }

    .timeline-symbol {
        margin-bottom: 16px;
    }

    .timeline-symbol h3 {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .timeline-events {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .timeline-event {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 4px 8px;
        background: var(--bg);
        border-radius: 4px;
    }

    .event-regime {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }

    .event-details {
        font-size: 11px;
        color: var(--text-muted);
    }

    .event-time {
        font-weight: 500;
    }
    """


def _get_theme_colors(theme: str) -> Dict[str, str]:
    """Get theme colors."""
    if theme == "dark":
        return {
            "bg": "#0f172a",
            "card_bg": "#1e293b",
            "text": "#e2e8f0",
            "text_muted": "#94a3b8",
            "border": "#334155",
        }
    return {
        "bg": "#f8fafc",
        "card_bg": "#ffffff",
        "text": "#1e293b",
        "text_muted": "#64748b",
        "border": "#e2e8f0",
    }


def _render_regime_level(
    level_name: str,
    symbols: List[tuple],
    colors: Dict[str, str],
) -> str:
    """Render a level section with regime cards."""
    html = f"""
    <div class="regime-level">
        <h3>{level_name}</h3>
        <div class="regime-cards">
    """

    for symbol, regime in symbols:
        rc = REGIME_COLORS.get(regime.market_regime.value, REGIME_COLORS["R1"])
        action_color = ACTION_COLORS.get(regime.action.display_name, ACTION_COLORS["No Go"])

        html += f"""
        <div class="regime-card" style="background: {colors['card_bg']}">
            <div class="regime-card-header">
                <span class="regime-card-symbol">{symbol}</span>
                <span class="regime-badge" style="background: {rc['bg']}; color: {rc['text']}">
                    {regime.market_regime.value}
                </span>
            </div>
            <div class="regime-card-details">
                <div>Confidence: {regime.market_confidence}%</div>
                <div style="margin-top: 4px;">
                    Action: <span class="action-badge" style="background: {action_color['bg']}; color: {action_color['text']}">{regime.action.display_name}</span>
                </div>
            </div>
        </div>
        """

    html += "</div></div>"
    return html


def _render_component_card(
    title: str,
    state: str,
    metrics: List[tuple],
    colors: Dict[str, str],
) -> str:
    """Render a component card with metrics."""
    # Determine state color
    state_colors = {
        "trend_up": "#22c55e",
        "trend_down": "#ef4444",
        "neutral": "#94a3b8",
        "vol_high": "#ef4444",
        "vol_normal": "#94a3b8",
        "vol_low": "#22c55e",
        "choppy": "#ef4444",
        "trending": "#22c55e",
        "overbought": "#ef4444",
        "oversold": "#22c55e",
        "slightly_high": "#ca8a04",
        "slightly_low": "#ca8a04",
        "iv_high": "#ef4444",
        "iv_elevated": "#ca8a04",
        "iv_low": "#22c55e",
    }
    state_color = state_colors.get(state, "#94a3b8")

    metrics_html = "\n".join(
        f'<div class="component-metric"><span class="metric-label">{label}</span><span class="metric-value">{value}</span></div>'
        for label, value in metrics
    )

    return f"""
    <div class="component-card">
        <div class="component-card-header">
            <span class="component-card-title">{title}</span>
            <span class="component-state" style="background: {state_color}20; color: {state_color}">{state.replace('_', ' ').title()}</span>
        </div>
        <div class="component-metrics">
            {metrics_html}
        </div>
    </div>
    """


def build_regime_data_json(
    hierarchical_regimes: Dict[str, HierarchicalRegime],
) -> str:
    """Build JSON data for JavaScript regime visualization."""
    data = {}
    for symbol, regime in hierarchical_regimes.items():
        data[symbol] = regime.to_dict()
    return json.dumps(data, default=str)
