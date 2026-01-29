"""
Regime Reporting Utilities.

Shared constants, colors, and utility functions for regime report generation.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List

from ..value_card import escape_html

if TYPE_CHECKING:
    from src.domain.services.regime import HierarchicalRegime


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


def get_state_color(state: str) -> str:
    """Get color for a component state."""
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
        "iv_normal": "#94a3b8",
        "iv_low": "#22c55e",
        "na": "#64748b",
    }
    return state_colors.get(state, "#94a3b8")


def get_theme_colors(theme: str) -> Dict[str, str]:
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


def get_score_gradient_color(score: float) -> str:
    """
    Map score (0-100) to a red-yellow-green gradient color.

    Uses HSL color space for smooth transitions:
    - Score 0 → red (#d9534f)
    - Score 50 → amber (#d9a634)
    - Score 100 → green (#5cb85c)
    """
    import colorsys

    # Clamp to 0-100
    score = max(0.0, min(100.0, score))

    # Piecewise hue interpolation:
    # 0-50: red (hue=0°) to yellow/amber (hue=45°)
    # 50-100: yellow (hue=45°) to green (hue=120°)
    if score <= 50:
        hue = (score / 50.0) * 45.0
    else:
        hue = 45.0 + ((score - 50.0) / 50.0) * 75.0

    # HSL to RGB (saturation=0.7, lightness=0.5)
    r, g, b = colorsys.hls_to_rgb(hue / 360.0, 0.5, 0.7)

    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def render_regime_level(
    level_name: str,
    symbols: List[tuple],
    colors: Dict[str, str],
) -> str:
    """Render a level section with regime cards."""
    html = f"""
    <div class="regime-level">
        <h3>{escape_html(level_name)}</h3>
        <div class="regime-cards">
    """

    for symbol, regime in symbols:
        rc = REGIME_COLORS.get(regime.market_regime.value, REGIME_COLORS["R1"])
        action_color = ACTION_COLORS.get(regime.action.display_name, ACTION_COLORS["No Go"])

        html += f"""
        <div class="regime-card" style="background: {colors['card_bg']}">
            <div class="regime-card-header">
                <span class="regime-card-symbol">{escape_html(symbol)}</span>
                <span class="regime-badge" style="background: {rc['bg']}; color: {rc['text']}">
                    {escape_html(regime.market_regime.value)}
                </span>
            </div>
            <div class="regime-card-details">
                <div>Confidence: {regime.market_confidence}%</div>
                <div style="margin-top: 4px;">
                    Action: <span class="action-badge" style="background: {action_color['bg']}; color: {action_color['text']}">{escape_html(regime.action.display_name)}</span>
                </div>
            </div>
        </div>
        """

    html += "</div></div>"
    return html


def render_component_card(
    title: str,
    state: str,
    metrics: List[tuple],
    colors: Dict[str, str],
) -> str:
    """Render a component card with metrics."""
    state_color = get_state_color(state)

    metrics_html = "\n".join(
        f'<div class="component-metric"><span class="metric-label">{escape_html(label)}</span><span class="metric-value">{escape_html(value)}</span></div>'
        for label, value in metrics
    )

    return f"""
    <div class="component-card">
        <div class="component-card-header">
            <span class="component-card-title">{escape_html(title)}</span>
            <span class="component-state" style="background: {state_color}20; color: {state_color}">{escape_html(state.replace('_', ' ').title())}</span>
        </div>
        <div class="component-metrics">
            {metrics_html}
        </div>
    </div>
    """


def build_regime_data_json(
    hierarchical_regimes: Dict[str, "HierarchicalRegime"],
) -> str:
    """Build JSON data for JavaScript regime visualization."""
    data = {}
    for symbol, regime in hierarchical_regimes.items():
        data[symbol] = regime.to_dict()
    return json.dumps(data, default=str)
