"""
Regime Report Header - Score card and sparkline display.

Provides the top-level regime header for per-symbol reports.
"""

from __future__ import annotations

from typing import List, Optional

from src.domain.signals.indicators.regime import RegimeOutput

from ..value_card import escape_html
from .utils import REGIME_COLORS


def _render_header_sparkline(points: List[float], width: int = 120, height: int = 32) -> str:
    """Render an inline SVG sparkline for the regime header."""
    if len(points) < 2:
        return ""

    delta = points[-1] - points[0]
    if delta > 3:
        color = "#10b981"
    elif delta < -3:
        color = "#ef4444"
    else:
        color = "#94a3b8"

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
    last_x = coords[-1].split(",")[0]
    last_y = coords[-1].split(",")[1]

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'style="display:block;" xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{polyline}" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
        f'<circle cx="{last_x}" cy="{last_y}" r="3" fill="{color}"/>'
        f"</svg>"
    )


def generate_report_header_html(
    regime_output: RegimeOutput,
    theme: str = "dark",
    score_sparkline: Optional[List[float]] = None,
    **kwargs: object,
) -> str:
    """
    Generate the Report Header as a score card with sparkline.

    Shows:
    - Symbol name, regime badge, score, and timestamp
    - Score history sparkline (if data available)
    """
    symbol = regime_output.symbol
    asof_ts = regime_output.asof_ts
    asof_str = asof_ts.strftime("%Y-%m-%d %H:%M") if asof_ts else "N/A"

    # Derive regime label from composite score for consistency
    composite_score = regime_output.composite_score
    if composite_score is not None:
        if composite_score >= 70:
            regime_code = "R0"
            regime_label = "Healthy Uptrend"
            score_color = "#10b981"
        elif composite_score >= 30:
            regime_code = "R1"
            regime_label = "Choppy/Extended"
            score_color = "#f59e0b"
        else:
            regime_code = "R2"
            regime_label = "Risk-Off"
            score_color = "#ef4444"
    else:
        regime_code = regime_output.final_regime.value
        regime_label = regime_output.final_regime.display_name
        score_color = "#94a3b8"

    regime_color = REGIME_COLORS.get(regime_code, REGIME_COLORS["R1"])

    # Theme colors
    bg_color = "#0f172a" if theme == "dark" else "#ffffff"
    text_color = "#e2e8f0" if theme == "dark" else "#1e293b"
    border_color = "#334155" if theme == "dark" else "#e2e8f0"
    muted_color = "#94a3b8" if theme == "dark" else "#64748b"

    # Score number display
    score_display = f"{composite_score:.0f}" if composite_score is not None else "—"

    # Sparkline SVG
    sparkline_html = ""
    sparkline_points = score_sparkline or []
    if len(sparkline_points) >= 2:
        svg = _render_header_sparkline(sparkline_points)
        sparkline_html = f"""
            <div style="display: flex; flex-direction: column; align-items: flex-end; gap: 2px;">
                {svg}
                <div style="font-size: 10px; color: {muted_color};">last {len(sparkline_points)} reports</div>
            </div>
        """

    return f"""
    <div class="report-header-section" style="
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 16px;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 22px; font-weight: 700; color: {text_color};">
                    {escape_html(symbol)}
                </div>
                <div style="
                    background: {regime_color['bg']};
                    color: {regime_color['text']};
                    padding: 4px 10px;
                    border-radius: 6px;
                    font-size: 13px;
                    font-weight: 600;
                ">{regime_code}: {regime_label}</div>
            </div>
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="text-align: center;">
                    <div style="font-size: 32px; font-weight: 700; color: {score_color}; line-height: 1;">
                        {score_display}
                    </div>
                    <div style="font-size: 11px; color: {muted_color};">score</div>
                </div>
                {sparkline_html}
                <div style="font-size: 12px; color: {muted_color}; text-align: right;">
                    {asof_str}
                </div>
            </div>
        </div>
    </div>
    """


def _render_component_breakdown(
    factors: dict,
    weights: dict,
    composite_score: float,
    theme: str = "dark",
) -> str:
    """
    Render the component score breakdown with calculation explanation.

    Shows each factor's:
    - Normalized score (0-1, displayed as 0-100)
    - Weight in the composite formula
    - Contribution to final score
    - Calculation methodology
    """
    bg_color = "#1e293b" if theme == "dark" else "#f8fafc"
    text_color = "#e2e8f0" if theme == "dark" else "#1e293b"
    border_color = "#334155" if theme == "dark" else "#e2e8f0"
    muted_color = "#94a3b8" if theme == "dark" else "#64748b"

    # Component definitions with calculation methodology
    component_info = {
        "trend": {
            "label": "Trend (Long)",
            "icon": "📈",
            "description": "EMA(20)/EMA(50) spread, percentile ranked over 252 bars",
            "formula": "(EMA₂₀ - EMA₅₀) / EMA₅₀ → percentile rank",
            "interpretation": "Higher = stronger uptrend relative to history",
        },
        "trend_short": {
            "label": "Trend (Short)",
            "icon": "🔥",
            "description": "EMA(10)/EMA(20) spread, percentile ranked - more sensitive",
            "formula": "(EMA₁₀ - EMA₂₀) / EMA₂₀ → percentile rank",
            "interpretation": "Higher = recent momentum acceleration",
        },
        "momentum": {
            "label": "RSI Momentum",
            "icon": "💪",
            "description": "RSI(14) percentile ranked over 63 bars",
            "formula": "RSI(14) → percentile rank",
            "interpretation": "Higher = more overbought vs recent history",
        },
        "volatility": {
            "label": "Volatility",
            "icon": "📊",
            "description": "ATR(14)/Price percentile ranked over 63 bars",
            "formula": "ATR(14) / Close → percentile rank",
            "interpretation": "Higher = more stress (inverted in score)",
        },
        "breadth": {
            "label": "Breadth",
            "icon": "🌐",
            "description": "20-day return vs benchmark, percentile ranked",
            "formula": "(Asset Return - Benchmark Return) → percentile rank",
            "interpretation": "Higher = outperforming benchmark",
        },
    }

    rows_html = ""
    for factor_key in [
        "trend",
        "trend_short",
        "momentum",
        "volatility",
        "breadth",
    ]:
        factor_value = factors.get(factor_key)
        weight = weights.get(factor_key, 0)
        info = component_info.get(factor_key, {})

        if factor_value is None:
            score_display = "N/A"
            contribution = 0
            bar_width = 0
            score_color = muted_color
        else:
            # Display as 0-100 for readability
            score_display = f"{factor_value * 100:.0f}"

            # Volatility is inverted in the composite formula
            if factor_key == "volatility":
                contribution = weight * (1 - factor_value) * 100
                # Show inverted score for clarity
                inverted_score = (1 - factor_value) * 100
                score_display = f"{factor_value * 100:.0f} → {inverted_score:.0f}"
            else:
                contribution = weight * factor_value * 100

            bar_width = min(100, max(0, factor_value * 100))

            # Color based on contribution (green = positive, red = negative for vol)
            if factor_key == "volatility":
                # For volatility, lower raw = better
                if factor_value <= 0.3:
                    score_color = "#10b981"  # green
                elif factor_value <= 0.6:
                    score_color = "#f59e0b"  # yellow
                else:
                    score_color = "#ef4444"  # red
            else:
                # For others, higher = better
                if factor_value >= 0.6:
                    score_color = "#10b981"  # green
                elif factor_value >= 0.3:
                    score_color = "#f59e0b"  # yellow
                else:
                    score_color = "#ef4444"  # red

        rows_html += f"""
        <div class="component-row" style="
            display: grid;
            grid-template-columns: 120px 80px 60px 80px 1fr;
            gap: 12px;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid {border_color};
        ">
            <div style="display: flex; align-items: center; gap: 6px;">
                <span>{info.get('icon', '•')}</span>
                <span style="font-weight: 500;">{info.get('label', factor_key)}</span>
            </div>
            <div style="text-align: center; color: {score_color}; font-weight: 600;">
                {score_display}
            </div>
            <div style="text-align: center; color: {muted_color};">
                ×{weight:.0%}
            </div>
            <div style="text-align: center; font-weight: 500;">
                +{contribution:.1f}
            </div>
            <div style="position: relative; height: 8px; background: {border_color}; border-radius: 4px; overflow: hidden;">
                <div style="
                    position: absolute;
                    left: 0;
                    top: 0;
                    height: 100%;
                    width: {bar_width}%;
                    background: {score_color};
                    border-radius: 4px;
                    transition: width 0.3s;
                "></div>
            </div>
        </div>
        """

    # Formula explanation
    formula_html = f"""
    <div style="margin-top: 12px; padding: 10px; background: {border_color}; border-radius: 6px;">
        <div style="font-size: 11px; color: {muted_color}; margin-bottom: 4px;">COMPOSITE FORMULA</div>
        <code style="font-size: 10px; color: {text_color};">
            Score = 0.13×Trend + 0.10×TrendShort + 0.35×RSI + 0.22×(1-Vol) + 0.20×Breadth
        </code>
        <div style="font-size: 11px; color: {muted_color}; margin-top: 8px;">
            • Score ≥70 → R0 (Healthy) | Score 30-70 → R1 (Choppy) | Score ≤30 → R2 (Risk-Off)
        </div>
    </div>
    """

    return f"""
    <div class="component-breakdown" style="
        margin-top: 16px;
        padding: 16px;
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: 8px;
        color: {text_color};
    ">
        <div style="font-weight: 600; margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
            <span>Component Scores</span>
            <span style="font-size: 12px; color: {muted_color};">Percentile Rank (0-100)</span>
        </div>
        <div class="component-header" style="
            display: grid;
            grid-template-columns: 120px 80px 60px 80px 1fr;
            gap: 12px;
            font-size: 11px;
            color: {muted_color};
            padding-bottom: 8px;
            border-bottom: 1px solid {border_color};
        ">
            <div>Factor</div>
            <div style="text-align: center;">Score</div>
            <div style="text-align: center;">Weight</div>
            <div style="text-align: center;">Contrib</div>
            <div>Distribution</div>
        </div>
        {rows_html}
        {formula_html}
    </div>
    """


def generate_composite_score_html(regime_output: RegimeOutput, theme: str = "dark") -> str:
    """
    Generate a dedicated Composite Score Analysis section.

    This is a standalone section showing the composite regime score with
    all component factors, weights, and contributions in detail.

    Args:
        regime_output: RegimeOutput containing composite_score and composite_factors
        theme: Color theme ("dark" or "light")

    Returns:
        HTML string for the composite score section
    """
    composite_score = regime_output.composite_score
    composite_factors = regime_output.composite_factors

    if composite_score is None:
        return ""

    # Theme colors
    text_color = "#e2e8f0" if theme == "dark" else "#1e293b"
    border_color = "#334155" if theme == "dark" else "#e2e8f0"
    muted_color = "#94a3b8" if theme == "dark" else "#64748b"
    card_bg = "#0f172a" if theme == "dark" else "#ffffff"

    # Score color based on regime bands
    if composite_score >= 70:
        score_color = "#10b981"  # green - R0 Healthy
        score_label = "Healthy (R0)"
        score_bg = "rgba(16, 185, 129, 0.15)"
    elif composite_score >= 30:
        score_color = "#f59e0b"  # yellow - R1 Choppy
        score_label = "Choppy (R1)"
        score_bg = "rgba(245, 158, 11, 0.15)"
    else:
        score_color = "#ef4444"  # red - R2 Risk-Off
        score_label = "Risk-Off (R2)"
        score_bg = "rgba(239, 68, 68, 0.15)"

    # Generate component breakdown if factors available
    component_html = ""
    if composite_factors:
        weights = {
            "trend": 0.13,
            "trend_short": 0.10,
            "momentum": 0.35,
            "volatility": 0.22,
            "breadth": 0.20,
        }
        component_html = _render_component_breakdown(
            composite_factors, weights, composite_score, theme
        )

    # Build the gauge visualization
    gauge_pct = min(100, max(0, composite_score))
    gauge_html = f"""
    <div style="position: relative; width: 100%; height: 24px; background: {border_color}; border-radius: 12px; overflow: hidden; margin: 16px 0;">
        <!-- R2 zone (0-30) -->
        <div style="position: absolute; left: 0; top: 0; width: 30%; height: 100%; background: rgba(239, 68, 68, 0.3);"></div>
        <!-- R1 zone (30-70) -->
        <div style="position: absolute; left: 30%; top: 0; width: 40%; height: 100%; background: rgba(245, 158, 11, 0.2);"></div>
        <!-- R0 zone (70-100) -->
        <div style="position: absolute; left: 70%; top: 0; width: 30%; height: 100%; background: rgba(16, 185, 129, 0.3);"></div>
        <!-- Score marker -->
        <div style="position: absolute; left: {gauge_pct}%; top: 0; transform: translateX(-50%); width: 4px; height: 100%; background: {score_color}; box-shadow: 0 0 8px {score_color};"></div>
        <!-- Zone labels -->
        <div style="position: absolute; left: 15%; top: 50%; transform: translate(-50%, -50%); font-size: 10px; color: {muted_color};">R2</div>
        <div style="position: absolute; left: 50%; top: 50%; transform: translate(-50%, -50%); font-size: 10px; color: {muted_color};">R1</div>
        <div style="position: absolute; left: 85%; top: 50%; transform: translate(-50%, -50%); font-size: 10px; color: {muted_color};">R0</div>
    </div>
    """

    return f"""
    <div class="composite-score-section" style="
        background: {card_bg};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
    ">
        <h3 style="
            color: {text_color};
            font-size: 16px;
            font-weight: 600;
            margin: 0 0 16px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        ">
            <span style="font-size: 20px;">📊</span>
            Composite Regime Score
        </h3>

        <!-- Score display -->
        <div style="
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 24px;
            padding: 20px;
            background: {score_bg};
            border-radius: 8px;
            margin-bottom: 16px;
        ">
            <div style="text-align: center;">
                <div style="font-size: 48px; font-weight: 700; color: {score_color}; line-height: 1;">
                    {composite_score:.0f}
                </div>
                <div style="font-size: 14px; color: {muted_color}; margin-top: 4px;">out of 100</div>
            </div>
            <div style="text-align: left;">
                <div style="font-size: 20px; font-weight: 600; color: {score_color};">
                    {score_label}
                </div>
                <div style="font-size: 12px; color: {muted_color}; margin-top: 4px;">
                    Based on 5 calibrated factors
                </div>
            </div>
        </div>

        <!-- Gauge visualization -->
        <div style="margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between; font-size: 11px; color: {muted_color}; margin-bottom: 4px;">
                <span>0 (Risk-Off)</span>
                <span>50 (Neutral)</span>
                <span>100 (Healthy)</span>
            </div>
            {gauge_html}
        </div>

        <!-- Component breakdown -->
        {component_html}
    </div>
    """
