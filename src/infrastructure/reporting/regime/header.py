"""
Regime Report Header - Report header and one-liner functions.

Provides the top-level metadata display for regime reports.
"""

from __future__ import annotations

from typing import Optional

from src.domain.services.regime import (
    ParamProvenanceSet,
    RecommenderResult,
)
from src.domain.signals.indicators.regime import RegimeOutput

from ..value_card import escape_html, render_one_liner_box
from .utils import REGIME_COLORS


def generate_report_header_html(
    regime_output: RegimeOutput,
    provenance_set: Optional[ParamProvenanceSet] = None,
    recommendations_result: Optional[RecommenderResult] = None,
    theme: str = "dark",
) -> str:
    """
    Generate the Report Header / Change Log section.

    Shows key metadata:
    - Schema version
    - Param set ID and source
    - Last training/verification dates
    - Recommendations status
    """
    symbol = regime_output.symbol
    asof_ts = regime_output.asof_ts
    asof_str = asof_ts.strftime("%Y-%m-%d %H:%M") if asof_ts else "N/A"
    schema_version = regime_output.schema_version

    # Provenance info
    param_set_id = "N/A"
    params_source = "default"
    last_training = "N/A"
    if provenance_set:
        param_set_id = provenance_set.provenance.param_set_id or "N/A"
        params_source = provenance_set.provenance.source
        if provenance_set.provenance.trained_data_end:
            last_training = provenance_set.provenance.trained_data_end.isoformat()

    # Recommendations status
    rec_status = "Not analyzed"
    rec_date = "N/A"
    if recommendations_result:
        rec_date = recommendations_result.analysis_date.isoformat()
        if recommendations_result.has_recommendations:
            rec_count = len(recommendations_result.recommendations)
            param_names = ", ".join(r.param_name for r in recommendations_result.recommendations)
            rec_status = f"{rec_count} pending ({param_names})"
        else:
            rec_status = "No changes suggested"

    # Regime color
    regime_code = regime_output.final_regime.value
    regime_color = REGIME_COLORS.get(regime_code, REGIME_COLORS["R1"])

    # Phase 5: Composite score display
    composite_score = regime_output.composite_score
    composite_html = ""

    if composite_score is not None:
        # Color based on score: green for high, yellow for mid, red for low
        if composite_score >= 70:
            score_color = "#10b981"  # green
        elif composite_score >= 30:
            score_color = "#f59e0b"  # yellow
        else:
            score_color = "#ef4444"  # red
        composite_html = f"""
            <div class="header-composite-score" style="color: {score_color}; font-weight: 600;">
                Score: {composite_score:.0f}/100
            </div>
        """

    body = f"""
    <div class="report-header-content">
        <div class="header-title-row">
            <div class="header-symbol">{escape_html(symbol)}</div>
            <div class="header-regime" style="background: {regime_color['bg']}; color: {regime_color['text']};">
                {regime_code}: {regime_output.final_regime.display_name}
            </div>
            {composite_html}
            <div class="header-timestamp">{asof_str}</div>
        </div>
        <div class="header-details">
            <div class="header-row">
                <span class="header-label">Schema Version</span>
                <span class="header-value"><code>{escape_html(schema_version)}</code></span>
            </div>
            <div class="header-row">
                <span class="header-label">Param Set ID</span>
                <span class="header-value"><code>{escape_html(param_set_id)}</code></span>
            </div>
            <div class="header-row">
                <span class="header-label">Params Source</span>
                <span class="header-value">{escape_html(params_source)}</span>
            </div>
            <div class="header-row">
                <span class="header-label">Last Training</span>
                <span class="header-value">{escape_html(last_training)}</span>
            </div>
            <div class="header-row">
                <span class="header-label">Last Recommender Run</span>
                <span class="header-value">{escape_html(rec_date)}</span>
            </div>
            <div class="header-row">
                <span class="header-label">Recommendations</span>
                <span class="header-value">{escape_html(rec_status)}</span>
            </div>
        </div>
    </div>
    """

    return f"""
    <div class="report-header-section">
        {body}
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
        "macd_trend": {
            "label": "MACD Trend",
            "icon": "📉",
            "description": "Long MACD (55/89) histogram, percentile ranked",
            "formula": "(EMA₅₅ - EMA₈₉) - Signal₉ → percentile rank",
            "interpretation": "Higher = stronger trend direction",
        },
        "macd_momentum": {
            "label": "MACD Momentum",
            "icon": "⚡",
            "description": "Short MACD (13/21) histogram, percentile ranked",
            "formula": "(EMA₁₃ - EMA₂₁) - Signal₉ → percentile rank",
            "interpretation": "Higher = momentum timing within trend",
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
        "macd_trend",
        "macd_momentum",
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
            Score = 0.10×Trend + 0.08×TrendShort + 0.12×MACDTrend + 0.10×MACDMom + 0.28×RSI + 0.17×(1-Vol) + 0.15×Breadth
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
            "trend": 0.10,
            "trend_short": 0.08,
            "macd_trend": 0.12,
            "macd_momentum": 0.10,
            "momentum": 0.28,
            "volatility": 0.17,
            "breadth": 0.15,
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
                    Based on 7 calibrated factors
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


def generate_regime_one_liner_html(regime_output: RegimeOutput) -> str:
    """
    Generate the UX one-liner showing decision vs final regime.

    This single element reduces 80% of user confusion by clearly showing
    when hysteresis is blocking a regime transition.
    """
    decision = regime_output.decision_regime.value
    final = regime_output.final_regime.value
    pending_count = regime_output.transition.pending_count
    entry_threshold = regime_output.transition.entry_threshold

    return render_one_liner_box(decision, final, pending_count, entry_threshold)
