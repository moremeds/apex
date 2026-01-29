"""
Regime Components - Composite score breakdown with summary table and detail tiles.

Shows:
1. Summary table with all factors at a glance
2. Four horizontal tiles with calculation details
"""

from __future__ import annotations

from typing import Dict, Optional

from src.domain.signals.indicators.regime import RegimeOutput

from ..value_card import render_section
from .utils import get_score_gradient_color

# Factor weights (must match composite_scorer.py)
FACTOR_WEIGHTS = {
    "trend": 0.13,
    "trend_short": 0.10,
    "momentum": 0.35,
    "volatility": 0.22,
    "breadth": 0.20,
}

FACTOR_INFO = [
    ("trend", "📈", "Trend (Long)"),
    ("trend_short", "🔥", "Trend (Short)"),
    ("momentum", "🚀", "Momentum"),
    ("volatility", "📉", "Volatility"),
    ("breadth", "🌐", "Breadth"),
]


def generate_components_4block_html(regime_output: RegimeOutput, theme: str = "dark") -> str:
    """Generate component breakdown with summary table and 4 detail tiles."""
    composite_score = regime_output.composite_score or 0
    factors = regime_output.composite_factors or {}

    # Theme colors
    bg_color = "#1e293b" if theme == "dark" else "#f8fafc"
    border_color = "#334155" if theme == "dark" else "#e2e8f0"

    # Build summary table
    summary_table = _render_summary_table(factors, composite_score, theme)

    # Build 4 horizontal tiles
    tiles_html = f"""
    <div class="component-tiles" style="
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 20px;
    ">
        {_render_trend_tile(factors, theme)}
        {_render_momentum_tile(factors, theme)}
        {_render_volatility_tile(factors, theme)}
        {_render_breadth_tile(factors, theme)}
    </div>
    """

    body = f"""
    <div class="composite-breakdown" style="
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 20px;
    ">
        {summary_table}
        {tiles_html}
    </div>
    """

    return render_section(
        title="Component Analysis",
        body=body,
        collapsed=False,
        icon="🔬",
        section_id="components-section",
    )


def _render_summary_table(
    factors: Dict[str, Optional[float]], composite_score: float, theme: str
) -> str:
    """Render the summary table with all factors."""
    bg = "#0f172a" if theme == "dark" else "#ffffff"
    text = "#e2e8f0" if theme == "dark" else "#1e293b"
    muted = "#94a3b8" if theme == "dark" else "#64748b"
    border = "#334155" if theme == "dark" else "#e2e8f0"
    bar_bg = "#334155" if theme == "dark" else "#e2e8f0"

    rows_html = []
    for key, icon, label in FACTOR_INFO:
        raw = factors.get(key, 0) or 0
        weight = FACTOR_WEIGHTS[key]

        # Volatility is inverted
        if key == "volatility":
            score = (1 - raw) * 100
        else:
            score = raw * 100

        contrib = score * weight
        color = get_score_gradient_color(score)

        rows_html.append(f"""
        <div class="factor-row" style="
            display: grid;
            grid-template-columns: 160px 70px 70px 70px 1fr;
            gap: 12px;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid {border};
        ">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span>{icon}</span>
                <span style="font-weight: 500; font-size: 13px; color: {text};">{label}</span>
            </div>
            <div style="text-align: center; color: {color}; font-weight: 600; font-size: 14px;">
                {score:.0f}
            </div>
            <div style="text-align: center; color: {muted}; font-size: 12px;">
                ×{weight:.0%}
            </div>
            <div style="text-align: center; font-weight: 500; font-size: 13px; color: {text};">
                +{contrib:.1f}
            </div>
            <div style="position: relative; height: 8px; background: {bar_bg}; border-radius: 4px; overflow: hidden;">
                <div style="
                    position: absolute;
                    left: 0;
                    top: 0;
                    height: 100%;
                    width: {score}%;
                    background: {color};
                    border-radius: 4px;
                "></div>
            </div>
        </div>
        """)

    # Composite score row
    composite_color = get_score_gradient_color(composite_score)

    return f"""
    <div class="summary-table" style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 16px;
    ">
        <div class="table-header" style="
            display: grid;
            grid-template-columns: 160px 70px 70px 70px 1fr;
            gap: 12px;
            font-size: 11px;
            color: {muted};
            padding-bottom: 8px;
            border-bottom: 2px solid {border};
            text-transform: uppercase;
        ">
            <div>Factor</div>
            <div style="text-align: center;">Score</div>
            <div style="text-align: center;">Weight</div>
            <div style="text-align: center;">Contrib</div>
            <div>Distribution</div>
        </div>
        {''.join(rows_html)}
        <div class="composite-row" style="
            display: grid;
            grid-template-columns: 160px 70px 70px 70px 1fr;
            gap: 12px;
            align-items: center;
            padding: 12px 0 4px 0;
            margin-top: 4px;
            border-top: 2px solid {border};
        ">
            <div style="font-weight: 600; font-size: 14px; color: {text};">
                Composite Score
            </div>
            <div style="text-align: center; color: {composite_color}; font-weight: 700; font-size: 16px;">
                {composite_score:.0f}
            </div>
            <div style="text-align: center; color: {muted}; font-size: 12px;">
                =
            </div>
            <div style="text-align: center; font-weight: 600; font-size: 14px; color: {text};">
                Σ
            </div>
            <div style="position: relative; height: 10px; background: {bar_bg}; border-radius: 5px; overflow: hidden;">
                <div style="
                    position: absolute;
                    left: 0;
                    top: 0;
                    height: 100%;
                    width: {composite_score}%;
                    background: {composite_color};
                    border-radius: 5px;
                "></div>
            </div>
        </div>
    </div>
    """


def _render_trend_tile(factors: Dict[str, Optional[float]], theme: str) -> str:
    """Render TREND tile."""
    bg = "#0f172a" if theme == "dark" else "#ffffff"
    text = "#e2e8f0" if theme == "dark" else "#1e293b"
    muted = "#94a3b8" if theme == "dark" else "#64748b"
    border = "#334155" if theme == "dark" else "#e2e8f0"

    trend_long = (factors.get("trend", 0) or 0) * 100
    trend_short = (factors.get("trend_short", 0) or 0) * 100

    return f"""
    <div class="tile" style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 12px;
    ">
        <div style="font-weight: 600; font-size: 12px; color: {text}; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid {border};">
            📈 TREND
        </div>
        <div style="font-size: 11px; color: {muted}; margin-bottom: 8px;">
            <div style="margin-bottom: 6px;">
                <strong style="color: {text};">Long ({trend_long:.0f})</strong><br>
                <code style="font-size: 10px;">(EMA20-EMA50)/EMA50</code><br>
                → percentile rank (252d)
            </div>
            <div>
                <strong style="color: {text};">Short ({trend_short:.0f})</strong><br>
                <code style="font-size: 10px;">(EMA10-EMA20)/EMA20</code><br>
                → percentile rank (252d)
            </div>
        </div>
    </div>
    """


def _render_momentum_tile(factors: Dict[str, Optional[float]], theme: str) -> str:
    """Render MOMENTUM tile."""
    bg = "#0f172a" if theme == "dark" else "#ffffff"
    text = "#e2e8f0" if theme == "dark" else "#1e293b"
    muted = "#94a3b8" if theme == "dark" else "#64748b"
    border = "#334155" if theme == "dark" else "#e2e8f0"

    rsi = (factors.get("momentum", 0) or 0) * 100

    return f"""
    <div class="tile" style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 12px;
    ">
        <div style="font-weight: 600; font-size: 12px; color: {text}; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid {border};">
            🚀 MOMENTUM
        </div>
        <div style="font-size: 11px; color: {muted};">
            <div>
                <strong style="color: {text};">RSI ({rsi:.0f})</strong><br>
                <code style="font-size: 10px;">RSI(14)</code> → pctl (63d)
            </div>
        </div>
    </div>
    """


def _render_volatility_tile(factors: Dict[str, Optional[float]], theme: str) -> str:
    """Render VOLATILITY tile."""
    bg = "#0f172a" if theme == "dark" else "#ffffff"
    text = "#e2e8f0" if theme == "dark" else "#1e293b"
    muted = "#94a3b8" if theme == "dark" else "#64748b"
    border = "#334155" if theme == "dark" else "#e2e8f0"

    vol_raw = (factors.get("volatility", 0) or 0) * 100
    vol_inv = 100 - vol_raw

    return f"""
    <div class="tile" style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 12px;
    ">
        <div style="font-weight: 600; font-size: 12px; color: {text}; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid {border};">
            📉 VOLATILITY
        </div>
        <div style="font-size: 11px; color: {muted};">
            <div style="margin-bottom: 6px;">
                <strong style="color: {text};">Score: {vol_inv:.0f}</strong><br>
                <code style="font-size: 10px;">1 - pctl(ATR14/price)</code>
            </div>
            <div style="margin-bottom: 6px;">
                Raw ATR pctl: {vol_raw:.0f}<br>
                Inverted: {vol_inv:.0f}
            </div>
            <div style="font-size: 10px; font-style: italic;">
                High vol → lower score
            </div>
        </div>
    </div>
    """


def _render_breadth_tile(factors: Dict[str, Optional[float]], theme: str) -> str:
    """Render BREADTH tile."""
    bg = "#0f172a" if theme == "dark" else "#ffffff"
    text = "#e2e8f0" if theme == "dark" else "#1e293b"
    muted = "#94a3b8" if theme == "dark" else "#64748b"
    border = "#334155" if theme == "dark" else "#e2e8f0"

    breadth = (factors.get("breadth", 0) or 0) * 100

    return f"""
    <div class="tile" style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 12px;
    ">
        <div style="font-weight: 600; font-size: 12px; color: {text}; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 1px solid {border};">
            🌐 BREADTH
        </div>
        <div style="font-size: 11px; color: {muted};">
            <div style="margin-bottom: 6px;">
                <strong style="color: {text};">Score: {breadth:.0f}</strong><br>
                <code style="font-size: 10px;">pctl(ret20d - SPY)</code>
            </div>
            <div style="margin-bottom: 6px;">
                20-day relative return<br>
                vs SPY benchmark
            </div>
            <div style="font-size: 10px; font-style: italic;">
                Outperform → higher score
            </div>
        </div>
    </div>
    """
