"""
Regime Components - Component 4-block analysis rendering.

Provides the detailed component breakdown with formula, values, thresholds,
and classification result for each regime component.
"""

from __future__ import annotations

from src.domain.signals.indicators.regime import (
    ComponentStates,
    ComponentValues,
    DerivedMetrics,
    RegimeOutput,
)

from ..value_card import escape_html, format_currency, format_percentage, render_section
from .utils import get_state_color


def generate_components_4block_html(regime_output: RegimeOutput, theme: str = "dark") -> str:
    """
    Generate component breakdown with consistent 4-block structure.

    Each component shows:
    1. Formula/Definition
    2. Current Values
    3. Thresholds
    4. Classification Result
    """
    components = regime_output.component_values
    states = regime_output.component_states
    derived = regime_output.derived_metrics

    body = f"""
    <div class="components-4block">
        {_render_trend_block(components, states, derived)}
        {_render_volatility_block(components, states, derived)}
        {_render_choppiness_block(components, states, derived)}
        {_render_extension_block(components, states, derived)}
        {_render_iv_block(components, states, derived) if states.iv_state.value != "na" else ""}
    </div>
    """

    return render_section(
        title="Component Analysis",
        body=body,
        collapsed=False,  # Start expanded - users want to see the values
        icon="🔬",
        section_id="components-section",
    )


def _render_trend_block(
    components: ComponentValues, states: ComponentStates, derived: DerivedMetrics
) -> str:
    """Render the Trend component 4-block."""
    state_color = get_state_color(states.trend_state.value)

    return f"""
    <div class="component-4block">
        <div class="block-header">
            <span class="block-title">TREND</span>
            <span class="block-state" style="background: {state_color}20; color: {state_color}">
                {escape_html(states.trend_state.value.replace('_', ' ').upper())}
            </span>
        </div>

        <div class="block-section">
            <div class="section-label">Formula</div>
            <div class="section-content">
                <code>UP: close > MA200 AND MA50_slope > 0 AND close > MA50</code><br>
                <code>DOWN: close < MA200 AND MA50_slope < 0</code>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Current Values</div>
            <div class="section-content values-grid">
                <div class="value-item">
                    <span class="value-label">Close</span>
                    <span class="value-num">{format_currency(components.close)}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">MA50</span>
                    <span class="value-num">{format_currency(components.ma50)}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">MA200</span>
                    <span class="value-num">{format_currency(components.ma200)}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">MA50 Slope</span>
                    <span class="value-num">{format_percentage(components.ma50_slope, multiply=True)}</span>
                </div>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Thresholds</div>
            <div class="section-content">
                close > MA200: <strong>{"✓" if components.close > components.ma200 else "✗"}</strong> |
                MA50_slope > 0: <strong>{"✓" if components.ma50_slope > 0 else "✗"}</strong> |
                close > MA50: <strong>{"✓" if components.close > components.ma50 else "✗"}</strong>
            </div>
        </div>
    </div>
    """


def _render_volatility_block(
    components: ComponentValues, states: ComponentStates, derived: DerivedMetrics
) -> str:
    """Render the Volatility component 4-block."""
    state_color = get_state_color(states.vol_state.value)

    return f"""
    <div class="component-4block">
        <div class="block-header">
            <span class="block-title">VOLATILITY</span>
            <span class="block-state" style="background: {state_color}20; color: {state_color}">
                {escape_html(states.vol_state.value.replace('_', ' ').upper())}
            </span>
        </div>

        <div class="block-section">
            <div class="section-label">Formula</div>
            <div class="section-content">
                <code>HIGH: ATR_pct_63 > 80 OR ATR_pct_252 > 85</code><br>
                <code>LOW: ATR_pct_63 < 20 AND ATR_pct_252 < 25</code>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Current Values</div>
            <div class="section-content values-grid">
                <div class="value-item">
                    <span class="value-label">ATR20</span>
                    <span class="value-num">{format_currency(components.atr20)}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">ATR % of Price</span>
                    <span class="value-num">{format_percentage(components.atr_pct, multiply=True)}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">ATR Pctile (63d)</span>
                    <span class="value-num">{derived.atr_pctile_short_window:.1f}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">ATR Pctile (252d)</span>
                    <span class="value-num">{derived.atr_pctile_long_window:.1f}</span>
                </div>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Thresholds</div>
            <div class="section-content">
                ATR_pct_63 > 80: <strong>{"✓" if derived.atr_pctile_short_window > 80 else "✗"}</strong> |
                ATR_pct_252 > 85: <strong>{"✓" if derived.atr_pctile_long_window > 85 else "✗"}</strong>
            </div>
        </div>
    </div>
    """


def _render_choppiness_block(
    components: ComponentValues, states: ComponentStates, derived: DerivedMetrics
) -> str:
    """Render the Choppiness component 4-block."""
    state_color = get_state_color(states.chop_state.value)

    return f"""
    <div class="component-4block">
        <div class="block-header">
            <span class="block-title">CHOPPINESS</span>
            <span class="block-state" style="background: {state_color}20; color: {state_color}">
                {escape_html(states.chop_state.value.upper())}
            </span>
        </div>

        <div class="block-section">
            <div class="section-label">Formula</div>
            <div class="section-content">
                <code>CHOPPY: CHOP_pct_252 > 70 OR MA20_crosses >= 4</code><br>
                <code>TRENDING: CHOP_pct_252 < 30 AND MA20_crosses <= 1</code>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Current Values</div>
            <div class="section-content values-grid">
                <div class="value-item">
                    <span class="value-label">CHOP Index</span>
                    <span class="value-num">{derived.chop_value:.1f}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">CHOP Pctile (252d)</span>
                    <span class="value-num">{derived.chop_pctile:.1f}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">MA20 Crosses (10 bars)</span>
                    <span class="value-num">{derived.ma20_crosses}</span>
                </div>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Thresholds</div>
            <div class="section-content">
                CHOP_pct > 70: <strong>{"✓" if derived.chop_pctile > 70 else "✗"}</strong> |
                MA20_crosses >= 4: <strong>{"✓" if derived.ma20_crosses >= 4 else "✗"}</strong>
            </div>
        </div>
    </div>
    """


def _render_extension_block(
    components: ComponentValues, states: ComponentStates, derived: DerivedMetrics
) -> str:
    """Render the Extension component 4-block."""
    state_color = get_state_color(states.ext_state.value)

    return f"""
    <div class="component-4block">
        <div class="block-header">
            <span class="block-title">EXTENSION</span>
            <span class="block-state" style="background: {state_color}20; color: {state_color}">
                {escape_html(states.ext_state.value.replace('_', ' ').upper())}
            </span>
        </div>

        <div class="block-section">
            <div class="section-label">Formula</div>
            <div class="section-content">
                <code>ext = (close - MA20) / ATR20</code><br>
                <code>OVERBOUGHT: ext > 2.0 | OVERSOLD: ext < -2.0</code>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Current Values</div>
            <div class="section-content values-grid">
                <div class="value-item">
                    <span class="value-label">Extension (ATR units)</span>
                    <span class="value-num">{derived.ext_atr_units:.2f}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">Distance from MA20</span>
                    <span class="value-num">{format_currency(components.close - components.ma20)}</span>
                </div>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Thresholds</div>
            <div class="section-content">
                ext > 2.0 (OB): <strong>{"✓" if derived.ext_atr_units > 2.0 else "✗"}</strong> |
                ext < -2.0 (OS): <strong>{"✓" if derived.ext_atr_units < -2.0 else "✗"}</strong>
            </div>
        </div>
    </div>
    """


def _render_iv_block(
    components: ComponentValues, states: ComponentStates, derived: DerivedMetrics
) -> str:
    """Render the IV component 4-block (market level only)."""
    state_color = get_state_color(states.iv_state.value)

    iv_value = derived.iv_value if derived.iv_value is not None else 0
    iv_pctile = derived.iv_pctile if derived.iv_pctile is not None else 0

    return f"""
    <div class="component-4block">
        <div class="block-header">
            <span class="block-title">IMPLIED VOLATILITY</span>
            <span class="block-state" style="background: {state_color}20; color: {state_color}">
                {escape_html(states.iv_state.value.replace('_', ' ').upper())}
            </span>
        </div>

        <div class="block-section">
            <div class="section-label">Formula</div>
            <div class="section-content">
                <code>HIGH: VIX_pct_63 > 75 | ELEVATED: 50-75 | NORMAL: 25-50 | LOW: < 25</code>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Current Values</div>
            <div class="section-content values-grid">
                <div class="value-item">
                    <span class="value-label">VIX/VXN Value</span>
                    <span class="value-num">{iv_value:.2f}</span>
                </div>
                <div class="value-item">
                    <span class="value-label">IV Pctile (63d)</span>
                    <span class="value-num">{iv_pctile:.1f}</span>
                </div>
            </div>
        </div>

        <div class="block-section">
            <div class="section-label">Thresholds</div>
            <div class="section-content">
                VIX_pct > 75 (HIGH): <strong>{"✓" if iv_pctile > 75 else "✗"}</strong> |
                VIX_pct > 50 (ELEVATED): <strong>{"✓" if iv_pctile > 50 else "✗"}</strong>
            </div>
        </div>
    </div>
    """
