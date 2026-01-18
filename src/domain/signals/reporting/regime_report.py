"""
Regime Report Generator - HTML sections for regime analysis.

Generates HTML sections for:
- Regime Dashboard: Color-coded regime cards
- Regime Timeline: Historical regime changes
- Component Breakdown: Detailed indicator values
- Action Summary: Trading decisions
- 4H Alerts: Early warning signals
- Decision Path: Full rule trace with PASS/FAIL (NEW - PR1)
- Methodology: Educational explanation of regime system (NEW - PR1)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.domain.services.regime import (
    HierarchicalRegime,
    ParamProvenance,
    ParamProvenanceSet,
    RecommenderResult,
    get_hierarchy_level,
)
from src.domain.signals.indicators.regime import (
    ComponentStates,
    ComponentValues,
    DerivedMetrics,
    MarketRegime,
    RegimeOutput,
    RuleTrace,
    generate_counterfactual,
)

from .value_card import (
    escape_html,
    format_currency,
    format_percentage,
    get_value_card_styles,
    render_info_row,
    render_one_liner_box,
    render_section,
)

if TYPE_CHECKING:
    pass


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


# =============================================================================
# NEW SECTIONS (PR1) - Explainability
# =============================================================================


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

    body = f"""
    <div class="report-header-content">
        <div class="header-title-row">
            <div class="header-symbol">{escape_html(symbol)}</div>
            <div class="header-regime" style="background: {regime_color['bg']}; color: {regime_color['text']};">
                {regime_code}: {regime_output.final_regime.display_name}
            </div>
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


def generate_methodology_html(theme: str = "dark") -> str:
    """
    Generate the Methodology section explaining how regime classification works.

    This is educational content that helps users understand the system.
    """
    body = """
    <div class="methodology-content">
        <h3>Overview</h3>
        <p>The regime classification system uses a <strong>priority-based decision tree</strong>
        to classify market conditions into one of four regimes:</p>

        <div class="regime-grid">
            <div class="regime-item r0">
                <div class="regime-code">R0</div>
                <div class="regime-label">Healthy Uptrend</div>
                <div class="regime-desc">Full trading allowed. TrendUp + NormalVol + Trending.</div>
            </div>
            <div class="regime-item r1">
                <div class="regime-code">R1</div>
                <div class="regime-label">Choppy/Extended</div>
                <div class="regime-desc">Reduced frequency, wider spreads. TrendUp but Choppy OR Overbought.</div>
            </div>
            <div class="regime-item r2">
                <div class="regime-code">R2</div>
                <div class="regime-label">Risk-Off</div>
                <div class="regime-desc">No new positions (veto). TrendDown OR (HighVol + close&lt;MA50) OR IV_HIGH.</div>
            </div>
            <div class="regime-item r3">
                <div class="regime-code">R3</div>
                <div class="regime-label">Rebound Window</div>
                <div class="regime-desc">Small defined-risk only. HighVol + Oversold + structural confirm.</div>
            </div>
        </div>

        <h3>Priority Order</h3>
        <p>The decision tree evaluates regimes in strict priority order (highest to lowest):</p>
        <ol class="priority-list">
            <li><strong>R2 (Risk-Off)</strong> - Veto power, always checked first</li>
            <li><strong>R3 (Rebound)</strong> - Only if NOT in active downtrend + structural confirm</li>
            <li><strong>R1 (Choppy)</strong> - Only if NOT in strong trend acceleration</li>
            <li><strong>R0 (Healthy)</strong> - Default when conditions are favorable</li>
        </ol>
        <p>If no regime conditions are fully met, the system defaults to R1 (Choppy).</p>

        <h3>Hysteresis (Stability)</h3>
        <p>To prevent whipsaw transitions, the system uses <strong>hysteresis</strong>:</p>
        <ul>
            <li><strong>Entry threshold:</strong> Bars needed to confirm a new regime</li>
            <li><strong>Exit threshold:</strong> Minimum bars before leaving current regime</li>
        </ul>
        <p>This is why <code>decision_regime</code> (raw tree output) may differ from
        <code>final_regime</code> (after hysteresis).</p>

        <h3>Components</h3>
        <table class="component-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Inputs</th>
                    <th>States</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Trend</td>
                    <td>Close, MA50, MA200, MA50 slope</td>
                    <td>UP, DOWN, NEUTRAL</td>
                </tr>
                <tr>
                    <td>Volatility</td>
                    <td>ATR20, ATR percentile (63d, 252d)</td>
                    <td>HIGH, NORMAL, LOW</td>
                </tr>
                <tr>
                    <td>Choppiness</td>
                    <td>CHOP index, CHOP percentile, MA20 crosses</td>
                    <td>CHOPPY, TRENDING, NEUTRAL</td>
                </tr>
                <tr>
                    <td>Extension</td>
                    <td>(Close - MA20) / ATR20</td>
                    <td>OVERBOUGHT, OVERSOLD, SLIGHTLY_HIGH, SLIGHTLY_LOW, NEUTRAL</td>
                </tr>
                <tr>
                    <td>IV (market only)</td>
                    <td>VIX/VXN percentile (63d)</td>
                    <td>HIGH, ELEVATED, NORMAL, LOW, NA</td>
                </tr>
            </tbody>
        </table>
    </div>
    """

    return render_section(
        title="Methodology",
        body=body,
        collapsed=True,
        icon="📖",
        section_id="methodology-section",
    )


def generate_decision_tree_html(regime_output: RegimeOutput, theme: str = "dark") -> str:
    """
    Generate the Decision Tree section showing the path taken.

    Shows each rule check with PASS/FAIL status and evidence.
    """
    rules = regime_output.rules_fired_decision
    decision = regime_output.decision_regime
    final = regime_output.final_regime

    # Group rules by regime target
    r2_rules = [r for r in rules if r.regime_target == "R2"]
    r3_rules = [r for r in rules if r.regime_target == "R3"]
    r1_rules = [r for r in rules if r.regime_target == "R1"]
    r0_rules = [r for r in rules if r.regime_target == "R0"]
    fallback_rules = [r for r in rules if r.regime_target not in ("R0", "R1", "R2", "R3")]

    body = f"""
    <div class="decision-tree-content">
        {_render_decision_one_liner(decision, final, regime_output)}

        <div class="tree-checks">
            {_render_rule_group("R2 RISK-OFF", r2_rules, 1, decision.value == "R2")}
            {_render_rule_group("R3 REBOUND WINDOW", r3_rules, 2, decision.value == "R3")}
            {_render_rule_group("R1 CHOPPY/EXTENDED", r1_rules, 3, decision.value == "R1")}
            {_render_rule_group("R0 HEALTHY UPTREND", r0_rules, 4, decision.value == "R0")}
            {_render_fallback_rules(fallback_rules) if fallback_rules else ""}
        </div>

        {_render_counterfactual_section(rules, decision.value)}
    </div>
    """

    return render_section(
        title="Decision Tree Path",
        body=body,
        collapsed=False,  # Start expanded - this is the key info
        icon="🌲",
        section_id="decision-tree-section",
    )


def _render_decision_one_liner(
    decision: MarketRegime, final: MarketRegime, output: RegimeOutput
) -> str:
    """Render the decision/final one-liner at the top of decision tree."""
    dc = REGIME_COLORS.get(decision.value, REGIME_COLORS["R1"])
    fc = REGIME_COLORS.get(final.value, REGIME_COLORS["R1"])

    if decision == final:
        status = f"""
        <div class="decision-result same">
            <span class="label">Decision Regime:</span>
            <span class="regime-badge" style="background: {dc['bg']}; color: {dc['text']}">{escape_html(decision.value)}</span>
            <span class="equals">=</span>
            <span class="label">Final Regime</span>
        </div>
        """
    else:
        pending = output.transition.pending_count
        threshold = output.transition.entry_threshold
        status = f"""
        <div class="decision-result different">
            <span class="label">Decision Regime (pre-hysteresis):</span>
            <span class="regime-badge" style="background: {dc['bg']}; color: {dc['text']}">{escape_html(decision.value)}</span>
            <span class="separator">→</span>
            <span class="label">Final Regime:</span>
            <span class="regime-badge" style="background: {fc['bg']}; color: {fc['text']}">{escape_html(final.value)}</span>
            <span class="pending-info">(pending {pending}/{threshold} bars to confirm {escape_html(decision.value)})</span>
        </div>
        """

    return status


def _render_rule_group(title: str, rules: List[RuleTrace], priority: int, matched: bool) -> str:
    """Render a group of rules for a specific regime."""
    if not rules:
        return ""

    # Check if any rule passed
    any_passed = any(r.passed for r in rules)

    status_class = "matched" if matched else ("partial" if any_passed else "skipped")
    status_text = "MATCHED" if matched else ("PARTIAL" if any_passed else "SKIPPED")

    rules_html = ""
    for rule in rules:
        status = "pass" if rule.passed else "fail"
        status_icon = "✓" if rule.passed else "✗"

        # Format evidence with actual values
        evidence_items = []
        for k, v in rule.evidence.items():
            if isinstance(v, float):
                evidence_items.append(f"{escape_html(k)}={v:.2f}")
            elif isinstance(v, bool):
                evidence_items.append(f"{escape_html(k)}={v}")
            else:
                evidence_items.append(f"{escape_html(k)}={escape_html(str(v))}")

        # Add threshold comparison if available (shows actual vs threshold)
        threshold_html = ""
        if rule.threshold_info:
            ti = rule.threshold_info
            unit = ti.unit if ti.unit else ""
            threshold_html = f"""
            <div class="rule-threshold">
                <span class="threshold-metric">{escape_html(ti.metric_name)}:</span>
                <span class="threshold-actual">{ti.current_value:.2f}{unit}</span>
                <span class="threshold-op">{escape_html(ti.operator)}</span>
                <span class="threshold-value">{ti.threshold:.2f}{unit}</span>
                <span class="threshold-gap">(gap: {ti.gap:.2f}{unit})</span>
            </div>
            """

        evidence_str = ", ".join(evidence_items)

        rules_html += f"""
        <div class="rule-item {status}">
            <div class="rule-main">
                <span class="rule-icon">{status_icon}</span>
                <span class="rule-id">{escape_html(rule.rule_id)}</span>
                <span class="rule-desc">{escape_html(rule.description)}</span>
                <span class="rule-status">{status.upper()}</span>
            </div>
            <div class="rule-details">
                <span class="rule-evidence">{escape_html(evidence_str)}</span>
                {threshold_html}
            </div>
        </div>
        """

    return f"""
    <div class="check-group {status_class}">
        <div class="check-header">
            <span class="check-priority">CHECK {priority}:</span>
            <span class="check-title">{escape_html(title)}</span>
            <span class="check-status">{status_text}</span>
        </div>
        <div class="check-rules">
            {rules_html}
        </div>
    </div>
    """


def _render_fallback_rules(rules: List[RuleTrace]) -> str:
    """Render fallback rules (NaN handling, etc.)."""
    rules_html = ""
    for rule in rules:
        status = "pass" if rule.passed else "fail"
        rules_html += f"""
        <div class="rule-item {status}">
            <span class="rule-id">{escape_html(rule.rule_id)}</span>
            <span class="rule-desc">{escape_html(rule.description)}</span>
        </div>
        """

    return f"""
    <div class="check-group fallback">
        <div class="check-header">
            <span class="check-title">FALLBACK RULES</span>
        </div>
        <div class="check-rules">
            {rules_html}
        </div>
    </div>
    """


def _render_counterfactual_section(rules: List[RuleTrace], current_regime: str) -> str:
    """Render the counterfactual section showing how to reach other regimes."""
    # Generate counterfactuals for regimes different from current
    other_regimes = [r for r in ["R0", "R2", "R3"] if r != current_regime]
    if current_regime not in ["R0", "R2", "R3"]:
        other_regimes = ["R0", "R2", "R3"]

    sections = []
    for target in other_regimes:
        counterfactuals = generate_counterfactual(rules, target)
        if not counterfactuals:
            sections.append(
                f"""
            <div class="counterfactual-item">
                <div class="cf-header">TO BECOME {escape_html(target)}:</div>
                <div class="cf-body">No simple path - multiple conditions needed</div>
            </div>
            """
            )
            continue

        conditions_html = ""
        for cf in counterfactuals:
            if cf.gap == float("inf"):
                conditions_html += f"""
                <div class="cf-condition">
                    • {escape_html(cf.metric_name)} would need to change (categorical)
                </div>
                """
            else:
                direction = "reach" if cf.operator in (">", ">=") else "drop below"
                conditions_html += f"""
                <div class="cf-condition">
                    • {escape_html(cf.metric_name)} needs to {direction} {cf.threshold:.2f}{escape_html(cf.unit)}
                    <span class="cf-current">(current: {cf.current_value:.2f}{escape_html(cf.unit)}, gap: {abs(cf.gap):.2f})</span>
                </div>
                """

        sections.append(
            f"""
        <div class="counterfactual-item">
            <div class="cf-header">TO BECOME {escape_html(target)}:</div>
            <div class="cf-body">{conditions_html}</div>
        </div>
        """
        )

    return f"""
    <div class="counterfactual-section">
        <h4>Counterfactual Analysis</h4>
        <div class="counterfactual-grid">
            {"".join(sections)}
        </div>
    </div>
    """


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
    state_color = _get_state_color(states.trend_state.value)

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
    state_color = _get_state_color(states.vol_state.value)

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
    state_color = _get_state_color(states.chop_state.value)

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
    state_color = _get_state_color(states.ext_state.value)

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
    state_color = _get_state_color(states.iv_state.value)

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


def _get_state_color(state: str) -> str:
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


# =============================================================================
# PR2: QUALITY & HYSTERESIS SECTIONS
# =============================================================================


def generate_quality_html(regime_output: RegimeOutput, theme: str = "dark") -> str:
    """
    Generate the Data Quality section showing warmup status and component validity.

    Displays:
    - Warmup status (bars needed vs available)
    - Component validity (which components have valid data)
    - Any fallback reasons or data issues
    """
    quality = regime_output.quality

    # Warmup status
    warmup_icon = "✓" if quality.warmup_ok else "⚠"
    warmup_class = "ok" if quality.warmup_ok else "warn"
    warmup_status = f"{quality.warmup_bars_available} / {quality.warmup_bars_needed} bars"

    # Component validity
    validity_rows = []
    for component, valid in quality.component_validity.items():
        icon = "✓" if valid else "⚠"
        status = "Valid" if valid else "N/A"
        issue = quality.component_issues.get(component, "")

        validity_rows.append(
            f"""
        <tr>
            <td>{escape_html(component.title())}</td>
            <td class="{'ok' if valid else 'na'}">{icon} {status}</td>
            <td class="issue">{escape_html(issue)}</td>
        </tr>
        """
        )

    # Fallback info
    fallback_section = ""
    if quality.fallback_active:
        fallback_section = f"""
        <div class="fallback-alert">
            <div class="fallback-header">⚠ FALLBACK ACTIVE</div>
            <div class="fallback-reason">
                Reason: {escape_html(quality.fallback_reason.value)}
            </div>
            {f'<div class="fallback-exception">Exception: {escape_html(quality.exception_msg)}</div>' if quality.exception_msg else ''}
            <div class="fallback-result">
                Default Regime: R1 (Choppy/Extended), Confidence: 0
            </div>
        </div>
        """

    body = f"""
    <div class="quality-content">
        <div class="quality-grid">
            <div class="quality-item">
                <div class="quality-label">Warmup Status</div>
                <div class="quality-value {warmup_class}">{warmup_icon} {warmup_status}</div>
            </div>
            <div class="quality-item">
                <div class="quality-label">NaN Counts</div>
                <div class="quality-value">{len(quality.nan_counts)} metrics with NaN</div>
            </div>
            <div class="quality-item">
                <div class="quality-label">Missing Columns</div>
                <div class="quality-value">{len(quality.missing_columns) if quality.missing_columns else 'None'}</div>
            </div>
        </div>

        {fallback_section}

        <h4>Component Validity</h4>
        <table class="validity-table">
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                    <th>Issue</th>
                </tr>
            </thead>
            <tbody>
                {''.join(validity_rows)}
            </tbody>
        </table>
    </div>
    """

    return render_section(
        title="Data Quality",
        body=body,
        collapsed=False,  # Start expanded - users want to see the values
        icon="📊",
        section_id="quality-section",
    )


def generate_hysteresis_html(regime_output: RegimeOutput, theme: str = "dark") -> str:
    """
    Generate the Hysteresis State Machine section.

    Shows:
    - Decision regime vs final regime
    - Current state (bars in regime, pending, thresholds)
    - Hysteresis rules that were evaluated
    - Transition reason
    """
    decision = regime_output.decision_regime
    final = regime_output.final_regime
    transition = regime_output.transition
    rules = regime_output.rules_fired_hysteresis

    dc = REGIME_COLORS.get(decision.value, REGIME_COLORS["R1"])
    fc = REGIME_COLORS.get(final.value, REGIME_COLORS["R1"])

    # Status based on whether there's a pending transition
    if transition.pending_regime:
        pc = REGIME_COLORS.get(transition.pending_regime.value, REGIME_COLORS["R1"])
        pending_html = f"""
        <div class="state-row">
            <span class="state-label">Pending Regime</span>
            <span class="regime-badge" style="background: {pc['bg']}; color: {pc['text']}">{escape_html(transition.pending_regime.value)}</span>
            <span class="pending-progress">({transition.pending_count} / {transition.entry_threshold} bars)</span>
        </div>
        """
        status_class = "accumulating"
        status_text = "ACCUMULATING"
    else:
        pending_html = """
        <div class="state-row">
            <span class="state-label">Pending Regime</span>
            <span class="state-value muted">None</span>
        </div>
        """
        status_class = "stable"
        status_text = "STABLE"

    # Hysteresis rules
    rules_html = ""
    if rules:
        for rule in rules:
            status = "pass" if rule.passed else "fail"
            status_icon = "✓" if rule.passed else "✗"
            evidence_items = []
            for k, v in list(rule.evidence.items())[:3]:
                if isinstance(v, float):
                    evidence_items.append(f"{escape_html(k)}={v:.2f}")
                else:
                    evidence_items.append(f"{escape_html(k)}={escape_html(str(v))}")
            evidence_str = ", ".join(evidence_items)

            rules_html += f"""
            <div class="hysteresis-rule {status}">
                <span class="rule-icon">{status_icon}</span>
                <span class="rule-id">{escape_html(rule.rule_id)}</span>
                <span class="rule-desc">{escape_html(rule.description)}</span>
                <span class="rule-evidence">({evidence_str})</span>
            </div>
            """

    # Last transition info
    last_transition = ""
    if transition.last_transition_ts:
        last_transition = f"""
        <div class="last-transition">
            Last Transition: {transition.last_transition_ts.strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """

    body = f"""
    <div class="hysteresis-content">
        <div class="hysteresis-summary {status_class}">
            <div class="summary-row">
                <span class="label">Decision Regime (raw):</span>
                <span class="regime-badge" style="background: {dc['bg']}; color: {dc['text']}">{escape_html(decision.value)}</span>
            </div>
            <div class="summary-row">
                <span class="label">Final Regime:</span>
                <span class="regime-badge" style="background: {fc['bg']}; color: {fc['text']}">{escape_html(final.value)}</span>
            </div>
            <div class="summary-row">
                <span class="label">Status:</span>
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
        </div>

        <div class="hysteresis-state">
            <h4>Current State</h4>
            <div class="state-row">
                <span class="state-label">Bars in {escape_html(final.value)}</span>
                <span class="state-value">{transition.bars_in_current}</span>
            </div>
            <div class="state-row">
                <span class="state-label">Exit Threshold</span>
                <span class="state-value">{transition.exit_threshold} bars</span>
            </div>
            {pending_html}
        </div>

        {f'<div class="transition-reason"><strong>Reason:</strong> {escape_html(transition.transition_reason)}</div>' if transition.transition_reason else ''}

        {f'''
        <div class="hysteresis-rules">
            <h4>Hysteresis Rules Evaluated</h4>
            {rules_html}
        </div>
        ''' if rules_html else ''}

        {last_transition}
    </div>
    """

    return render_section(
        title="Hysteresis State Machine",
        body=body,
        collapsed=False,  # Start expanded - users want to see the values
        icon="⚙️",
        section_id="hysteresis-section",
    )


# =============================================================================
# TURNING POINT DETECTION (Phase 4)
# =============================================================================


def _load_experiment_result(symbol: str) -> Optional[Dict[str, Any]]:
    """Load experiment result for a symbol if available."""
    from pathlib import Path

    exp_path = Path(f"experiments/turning_point/{symbol.lower()}_latest.json")
    if exp_path.exists():
        try:
            with open(exp_path) as f:
                result: Dict[str, Any] = json.load(f)
                return result
        except Exception:
            return None
    return None


def generate_turning_point_html(regime_output: RegimeOutput, theme: str = "dark") -> str:
    """
    Generate HTML for Turning Point Detection section (Phase 4).

    Shows:
    - Current turning point state (NONE, TOP_RISK, BOTTOM_RISK)
    - Confidence level with visual indicator
    - Top contributing features
    - Gating actions (block R0, accelerate R3)
    - Walk-forward backtest metrics (if available)
    """
    turning_point = regime_output.turning_point
    symbol = regime_output.symbol

    # If no turning point data, show placeholder
    if turning_point is None:
        return render_section(
            title="Turning Point Detection",
            body="""
            <div class="turning-point-placeholder">
                <p class="muted">Turning point model not active. Train model to enable predictive gating.</p>
                <p class="muted" style="font-size: 11px; margin-top: 8px;">
                    Run <code>python scripts/train_turning_point_model.py</code> to train the model.
                </p>
            </div>
            """,
            collapsed=False,
            icon="🔄",
            section_id="turning-point-section",
        )

    # Determine state styling
    state = turning_point.turn_state.value
    confidence = turning_point.turn_confidence

    state_colors = {
        "none": {"bg": "#374151", "text": "#9ca3af", "label": "NONE"},
        "top_risk": {"bg": "#dc2626", "text": "#ffffff", "label": "TOP RISK"},
        "bottom_risk": {"bg": "#16a34a", "text": "#ffffff", "label": "BOTTOM RISK"},
    }
    sc = state_colors.get(state, state_colors["none"])

    # Confidence bar
    conf_pct = int(confidence * 100)
    if conf_pct >= 70:
        conf_class = "high"
        conf_color = "#16a34a"
    elif conf_pct >= 50:
        conf_class = "medium"
        conf_color = "#ca8a04"
    else:
        conf_class = "low"
        conf_color = "#6b7280"

    # Top features
    features_html = ""
    if turning_point.top_features:
        features_html = """
        <div class="tp-features">
            <h4>Top Contributing Features</h4>
            <div class="feature-list">
        """
        for name, contrib in turning_point.top_features[:3]:
            direction = "+" if contrib >= 0 else ""
            features_html += f"""
                <div class="feature-item">
                    <span class="feature-name">{escape_html(name)}</span>
                    <span class="feature-contrib" style="color: {'#16a34a' if contrib >= 0 else '#dc2626'}">{direction}{contrib:.3f}</span>
                </div>
            """
        features_html += "</div></div>"

    # Gating actions
    gating_html = ""
    if state == "top_risk" and confidence >= 0.7:
        gating_html = """
        <div class="tp-gating warning">
            <span class="gating-icon">⚠️</span>
            <span class="gating-text"><strong>R0 BLOCKED</strong> - High TOP_RISK gates Healthy Uptrend entry</span>
        </div>
        """
    elif state == "bottom_risk" and confidence >= 0.7:
        gating_html = """
        <div class="tp-gating success">
            <span class="gating-icon">🚀</span>
            <span class="gating-text"><strong>R3 ACCELERATED</strong> - High BOTTOM_RISK accelerates Rebound entry</span>
        </div>
        """

    # Inference time
    inference_time = turning_point.inference_time_ms
    inference_status = "fast" if inference_time < 1.0 else "slow"

    # Load experiment/backtest results if available
    exp_result = _load_experiment_result(symbol)
    backtest_html = ""
    if exp_result:
        # Format metrics
        top_roc = exp_result.get("median_top_roc_auc", 0)
        top_pr = exp_result.get("median_top_pr_auc", 0)
        bottom_roc = exp_result.get("median_bottom_roc_auc", 0)
        bottom_pr = exp_result.get("median_bottom_pr_auc", 0)
        std_top = exp_result.get("std_top_roc_auc", 0)
        std_bottom = exp_result.get("std_bottom_roc_auc", 0)
        n_windows = len(exp_result.get("window_metrics", []))
        created_at = exp_result.get("created_at", "N/A")[:10]

        # Color code by quality
        def _metric_color(val: float) -> str:
            if val >= 0.7:
                return "#16a34a"  # Good
            elif val >= 0.55:
                return "#ca8a04"  # Medium
            else:
                return "#dc2626"  # Poor

        backtest_html = f"""
        <div class="tp-backtest">
            <h4>Walk-Forward Backtest ({n_windows} windows)</h4>
            <div class="backtest-grid">
                <div class="backtest-model">
                    <span class="model-label">TOP_RISK</span>
                    <div class="metric-row">
                        <span class="metric-name">ROC-AUC:</span>
                        <span class="metric-value" style="color: {_metric_color(top_roc)}">{top_roc:.3f} ± {std_top:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">PR-AUC:</span>
                        <span class="metric-value" style="color: {_metric_color(top_pr)}">{top_pr:.3f}</span>
                    </div>
                </div>
                <div class="backtest-model">
                    <span class="model-label">BOTTOM_RISK</span>
                    <div class="metric-row">
                        <span class="metric-name">ROC-AUC:</span>
                        <span class="metric-value" style="color: {_metric_color(bottom_roc)}">{bottom_roc:.3f} ± {std_bottom:.3f}</span>
                    </div>
                    <div class="metric-row">
                        <span class="metric-name">PR-AUC:</span>
                        <span class="metric-value" style="color: {_metric_color(bottom_pr)}">{bottom_pr:.3f}</span>
                    </div>
                </div>
            </div>
            <div class="backtest-meta">
                Last evaluated: {created_at} |
                <a href="#" onclick="alert('Run: python scripts/retrain_turning_point_models.py --symbol {symbol}'); return false;">Retrain</a>
            </div>
        </div>
        """

    body = f"""
    <div class="turning-point-content">
        <div class="tp-summary">
            <div class="tp-state-row">
                <span class="label">Turn State:</span>
                <span class="tp-state-badge" style="background: {sc['bg']}; color: {sc['text']}">{sc['label']}</span>
            </div>
            <div class="tp-confidence-row">
                <span class="label">Confidence:</span>
                <div class="confidence-bar-container">
                    <div class="confidence-bar {conf_class}" style="width: {conf_pct}%; background: {conf_color}"></div>
                    <span class="confidence-value">{conf_pct}%</span>
                </div>
            </div>
            <div class="tp-meta-row">
                <span class="label">Model:</span>
                <span class="tp-version">{escape_html(turning_point.model_version)}</span>
                <span class="tp-inference {inference_status}">({inference_time:.2f}ms)</span>
            </div>
        </div>

        {gating_html}
        {features_html}
        {backtest_html}
    </div>
    """

    return render_section(
        title="Turning Point Detection",
        body=body,
        collapsed=False,
        icon="🔄",
        section_id="turning-point-section",
    )


# =============================================================================
# EXISTING SECTIONS (maintained for backward compatibility)
# =============================================================================


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
    _get_theme_colors(theme)

    html = f"""
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
    colors = _get_theme_colors(theme)
    components = regime_output.component_values
    states = regime_output.component_states

    html = f"""
    <div class="component-breakdown">
        <h2 class="section-header">Component Breakdown - {escape_html(regime_output.symbol)}</h2>
        <div class="component-grid">
    """

    # Trend Component
    html += _render_component_card(
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
    html += _render_component_card(
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
            ("Distance from MA20", format_currency(components.close - components.ma20)),
        ],
        colors,
    )

    # IV Component (if available)
    if states.iv_state.value != "na":
        html += _render_component_card(
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
    _get_theme_colors(theme)

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
    _get_theme_colors(theme)

    # Collect all alerts
    all_alerts: List[Dict[str, str]] = []
    for symbol, regime in hierarchical_regimes.items():
        for alert_msg in regime.alerts:
            all_alerts.append({"symbol": symbol, "message": alert_msg})

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
            <span class="alert-symbol">{escape_html(alert['symbol'])}</span>
            <span class="alert-message">{escape_html(alert['message'])}</span>
        </div>
        """

    html += "</div></div>"
    return html


def generate_regime_styles() -> str:
    """Generate CSS styles for regime report sections."""
    base_styles = """
    /* Report Header Section (PR4) */
    .report-header-section {
        margin-bottom: 20px;
        padding: 16px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
    }

    .report-header-content {
        display: flex;
        flex-direction: column;
        gap: 16px;
    }

    .header-title-row {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }

    .header-symbol {
        font-size: 24px;
        font-weight: 700;
        color: var(--text);
    }

    .header-regime {
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 13px;
        font-weight: 600;
    }

    .header-timestamp {
        margin-left: auto;
        font-size: 13px;
        color: var(--text-muted);
    }

    .header-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 8px 24px;
        padding: 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
    }

    .header-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 12px;
    }

    .header-label {
        color: var(--text-muted);
    }

    .header-value {
        font-weight: 500;
    }

    .header-value code {
        background: rgba(59, 130, 246, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 11px;
    }

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

    # Add new PR1 styles
    pr1_styles = """
    /* Methodology Section */
    .methodology-content {
        line-height: 1.6;
    }

    .methodology-content h3 {
        margin-top: 20px;
        margin-bottom: 10px;
        font-size: 16px;
        font-weight: 600;
    }

    .methodology-content p {
        margin-bottom: 12px;
    }

    .regime-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 12px;
        margin: 16px 0;
    }

    .regime-item {
        padding: 12px;
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .regime-item.r0 { border-left: 4px solid #166534; }
    .regime-item.r1 { border-left: 4px solid #ca8a04; }
    .regime-item.r2 { border-left: 4px solid #dc2626; }
    .regime-item.r3 { border-left: 4px solid #2563eb; }

    .regime-code {
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .regime-label {
        font-weight: 600;
        margin-bottom: 4px;
    }

    .regime-desc {
        font-size: 12px;
        color: var(--text-muted);
    }

    .priority-list {
        margin: 12px 0;
        padding-left: 24px;
    }

    .priority-list li {
        margin-bottom: 8px;
    }

    .component-table {
        width: 100%;
        border-collapse: collapse;
        margin: 16px 0;
        font-size: 13px;
    }

    .component-table th,
    .component-table td {
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .component-table th {
        background: var(--header-bg);
        font-weight: 600;
    }

    /* Decision Tree Section */
    .decision-tree-content {
        font-family: monospace;
    }

    .decision-result {
        padding: 16px;
        background: var(--highlight-bg);
        border-radius: 8px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }

    .decision-result.same .label {
        color: var(--text-muted);
    }

    .decision-result.different {
        border: 2px solid #ca8a04;
    }

    .decision-result .separator {
        font-size: 18px;
    }

    .decision-result .pending-info {
        font-size: 12px;
        color: #ca8a04;
    }

    .check-group {
        margin-bottom: 16px;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    .check-group.matched {
        border-color: #22c55e;
    }

    .check-group.skipped {
        opacity: 0.6;
    }

    .check-header {
        padding: 12px 16px;
        background: var(--header-bg);
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .check-priority {
        font-weight: 600;
        color: var(--text-muted);
    }

    .check-title {
        font-weight: 600;
        flex: 1;
    }

    .check-status {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }

    .check-group.matched .check-status {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }

    .check-group.skipped .check-status {
        background: rgba(148, 163, 184, 0.2);
        color: #94a3b8;
    }

    .check-rules {
        padding: 12px 16px;
    }

    .rule-item {
        padding: 8px 0;
        font-size: 13px;
        border-bottom: 1px solid var(--border);
    }

    .rule-item:last-child {
        border-bottom: none;
    }

    .rule-main {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .rule-details {
        margin-left: 28px;
        margin-top: 4px;
    }

    .rule-icon {
        width: 20px;
        text-align: center;
    }

    .rule-item.pass .rule-icon { color: #22c55e; }
    .rule-item.fail .rule-icon { color: #ef4444; }

    .rule-id {
        width: 180px;
        font-weight: 500;
        font-family: monospace;
        font-size: 12px;
    }

    .rule-desc {
        flex: 1;
        color: var(--text-muted);
    }

    .rule-status {
        width: 50px;
        font-size: 11px;
        font-weight: 600;
    }

    .rule-item.pass .rule-status { color: #22c55e; }
    .rule-item.fail .rule-status { color: #ef4444; }

    .rule-evidence {
        font-size: 12px;
        color: var(--text-muted);
        font-family: monospace;
    }

    .rule-threshold {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 4px;
        padding: 6px 10px;
        background: var(--bg);
        border-radius: 4px;
        font-size: 12px;
        font-family: monospace;
    }

    .threshold-metric {
        color: var(--text-muted);
    }

    .threshold-actual {
        font-weight: 600;
        color: var(--text);
    }

    .threshold-op {
        color: var(--text-muted);
    }

    .threshold-value {
        font-weight: 600;
        color: #3b82f6;
    }

    .threshold-gap {
        color: var(--text-muted);
        font-size: 11px;
    }

    .rule-item.pass .threshold-actual { color: #22c55e; }
    .rule-item.fail .threshold-actual { color: #ef4444; }

    /* Counterfactual Section */
    .counterfactual-section {
        margin-top: 24px;
        padding-top: 16px;
        border-top: 2px dashed var(--border);
    }

    .counterfactual-section h4 {
        margin-bottom: 12px;
        font-size: 14px;
    }

    .counterfactual-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
    }

    .counterfactual-item {
        padding: 12px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .cf-header {
        font-weight: 600;
        margin-bottom: 8px;
    }

    .cf-condition {
        font-size: 13px;
        padding: 4px 0;
    }

    .cf-current {
        font-size: 11px;
        color: var(--text-muted);
    }

    /* Component 4-Block */
    .components-4block {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 16px;
    }

    .component-4block {
        padding: 16px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .block-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid var(--border);
    }

    .block-title {
        font-weight: 700;
        font-size: 14px;
    }

    .block-state {
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }

    .block-section {
        margin-bottom: 12px;
    }

    .section-label {
        font-size: 10px;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 4px;
    }

    .section-content {
        font-size: 13px;
    }

    .section-content code {
        font-size: 11px;
        background: var(--code-bg);
        padding: 2px 4px;
        border-radius: 4px;
    }

    .values-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
    }

    .value-item {
        display: flex;
        flex-direction: column;
    }

    .value-label {
        font-size: 11px;
        color: var(--text-muted);
    }

    .value-num {
        font-weight: 600;
        font-size: 14px;
    }
    """

    # PR2 styles for Quality and Hysteresis sections
    pr2_styles = """
    /* Quality Section */
    .quality-content {
        padding: 8px 0;
    }

    .quality-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
        margin-bottom: 20px;
    }

    .quality-item {
        padding: 12px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .quality-label {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .quality-value {
        font-size: 14px;
        font-weight: 600;
    }

    .quality-value.ok { color: #22c55e; }
    .quality-value.warn { color: #ca8a04; }

    .validity-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .validity-table th,
    .validity-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .validity-table th {
        font-weight: 600;
        color: var(--text-muted);
        font-size: 11px;
        text-transform: uppercase;
    }

    .validity-table .ok { color: #22c55e; }
    .validity-table .na { color: #ca8a04; }
    .validity-table .issue { color: var(--text-muted); font-size: 12px; }

    .fallback-alert {
        padding: 16px;
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .fallback-header {
        font-weight: 700;
        color: #ef4444;
        margin-bottom: 8px;
    }

    .fallback-reason,
    .fallback-exception,
    .fallback-result {
        font-size: 13px;
        margin-top: 4px;
    }

    /* Hysteresis Section */
    .hysteresis-content {
        padding: 8px 0;
    }

    .hysteresis-summary {
        padding: 16px;
        background: var(--highlight-bg);
        border-radius: 8px;
        margin-bottom: 20px;
        border: 2px solid var(--border);
    }

    .hysteresis-summary.stable {
        border-color: #22c55e;
    }

    .hysteresis-summary.accumulating {
        border-color: #ca8a04;
    }

    .summary-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 8px;
    }

    .summary-row:last-child {
        margin-bottom: 0;
    }

    .summary-row .label {
        min-width: 180px;
        color: var(--text-muted);
    }

    .status-badge {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    .status-badge.stable {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }

    .status-badge.accumulating {
        background: rgba(202, 138, 4, 0.2);
        color: #ca8a04;
    }

    .hysteresis-state {
        margin-bottom: 20px;
    }

    .hysteresis-state h4 {
        font-size: 14px;
        margin-bottom: 12px;
    }

    .state-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 0;
        border-bottom: 1px solid var(--border);
    }

    .state-label {
        min-width: 150px;
        color: var(--text-muted);
        font-size: 13px;
    }

    .state-value {
        font-weight: 600;
        font-size: 14px;
    }

    .state-value.muted {
        color: var(--text-muted);
        font-weight: normal;
    }

    .pending-progress {
        font-size: 12px;
        color: #ca8a04;
    }

    .transition-reason {
        padding: 12px;
        background: var(--bg);
        border-radius: 8px;
        font-size: 13px;
        margin-bottom: 16px;
    }

    .hysteresis-rules {
        margin-top: 16px;
    }

    .hysteresis-rules h4 {
        font-size: 14px;
        margin-bottom: 12px;
    }

    .hysteresis-rule {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 0;
        font-size: 13px;
        border-bottom: 1px solid var(--border);
    }

    .hysteresis-rule .rule-icon {
        width: 20px;
        text-align: center;
    }

    .hysteresis-rule.pass .rule-icon { color: #22c55e; }
    .hysteresis-rule.fail .rule-icon { color: #ef4444; }

    .hysteresis-rule .rule-id {
        width: 200px;
        font-weight: 500;
    }

    .hysteresis-rule .rule-desc {
        flex: 1;
        color: var(--text-muted);
    }

    .hysteresis-rule .rule-evidence {
        font-size: 11px;
        color: var(--text-muted);
        max-width: 250px;
    }

    .last-transition {
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid var(--border);
        font-size: 12px;
        color: var(--text-muted);
    }
    """

    # PR3 styles for Optimization and Recommendations sections
    pr3_styles = """
    /* Optimization Section */
    .optimization-content {
        padding: 8px 0;
    }

    .provenance-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px;
        background: var(--highlight-bg);
        border-radius: 8px;
        margin-bottom: 20px;
    }

    .provenance-id {
        font-family: monospace;
        font-size: 16px;
        font-weight: 600;
    }

    .provenance-source {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
    }

    .provenance-source.symbol-specific {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }

    .provenance-source.group {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
    }

    .provenance-source.default {
        background: rgba(148, 163, 184, 0.2);
        color: #94a3b8;
    }

    .param-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        margin-bottom: 20px;
    }

    .param-table th,
    .param-table td {
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .param-table th {
        font-weight: 600;
        color: var(--text-muted);
        font-size: 11px;
        text-transform: uppercase;
        background: var(--header-bg);
    }

    .param-source {
        font-size: 11px;
        padding: 2px 6px;
        border-radius: 3px;
    }

    .validation-section {
        margin-top: 20px;
        padding-top: 16px;
        border-top: 1px solid var(--border);
    }

    .validation-section h4 {
        font-size: 14px;
        margin-bottom: 12px;
    }

    .validation-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 12px;
    }

    .validation-item {
        padding: 12px;
        background: var(--bg);
        border-radius: 8px;
        border: 1px solid var(--border);
    }

    .validation-label {
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .validation-value {
        font-size: 14px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    .validation-value.pass { color: #22c55e; }
    .validation-value.fail { color: #ef4444; }

    /* Recommendations Section */
    .recommendations-content {
        padding: 8px 0;
    }

    .no-recommendations {
        padding: 20px;
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 8px;
        text-align: center;
    }

    .no-recommendations .header {
        font-size: 16px;
        font-weight: 600;
        color: #22c55e;
        margin-bottom: 12px;
    }

    .no-recommendations .reason {
        font-size: 13px;
        color: var(--text-muted);
    }

    .no-recommendations .metrics {
        display: flex;
        justify-content: center;
        gap: 24px;
        margin-top: 12px;
        font-size: 12px;
    }

    /* Current params table */
    .current-params {
        margin-top: 20px;
        text-align: left;
    }

    .current-params h4 {
        margin-bottom: 12px;
        font-size: 14px;
        color: var(--text-muted);
    }

    .params-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }

    .params-table th, .params-table td {
        padding: 8px 12px;
        text-align: left;
        border-bottom: 1px solid var(--border);
    }

    .params-table th {
        background: rgba(255, 255, 255, 0.05);
        font-weight: 500;
    }

    .params-table code {
        background: rgba(59, 130, 246, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 12px;
    }

    /* Analysis details section */
    .analysis-details {
        margin-top: 24px;
        padding: 16px;
        background: rgba(59, 130, 246, 0.05);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 8px;
        text-align: left;
    }

    .analysis-details h4 {
        margin-bottom: 16px;
        font-size: 14px;
        color: #3b82f6;
    }

    .analysis-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
        margin-bottom: 16px;
    }

    .analysis-section {
        padding: 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 6px;
    }

    .analysis-title {
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 12px;
        color: var(--text);
    }

    .analysis-table {
        width: 100%;
        font-size: 12px;
        border-collapse: collapse;
    }

    .analysis-table td {
        padding: 6px 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .analysis-table td:first-child {
        color: var(--text-muted);
        width: 45%;
    }

    .analysis-table td:last-child {
        font-family: monospace;
    }

    .analysis-methodology {
        font-size: 11px;
        color: var(--text-muted);
        padding: 12px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 4px;
        line-height: 1.5;
    }

    .recommendation-card {
        padding: 16px;
        background: rgba(202, 138, 4, 0.1);
        border: 1px solid rgba(202, 138, 4, 0.3);
        border-radius: 8px;
        margin-bottom: 16px;
    }

    .recommendation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .recommendation-param {
        font-family: monospace;
        font-size: 14px;
        font-weight: 600;
    }

    .recommendation-change {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .recommendation-current {
        font-size: 16px;
        color: var(--text-muted);
    }

    .recommendation-arrow {
        font-size: 14px;
        color: #ca8a04;
    }

    .recommendation-suggested {
        font-size: 16px;
        font-weight: 600;
        color: #ca8a04;
    }

    .recommendation-delta {
        font-size: 12px;
        padding: 2px 6px;
        background: rgba(202, 138, 4, 0.2);
        border-radius: 4px;
        color: #ca8a04;
    }

    .recommendation-confidence {
        font-size: 12px;
        color: var(--text-muted);
    }

    .recommendation-reason {
        font-size: 13px;
        margin-bottom: 12px;
        padding: 8px;
        background: var(--bg);
        border-radius: 4px;
    }

    .recommendation-evidence {
        font-size: 12px;
        color: var(--text-muted);
    }

    .evidence-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 8px;
        margin-top: 8px;
    }

    .evidence-item {
        display: flex;
        justify-content: space-between;
    }

    .manual-review-badge {
        padding: 4px 8px;
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 12px;
    }
    """

    # Phase 4: Turning Point Detection styles
    turning_point_styles = """
    /* Turning Point Detection Section */
    .turning-point-content {
        padding: 16px;
    }

    .turning-point-placeholder {
        text-align: center;
        padding: 24px;
    }

    .turning-point-placeholder .muted {
        color: var(--text-muted);
        font-size: 13px;
    }

    .tp-summary {
        margin-bottom: 16px;
    }

    .tp-state-row, .tp-confidence-row, .tp-meta-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
    }

    .tp-state-badge {
        padding: 6px 14px;
        border-radius: 6px;
        font-size: 14px;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    .confidence-bar-container {
        flex: 1;
        height: 24px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        position: relative;
        overflow: hidden;
        max-width: 200px;
    }

    .confidence-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .confidence-value {
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 12px;
        font-weight: 600;
        color: #ffffff;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }

    .tp-version {
        font-family: monospace;
        font-size: 11px;
        color: var(--text-muted);
    }

    .tp-inference {
        font-size: 11px;
        color: var(--text-muted);
    }

    .tp-inference.fast {
        color: #16a34a;
    }

    .tp-inference.slow {
        color: #ca8a04;
    }

    .tp-gating {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 16px;
    }

    .tp-gating.warning {
        background: rgba(220, 38, 38, 0.15);
        border: 1px solid rgba(220, 38, 38, 0.3);
    }

    .tp-gating.success {
        background: rgba(22, 163, 74, 0.15);
        border: 1px solid rgba(22, 163, 74, 0.3);
    }

    .gating-icon {
        font-size: 18px;
    }

    .gating-text {
        font-size: 13px;
    }

    .tp-features {
        margin-top: 16px;
    }

    .tp-features h4 {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
        color: var(--text-muted);
    }

    .feature-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }

    .feature-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 4px;
    }

    .feature-name {
        font-family: monospace;
        font-size: 12px;
    }

    .feature-contrib {
        font-family: monospace;
        font-size: 12px;
        font-weight: 600;
    }

    /* Backtest Metrics Section */
    .tp-backtest {
        margin-top: 20px;
        padding: 16px;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .tp-backtest h4 {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
        color: var(--text-muted);
    }

    .backtest-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }

    .backtest-model {
        padding: 12px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 6px;
    }

    .model-label {
        display: block;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .metric-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }

    .metric-name {
        font-size: 12px;
        color: var(--text-muted);
    }

    .metric-value {
        font-family: monospace;
        font-size: 12px;
        font-weight: 600;
    }

    .backtest-meta {
        margin-top: 12px;
        font-size: 11px;
        color: var(--text-muted);
        text-align: right;
    }

    .backtest-meta a {
        color: #3b82f6;
        text-decoration: none;
    }

    .backtest-meta a:hover {
        text-decoration: underline;
    }
    """

    return (
        base_styles
        + pr1_styles
        + pr2_styles
        + pr3_styles
        + turning_point_styles
        + get_value_card_styles()
    )


# =============================================================================
# PR3: OPTIMIZATION & RECOMMENDATIONS SECTIONS
# =============================================================================


def generate_optimization_html(
    provenance: Optional[ParamProvenance] = None,
    provenance_set: Optional[ParamProvenanceSet] = None,
    theme: str = "dark",
) -> str:
    """
    Generate the Parameter Provenance (Optimization) section.

    Shows:
    - Param set ID and source
    - Individual parameter sources
    - Training validation metrics (PBO, DSR, OOS Sharpe)
    """
    if provenance is None and provenance_set is None:
        # No provenance data available
        body = """
        <div class="optimization-content">
            <div class="no-provenance">
                <p>Parameter provenance data not available.</p>
                <p class="muted">Run parameter optimization to generate provenance tracking.</p>
            </div>
        </div>
        """
        return render_section(
            title="Parameter Provenance",
            body=body,
            collapsed=True,
            icon="⚙️",
            section_id="optimization-section",
        )

    # Use provenance from set if available
    prov = provenance_set.provenance if provenance_set else provenance

    if prov is None:
        # Provenance set exists but has no provenance data
        body = """
        <div class="optimization-content">
            <div class="no-provenance">
                <p>Parameter provenance data not available.</p>
            </div>
        </div>
        """
        return render_section(
            title="Parameter Provenance",
            body=body,
            collapsed=True,
            icon="⚙️",
            section_id="optimization-section",
        )

    # Source badge class
    source_class = prov.source.replace("-", "_").replace(" ", "_")

    # Build validation metrics
    validation_html = ""
    if prov.is_validated:
        pbo_status = "pass" if prov.pbo_ok else "fail"
        pbo_icon = "✓" if prov.pbo_ok else "✗"
        dsr_status = "pass" if prov.dsr_ok else "fail"
        dsr_icon = "✓" if prov.dsr_ok else "✗"
        oos_status = "pass" if prov.oos_ok else "fail"
        oos_icon = "✓" if prov.oos_ok else "✗"

        validation_html = f"""
        <div class="validation-section">
            <h4>Validation Metrics (Last Training)</h4>
            <div class="validation-grid">
                <div class="validation-item">
                    <div class="validation-label">Walk-Forward Folds</div>
                    <div class="validation-value">{prov.walk_forward_folds}</div>
                </div>
                <div class="validation-item">
                    <div class="validation-label">PBO (< {prov.pbo_threshold})</div>
                    <div class="validation-value {pbo_status}">{pbo_icon} {prov.pbo_value:.2f}</div>
                </div>
                <div class="validation-item">
                    <div class="validation-label">DSR (> {prov.dsr_threshold})</div>
                    <div class="validation-value {dsr_status}">{dsr_icon} {prov.dsr_value:.2f}</div>
                </div>
                <div class="validation-item">
                    <div class="validation-label">OOS Sharpe (> 0)</div>
                    <div class="validation-value {oos_status}">{oos_icon} {prov.oos_sharpe:.2f}</div>
                </div>
            </div>
        </div>
        """

    # Build parameter table if provenance_set has param_sources
    param_table_html = ""
    if provenance_set and provenance_set.param_sources:
        param_rows = []
        for name, ps in provenance_set.param_sources.items():
            source_badge_class = ps.source.replace("-", "_").replace(" ", "_")
            trained_on = ps.trained_on.isoformat() if ps.trained_on else "N/A"
            param_rows.append(
                f"""
            <tr>
                <td><code>{escape_html(name)}</code></td>
                <td>{ps.value}</td>
                <td><span class="param-source provenance-source {source_badge_class}">{escape_html(ps.source)}</span></td>
                <td>{trained_on}</td>
            </tr>
            """
            )

        param_table_html = f"""
        <table class="param-table">
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                    <th>Source</th>
                    <th>Trained On</th>
                </tr>
            </thead>
            <tbody>
                {''.join(param_rows)}
            </tbody>
        </table>
        """

    trained_date = prov.trained_data_end.isoformat() if prov.trained_data_end else "N/A"
    trainer = prov.trainer_version if prov.trainer_version else "N/A"

    body = f"""
    <div class="optimization-content">
        <div class="provenance-header">
            <div>
                <div class="provenance-label">Param Set ID</div>
                <div class="provenance-id">{escape_html(prov.param_set_id)}</div>
            </div>
            <div>
                <span class="provenance-source {source_class}">{escape_html(prov.source)}</span>
            </div>
        </div>

        <div class="provenance-meta">
            {render_info_row("Symbol", prov.symbol)}
            {render_info_row("Group", prov.group or "N/A")}
            {render_info_row("Trained Data End", trained_date)}
            {render_info_row("Trainer Version", trainer)}
            {render_info_row("Validation Status", "VALIDATED" if prov.validation_passed else "NOT VALIDATED")}
        </div>

        {param_table_html}
        {validation_html}
    </div>
    """

    return render_section(
        title="Parameter Provenance",
        body=body,
        collapsed=True,
        icon="⚙️",
        section_id="optimization-section",
    )


def generate_recommendations_html(
    result: Optional[RecommenderResult] = None,
    theme: str = "dark",
) -> str:
    """
    Generate the Parameter Recommendations section.

    Shows either:
    - "No changes suggested" with explicit reasons, or
    - List of recommendations with evidence
    """
    if result is None:
        body = """
        <div class="recommendations-content">
            <div class="no-recommendations">
                <div class="header">NO ANALYSIS AVAILABLE</div>
                <div class="reason">Run the parameter recommender to analyze thresholds.</div>
            </div>
        </div>
        """
        return render_section(
            title="Parameter Recommendations",
            body=body,
            collapsed=True,
            icon="💡",
            section_id="recommendations-section",
        )

    if not result.has_recommendations:
        # No recommendations - show "well calibrated" message with analysis details
        analysis_html = ""
        if result.analysis_metrics:
            m = result.analysis_metrics
            vol_status = "✓" if m.vol_boundary_density < m.concern_level else "⚠"
            chop_status = "✓" if m.chop_boundary_density < m.concern_level else "⚠"

            analysis_html = f"""
            <div class="analysis-details">
                <h4>Analysis Details</h4>
                <div class="analysis-grid">
                    <div class="analysis-section">
                        <div class="analysis-title">Volatility Threshold Analysis</div>
                        <table class="analysis-table">
                            <tr><td>Current Threshold</td><td><strong>{m.vol_threshold:.0f}</strong> percentile</td></tr>
                            <tr><td>Proxy (rolling stdev returns)</td><td>Current: {m.vol_proxy_current:.1f} | Mean: {m.vol_proxy_mean:.1f}</td></tr>
                            <tr><td>Boundary Density</td><td>{vol_status} {m.vol_boundary_density:.1%} (concern if &gt;{m.concern_level:.0%})</td></tr>
                            <tr><td>Above Threshold</td><td>{m.vol_above_threshold_pct:.1%} of time</td></tr>
                        </table>
                    </div>
                    <div class="analysis-section">
                        <div class="analysis-title">Choppiness Threshold Analysis</div>
                        <table class="analysis-table">
                            <tr><td>Current Threshold</td><td><strong>{m.chop_threshold:.0f}</strong> percentile</td></tr>
                            <tr><td>Proxy (CHOP index pctile)</td><td>Current: {m.chop_proxy_current:.1f} | Mean: {m.chop_proxy_mean:.1f}</td></tr>
                            <tr><td>Boundary Density</td><td>{chop_status} {m.chop_boundary_density:.1%} (concern if &gt;{m.concern_level:.0%})</td></tr>
                            <tr><td>Above Threshold</td><td>{m.chop_above_threshold_pct:.1%} of time</td></tr>
                        </table>
                    </div>
                </div>
                <div class="analysis-methodology">
                    <strong>Methodology:</strong> Boundary density measures % of time values are within ±{m.boundary_tolerance:.0f} points of threshold.
                    High density (&gt;{m.concern_level:.0%}) suggests threshold is at a decision boundary and may need adjustment.
                </div>
            </div>
            """

        # Show current params
        params_html = ""
        if result.current_params:
            params_rows = "".join(
                f"<tr><td><code>{escape_html(k)}</code></td><td>{v}</td></tr>"
                for k, v in sorted(result.current_params.items())
            )
            params_html = f"""
            <div class="current-params">
                <h4>Current Parameters</h4>
                <table class="params-table">
                    <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
                    <tbody>{params_rows}</tbody>
                </table>
            </div>
            """

        body = f"""
        <div class="recommendations-content">
            <div class="no-recommendations">
                <div class="header">NO CHANGES SUGGESTED</div>
                <div class="reason">{escape_html(result.no_change_reason or "Parameters appear well-calibrated")}</div>
                <div class="metrics">
                    <span>Lookback: {result.lookback_days} days</span>
                    <span>Analysis Date: {result.analysis_date.isoformat()}</span>
                </div>
            </div>
            {params_html}
            {analysis_html}
        </div>
        """
        return render_section(
            title="Parameter Recommendations",
            body=body,
            collapsed=True,
            icon="💡",
            section_id="recommendations-section",
        )

    # Build recommendation cards
    cards_html = ""
    for rec in result.recommendations:
        delta_sign = "↑" if rec.change > 0 else "↓"
        ev = rec.evidence

        evidence_html = f"""
        <div class="recommendation-evidence">
            <strong>Evidence:</strong>
            <div class="evidence-grid">
                <div class="evidence-item">
                    <span>Lookback:</span>
                    <span>{ev.lookback_days} days</span>
                </div>
                <div class="evidence-item">
                    <span>Vol Proxy:</span>
                    <span>{escape_html(ev.vol_proxy_name)}</span>
                </div>
                <div class="evidence-item">
                    <span>Boundary Density:</span>
                    <span>{ev.boundary_density:.1%}</span>
                </div>
                <div class="evidence-item">
                    <span>Boundary Tolerance:</span>
                    <span>±{ev.boundary_tolerance} pts</span>
                </div>
            </div>
        </div>
        """

        manual_review = (
            '<div class="manual-review-badge">⚠ MANUAL REVIEW REQUIRED</div>'
            if rec.requires_manual_review
            else ""
        )

        cards_html += f"""
        <div class="recommendation-card">
            <div class="recommendation-header">
                <span class="recommendation-param">{escape_html(rec.param_name)}</span>
                <span class="recommendation-confidence">Confidence: {rec.confidence:.0f}%</span>
            </div>
            <div class="recommendation-change">
                <span class="recommendation-current">{rec.current_value:.0f}</span>
                <span class="recommendation-arrow">→</span>
                <span class="recommendation-suggested">{rec.suggested_value:.0f}</span>
                <span class="recommendation-delta">{delta_sign}{abs(rec.change):.0f}</span>
            </div>
            <div class="recommendation-reason">{escape_html(rec.reason)}</div>
            {evidence_html}
            {manual_review}
        </div>
        """

    body = f"""
    <div class="recommendations-content">
        <div class="recommendations-header">
            <p>Analysis Date: {result.analysis_date.isoformat()} | Lookback: {result.lookback_days} days</p>
        </div>
        {cards_html}
    </div>
    """

    return render_section(
        title=f"Parameter Recommendations ({len(result.recommendations)})",
        body=body,
        collapsed=False,
        icon="💡",
        section_id="recommendations-section",
    )


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


def _render_component_card(
    title: str,
    state: str,
    metrics: List[tuple],
    colors: Dict[str, str],
) -> str:
    """Render a component card with metrics."""
    state_color = _get_state_color(state)

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
    hierarchical_regimes: Dict[str, HierarchicalRegime],
) -> str:
    """Build JSON data for JavaScript regime visualization."""
    data = {}
    for symbol, regime in hierarchical_regimes.items():
        data[symbol] = regime.to_dict()
    return json.dumps(data, default=str)
