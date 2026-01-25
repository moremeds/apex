"""
Regime Decision Tree - Decision path visualization.

Shows the path taken through the regime decision tree with PASS/FAIL status.
"""

from __future__ import annotations

from typing import List

from src.domain.signals.indicators.regime import (
    MarketRegime,
    RegimeOutput,
    RuleTrace,
    generate_counterfactual,
)

from ..value_card import escape_html, render_section
from .utils import REGIME_COLORS


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
            sections.append(f"""
            <div class="counterfactual-item">
                <div class="cf-header">TO BECOME {escape_html(target)}:</div>
                <div class="cf-body">No simple path - multiple conditions needed</div>
            </div>
            """)
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

        sections.append(f"""
        <div class="counterfactual-item">
            <div class="cf-header">TO BECOME {escape_html(target)}:</div>
            <div class="cf-body">{conditions_html}</div>
        </div>
        """)

    return f"""
    <div class="counterfactual-section">
        <h4>Counterfactual Analysis</h4>
        <div class="counterfactual-grid">
            {"".join(sections)}
        </div>
    </div>
    """
