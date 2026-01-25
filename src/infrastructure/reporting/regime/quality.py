"""
Regime Quality - Data quality and hysteresis state machine sections.

Provides visibility into data quality issues and hysteresis behavior.
"""

from __future__ import annotations

from src.domain.signals.indicators.regime import RegimeOutput

from ..value_card import escape_html, render_section
from .utils import REGIME_COLORS


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

        validity_rows.append(f"""
        <tr>
            <td>{escape_html(component.title())}</td>
            <td class="{'ok' if valid else 'na'}">{icon} {status}</td>
            <td class="issue">{escape_html(issue)}</td>
        </tr>
        """)

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
        <div class="hysteresis-status {status_class}">
            <span class="status-indicator"></span>
            <span class="status-text">{status_text}</span>
        </div>

        <div class="state-grid">
            <div class="state-row">
                <span class="state-label">Decision Regime</span>
                <span class="regime-badge" style="background: {dc['bg']}; color: {dc['text']}">{escape_html(decision.value)}</span>
            </div>
            <div class="state-row">
                <span class="state-label">Final Regime</span>
                <span class="regime-badge" style="background: {fc['bg']}; color: {fc['text']}">{escape_html(final.value)}</span>
            </div>
            {pending_html}
            <div class="state-row">
                <span class="state-label">Bars in Regime</span>
                <span class="state-value">{transition.bars_in_current}</span>
            </div>
            <div class="state-row">
                <span class="state-label">Entry Threshold</span>
                <span class="state-value">{transition.entry_threshold} bars</span>
            </div>
            <div class="state-row">
                <span class="state-label">Exit Threshold</span>
                <span class="state-value">{transition.exit_threshold} bars</span>
            </div>
        </div>

        {last_transition}

        <div class="hysteresis-rules">
            <h4>Hysteresis Rules Evaluated</h4>
            {rules_html if rules_html else '<div class="no-rules">No hysteresis rules fired</div>'}
        </div>

        <div class="transition-reason">
            <h4>Transition Reason</h4>
            <div class="reason-text">{escape_html(transition.transition_reason or "No recent transition")}</div>
        </div>
    </div>
    """

    return render_section(
        title="Hysteresis State Machine",
        body=body,
        collapsed=False,
        icon="⚙️",
        section_id="hysteresis-section",
    )
