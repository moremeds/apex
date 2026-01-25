"""
Regime Optimization - Parameter provenance and recommendations.

Provides parameter tracking and tuning recommendation sections.
"""

from __future__ import annotations

from typing import Optional

from src.domain.services.regime import (
    ParamProvenance,
    ParamProvenanceSet,
    RecommenderResult,
)

from ..value_card import escape_html, render_info_row, render_section


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
            param_rows.append(f"""
            <tr>
                <td><code>{escape_html(name)}</code></td>
                <td>{ps.value}</td>
                <td><span class="param-source provenance-source {source_badge_class}">{escape_html(ps.source)}</span></td>
                <td>{trained_on}</td>
            </tr>
            """)

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
