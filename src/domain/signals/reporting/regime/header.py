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
