"""
Signal Reporting Package.

Provides HTML report generation for signal analysis:
- SignalReportGenerator: Interactive HTML with Plotly charts
- Description generation for indicators and rules
- Regime reporting: Dashboard, timeline, component breakdown
"""

from .description_generator import (
    generate_indicator_description,
    generate_rule_description,
)
from .regime_report import (
    build_regime_data_json,
    generate_action_summary_html,
    generate_alerts_html,
    generate_component_breakdown_html,
    generate_regime_dashboard_html,
    generate_regime_styles,
    generate_regime_timeline_html,
)
from .signal_report_generator import SignalReportGenerator

__all__ = [
    # Signal reports
    "SignalReportGenerator",
    "generate_indicator_description",
    "generate_rule_description",
    # Regime reports
    "generate_regime_dashboard_html",
    "generate_regime_timeline_html",
    "generate_component_breakdown_html",
    "generate_action_summary_html",
    "generate_alerts_html",
    "generate_regime_styles",
    "build_regime_data_json",
]
