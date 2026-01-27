"""
Regime Report Generator - HTML sections for regime analysis.

This module has been refactored into the `regime/` subpackage for better modularity.
This file maintains backward compatibility by re-exporting all public functions.

For new code, prefer importing directly from the subpackage:
    from src.infrastructure.reporting.regime import (
        generate_report_header_html,
        generate_methodology_html,
        generate_decision_tree_html,
        ...
    )

Generates HTML sections for:
- Regime Dashboard: Color-coded regime cards
- Regime Timeline: Historical regime changes
- Component Breakdown: Detailed indicator values
- Action Summary: Trading decisions
- 4H Alerts: Early warning signals
- Decision Path: Full rule trace with PASS/FAIL (NEW - PR1)
- Methodology: Educational explanation of regime system (NEW - PR1)
"""

# Re-export all public functions from the regime subpackage for backward compatibility
from .regime import (
    ACTION_COLORS,
    REGIME_COLORS,
    build_regime_data_json,
    generate_action_summary_html,
    generate_alerts_html,
    generate_component_breakdown_html,
    generate_components_4block_html,
    generate_composite_score_html,
    generate_decision_tree_html,
    generate_hysteresis_html,
    generate_methodology_html,
    generate_optimization_html,
    generate_quality_html,
    generate_recommendations_html,
    generate_regime_dashboard_html,
    generate_regime_one_liner_html,
    generate_regime_styles,
    generate_regime_timeline_html,
    generate_report_header_html,
    generate_turning_point_html,
)

__all__ = [
    # Header
    "generate_report_header_html",
    "generate_regime_one_liner_html",
    "generate_composite_score_html",
    # Methodology
    "generate_methodology_html",
    # Decision Tree
    "generate_decision_tree_html",
    # Components
    "generate_components_4block_html",
    # Quality
    "generate_quality_html",
    "generate_hysteresis_html",
    # Turning Point
    "generate_turning_point_html",
    # Dashboard
    "generate_regime_dashboard_html",
    "generate_regime_timeline_html",
    "generate_component_breakdown_html",
    "generate_action_summary_html",
    "generate_alerts_html",
    # Optimization
    "generate_optimization_html",
    "generate_recommendations_html",
    # Styles
    "generate_regime_styles",
    # Utils
    "build_regime_data_json",
    "REGIME_COLORS",
    "ACTION_COLORS",
]
