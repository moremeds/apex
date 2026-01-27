"""
Regime Reporting Package - Modular HTML generation for regime analysis.

This package contains the regime report components broken down into logical modules:
- header: Report header and one-liner functions
- methodology: Educational methodology explanation
- decision_tree: Decision tree path rendering
- components: Component 4-block analysis
- quality: Data quality and hysteresis sections
- turning_point: Turning point analysis
- dashboard: Legacy dashboard functions
- optimization: Parameter provenance and recommendations
- styles: CSS styles for all regime sections
- utils: Shared utilities and helpers

Usage:
    from src.infrastructure.reporting.regime import (
        generate_report_header_html,
        generate_regime_one_liner_html,
        generate_methodology_html,
        generate_decision_tree_html,
        generate_components_4block_html,
        generate_quality_html,
        generate_hysteresis_html,
        generate_turning_point_html,
        generate_regime_dashboard_html,
        generate_regime_timeline_html,
        generate_component_breakdown_html,
        generate_action_summary_html,
        generate_alerts_html,
        generate_optimization_html,
        generate_recommendations_html,
        generate_regime_styles,
        build_regime_data_json,
    )
"""

from .components import generate_components_4block_html
from .dashboard import (
    generate_action_summary_html,
    generate_alerts_html,
    generate_component_breakdown_html,
    generate_regime_dashboard_html,
    generate_regime_timeline_html,
)
from .decision_tree import generate_decision_tree_html
from .header import (
    generate_composite_score_html,
    generate_regime_one_liner_html,
    generate_report_header_html,
)
from .methodology import generate_methodology_html
from .optimization import generate_optimization_html, generate_recommendations_html
from .quality import generate_hysteresis_html, generate_quality_html
from .styles import generate_regime_styles
from .turning_point import generate_turning_point_html
from .utils import ACTION_COLORS, REGIME_COLORS, build_regime_data_json

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
