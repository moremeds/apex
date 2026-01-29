"""
Regime Report Generator - HTML sections for regime analysis.

Simplified to show only essential regime information:
- Report Header: Symbol and regime classification
- Composite Score: 0-100 score gauge
- Components: Factor breakdown (trend, momentum, volatility, breadth)
- Turning Point: Early warning detection

For new code, prefer importing directly from the subpackage:
    from src.infrastructure.reporting.regime import (
        generate_report_header_html,
        generate_composite_score_html,
        generate_components_4block_html,
        generate_turning_point_html,
    )
"""

# Re-export all public functions from the regime subpackage for backward compatibility
from .regime import (
    REGIME_COLORS,
    build_regime_data_json,
    generate_components_4block_html,
    generate_composite_score_html,
    generate_recommendations_html,
    generate_regime_one_liner_html,
    generate_regime_styles,
    generate_report_header_html,
    generate_turning_point_html,
)

__all__ = [
    # Header
    "generate_report_header_html",
    "generate_composite_score_html",
    # Components
    "generate_components_4block_html",
    # Turning Point
    "generate_turning_point_html",
    # Styles
    "generate_regime_styles",
    # Utils
    "build_regime_data_json",
    "REGIME_COLORS",
]
