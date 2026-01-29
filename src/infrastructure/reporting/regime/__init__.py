"""
Regime Reporting Package - Modular HTML generation for regime analysis.

Simplified to show only essential regime information:
- header: Report header with regime classification
- components: Component 4-block analysis (composite score factors)
- turning_point: Turning point detection
- styles: CSS styles for all regime sections

Usage:
    from src.infrastructure.reporting.regime import (
        generate_report_header_html,
        generate_composite_score_html,
        generate_components_4block_html,
        generate_turning_point_html,
        generate_regime_styles,
    )
"""

from .components import generate_components_4block_html
from .header import (
    generate_composite_score_html,
    generate_report_header_html,
)
from .styles import generate_regime_styles
from .turning_point import generate_turning_point_html
from .utils import REGIME_COLORS, build_regime_data_json

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
