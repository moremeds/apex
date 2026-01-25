"""
Regime Report Styles - CSS orchestrator for all regime report sections.

This module aggregates CSS from specialized style modules:
- styles_base.py: Header section and common components
- styles_regime_analysis.py: Methodology and Decision Tree sections
- styles_validation_status.py: Quality and Hysteresis sections
- styles_parameters.py: Optimization and Recommendations sections
- styles_turning_point.py: Turning Point Detection section
"""

from __future__ import annotations

from ..value_card import get_value_card_styles
from .styles_base import BASE_STYLES
from .styles_parameters import PARAMETERS_STYLES
from .styles_regime_analysis import REGIME_ANALYSIS_STYLES
from .styles_turning_point import TURNING_POINT_STYLES
from .styles_validation_status import VALIDATION_STATUS_STYLES


def generate_regime_styles() -> str:
    """Generate CSS styles for regime report sections."""
    return (
        BASE_STYLES
        + REGIME_ANALYSIS_STYLES
        + VALIDATION_STATUS_STYLES
        + PARAMETERS_STYLES
        + TURNING_POINT_STYLES
        + get_value_card_styles()
    )
