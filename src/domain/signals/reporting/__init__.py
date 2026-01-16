"""
Signal Reporting Package.

Provides HTML report generation for signal analysis:
- SignalReportGenerator: Interactive HTML with Plotly charts
- Description generation for indicators and rules
"""

from .description_generator import (
    generate_indicator_description,
    generate_rule_description,
)
from .signal_report_generator import SignalReportGenerator

__all__ = [
    "SignalReportGenerator",
    "generate_indicator_description",
    "generate_rule_description",
]
