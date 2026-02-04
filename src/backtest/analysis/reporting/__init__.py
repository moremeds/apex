"""
Reporting module for generating experiment reports.

Generates:
- HTML tearsheets with interactive charts (Plotly)
- Tabbed interface with per-asset breakdown
- All metric categories in sub-tabs
"""

from .experiment_report import (
    build_equity_curve,
    generate_experiment_report,
    query_per_symbol_metrics,
    query_per_window_metrics,
    query_trade_summary,
)
from .html_report import HTMLReportGenerator, ReportConfig, ReportData

__all__ = [
    "HTMLReportGenerator",
    "ReportConfig",
    "ReportData",
    # Experiment report generation
    "generate_experiment_report",
    "query_per_symbol_metrics",
    "query_per_window_metrics",
    "build_equity_curve",
    "query_trade_summary",
]
