"""
Reporting module for generating experiment reports.

Generates:
- HTML tearsheets with interactive charts (Plotly)
- Tabbed interface with per-asset breakdown
- All metric categories in sub-tabs
"""

from .html_report import HTMLReportGenerator, ReportConfig, ReportData

__all__ = [
    "HTMLReportGenerator",
    "ReportConfig",
    "ReportData",
]
