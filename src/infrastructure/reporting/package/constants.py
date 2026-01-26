"""
Package Builder Constants.

Shared constants used across package builder modules.
"""

from __future__ import annotations

# Indicator grouping for chart layout (same as SignalReportGenerator)
OVERLAY_INDICATORS = {
    "bollinger",
    "supertrend",
    "sma",
    "ema",
    "vwap",
    "keltner",
    "donchian",
    "ichimoku",
}
BOUNDED_OSCILLATORS = {"rsi", "stochastic", "kdj", "williams_r", "mfi", "cci", "adx"}
UNBOUNDED_OSCILLATORS = {"macd", "momentum", "roc", "cmf", "pvo", "force_index"}

# Version for package format
PACKAGE_FORMAT_VERSION = "1.0"

# Size budget constants (KB) - M3 PR-03
SUMMARY_BUDGET_KB = 200
MARKET_BUDGET_KB = 8
SECTORS_BUDGET_KB = 20
TICKERS_BUDGET_KB = 100
HIGHLIGHTS_BUDGET_KB = 40
CONFLUENCE_BUDGET_KB = 30
DATA_QUALITY_BUDGET_KB = 2

# Theme color schemes
THEMES = {
    "dark": {
        "bg": "#0f172a",
        "card_bg": "#1e293b",
        "border": "#334155",
        "text": "#f8fafc",
        "text_muted": "#94a3b8",
        "primary": "#3b82f6",
        "success": "#22c55e",
        "warning": "#eab308",
        "danger": "#ef4444",
    },
    "light": {
        "bg": "#ffffff",
        "card_bg": "#f8fafc",
        "border": "#e2e8f0",
        "text": "#1e293b",
        "text_muted": "#64748b",
        "primary": "#2563eb",
        "success": "#16a34a",
        "warning": "#ca8a04",
        "danger": "#dc2626",
    },
}
