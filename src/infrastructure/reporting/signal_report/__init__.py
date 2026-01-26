"""
Signal Report Sub-Package.

Interactive HTML reports for signal analysis.
"""

from __future__ import annotations

from .confluence_analyzer import calculate_confluence, derive_indicator_states
from .constants import (
    BOUNDED_OSCILLATORS,
    OVERLAY_INDICATORS,
    TIMEFRAME_SECONDS,
    UNBOUNDED_OSCILLATORS,
    VOLUME_INDICATORS,
)
from .generator import SignalReportGenerator
from .html_renderer import (
    render_indicator_cards,
    render_rules,
    render_symbol_options,
    render_timeframe_buttons,
)
from .plotly_scripts import get_scripts
from .regime_renderer import (
    compute_param_analysis,
    compute_regime_outputs,
    render_regime_sections,
)
from .signal_detection import detect_historical_signals
from .theme_styles import get_styles, get_theme_colors

__all__ = [
    # Main generator
    "SignalReportGenerator",
    # Confluence
    "calculate_confluence",
    "derive_indicator_states",
    # Signal detection
    "detect_historical_signals",
    # Constants
    "TIMEFRAME_SECONDS",
    "OVERLAY_INDICATORS",
    "BOUNDED_OSCILLATORS",
    "UNBOUNDED_OSCILLATORS",
    "VOLUME_INDICATORS",
    # Theme
    "get_theme_colors",
    "get_styles",
    # HTML rendering
    "render_symbol_options",
    "render_timeframe_buttons",
    "render_rules",
    "render_indicator_cards",
    # Regime
    "compute_regime_outputs",
    "compute_param_analysis",
    "render_regime_sections",
    # Scripts
    "get_scripts",
]
