"""
Signal Reporting Package.

Provides HTML report generation for signal analysis:
- SignalReportGenerator: Interactive HTML with Plotly charts (legacy singlefile)
- PackageBuilder: Directory-based package with lazy loading (PR-02)
- SnapshotBuilder: Machine-readable payload snapshot for diffing
- Description generation for indicators and rules
- Regime reporting: Dashboard, timeline, component breakdown
"""

from .description_generator import (
    generate_indicator_description,
    generate_rule_description,
)

# Heatmap sub-package
from .heatmap import HeatmapBuilder
from .heatmap.model import (
    ColorMetric,
    HeatmapModel,
    SectorGroup,
    SizeMetric,
    TreemapNode,
    get_alignment_color,
    get_daily_change_color,
    get_regime_color,
)

# Package sub-package
from .package import PackageBuilder

# Regime reports (simplified)
from .regime_report import (
    build_regime_data_json,
    generate_regime_styles,
)

# Signal report sub-package
from .signal_report import SignalReportGenerator
from .signal_report.confluence_analyzer import calculate_confluence, derive_indicator_states
from .signal_report.signal_detection import detect_historical_signals
from .snapshot_builder import SnapshotBuilder, SnapshotDiff

__all__ = [
    # Signal reports
    "SignalReportGenerator",
    "PackageBuilder",
    "SnapshotBuilder",
    "SnapshotDiff",
    "generate_indicator_description",
    "generate_rule_description",
    # Confluence and signal detection
    "calculate_confluence",
    "derive_indicator_states",
    "detect_historical_signals",
    # Regime reports (simplified)
    "generate_regime_styles",
    "build_regime_data_json",
    # Heatmap (PR-C)
    "HeatmapBuilder",
    "HeatmapModel",
    "TreemapNode",
    "SectorGroup",
    "SizeMetric",
    "ColorMetric",
    "get_regime_color",
    "get_daily_change_color",
    "get_alignment_color",
]
