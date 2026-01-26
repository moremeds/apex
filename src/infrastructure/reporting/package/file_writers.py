"""
File Writers - JSON and HTML file I/O operations.

Handles writing data files, indicators, regime HTML, and snapshots.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger

from .data_aggregator import build_indicators_data, df_to_chart_data

if TYPE_CHECKING:
    from src.domain.signals.indicators.base import Indicator
    from src.domain.signals.indicators.regime import RegimeOutput
    from src.domain.signals.models import SignalRule

logger = get_logger(__name__)


def write_data_files(
    data: Dict[Tuple[str, str], pd.DataFrame],
    indicators: List["Indicator"],
    rules: List["SignalRule"],
    data_dir: Path,
) -> List[str]:
    """
    Write individual JSON data files for each symbol/timeframe.

    Args:
        data: Dict mapping (symbol, timeframe) to DataFrame
        indicators: List of computed indicators
        rules: List of signal rules
        data_dir: Directory to write data files

    Returns:
        List of data file keys (e.g., ["AAPL_1d", "SPY_1d"])
    """
    from ..signal_report.signal_detection import detect_historical_signals

    files_written = []

    for (symbol, timeframe), df in data.items():
        key = f"{symbol}_{timeframe}"

        # Convert DataFrame to JSON-serializable format
        chart_data = df_to_chart_data(df)

        # Detect signals for this symbol/timeframe
        signals = detect_historical_signals(df, rules, symbol, timeframe)

        file_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "generated_at": datetime.now().isoformat(),
            "bar_count": len(df),
            "chart_data": chart_data,
            "signals": signals,
        }

        file_path = data_dir / f"{key}.json"
        file_path.write_text(
            json.dumps(file_data, indent=2, default=str),
            encoding="utf-8",
        )
        files_written.append(key)

    return files_written


def write_indicators_file(
    indicators: List["Indicator"],
    rules: List["SignalRule"],
    data_dir: Path,
) -> None:
    """
    Write indicators.json with indicator and rule information.

    Args:
        indicators: List of computed indicators
        rules: List of signal rules
        data_dir: Directory to write indicators.json
    """
    indicators_data = build_indicators_data(indicators, rules)

    file_path = data_dir / "indicators.json"
    file_path.write_text(
        json.dumps(indicators_data, indent=2, default=str),
        encoding="utf-8",
    )


def write_regime_html_files(
    regime_outputs: Dict[str, "RegimeOutput"],
    regime_dir: Path,
    theme: str = "dark",
) -> List[str]:
    """
    Write pre-rendered regime HTML files for each symbol.

    This provides 1:1 feature parity with SignalReportGenerator by using
    the same HTML generation functions from the regime package.

    Each file contains:
    - Report header
    - One-liner summary
    - Methodology section
    - Decision tree
    - Components 4-block
    - Quality section
    - Hysteresis section
    - Turning point section
    - Optimization section (placeholder if no provenance)
    - Recommendations section (placeholder if no results)

    Args:
        regime_outputs: Dict mapping symbol to RegimeOutput
        regime_dir: Directory to write regime HTML files
        theme: Color theme ("dark" or "light")

    Returns:
        List of symbols for which HTML was written
    """
    from ..regime import (
        generate_components_4block_html,
        generate_decision_tree_html,
        generate_hysteresis_html,
        generate_methodology_html,
        generate_optimization_html,
        generate_quality_html,
        generate_recommendations_html,
        generate_regime_one_liner_html,
        generate_report_header_html,
        generate_turning_point_html,
    )

    files_written = []

    for symbol, regime_output in regime_outputs.items():
        try:
            html_sections = []

            # Generate all regime sections using the regime package functions
            # Note: Some functions don't take theme arg, they use CSS variables
            html_sections.append(generate_report_header_html(regime_output, theme=theme))
            html_sections.append(generate_regime_one_liner_html(regime_output))
            html_sections.append(generate_methodology_html(theme=theme))
            html_sections.append(generate_decision_tree_html(regime_output, theme=theme))
            html_sections.append(generate_components_4block_html(regime_output, theme=theme))
            html_sections.append(generate_quality_html(regime_output, theme=theme))
            html_sections.append(generate_hysteresis_html(regime_output, theme=theme))
            html_sections.append(generate_turning_point_html(regime_output, theme=theme))

            # Optimization sections (placeholders - actual data comes from param services)
            html_sections.append(generate_optimization_html(provenance=None, theme=theme))
            html_sections.append(generate_recommendations_html(result=None, theme=theme))

            # Combine into full regime HTML
            regime_html = f"""
<!-- Regime Analysis for {symbol} - Generated by PackageBuilder -->
<div class="regime-report-container" data-symbol="{symbol}">
    {''.join(html_sections)}
</div>
"""
            # Write to file
            file_path = regime_dir / f"{symbol}.html"
            file_path.write_text(regime_html, encoding="utf-8")
            files_written.append(symbol)

            logger.debug(f"Wrote regime HTML: {file_path}")
        except Exception as e:
            # Log warning but continue - don't fail entire build for regime HTML
            logger.warning(f"Failed to generate regime HTML for {symbol}: {e}")

    return files_written


def write_summary_file(
    summary: Dict[str, Any],
    output_dir: Path,
) -> float:
    """
    Write summary.json to output directory.

    Args:
        summary: Summary data dictionary
        output_dir: Package output directory

    Returns:
        Size of summary.json in KB
    """
    summary_path = output_dir / "data" / "summary.json"
    summary_json = json.dumps(summary, indent=2, default=str)
    summary_path.write_text(summary_json, encoding="utf-8")
    return len(summary_json.encode("utf-8")) / 1024


def write_manifest_file(
    manifest: Any,
    output_dir: Path,
) -> None:
    """
    Write manifest.json to output directory.

    Args:
        manifest: PackageManifest object
        output_dir: Package output directory
    """
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2),
        encoding="utf-8",
    )


def write_snapshot_file(
    data: Dict[Tuple[str, str], pd.DataFrame],
    regime_outputs: Dict[str, "RegimeOutput"],
    symbols: List[str],
    timeframes: List[str],
    output_dir: Path,
) -> None:
    """
    Write payload snapshot for diffing.

    Args:
        data: Dict mapping (symbol, timeframe) to DataFrame
        regime_outputs: Dict mapping symbol to RegimeOutput
        symbols: List of symbols
        timeframes: List of timeframes
        output_dir: Package output directory
    """
    from ..snapshot_builder import SnapshotBuilder

    snapshot_builder = SnapshotBuilder()
    snapshot = snapshot_builder.build(
        data=data,
        regime_outputs=regime_outputs,
        symbols=symbols,
        timeframes=timeframes,
    )
    snapshot_path = output_dir / "snapshots" / "payload_snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, default=str),
        encoding="utf-8",
    )
