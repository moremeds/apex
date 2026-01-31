"""
File Writers - JSON and HTML file I/O operations.

Handles writing data files, indicators, regime HTML, and snapshots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logging_setup import get_logger
from src.utils.timezone import now_utc

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
    display_timezone: str = "US/Eastern",
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

        # Compute DualMACD history for verification table
        dual_macd_history = _compute_dual_macd_history_for_key(df, timeframe, display_timezone)

        # Compute TrendPulse history for verification table
        trend_pulse_history = _compute_trend_pulse_history_for_key(df, timeframe)

        file_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "generated_at": now_utc().isoformat(),
            "bar_count": len(df),
            "chart_data": chart_data,
            "signals": signals,
            "dual_macd_history": dual_macd_history,
            "trend_pulse_history": trend_pulse_history,
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
    all_symbols: Optional[List[str]] = None,
    score_sparklines: Optional[Dict[str, List[float]]] = None,
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

    For symbols in all_symbols but not in regime_outputs, generates a
    placeholder HTML explaining insufficient data.

    Args:
        regime_outputs: Dict mapping symbol to RegimeOutput
        regime_dir: Directory to write regime HTML files
        theme: Color theme ("dark" or "light")
        all_symbols: Optional list of all symbols to generate HTML for

    Returns:
        List of symbols for which HTML was written
    """
    from ..regime import (
        generate_components_4block_html,
        generate_composite_score_html,
        generate_report_header_html,
        generate_turning_point_html,
    )

    files_written = []

    for symbol, regime_output in regime_outputs.items():
        try:
            html_sections = []

            # Simplified regime sections: header, composite score, components, turning point
            sparkline = (score_sparklines or {}).get(symbol, [])
            html_sections.append(
                generate_report_header_html(regime_output, theme=theme, score_sparkline=sparkline)
            )
            html_sections.append(generate_composite_score_html(regime_output, theme=theme))
            html_sections.append(generate_components_4block_html(regime_output, theme=theme))
            html_sections.append(generate_turning_point_html(regime_output, theme=theme))

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

    # Generate placeholder HTML for symbols without regime data
    if all_symbols:
        missing_symbols = set(all_symbols) - set(regime_outputs.keys())
        for symbol in missing_symbols:
            _write_placeholder_regime_html(symbol, regime_dir, theme)
            files_written.append(symbol)

    return files_written


def _write_placeholder_regime_html(symbol: str, regime_dir: Path, theme: str) -> None:
    """
    Write placeholder HTML for symbols without regime data.

    This prevents 404 errors when users click on symbols that don't have
    enough data for regime calculation.
    """
    bg_color = "#1e293b" if theme == "dark" else "#f8fafc"
    text_color = "#e2e8f0" if theme == "dark" else "#1e293b"
    muted_color = "#94a3b8" if theme == "dark" else "#64748b"
    border_color = "#334155" if theme == "dark" else "#e2e8f0"
    warning_color = "#f59e0b"

    placeholder_html = f"""
<!-- Regime Analysis for {symbol} - Insufficient Data -->
<div class="regime-report-container" data-symbol="{symbol}">
    <div style="
        background: {bg_color};
        border: 1px solid {border_color};
        border-radius: 12px;
        padding: 32px;
        margin: 16px 0;
        text-align: center;
    ">
        <div style="font-size: 48px; margin-bottom: 16px;">⚠️</div>
        <h2 style="color: {text_color}; margin: 0 0 12px 0; font-size: 20px;">
            Ticker Too New: {symbol}
        </h2>
        <p style="color: {muted_color}; margin: 0 0 16px 0; font-size: 14px;">
            Regime analysis requires at least 6 months (~126 bars) of trading history.<br>
            This ticker may be recently listed, have limited data, or has been delisted.
        </p>
        <div style="
            display: inline-block;
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid {warning_color};
            border-radius: 8px;
            padding: 12px 20px;
            color: {warning_color};
            font-size: 13px;
        ">
            📅 Check back after more trading data accumulates
        </div>
    </div>
</div>
"""
    file_path = regime_dir / f"{symbol}.html"
    file_path.write_text(placeholder_html, encoding="utf-8")
    logger.debug(f"Wrote placeholder regime HTML: {file_path}")


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


def _compute_dual_macd_history_for_key(
    df: pd.DataFrame, timeframe: str = "1d", display_timezone: str = "US/Eastern"
) -> List[Dict[str, Any]]:
    """Compute DualMACD state history for a single DataFrame (last 60 bars)."""
    try:
        from src.domain.signals.indicators.momentum.dual_macd import DualMACDIndicator

        indicator = DualMACDIndicator()
        if len(df) < indicator.warmup_periods:
            return []

        close_df = df[["close"]].copy()
        result = indicator.calculate(close_df, indicator.default_params)
        if result.empty:
            return []

        last_n = 60
        start_idx = max(0, len(result) - last_n)
        rows: List[Dict[str, Any]] = []

        for i in range(start_idx, len(result)):
            current = result.iloc[i]
            previous = result.iloc[i - 1] if i > 0 else None
            state = indicator._get_state(current, previous, indicator.default_params)

            ts = result.index[i]
            # Convert to display timezone for intraday display
            is_daily = timeframe in ("1d", "1w", "1D", "1W")
            if is_daily:
                # Daily bars: just show date, no time conversion needed
                date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
            else:
                # Intraday: convert to display timezone
                if hasattr(ts, "tz") and ts.tz is not None:
                    ts_local = ts.tz_convert(display_timezone)
                elif hasattr(ts, "tz_localize"):
                    ts_local = ts.tz_localize("UTC").tz_convert(display_timezone)
                else:
                    ts_local = ts
                date_str = (
                    ts_local.strftime("%Y-%m-%d %H:%M")
                    if hasattr(ts_local, "strftime")
                    else str(ts_local)
                )
            rows.append({"date": date_str, **state})

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to compute DualMACD history: {e}")
        return []


def _compute_trend_pulse_history_for_key(
    df: pd.DataFrame, timeframe: str = "1d"
) -> List[Dict[str, Any]]:
    """Compute TrendPulse state history for a single DataFrame (last 60 bars)."""
    try:
        from src.domain.signals.indicators.trend.trend_pulse import TrendPulseIndicator

        indicator = TrendPulseIndicator()
        params = indicator.default_params
        ema_periods = params.get("ema_periods", (14, 25, 99, 144, 453))
        min_bars = max(ema_periods[-1] + 50, 200)
        if len(df) < min_bars:
            return []

        required = ["high", "low", "close"]
        if not all(c in df.columns for c in required):
            return []

        hlc_df = df[required].copy()
        result = indicator.calculate(hlc_df, params)
        if result.empty:
            return []

        last_n = 60
        start_idx = max(0, len(result) - last_n)
        rows: List[Dict[str, Any]] = []

        for i in range(start_idx, len(result)):
            current = result.iloc[i]
            previous = result.iloc[i - 1] if i > 0 else None
            state = indicator._get_state(current, previous, params)

            ts = result.index[i]
            is_daily = timeframe in ("1d", "1w", "1D", "1W")
            if is_daily:
                date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
            else:
                if hasattr(ts, "tz") and ts.tz is not None:
                    ts_et = ts.tz_convert("US/Eastern")
                elif hasattr(ts, "tz_localize"):
                    ts_et = ts.tz_localize("UTC").tz_convert("US/Eastern")
                else:
                    ts_et = ts
                date_str = (
                    ts_et.strftime("%Y-%m-%d %H:%M") if hasattr(ts_et, "strftime") else str(ts_et)
                )
            rows.append({"date": date_str, **state})

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to compute TrendPulse history: {e}")
        return []
