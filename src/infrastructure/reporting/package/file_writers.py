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
        trend_pulse_history = _compute_trend_pulse_history_for_key(df, timeframe, display_timezone)

        # Compute strategy signal histories
        pulse_dip_history = _compute_pulse_dip_history(df, timeframe, display_timezone)
        squeeze_play_history = _compute_squeeze_play_history(df, timeframe, display_timezone)
        regime_flex_history = _compute_regime_flex_history(df, timeframe, display_timezone)
        sector_pulse_history = _compute_sector_pulse_history(
            symbol, df, timeframe, display_timezone
        )

        file_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "generated_at": now_utc().isoformat(),
            "bar_count": len(df),
            "chart_data": chart_data,
            "signals": signals,
            "dual_macd_history": dual_macd_history,
            "trend_pulse_history": trend_pulse_history,
            "pulse_dip_history": pulse_dip_history,
            "squeeze_play_history": squeeze_play_history,
            "regime_flex_history": regime_flex_history,
            "sector_pulse_history": sector_pulse_history,
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
    df: pd.DataFrame, timeframe: str = "1d", display_timezone: str = "US/Eastern"
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
        logger.warning(f"Failed to compute TrendPulse history: {e}")
        return []


def _format_timestamp(ts: Any, timeframe: str, display_timezone: str) -> str:
    """Format a pandas Timestamp for display."""
    is_daily = timeframe in ("1d", "1w", "1D", "1W")
    if is_daily:
        return ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
    if hasattr(ts, "tz") and ts.tz is not None:
        ts_local = ts.tz_convert(display_timezone)
    elif hasattr(ts, "tz_localize"):
        ts_local = ts.tz_localize("UTC").tz_convert(display_timezone)
    else:
        ts_local = ts
    return ts_local.strftime("%Y-%m-%d %H:%M") if hasattr(ts_local, "strftime") else str(ts_local)


def _compute_pulse_dip_history(
    df: pd.DataFrame, timeframe: str = "1d", display_timezone: str = "US/Eastern"
) -> List[Dict[str, Any]]:
    """Compute PulseDip strategy signal history for a single DataFrame (last 60 bars).

    DRIFT RISK: This re-implements PulseDip entry/exit logic for display-only
    per-bar annotations (ENTRY/EXIT/WATCH + stop levels, PnL, bars held).
    The vectorized SignalGenerator only produces boolean series and cannot
    provide these annotations. Keep in sync with:
    - src/domain/strategy/signals/pulse_dip.py (entry/exit conditions)
    - src/domain/strategy/playbook/pulse_dip.py (full event-driven logic)
    """
    try:
        from src.domain.strategy.signals.indicators import atr, ema, rsi

        if len(df) < 120:
            return []

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        # Default PulseDip parameters
        ema_trend_period = 99
        rsi_period = 14
        rsi_entry_threshold = 35.0
        atr_stop_mult = 3.0
        hard_stop_pct = 0.08

        ema_vals = ema(pd.Series(close, index=df.index), ema_trend_period)
        rsi_vals = rsi(pd.Series(close, index=df.index), rsi_period)
        atr_vals = atr(
            pd.Series(high, index=df.index),
            pd.Series(low, index=df.index),
            pd.Series(close, index=df.index),
            14,
        )

        last_n = 60
        start_idx = max(0, len(df) - last_n)
        rows: List[Dict[str, Any]] = []

        # Simple signal simulation for display (no position tracking)
        in_position = False
        entry_price = 0.0
        peak_price = 0.0
        bars_held = 0

        # Pre-scan from warmup for position context
        for i in range(max(ema_trend_period, start_idx - 60), start_idx):
            if i < 0 or i >= len(df):
                continue
            c = close[i]
            r = rsi_vals.iloc[i] if i < len(rsi_vals) and not pd.isna(rsi_vals.iloc[i]) else 50.0
            e = ema_vals.iloc[i] if i < len(ema_vals) and not pd.isna(ema_vals.iloc[i]) else c
            a = atr_vals.iloc[i] if i < len(atr_vals) and not pd.isna(atr_vals.iloc[i]) else 0.0

            if not in_position:
                if c > e and r < rsi_entry_threshold:
                    in_position = True
                    entry_price = c
                    peak_price = c
                    bars_held = 0
            else:
                bars_held += 1
                peak_price = max(peak_price, c)
                trail_stop = peak_price - atr_stop_mult * a
                hard_stop = entry_price * (1 - hard_stop_pct)
                if c < hard_stop or c < trail_stop or r > 65 or bars_held >= 40:
                    in_position = False

        for i in range(start_idx, len(df)):
            c = close[i]
            r_val = (
                rsi_vals.iloc[i] if i < len(rsi_vals) and not pd.isna(rsi_vals.iloc[i]) else 50.0
            )
            e_val = ema_vals.iloc[i] if i < len(ema_vals) and not pd.isna(ema_vals.iloc[i]) else c
            a_val = atr_vals.iloc[i] if i < len(atr_vals) and not pd.isna(atr_vals.iloc[i]) else 0.0

            ema_ok = c > e_val
            rsi_below = r_val < rsi_entry_threshold

            signal = ""
            exit_reason: Optional[str] = None
            pnl_pct: Optional[float] = None

            if not in_position:
                if ema_ok and rsi_below:
                    signal = "ENTRY"
                    in_position = True
                    entry_price = c
                    peak_price = c
                    bars_held = 0
                elif rsi_below:
                    signal = "WATCH"
            else:
                bars_held += 1
                peak_price = max(peak_price, c)
                trail_stop = peak_price - atr_stop_mult * a_val
                hard_stop = entry_price * (1 - hard_stop_pct)

                if c < hard_stop:
                    exit_reason = "HARD_STOP"
                elif c < trail_stop:
                    exit_reason = "ATR_TRAIL"
                elif r_val > 65:
                    exit_reason = "RSI_EXIT"
                elif bars_held >= 40:
                    exit_reason = "TIME_STOP"

                if exit_reason:
                    signal = "EXIT"
                    pnl_pct = round((c / entry_price - 1) * 100, 2)
                    in_position = False
                else:
                    signal = "LONG"

            ts = df.index[i]
            date_str = _format_timestamp(ts, timeframe, display_timezone)

            row: Dict[str, Any] = {
                "date": date_str,
                "rsi": round(float(r_val), 1),
                "ema_ok": ema_ok,
                "signal": signal,
                "exit_reason": exit_reason,
                "pnl_pct": pnl_pct,
                "bars_held": bars_held if in_position or signal == "EXIT" else None,
                "stop_level": (
                    round(float(peak_price - atr_stop_mult * a_val), 2) if in_position else None
                ),
            }
            rows.append(row)

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to compute PulseDip history: {e}")
        return []


def _compute_squeeze_play_history(
    df: pd.DataFrame, timeframe: str = "1d", display_timezone: str = "US/Eastern"
) -> List[Dict[str, Any]]:
    """Compute SqueezePlay strategy signal history for a single DataFrame (last 60 bars).

    DRIFT RISK: This re-implements SqueezePlay signal logic for display-only
    per-bar annotations (SQUEEZE/BREAK/HOLD + direction, ADX, stop levels).
    Keep in sync with:
    - src/domain/strategy/signals/squeeze_play.py (entry/exit conditions)
    - src/domain/strategy/playbook/squeeze_play.py (full event-driven logic)
    """
    try:
        from src.domain.strategy.signals.indicators import adx, atr, bbands

        if len(df) < 50:
            return []

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values

        close_s = pd.Series(close, index=df.index)
        high_s = pd.Series(high, index=df.index)
        low_s = pd.Series(low, index=df.index)

        # Default SqueezePlay parameters
        bb_period = 20
        bb_std = 2.0
        kc_multiplier = 1.5
        release_persist_bars = 2
        close_outside_bars = 2
        adx_min = 20.0

        bb_result = bbands(close_s, bb_period, bb_std)
        bb_upper, bb_middle, bb_lower = bb_result
        atr_vals = atr(high_s, low_s, close_s, bb_period)
        kc_upper = bb_middle + kc_multiplier * atr_vals
        kc_lower = bb_middle - kc_multiplier * atr_vals
        adx_vals = adx(high_s, low_s, close_s, 14)

        # Squeeze detection
        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        last_n = 60
        start_idx = max(0, len(df) - last_n)
        rows: List[Dict[str, Any]] = []

        release_count = 0
        outside_count = 0
        in_position = False

        # Pre-scan for state
        for i in range(max(bb_period, start_idx - 60), start_idx):
            if i < 0 or i >= len(df):
                continue
            sq = (
                bool(squeeze_on.iloc[i])
                if i < len(squeeze_on) and not pd.isna(squeeze_on.iloc[i])
                else True
            )
            if sq:
                release_count = 0
                outside_count = 0
            else:
                release_count += 1
                bbu = bb_upper.iloc[i] if not pd.isna(bb_upper.iloc[i]) else float("inf")
                bbl = bb_lower.iloc[i] if not pd.isna(bb_lower.iloc[i]) else float("-inf")
                if close[i] > bbu or close[i] < bbl:
                    outside_count += 1
                else:
                    outside_count = 0

        for i in range(start_idx, len(df)):
            c = close[i]
            sq = (
                bool(squeeze_on.iloc[i])
                if i < len(squeeze_on) and not pd.isna(squeeze_on.iloc[i])
                else True
            )
            adx_v = (
                float(adx_vals.iloc[i])
                if i < len(adx_vals) and not pd.isna(adx_vals.iloc[i])
                else 0.0
            )
            bbu = float(bb_upper.iloc[i]) if not pd.isna(bb_upper.iloc[i]) else c + 1
            bbl = float(bb_lower.iloc[i]) if not pd.isna(bb_lower.iloc[i]) else c - 1

            if sq:
                release_count = 0
                outside_count = 0
            else:
                release_count += 1
                if c > bbu or c < bbl:
                    outside_count += 1
                else:
                    outside_count = 0

            # Long-only strategy: show bullish bias context
            bbm = float(bb_middle.iloc[i]) if not pd.isna(bb_middle.iloc[i]) else c
            direction = "BULL" if c > bbm else "NEUT"

            signal = ""
            if sq:
                signal = "SQUEEZE"
            elif (
                release_count >= release_persist_bars
                and outside_count >= close_outside_bars
                and adx_v >= adx_min
            ):
                if not in_position:
                    signal = "LONG ENTRY"
                    in_position = True
                else:
                    signal = "HOLD"
            elif in_position:
                signal = "HOLD"

            ts = df.index[i]
            date_str = _format_timestamp(ts, timeframe, display_timezone)

            row: Dict[str, Any] = {
                "date": date_str,
                "squeeze_on": sq,
                "release_count": release_count,
                "outside_bb": outside_count,
                "adx": round(adx_v, 1),
                "direction": direction,
                "signal": signal,
            }
            rows.append(row)

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to compute SqueezePlay history: {e}")
        return []


def _compute_regime_flex_history(
    df: pd.DataFrame, timeframe: str = "1d", display_timezone: str = "US/Eastern"
) -> List[Dict[str, Any]]:
    """Compute RegimeFlex strategy signal history for a single DataFrame (last 60 bars).

    Uses the RegimeDetector to classify each bar's regime, then maps regime to
    target exposure using the strategy's YAML params (r0/r1/r3_gross_pct).
    Generates BUY/SELL/HOLD signals based on exposure changes.

    DRIFT RISK: This re-implements RegimeFlex exposure logic for display-only
    per-bar annotations. Keep in sync with:
    - src/domain/strategy/signals/regime_flex.py (signal conditions)
    - src/domain/strategy/playbook/regime_flex.py (full event-driven logic)
    """
    try:
        from src.domain.strategy.param_loader import get_strategy_params

        if len(df) < 130:
            return []

        required = ["high", "low", "close"]
        if not all(c in df.columns for c in required):
            return []

        # Load strategy params
        params = get_strategy_params("regime_flex")
        r0_pct = params.get("r0_gross_pct", 1.0)
        r1_pct = params.get("r1_gross_pct", 0.6)
        r3_pct = params.get("r3_gross_pct", 0.3)

        exposure_map: Dict[str, float] = {
            "R0": r0_pct,
            "R1": r1_pct,
            "R2": 0.0,
            "R3": r3_pct,
        }

        # Try to compute regime series via RegimeDetector
        regime_values: List[str] = []
        try:
            from src.domain.signals.indicators.regime.regime_detector import (
                RegimeDetectorIndicator,
            )

            detector = RegimeDetectorIndicator()
            result = detector.calculate(df[required].copy(), detector.default_params)
            if not result.empty and "regime" in result.columns:
                regime_values = [str(v) for v in result["regime"].values]
            else:
                return []
        except Exception:
            return []

        if len(regime_values) != len(df):
            return []

        last_n = 60
        start_idx = max(0, len(df) - last_n)
        rows: List[Dict[str, Any]] = []

        prev_exposure: Optional[float] = None

        for i in range(start_idx, len(df)):
            regime = regime_values[i]
            target_exposure = exposure_map.get(regime, 0.0)

            # Determine signal based on exposure change
            if prev_exposure is None:
                signal = "HOLD"
            elif target_exposure > prev_exposure:
                signal = "BUY"
            elif target_exposure < prev_exposure:
                signal = "SELL"
            else:
                signal = "HOLD"

            prev_exposure = target_exposure

            ts = df.index[i]
            date_str = _format_timestamp(ts, timeframe, display_timezone)

            row: Dict[str, Any] = {
                "date": date_str,
                "regime": regime,
                "target_exposure": round(target_exposure * 100, 1),
                "signal": signal,
            }
            rows.append(row)

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to compute RegimeFlex history: {e}")
        return []


def _compute_sector_pulse_history(
    symbol: str,
    df: pd.DataFrame,
    timeframe: str = "1d",
    display_timezone: str = "US/Eastern",
) -> List[Dict[str, Any]]:
    """Compute SectorPulse strategy signal history for a single DataFrame (last 60 bars).

    Computes a simple 20-day momentum score for the symbol and overlays the
    regime classification. This is a per-symbol approximation of the cross-sectional
    SectorPulse strategy (which normally ranks all sectors together).

    DRIFT RISK: This re-implements SectorPulse momentum logic for display-only
    per-bar annotations. Keep in sync with:
    - src/domain/strategy/signals/sector_pulse.py (signal conditions)
    - src/domain/strategy/playbook/sector_pulse.py (full event-driven logic)
    """
    try:
        if len(df) < 130:
            return []

        required = ["high", "low", "close"]
        if not all(c in df.columns for c in required):
            return []

        close = df["close"]
        momentum_period = 20

        # Compute momentum as 20-day return
        momentum = close.pct_change(momentum_period) * 100  # as percentage

        # Try to compute regime series via RegimeDetector
        regime_values: List[str] = []
        try:
            from src.domain.signals.indicators.regime.regime_detector import (
                RegimeDetectorIndicator,
            )

            detector = RegimeDetectorIndicator()
            result = detector.calculate(df[required].copy(), detector.default_params)
            if not result.empty and "regime" in result.columns:
                regime_values = [str(v) for v in result["regime"].values]
            else:
                # Fall back to empty regime
                regime_values = ["--"] * len(df)
        except Exception:
            regime_values = ["--"] * len(df)

        last_n = 60
        start_idx = max(0, len(df) - last_n)
        rows: List[Dict[str, Any]] = []

        for i in range(start_idx, len(df)):
            mom_val = float(momentum.iloc[i]) if not pd.isna(momentum.iloc[i]) else 0.0
            regime = regime_values[i] if i < len(regime_values) else "--"

            ts = df.index[i]
            date_str = _format_timestamp(ts, timeframe, display_timezone)

            row: Dict[str, Any] = {
                "date": date_str,
                "momentum_score": round(mom_val, 2),
                "regime": regime,
            }
            rows.append(row)

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to compute SectorPulse history: {e}")
        return []
