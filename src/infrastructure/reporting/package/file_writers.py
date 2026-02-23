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
    max_workers: int = 1,
    regime_series: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """
    Write individual JSON data files for each symbol/timeframe.

    Pre-computes all indicator history data once (CPU-bound), then writes
    files in parallel (I/O-bound). This avoids creating 435 duplicate
    indicator instances and recomputing 40+ indicators per file.

    Args:
        data: Dict mapping (symbol, timeframe) to DataFrame
        indicators: List of computed indicators
        rules: List of signal rules
        data_dir: Directory to write data files
        display_timezone: IANA timezone for display timestamps
        max_workers: ThreadPool workers (1 = serial, >1 = parallel)
        regime_series: Pre-computed regime series per key (skips recompute)

    Returns:
        List of data file keys (e.g., ["AAPL_1d", "SPY_1d"])
    """
    # Pre-warm strategy params cache (file I/O, not thread-safe)
    from src.domain.strategy.param_loader import get_strategy_params

    for name in ["trend_pulse", "regime_flex", "sector_pulse", "rsi_mean_reversion"]:
        try:
            get_strategy_params(name)
        except Exception:
            pass

    # Phase 1: Pre-compute all history data once (CPU-bound, sequential)
    history_cache = _precompute_all_history(data, display_timezone, regime_series)

    # Phase 2: Write files (I/O-bound, optionally parallel)
    if max_workers <= 1:
        return _write_data_files_sequential(data, rules, data_dir, display_timezone, history_cache)
    return _write_data_files_parallel(
        data, rules, data_dir, display_timezone, max_workers, history_cache
    )


# Type alias for pre-computed history data per key
HistoryCache = Dict[str, Dict[str, List[Dict[str, Any]]]]


def _precompute_all_history(
    data: Dict[Tuple[str, str], pd.DataFrame],
    display_timezone: str,
    regime_series: Optional[Dict[str, List[str]]] = None,
) -> HistoryCache:
    """Pre-compute indicator history for all symbol/timeframe pairs.

    Creates indicator instances once and reuses them across all DataFrames,
    avoiding ~435 redundant constructor calls per indicator type.

    Args:
        regime_series: Pre-computed regime series from processor (Fix B).
            When available, skips the expensive _compute_regime_series() call.

    Returns:
        Dict mapping "SYMBOL_TF" keys to their pre-computed history dicts.
    """
    # Create indicator instances once (constructors load YAML params etc.)
    dual_macd = _create_dual_macd_indicator()
    trend_pulse = _create_trend_pulse_indicator()
    regime_detector = _create_regime_detector()

    cache: HistoryCache = {}
    for (symbol, timeframe), df in data.items():
        key = f"{symbol}_{timeframe}"

        # Use pre-computed regime if available (Fix B), skip expensive recompute
        if regime_series and key in regime_series:
            regime_values: Optional[List[str]] = regime_series[key]
        else:
            regime_values = _compute_regime_series(regime_detector, df)

        cache[key] = {
            "dual_macd": _compute_dual_macd_with_instance(
                dual_macd, df, timeframe, display_timezone
            ),
            "trend_pulse": _compute_trend_pulse_with_instance(
                trend_pulse, df, timeframe, display_timezone
            ),
            "regime_flex": _build_regime_flex_history(
                regime_values, df, timeframe, display_timezone
            ),
            "sector_pulse": _build_sector_pulse_history(
                regime_values, symbol, df, timeframe, display_timezone
            ),
        }
    return cache


def _write_one_data_file(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    rules: List["SignalRule"],
    data_dir: Path,
    display_timezone: str,
    history_cache: Optional[HistoryCache] = None,
) -> str:
    """Write a single symbol/timeframe JSON data file. Thread-safe."""
    from .signal_detection import detect_historical_signals

    key = f"{symbol}_{timeframe}"

    chart_data = df_to_chart_data(df)
    signals = detect_historical_signals(df, rules, symbol, timeframe)

    # Use pre-computed history if available, otherwise compute on-the-fly
    if history_cache and key in history_cache:
        precomputed = history_cache[key]
        dual_macd_history = precomputed["dual_macd"]
        trend_pulse_history = precomputed["trend_pulse"]
        regime_flex_history = precomputed["regime_flex"]
        sector_pulse_history = precomputed["sector_pulse"]
    else:
        dual_macd_history = _compute_dual_macd_history_for_key(df, timeframe, display_timezone)
        trend_pulse_history = _compute_trend_pulse_history_for_key(df, timeframe, display_timezone)
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
        "regime_flex_history": regime_flex_history,
        "sector_pulse_history": sector_pulse_history,
    }

    file_path = data_dir / f"{key}.json"
    file_path.write_text(
        json.dumps(file_data, default=str),
        encoding="utf-8",
    )
    return key


def _write_data_files_sequential(
    data: Dict[Tuple[str, str], pd.DataFrame],
    rules: List["SignalRule"],
    data_dir: Path,
    display_timezone: str,
    history_cache: Optional[HistoryCache] = None,
) -> List[str]:
    """Sequential write path (original behavior)."""
    files_written = []
    for (symbol, timeframe), df in data.items():
        key = _write_one_data_file(
            symbol, timeframe, df, rules, data_dir, display_timezone, history_cache
        )
        files_written.append(key)
    return files_written


def _write_data_files_parallel(
    data: Dict[Tuple[str, str], pd.DataFrame],
    rules: List["SignalRule"],
    data_dir: Path,
    display_timezone: str,
    max_workers: int,
    history_cache: Optional[HistoryCache] = None,
) -> List[str]:
    """Parallel write path using ThreadPoolExecutor.

    History data is pre-computed before this function is called, so threads
    only do signal detection + JSON serialization + file I/O (all thread-safe).

    Raises RuntimeError if any file write fails (fail-fast for CI).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    files_written: List[str] = []
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _write_one_data_file,
                sym,
                tf,
                df,
                rules,
                data_dir,
                display_timezone,
                history_cache,
            ): (sym, tf)
            for (sym, tf), df in data.items()
        }
        for future in as_completed(futures):
            sym, tf = futures[future]
            try:
                files_written.append(future.result())
            except Exception as e:
                logger.error(f"Failed to write data file {sym}/{tf}: {e}")
                errors.append(f"{sym}/{tf}: {e}")

    if errors:
        raise RuntimeError(f"Failed to write {len(errors)} data file(s): {'; '.join(errors)}")

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
    summary_json = json.dumps(summary, default=str)
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
        json.dumps(manifest.to_dict()),
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
        json.dumps(snapshot, default=str),
        encoding="utf-8",
    )


# -------------------------------------------------------------------------
# Indicator instance factories (create once, reuse across all files)
# -------------------------------------------------------------------------


def _create_dual_macd_indicator() -> Any:
    """Create a DualMACDIndicator instance, or None if import fails."""
    try:
        from src.domain.signals.indicators.momentum.dual_macd import DualMACDIndicator

        return DualMACDIndicator()
    except Exception as e:
        logger.warning(f"Failed to create DualMACDIndicator: {e}")
        return None


def _create_trend_pulse_indicator() -> Any:
    """Create a TrendPulseIndicator instance, or None if import fails."""
    try:
        from src.domain.signals.indicators.trend.trend_pulse import TrendPulseIndicator

        return TrendPulseIndicator()
    except Exception as e:
        logger.warning(f"Failed to create TrendPulseIndicator: {e}")
        return None


def _create_regime_detector() -> Any:
    """Create a RegimeDetectorIndicator instance, or None if import fails."""
    try:
        from src.domain.signals.indicators.regime.regime_detector import (
            RegimeDetectorIndicator,
        )

        return RegimeDetectorIndicator()
    except Exception as e:
        logger.warning(f"Failed to create RegimeDetectorIndicator: {e}")
        return None


# -------------------------------------------------------------------------
# Instance-based compute functions (reuse pre-created indicator instances)
# -------------------------------------------------------------------------


def _compute_dual_macd_with_instance(
    indicator: Any,
    df: pd.DataFrame,
    timeframe: str,
    display_timezone: str,
) -> List[Dict[str, Any]]:
    """Compute DualMACD history reusing a pre-created indicator instance."""
    if indicator is None:
        return []
    try:
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
            date_str = _format_timestamp(ts, timeframe, display_timezone)
            rows.append({"date": date_str, **state})

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to compute DualMACD history: {e}")
        return []


def _compute_trend_pulse_with_instance(
    indicator: Any,
    df: pd.DataFrame,
    timeframe: str,
    display_timezone: str,
) -> List[Dict[str, Any]]:
    """Compute TrendPulse history reusing a pre-created indicator instance."""
    if indicator is None:
        return []
    try:
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
            date_str = _format_timestamp(ts, timeframe, display_timezone)
            rows.append({"date": date_str, **state})

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to compute TrendPulse history: {e}")
        return []


def _compute_regime_series(
    detector: Any,
    df: pd.DataFrame,
) -> Optional[List[str]]:
    """Compute regime classification once for all bars.

    Returns list of regime strings (e.g., ["R0", "R1", ...]) or None if
    computation fails or data is insufficient.
    """
    if detector is None or len(df) < 130:
        return None

    required = ["high", "low", "close"]
    if not all(c in df.columns for c in required):
        return None

    try:
        result = detector.calculate(df[required].copy(), detector.default_params)
        if not result.empty and "regime" in result.columns:
            values = [str(v) for v in result["regime"].values]
            if len(values) == len(df):
                return values
    except Exception as e:
        logger.warning(f"Failed to compute regime series: {e}")

    return None


def _build_regime_flex_history(
    regime_values: Optional[List[str]],
    df: pd.DataFrame,
    timeframe: str,
    display_timezone: str,
) -> List[Dict[str, Any]]:
    """Build RegimeFlex history from pre-computed regime values."""
    if regime_values is None:
        return []

    try:
        from src.domain.strategy.param_loader import get_strategy_params

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

        last_n = 60
        start_idx = max(0, len(df) - last_n)
        rows: List[Dict[str, Any]] = []
        prev_exposure: Optional[float] = None

        for i in range(start_idx, len(df)):
            regime = regime_values[i]
            target_exposure = exposure_map.get(regime, 0.0)

            if prev_exposure is None:
                sig = "HOLD"
            elif target_exposure > prev_exposure:
                sig = "BUY"
            elif target_exposure < prev_exposure:
                sig = "SELL"
            else:
                sig = "HOLD"

            prev_exposure = target_exposure
            ts = df.index[i]
            date_str = _format_timestamp(ts, timeframe, display_timezone)
            rows.append(
                {
                    "date": date_str,
                    "regime": regime,
                    "target_exposure": round(target_exposure * 100, 1),
                    "signal": sig,
                }
            )

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to build RegimeFlex history: {e}")
        return []


def _build_sector_pulse_history(
    regime_values: Optional[List[str]],
    symbol: str,
    df: pd.DataFrame,
    timeframe: str,
    display_timezone: str,
) -> List[Dict[str, Any]]:
    """Build SectorPulse history from pre-computed regime values."""
    try:
        if len(df) < 130:
            return []

        required = ["high", "low", "close"]
        if not all(c in df.columns for c in required):
            return []

        close = df["close"]
        momentum = close.pct_change(20) * 100

        # Use pre-computed regime or fall back to placeholder
        effective_regime = regime_values if regime_values is not None else ["--"] * len(df)

        last_n = 60
        start_idx = max(0, len(df) - last_n)
        rows: List[Dict[str, Any]] = []

        for i in range(start_idx, len(df)):
            mom_val = float(momentum.iloc[i]) if not pd.isna(momentum.iloc[i]) else 0.0
            regime = effective_regime[i] if i < len(effective_regime) else "--"
            ts = df.index[i]
            date_str = _format_timestamp(ts, timeframe, display_timezone)
            rows.append(
                {
                    "date": date_str,
                    "momentum_score": round(mom_val, 2),
                    "regime": regime,
                }
            )

        rows.reverse()
        return rows
    except Exception as e:
        logger.warning(f"Failed to build SectorPulse history: {e}")
        return []


# -------------------------------------------------------------------------
# Legacy per-call compute functions (used when history_cache is not provided)
# -------------------------------------------------------------------------


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
