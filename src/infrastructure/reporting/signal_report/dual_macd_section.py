"""
DualMACD Historical State Section - Collapsible table for signal verification.

Renders a per-symbol, per-timeframe table showing the last N bars of DualMACD state:
  Date | H_slow | H_fast | ΔH_slow | ΔH_fast | Trend State | Tactical | Mom.Bal | Conf

Row color coding:
  - DIP_BUY rows: green-10% background
  - RALLY_SELL rows: red-10% background
  - DETERIORATING trend: amber cell
  - IMPROVING trend: teal cell
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.domain.signals.indicators.momentum.dual_macd import DualMACDIndicator


def compute_dual_macd_history(
    data: Dict[Tuple[str, str], pd.DataFrame],
    last_n_bars: int = 60,
    display_timezone: str = "US/Eastern",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute DualMACD state for each (symbol, timeframe) pair.

    Args:
        data: Dict mapping (symbol, timeframe) to OHLCV+indicator DataFrame
        last_n_bars: Number of most recent bars to include

    Returns:
        Dict mapping "{symbol}_{timeframe}" to list of state dicts (newest first)
    """
    indicator = DualMACDIndicator()
    params = indicator.default_params
    history: Dict[str, List[Dict[str, Any]]] = {}

    for (symbol, timeframe), df in data.items():
        if len(df) < indicator.warmup_periods:
            continue

        # Calculate indicator on close prices
        close_df = df[["close"]].copy()
        result = indicator.calculate(close_df, params)

        if result.empty:
            continue

        # Extract last N bars of state
        start_idx = max(0, len(result) - last_n_bars)
        rows: List[Dict[str, Any]] = []

        for i in range(start_idx, len(result)):
            current = result.iloc[i]
            previous = result.iloc[i - 1] if i > 0 else None
            state = indicator._get_state(current, previous, params)

            # Add timestamp with timezone conversion for intraday
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

            rows.append(
                {
                    "date": date_str,
                    **state,
                }
            )

        # Reverse so newest is first
        rows.reverse()

        key = f"{symbol}_{timeframe}"
        history[key] = rows

    return history


def render_dual_macd_history_html(
    history: Dict[str, List[Dict[str, Any]]],
    theme: str = "dark",
) -> str:
    """
    Render DualMACD historical state as a collapsible HTML section.

    Each (symbol, timeframe) pair gets a sub-section filtered by JavaScript
    based on the active symbol/timeframe selection.

    Args:
        history: Dict from compute_dual_macd_history()
        theme: Color theme

    Returns:
        HTML string for the DualMACD history section
    """
    if not history:
        return ""

    # Theme colors
    bg = "#0f172a" if theme == "dark" else "#ffffff"
    text = "#e2e8f0" if theme == "dark" else "#1e293b"
    muted = "#94a3b8" if theme == "dark" else "#64748b"
    border = "#334155" if theme == "dark" else "#e2e8f0"
    header_bg = "#1e293b" if theme == "dark" else "#f1f5f9"

    # Row background colors for tactical signals
    row_colors = {
        "DIP_BUY": "rgba(16, 185, 129, 0.10)",
        "RALLY_SELL": "rgba(239, 68, 68, 0.10)",
        "NONE": "transparent",
    }

    # Trend state cell colors
    trend_colors = {
        "BULLISH": "#10b981",
        "BEARISH": "#ef4444",
        "DETERIORATING": "#f59e0b",
        "IMPROVING": "#06b6d4",
    }

    # Build tables for each symbol_timeframe
    tables_html = []
    for key, rows in sorted(history.items()):
        if not rows:
            continue

        # Parse symbol_timeframe
        parts = key.rsplit("_", 1)
        symbol = parts[0] if len(parts) == 2 else key
        timeframe = parts[1] if len(parts) == 2 else ""

        body_rows = []
        for row in rows:
            tactical = row.get("tactical_signal", "NONE")
            trend = row.get("trend_state", "BEARISH")
            row_bg = row_colors.get(tactical, "transparent")
            trend_color = trend_colors.get(trend, muted)
            confidence = row.get("confidence", 0.0)

            # Format tactical signal with color
            if tactical == "DIP_BUY":
                tactical_html = f'<span style="color: #10b981; font-weight: 600;">DIP_BUY</span>'
            elif tactical == "RALLY_SELL":
                tactical_html = f'<span style="color: #ef4444; font-weight: 600;">RALLY_SELL</span>'
            else:
                tactical_html = f'<span style="color: {muted};">—</span>'

            # Confidence bar
            conf_pct = int(confidence * 100)
            conf_color = "#10b981" if conf_pct >= 50 else "#f59e0b" if conf_pct >= 25 else muted
            conf_html = (
                f'<div style="display:flex;align-items:center;gap:4px;">'
                f'<div style="width:40px;height:6px;background:{border};border-radius:3px;overflow:hidden;">'
                f'<div style="width:{conf_pct}%;height:100%;background:{conf_color};"></div>'
                f"</div>"
                f'<span style="font-size:10px;">{conf_pct}</span>'
                f"</div>"
            )

            body_rows.append(f"""
                <tr style="background:{row_bg};border-bottom:1px solid {border};">
                    <td style="padding:4px 8px;font-size:11px;white-space:nowrap;">{row['date']}</td>
                    <td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:11px;">
                        {row['slow_histogram']:+.3f}
                    </td>
                    <td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:11px;">
                        {row['fast_histogram']:+.3f}
                    </td>
                    <td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:11px;">
                        {row['slow_hist_delta']:+.3f}
                    </td>
                    <td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:11px;">
                        {row['fast_hist_delta']:+.3f}
                    </td>
                    <td style="padding:4px 6px;color:{trend_color};font-weight:500;font-size:11px;">
                        {trend}
                    </td>
                    <td style="padding:4px 6px;font-size:11px;">{tactical_html}</td>
                    <td style="padding:4px 6px;font-size:11px;color:{muted};">
                        {row.get('momentum_balance', '—')}
                    </td>
                    <td style="padding:4px 6px;font-size:11px;">{conf_html}</td>
                </tr>
            """)

        table_html = f"""
        <div class="dual-macd-table" data-symbol="{symbol}" data-timeframe="{timeframe}"
             style="display:none;">
            <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;color:{text};font-size:12px;">
                    <thead>
                        <tr style="background:{header_bg};border-bottom:2px solid {border};">
                            <th style="padding:6px 8px;text-align:left;font-size:11px;color:{muted};">Date</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};">H_slow</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};">H_fast</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};">\u0394H_slow</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};">\u0394H_fast</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">Trend</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">Tactical</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">Mom.Bal</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">Conf</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(body_rows)}
                    </tbody>
                </table>
            </div>
        </div>
        """
        tables_html.append(table_html)

    return f"""
    <div class="dual-macd-history-section">
        <h2 class="section-header" onclick="toggleSection('dual-macd-history-content')">
            <span class="toggle-icon">\u25b6</span> DualMACD Historical State
        </h2>
        <div id="dual-macd-history-content" class="section-content collapsed">
            <div style="
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 16px;
            ">
                <div style="font-size:12px;color:{muted};margin-bottom:12px;">
                    Last 60 bars \u00b7 Newest first \u00b7
                    <span style="display:inline-block;width:10px;height:10px;background:rgba(16,185,129,0.10);border:1px solid #10b981;border-radius:2px;vertical-align:middle;"></span> DIP_BUY
                    <span style="display:inline-block;width:10px;height:10px;background:rgba(239,68,68,0.10);border:1px solid #ef4444;border-radius:2px;vertical-align:middle;margin-left:8px;"></span> RALLY_SELL
                </div>
                <div id="dual-macd-tables-container">
                    {''.join(tables_html)}
                </div>
            </div>
        </div>
    </div>
    """
