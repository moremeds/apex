"""
TrendPulse Historical State Section - Collapsible table for signal verification.

Renders a per-symbol, per-timeframe table showing the last N bars of TrendPulse state:
  Date | Swing | Trend | Strength | Label | Top | EMA Align | Score | Conf

Row color coding:
  - BUY rows: green-10% background
  - SELL rows: red-10% background
  - TOP_DETECTED: purple-10% background
  - TOP_ZONE: amber-10% background
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from src.domain.signals.indicators.trend.trend_pulse import TrendPulseIndicator


def compute_trend_pulse_history(
    data: Dict[Tuple[str, str], pd.DataFrame],
    last_n_bars: int = 60,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute TrendPulse state for each (symbol, timeframe) pair.

    Args:
        data: Dict mapping (symbol, timeframe) to OHLCV+indicator DataFrame
        last_n_bars: Number of most recent bars to include

    Returns:
        Dict mapping "{symbol}_{timeframe}" to list of state dicts (newest first)
    """
    indicator = TrendPulseIndicator()
    params = indicator.default_params
    history: Dict[str, List[Dict[str, Any]]] = {}

    for (symbol, timeframe), df in data.items():
        # Lower threshold than warmup_periods (500) so the section renders
        # with limited history. EMA-453 needs ~453 bars to converge.
        ema_periods = params.get("ema_periods", (14, 25, 99, 144, 453))
        min_bars = max(ema_periods[-1] + 50, 200)
        if len(df) < min_bars:
            continue

        required = ["high", "low", "close"]
        if not all(c in df.columns for c in required):
            continue

        hlc_df = df[required].copy()
        result = indicator.calculate(hlc_df, params)

        if result.empty:
            continue

        start_idx = max(0, len(result) - last_n_bars)
        rows: List[Dict[str, Any]] = []

        for i in range(start_idx, len(result)):
            current = result.iloc[i]
            previous = result.iloc[i - 1] if i > 0 else None
            state = indicator._get_state(current, previous, params)

            ts = result.index[i]
            date_str = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)
            rows.append({"date": date_str, **state})

        rows.reverse()
        history[f"{symbol}_{timeframe}"] = rows

    return history


def render_trend_pulse_history_html(
    history: Dict[str, List[Dict[str, Any]]],
    theme: str = "dark",
) -> str:
    """Render TrendPulse historical state as a collapsible HTML section."""
    if not history:
        return ""

    bg = "#0f172a" if theme == "dark" else "#ffffff"
    text = "#e2e8f0" if theme == "dark" else "#1e293b"
    muted = "#94a3b8" if theme == "dark" else "#64748b"
    border = "#334155" if theme == "dark" else "#e2e8f0"
    header_bg = "#1e293b" if theme == "dark" else "#f1f5f9"

    row_colors = {
        "BUY": "rgba(16, 185, 129, 0.10)",
        "SELL": "rgba(239, 68, 68, 0.10)",
        "TOP_DETECTED": "rgba(168, 85, 247, 0.10)",
        "TOP_ZONE": "rgba(245, 158, 11, 0.10)",
    }

    signal_colors = {
        "BUY": "#10b981",
        "SELL": "#ef4444",
        "NONE": muted,
    }

    trend_colors = {
        "BULLISH": "#10b981",
        "BEARISH": "#ef4444",
        "NEUTRAL": muted,
    }

    top_colors = {
        "TOP_DETECTED": "#a855f7",
        "TOP_ZONE": "#f59e0b",
        "TOP_PENDING": "#facc15",
        "NONE": muted,
    }

    tables_html = []
    for key, rows in sorted(history.items()):
        if not rows:
            continue

        parts = key.rsplit("_", 1)
        symbol = parts[0] if len(parts) == 2 else key
        timeframe = parts[1] if len(parts) == 2 else ""

        body_rows = []
        for row in rows:
            swing = row.get("swing_signal", "NONE")
            top = row.get("top_warning", "NONE")
            score = row.get("score", 0.0)
            confidence = row.get("confidence", 0.0)

            # Row background: swing signal or top warning
            if swing != "NONE":
                row_bg = row_colors.get(swing, "transparent")
            elif top in ("TOP_DETECTED", "TOP_ZONE"):
                row_bg = row_colors.get(top, "transparent")
            else:
                row_bg = "transparent"

            sig_color = signal_colors.get(swing, muted)
            swing_html = (
                f'<span style="color:{sig_color};font-weight:600;">{swing}</span>'
                if swing != "NONE"
                else f'<span style="color:{muted};">\u2014</span>'
            )

            trend_color = trend_colors.get(row.get("trend_filter", "NEUTRAL"), muted)
            top_color = top_colors.get(top, muted)
            top_html = (
                f'<span style="color:{top_color};font-weight:600;">{top}</span>'
                if top != "NONE"
                else f'<span style="color:{muted};">\u2014</span>'
            )

            # Score heatmap
            if score >= 80:
                score_color = "#10b981"
            elif score >= 50:
                score_color = "#f59e0b"
            else:
                score_color = muted

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
                    <td style="padding:4px 6px;font-size:11px;">{swing_html}</td>
                    <td style="padding:4px 6px;color:{trend_color};font-weight:500;font-size:11px;">
                        {row.get('trend_filter', 'NEUTRAL')}
                    </td>
                    <td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:11px;">
                        {row.get('trend_strength', 0):.2f}
                    </td>
                    <td style="padding:4px 6px;font-size:11px;color:{muted};">
                        {row.get('trend_strength_label', 'WEAK')}
                    </td>
                    <td style="padding:4px 6px;font-size:11px;">{top_html}</td>
                    <td style="padding:4px 6px;font-size:11px;color:{muted};">
                        {row.get('ema_alignment', 'MIXED')}
                    </td>
                    <td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:11px;color:{score_color};font-weight:600;">
                        {score:.0f}
                    </td>
                    <td style="padding:4px 6px;font-size:11px;">{conf_html}</td>
                </tr>
            """)

        table_html = f"""
        <div class="trend-pulse-table" data-symbol="{symbol}" data-timeframe="{timeframe}"
             style="display:none;">
            <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;color:{text};font-size:12px;">
                    <thead>
                        <tr style="background:{header_bg};border-bottom:2px solid {border};">
                            <th style="padding:6px 8px;text-align:left;font-size:11px;color:{muted};">Date</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">Swing</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">Trend</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};">Strength</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">Label</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">Top</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};">EMA Align</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};">Score</th>
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
    <div class="trend-pulse-history-section">
        <h2 class="section-header" onclick="toggleSection('trend-pulse-history-content')">
            <span class="toggle-icon">\u25b6</span> TrendPulse Historical State
        </h2>
        <div id="trend-pulse-history-content" class="section-content collapsed">
            <div style="
                background: {bg};
                border: 1px solid {border};
                border-radius: 8px;
                padding: 16px;
            ">
                <div style="font-size:12px;color:{muted};margin-bottom:12px;">
                    Last 60 bars \u00b7 Newest first \u00b7
                    <span style="display:inline-block;width:10px;height:10px;background:rgba(16,185,129,0.10);border:1px solid #10b981;border-radius:2px;vertical-align:middle;"></span> BUY
                    <span style="display:inline-block;width:10px;height:10px;background:rgba(239,68,68,0.10);border:1px solid #ef4444;border-radius:2px;vertical-align:middle;margin-left:8px;"></span> SELL
                    <span style="display:inline-block;width:10px;height:10px;background:rgba(168,85,247,0.10);border:1px solid #a855f7;border-radius:2px;vertical-align:middle;margin-left:8px;"></span> TOP
                </div>
                <div id="trend-pulse-tables-container">
                    {''.join(tables_html)}
                </div>
            </div>
        </div>
    </div>
    """
