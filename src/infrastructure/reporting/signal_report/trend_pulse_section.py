"""
TrendPulse Historical State Section - Collapsible table for signal verification.

Renders a per-symbol, per-timeframe table showing the last N bars of TrendPulse state:
  Date | Swing | Entry | MACD Trend | ADX | Trend | Strength | Top | Conf(4f) | ATR Stop | CD | Exit

Row color coding:
  - BUY rows: green-10% background
  - SELL rows: red-10% background
  - Entry signal: green-15% background
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
        "ENTRY": "rgba(16, 185, 129, 0.15)",
        "TOP_DETECTED": "rgba(168, 85, 247, 0.10)",
        "TOP_ZONE": "rgba(245, 158, 11, 0.10)",
    }

    signal_colors = {
        "BUY": "#10b981",
        "SELL": "#ef4444",
        "NONE": muted,
    }

    top_colors = {
        "TOP_DETECTED": "#a855f7",
        "TOP_ZONE": "#f59e0b",
        "TOP_PENDING": "#facc15",
        "NONE": muted,
    }

    dm_state_colors = {
        "BULLISH": "#10b981",
        "IMPROVING": "#06b6d4",
        "DETERIORATING": "#f59e0b",
        "BEARISH": "#ef4444",
    }

    exit_colors = {
        "atr_stop": "#ef4444",
        "dm_regime": "#f59e0b",
        "zig_sell": "#a855f7",
        "top_detected": "#ec4899",
        "none": muted,
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
            entry = row.get("entry_signal", False)
            exit_sig = row.get("exit_signal", "none")
            dm_state = row.get("dm_state", "BEARISH")
            adx_val = row.get("adx", 0.0)
            adx_ok = row.get("adx_ok", False)
            conf_4f = row.get("confidence_4f", 0.0)
            atr_stop = row.get("atr_stop_level", 0.0)
            cooldown = row.get("cooldown_left", 0)

            # Row background priority: entry > swing > top
            if entry:
                row_bg = row_colors["ENTRY"]
            elif swing != "NONE":
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

            # Entry badge
            entry_html = (
                f'<span style="color:#10b981;font-weight:700;">\u2713</span>'
                if entry
                else f'<span style="color:{muted};">\u2014</span>'
            )

            # MACD Trend colored
            dm_color = dm_state_colors.get(dm_state, muted)
            dm_html = f'<span style="color:{dm_color};font-weight:500;">{dm_state}</span>'

            # ADX with chop zone highlight
            adx_bg = "rgba(239,68,68,0.15)" if not adx_ok else "transparent"
            adx_html = (
                f'<span style="background:{adx_bg};padding:1px 3px;border-radius:2px;'
                f'font-family:monospace;">{adx_val:.0f}</span>'
            )

            top_color = top_colors.get(top, muted)
            top_html = (
                f'<span style="color:{top_color};font-weight:600;">{top}</span>'
                if top != "NONE"
                else f'<span style="color:{muted};">\u2014</span>'
            )

            # 4-factor confidence bar
            conf_pct = int(conf_4f * 100)
            conf_color = "#10b981" if conf_pct >= 50 else "#f59e0b" if conf_pct >= 25 else muted
            conf_html = (
                f'<div style="display:flex;align-items:center;gap:4px;">'
                f'<div style="width:40px;height:6px;background:{border};border-radius:3px;'
                f'overflow:hidden;">'
                f'<div style="width:{conf_pct}%;height:100%;background:{conf_color};"></div>'
                f"</div>"
                f'<span style="font-size:10px;">{conf_pct}</span>'
                f"</div>"
            )

            # ATR stop level
            atr_html = (
                f'<span style="font-family:monospace;">${atr_stop:.2f}</span>'
                if atr_stop > 0
                else f'<span style="color:{muted};">\u2014</span>'
            )

            # Cooldown
            cd_color = "#10b981" if cooldown == 0 else "#ef4444"
            cd_html = f'<span style="color:{cd_color};font-weight:600;">{cooldown}</span>'

            # Exit badge
            exit_color = exit_colors.get(exit_sig, muted)
            exit_html = (
                f'<span style="color:{exit_color};font-weight:600;'
                f'background:rgba(255,255,255,0.05);padding:1px 4px;border-radius:3px;">'
                f"{exit_sig}</span>"
                if exit_sig != "none"
                else f'<span style="color:{muted};">\u2014</span>'
            )

            body_rows.append(f"""
                <tr style="background:{row_bg};border-bottom:1px solid {border};">
                    <td style="padding:4px 8px;font-size:11px;white-space:nowrap;">{row['date']}</td>
                    <td style="padding:4px 6px;font-size:11px;">{swing_html}</td>
                    <td style="padding:4px 6px;font-size:11px;text-align:center;">{entry_html}</td>
                    <td style="padding:4px 6px;font-size:11px;">{dm_html}</td>
                    <td style="padding:4px 6px;font-size:11px;text-align:right;">{adx_html}</td>
                    <td style="padding:4px 6px;text-align:right;font-family:monospace;font-size:11px;">
                        {row.get('trend_strength', 0):.2f}
                    </td>
                    <td style="padding:4px 6px;font-size:11px;">{top_html}</td>
                    <td style="padding:4px 6px;font-size:11px;">{conf_html}</td>
                    <td style="padding:4px 6px;font-size:11px;text-align:right;">{atr_html}</td>
                    <td style="padding:4px 6px;font-size:11px;text-align:center;">{cd_html}</td>
                    <td style="padding:4px 6px;font-size:11px;">{exit_html}</td>
                </tr>
            """)

        table_html = f"""
        <div class="trend-pulse-table" data-symbol="{symbol}" data-timeframe="{timeframe}"
             style="display:none;">
            <div style="overflow-x:auto;">
                <table style="width:100%;border-collapse:collapse;color:{text};font-size:12px;">
                    <thead>
                        <tr style="background:{header_bg};border-bottom:2px solid {border};">
                            <th style="padding:6px 8px;text-align:left;font-size:11px;color:{muted};cursor:help;" title="Bar timestamp">Date</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};cursor:help;" title="Causal ZIG/MA crossover signal.&#10;BUY: ZIG crosses above MA (bullish trend filter).&#10;SELL: ZIG crosses below MA.&#10;Cooldown: BUY suppressed for N bars after last BUY.">Swing</th>
                            <th style="padding:6px 6px;text-align:center;font-size:11px;color:{muted};cursor:help;" title="Composite entry signal. All conditions must be true:&#10;1. Swing = BUY&#10;2. Price > EMA99 (bullish trend)&#10;3. ADX >= 15 (no chop)&#10;4. Trend strength >= 0.30 (moderate+)&#10;5. MACD Trend = BULLISH or IMPROVING&#10;6. Cooldown = 0 (no recent exit)">Entry</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};cursor:help;" title="Dual MACD (55/89/34) structural trend state.&#10;Histogram = 2 &times; (EMA55 - EMA89 - Signal34).&#10;BULLISH: histogram > 0, slope > 0&#10;DETERIORATING: histogram > 0, slope < 0&#10;IMPROVING: histogram < 0, slope > 0&#10;BEARISH: histogram < 0, slope < 0">MACD Trend</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};cursor:help;" title="Average Directional Index (25-period).&#10;Measures trend strength regardless of direction.&#10;Red background when < 15 (chop zone, entry blocked).&#10;> 25 = trending, > 40 = strong trend.">ADX</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};cursor:help;" title="Normalized trend strength = ADX / 50.&#10;STRONG >= 0.60, MODERATE >= 0.30, WEAK < 0.30.&#10;Entry requires >= 0.30 (moderate).">Strength</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};cursor:help;" title="Williams %R top detection system.&#10;TOP_PENDING: W%R(13) > 70&#10;TOP_ZONE: W%R mid declining from >80, short near peak&#10;TOP_DETECTED: W%R long crosses above short + ADX declining&#10;Resets after W%R mid < 60 for 3 bars.">Top</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};cursor:help;" title="4-factor confidence score (0-100%).&#10;= 30% &times; ZIG strength (ADX/50)&#10;+ 25% &times; MACD health (BULL=1, IMPR=0.7, DET=0.3, BEAR=0)&#10;+ 25% &times; EMA alignment (5-EMA stack ordering)&#10;+ 20% &times; Vol quality (ADX filter &times; top penalty)">Conf(4f)</th>
                            <th style="padding:6px 6px;text-align:right;font-size:11px;color:{muted};cursor:help;" title="Informational trailing stop level.&#10;= Rolling max(close, 20 bars) - 3.5 &times; ATR(20).&#10;Not position-aware; shows per-bar level.&#10;Exit triggers when close < ATR stop.">ATR Stop</th>
                            <th style="padding:6px 6px;text-align:center;font-size:11px;color:{muted};cursor:help;" title="Cooldown bars remaining after an exit signal.&#10;Counts down from 5 to 0 after each exit.&#10;Green (0) = ready for new entry.&#10;Red (>0) = entry blocked.">CD</th>
                            <th style="padding:6px 6px;text-align:left;font-size:11px;color:{muted};cursor:help;" title="Exit signal reason (first match wins):&#10;atr_stop: close < trailing ATR stop level&#10;dm_regime: 3 consecutive MACD BEARISH bars&#10;zig_sell: ZIG/MA cross down&#10;top_detected: Williams %R top confirmed&#10;none: no exit condition">Exit</th>
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
