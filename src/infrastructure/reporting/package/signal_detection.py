"""
Signal Detection - Historical signal detection from indicator data.

This module provides functions for detecting historical signals from
DataFrame indicator columns, including:
- Cross up/down signals (line_a crosses line_b)
- Threshold cross signals (value crosses threshold)
- MACD-specific signal/zero line crosses
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import pandas as pd

if TYPE_CHECKING:
    from src.domain.signals.models import SignalRule


def _detect_cross_up_signals(
    df: pd.DataFrame,
    rule: "SignalRule",
    ind_cols: List[str],
    timestamps: List,
    symbol: str,
) -> List[Dict[str, Any]]:
    """Detect CROSS_UP signals where line_a crosses above line_b."""
    signals: List[Dict[str, Any]] = []
    cond = rule.condition_config
    line_a = cond.get("line_a", "")
    line_b = cond.get("line_b", "")
    col_a = next((c for c in ind_cols if line_a in c.lower()), None)
    col_b = next((c for c in ind_cols if line_b in c.lower()), None)

    if not (col_a and col_b):
        return signals

    a_vals = df[col_a].values
    b_vals = df[col_b].values
    for i in range(1, len(df)):
        if not all(pd.notna(v) for v in [a_vals[i], b_vals[i], a_vals[i - 1], b_vals[i - 1]]):
            continue
        if a_vals[i - 1] <= b_vals[i - 1] and a_vals[i] > b_vals[i]:
            signals.append(
                {
                    "timestamp": timestamps[i],
                    "rule": rule.name,
                    "direction": rule.direction.value,
                    "indicator": rule.indicator,
                    "message": rule.message_template.format(symbol=symbol),
                    "value": float(a_vals[i]),
                }
            )
    return signals


def _detect_cross_down_signals(
    df: pd.DataFrame,
    rule: "SignalRule",
    ind_cols: List[str],
    timestamps: List,
    symbol: str,
) -> List[Dict[str, Any]]:
    """Detect CROSS_DOWN signals where line_a crosses below line_b."""
    signals: List[Dict[str, Any]] = []
    cond = rule.condition_config
    line_a = cond.get("line_a", "")
    line_b = cond.get("line_b", "")
    col_a = next((c for c in ind_cols if line_a in c.lower()), None)
    col_b = next((c for c in ind_cols if line_b in c.lower()), None)

    if not (col_a and col_b):
        return signals

    a_vals = df[col_a].values
    b_vals = df[col_b].values
    for i in range(1, len(df)):
        if not all(pd.notna(v) for v in [a_vals[i], b_vals[i], a_vals[i - 1], b_vals[i - 1]]):
            continue
        if a_vals[i - 1] >= b_vals[i - 1] and a_vals[i] < b_vals[i]:
            signals.append(
                {
                    "timestamp": timestamps[i],
                    "rule": rule.name,
                    "direction": rule.direction.value,
                    "indicator": rule.indicator,
                    "message": rule.message_template.format(symbol=symbol),
                    "value": float(a_vals[i]),
                }
            )
    return signals


def _detect_threshold_cross_up_signals(
    df: pd.DataFrame,
    rule: "SignalRule",
    ind_cols: List[str],
    timestamps: List,
    symbol: str,
) -> List[Dict[str, Any]]:
    """Detect THRESHOLD_CROSS_UP signals where value crosses above threshold."""
    signals: List[Dict[str, Any]] = []
    cond = rule.condition_config
    field = cond.get("field", "value")
    threshold = cond.get("threshold")
    col = next((c for c in ind_cols if field in c.lower()), None)

    if not (col and threshold is not None):
        return signals

    vals = df[col].values
    for i in range(1, len(df)):
        if pd.notna(vals[i]) and pd.notna(vals[i - 1]) and vals[i - 1] <= threshold < vals[i]:
            signals.append(
                {
                    "timestamp": timestamps[i],
                    "rule": rule.name,
                    "direction": rule.direction.value,
                    "indicator": rule.indicator,
                    "message": rule.message_template.format(
                        symbol=symbol, value=vals[i], threshold=threshold
                    ),
                    "value": float(vals[i]),
                    "threshold": threshold,
                }
            )
    return signals


def _detect_threshold_cross_down_signals(
    df: pd.DataFrame,
    rule: "SignalRule",
    ind_cols: List[str],
    timestamps: List,
    symbol: str,
) -> List[Dict[str, Any]]:
    """Detect THRESHOLD_CROSS_DOWN signals where value crosses below threshold."""
    signals: List[Dict[str, Any]] = []
    cond = rule.condition_config
    field = cond.get("field", "value")
    threshold = cond.get("threshold")
    col = next((c for c in ind_cols if field in c.lower()), None)

    if not (col and threshold is not None):
        return signals

    vals = df[col].values
    for i in range(1, len(df)):
        if pd.notna(vals[i]) and pd.notna(vals[i - 1]) and vals[i - 1] >= threshold > vals[i]:
            signals.append(
                {
                    "timestamp": timestamps[i],
                    "rule": rule.name,
                    "direction": rule.direction.value,
                    "indicator": rule.indicator,
                    "message": rule.message_template.format(
                        symbol=symbol, value=vals[i], threshold=threshold
                    ),
                    "value": float(vals[i]),
                    "threshold": threshold,
                }
            )
    return signals


def _detect_state_change_signals(
    df: pd.DataFrame,
    rule: "SignalRule",
    ind_cols: List[str],
    timestamps: List,
    symbol: str,
) -> List[Dict[str, Any]]:
    """Detect STATE_CHANGE signals where field transitions between specified values."""
    signals: List[Dict[str, Any]] = []
    cond = rule.condition_config
    field = cond.get("field", "")
    from_values = cond.get("from", [])
    to_values = cond.get("to", [])

    # Find the column matching the field
    col = next((c for c in ind_cols if field in c.lower()), None)

    # Also try exact match with prefix (e.g., cp_pattern for chart_patterns)
    if col is None:
        prefix = rule.indicator.lower()
        # Try common prefixes
        for p in [f"{prefix}_{field}", f"cp_{field}", field]:
            col = next((c for c in df.columns if c.lower() == p.lower()), None)
            if col:
                break

    if col is None:
        return signals

    vals = df[col].values
    for i in range(1, len(df)):
        prev_val = vals[i - 1]
        curr_val = vals[i]

        # Handle None/NaN comparisons
        prev_in_from = (
            prev_val in from_values
            or (pd.isna(prev_val) and None in from_values)
            or (prev_val is None and None in from_values)
        )
        curr_in_to = curr_val in to_values

        if prev_in_from and curr_in_to:
            signals.append(
                {
                    "timestamp": timestamps[i],
                    "rule": rule.name,
                    "direction": rule.direction.value,
                    "indicator": rule.indicator,
                    "message": rule.message_template.format(symbol=symbol),
                    "value": str(curr_val) if curr_val is not None else None,
                    "pattern": str(curr_val) if curr_val is not None else None,
                }
            )
    return signals


def _detect_macd_crosses(
    df: pd.DataFrame,
    rule: "SignalRule",
    timestamps: List,
    symbol: str,
    existing_signals: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect MACD signal/zero line crosses."""
    signals: List[Dict[str, Any]] = []
    macd_col = next((c for c in df.columns if "macd_macd" in c.lower()), None)
    signal_col = next((c for c in df.columns if "macd_signal" in c.lower()), None)

    if not (macd_col and signal_col) or "cross" in rule.name.lower():
        return signals

    macd_vals = df[macd_col].values
    signal_vals = df[signal_col].values

    for i in range(1, len(df)):
        vals_to_check = [macd_vals[i], macd_vals[i - 1], signal_vals[i], signal_vals[i - 1]]
        if not all(pd.notna(v) for v in vals_to_check):
            continue

        # Check for duplicate at this timestamp
        has_macd_signal = any(
            s["timestamp"] == timestamps[i] and "macd" in s["rule"].lower()
            for s in existing_signals + signals
        )
        if has_macd_signal:
            continue

        # Bullish cross
        if macd_vals[i - 1] <= signal_vals[i - 1] and macd_vals[i] > signal_vals[i]:
            signals.append(
                {
                    "timestamp": timestamps[i],
                    "rule": "macd_bullish_cross",
                    "direction": "buy",
                    "indicator": "macd",
                    "message": f"{symbol} MACD crossed above signal line",
                    "value": float(macd_vals[i]),
                }
            )
        # Bearish cross
        elif macd_vals[i - 1] >= signal_vals[i - 1] and macd_vals[i] < signal_vals[i]:
            signals.append(
                {
                    "timestamp": timestamps[i],
                    "rule": "macd_bearish_cross",
                    "direction": "sell",
                    "indicator": "macd",
                    "message": f"{symbol} MACD crossed below signal line",
                    "value": float(macd_vals[i]),
                }
            )
    return signals


def detect_historical_signals(
    df: pd.DataFrame,
    rules: List["SignalRule"],
    symbol: str,
    timeframe: str,
    calculate_outcomes: bool = True,
    outcome_forward_bars: int = 10,
) -> List[Dict[str, Any]]:
    """
    Detect where signal rules would have triggered in historical data.

    Scans the DataFrame for cross events, threshold breaches, and state changes
    that match the rule conditions.

    Args:
        df: DataFrame with OHLCV and indicator columns
        rules: List of signal rules to check
        symbol: Trading symbol
        timeframe: Timeframe string
        calculate_outcomes: Whether to calculate outcomes for historical signals
        outcome_forward_bars: Number of bars to look forward for outcome calculation

    Returns:
        List of detected signal events with timestamps and details
    """
    from src.domain.signals.models import ConditionType

    signals: List[Dict[str, Any]] = []
    timestamps = df.index.tolist()

    for rule in rules:
        if not rule.enabled or timeframe not in rule.timeframes:
            continue

        indicator = rule.indicator.lower()
        prefix = f"{indicator}_"
        ind_cols = [c for c in df.columns if c.lower().startswith(prefix)]

        # Dispatch to appropriate handler based on condition type
        if rule.condition_type == ConditionType.CROSS_UP:
            signals.extend(_detect_cross_up_signals(df, rule, ind_cols, timestamps, symbol))
        elif rule.condition_type == ConditionType.CROSS_DOWN:
            signals.extend(_detect_cross_down_signals(df, rule, ind_cols, timestamps, symbol))
        elif rule.condition_type == ConditionType.THRESHOLD_CROSS_UP:
            signals.extend(
                _detect_threshold_cross_up_signals(df, rule, ind_cols, timestamps, symbol)
            )
        elif rule.condition_type == ConditionType.THRESHOLD_CROSS_DOWN:
            signals.extend(
                _detect_threshold_cross_down_signals(df, rule, ind_cols, timestamps, symbol)
            )
        elif rule.condition_type == ConditionType.STATE_CHANGE:
            signals.extend(_detect_state_change_signals(df, rule, ind_cols, timestamps, symbol))

        # MACD special handling
        if indicator == "macd":
            signals.extend(_detect_macd_crosses(df, rule, timestamps, symbol, signals))

    signals.sort(key=lambda x: x["timestamp"])

    # Calculate outcomes for historical signals
    if calculate_outcomes and signals:
        signals = calculate_signal_outcomes(df, signals, outcome_forward_bars)

    return signals


def detect_signals_with_frequency(
    df: pd.DataFrame,
    rules: List["SignalRule"],
    symbol: str,
    timeframe: str,
    lookback_bars: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Detect historical signals and compute frequency counts.

    Uses existing column-based detection logic (CROSS_UP, THRESHOLD_CROSS, etc.)
    applied to indicator columns in the DataFrame.

    Args:
        df: DataFrame with OHLCV and indicator columns
        rules: List of signal rules to check
        symbol: Trading symbol
        timeframe: Timeframe string
        lookback_bars: Optional limit to last N bars (None = all bars)

    Returns:
        Tuple of:
            - signals: List of signal dicts (filtered to lookback window)
            - frequency: Dict mapping rule_name -> count
    """
    # Use existing detection function for all signals
    all_signals = detect_historical_signals(df, rules, symbol, timeframe)

    # Filter to lookback window if specified
    if lookback_bars and len(df) > lookback_bars:
        cutoff_time = df.index[-lookback_bars]
        signals = [s for s in all_signals if s["timestamp"] >= cutoff_time]
    else:
        signals = all_signals

    # Aggregate frequency by rule
    frequency: Dict[str, int] = defaultdict(int)
    for sig in signals:
        frequency[sig["rule"]] += 1

    return signals, dict(frequency)


def calculate_signal_outcomes(
    df: pd.DataFrame,
    signals: List[Dict[str, Any]],
    forward_bars: int = 10,
) -> List[Dict[str, Any]]:
    """
    Calculate outcomes for historical signals.

    For each signal, looks forward N bars to determine:
    - Price change % from signal
    - Whether prediction was correct (buy signal -> price up, sell -> price down)
    - Max favorable/adverse excursion

    Args:
        df: DataFrame with OHLCV data
        signals: List of detected signals with timestamps
        forward_bars: Number of bars to look forward for outcome

    Returns:
        Updated signals list with outcome data added
    """
    if "close" not in df.columns or df.empty:
        return signals

    timestamps = df.index.tolist()
    close_values = df["close"].values
    high_values = df["high"].values if "high" in df.columns else close_values
    low_values = df["low"].values if "low" in df.columns else close_values

    for signal in signals:
        sig_ts = signal.get("timestamp")
        if sig_ts is None:
            continue

        # Find index of signal timestamp
        try:
            sig_idx = timestamps.index(sig_ts)
        except ValueError:
            continue

        # Check if we have enough forward data
        end_idx = min(sig_idx + forward_bars, len(df) - 1)
        is_current = end_idx >= len(df) - 2  # Signal is recent (within last 2 bars)

        if is_current:
            # Current/forming signal - skip outcome (not enough data yet)
            signal["outcome"] = {
                "status": "forming",
                "bars_elapsed": len(df) - 1 - sig_idx,
                "is_current": True,
            }
            continue

        # Calculate outcome for historical signal
        entry_price = close_values[sig_idx]
        exit_price = close_values[end_idx]

        if entry_price <= 0 or pd.isna(entry_price):
            continue

        # Price change
        price_change_pct = ((exit_price - entry_price) / entry_price) * 100

        # Max favorable/adverse excursion
        forward_highs = high_values[sig_idx + 1 : end_idx + 1]
        forward_lows = low_values[sig_idx + 1 : end_idx + 1]

        max_high = float(forward_highs.max()) if len(forward_highs) > 0 else entry_price
        min_low = float(forward_lows.min()) if len(forward_lows) > 0 else entry_price

        mfe = ((max_high - entry_price) / entry_price) * 100  # Max Favorable Excursion
        mae = ((entry_price - min_low) / entry_price) * 100  # Max Adverse Excursion

        # Determine if prediction was correct
        direction = signal.get("direction", "alert")
        if direction == "buy":
            correct = price_change_pct > 0
            outcome_label = "✓ Correct" if correct else "✗ Wrong"
        elif direction == "sell":
            correct = price_change_pct < 0
            outcome_label = "✓ Correct" if correct else "✗ Wrong"
        else:
            correct = None
            outcome_label = "—"

        signal["outcome"] = {
            "status": "completed",
            "is_current": False,
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "price_change_pct": round(price_change_pct, 2),
            "mfe_pct": round(mfe, 2),
            "mae_pct": round(mae, 2),
            "bars_forward": forward_bars,
            "correct": correct,
            "outcome_label": outcome_label,
        }

    return signals


def aggregate_rule_frequency(
    all_signals: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """
    Aggregate rule fire counts across all symbols/timeframes.

    Args:
        all_signals: Dict mapping "SYMBOL_TF" to list of detected signals

    Returns:
        Dict with aggregated frequency data:
            - by_symbol: {symbol: total_count}
            - buy_by_symbol: {symbol: buy_count}
            - sell_by_symbol: {symbol: sell_count}
            - by_rule: {rule_name: total_count}
            - top_symbols: [(symbol, count), ...] sorted desc
            - top_rules: [(rule, count), ...] sorted desc
            - total_signals: int
    """
    by_symbol: Dict[str, int] = defaultdict(int)
    buy_by_symbol: Dict[str, int] = defaultdict(int)
    sell_by_symbol: Dict[str, int] = defaultdict(int)
    by_rule: Dict[str, int] = defaultdict(int)

    for key, signals in all_signals.items():
        # Extract symbol from key (e.g., "AAPL_1h" -> "AAPL")
        symbol = key.rsplit("_", 1)[0] if "_" in key else key

        for signal in signals:
            rule_name = signal.get("rule", "unknown")
            direction = signal.get("direction", "alert")
            by_symbol[symbol] += 1
            by_rule[rule_name] += 1

            # Track buy/sell separately for direction mode
            if direction == "buy":
                buy_by_symbol[symbol] += 1
            elif direction == "sell":
                sell_by_symbol[symbol] += 1

    # Sort for top lists
    top_symbols = sorted(by_symbol.items(), key=lambda x: x[1], reverse=True)
    top_rules = sorted(by_rule.items(), key=lambda x: x[1], reverse=True)

    return {
        "by_symbol": dict(by_symbol),
        "buy_by_symbol": dict(buy_by_symbol),
        "sell_by_symbol": dict(sell_by_symbol),
        "by_rule": dict(by_rule),
        "top_symbols": top_symbols[:20],
        "top_rules": top_rules[:20],
        "total_signals": sum(by_symbol.values()),
    }
