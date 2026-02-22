"""
Confluence Analyzer - Indicator state derivation and confluence scoring.

This module provides:
1. Indicator state derivation from DataFrame values
2. Confluence score calculation across indicators

The derived states are used by CrossIndicatorAnalyzer to compute
alignment scores showing bullish/bearish confluence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.domain.signals.divergence.cross_divergence import CrossIndicatorAnalyzer
from src.domain.signals.models import ConfluenceScore
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper functions for derive_indicator_states
# =============================================================================


def _derive_rsi_state(df: pd.DataFrame, last_row: pd.Series) -> Optional[Dict[str, Any]]:
    """Derive RSI indicator state."""
    rsi_col = next((c for c in df.columns if c.lower().startswith("rsi_rsi")), None)
    if rsi_col and pd.notna(last_row[rsi_col]):
        rsi_val = float(last_row[rsi_col])
        zone = "oversold" if rsi_val < 30 else "overbought" if rsi_val > 70 else "neutral"
        return {"value": rsi_val, "zone": zone}
    return None


def _derive_macd_state(df: pd.DataFrame, last_row: pd.Series) -> Optional[Dict[str, Any]]:
    """Derive MACD indicator state."""
    macd_col = next((c for c in df.columns if "macd_macd" in c.lower()), None)
    signal_col = next((c for c in df.columns if "macd_signal" in c.lower()), None)
    hist_col = next((c for c in df.columns if "macd_histogram" in c.lower()), None)

    if not hist_col or not pd.notna(last_row[hist_col]):
        return None

    hist_val = float(last_row[hist_col])
    macd_val = float(last_row[macd_col]) if macd_col and pd.notna(last_row[macd_col]) else 0
    signal_val = float(last_row[signal_col]) if signal_col and pd.notna(last_row[signal_col]) else 0

    cross = _detect_line_cross(df, macd_col, signal_col, macd_val, signal_val)
    return {"histogram": hist_val, "macd": macd_val, "signal": signal_val, "cross": cross}


def _detect_line_cross(
    df: pd.DataFrame,
    col_a: Optional[str],
    col_b: Optional[str],
    val_a: float,
    val_b: float,
) -> str:
    """Detect bullish/bearish cross from last two rows."""
    if len(df) < 2 or not col_a or not col_b:
        return "neutral"

    prev_a = df[col_a].iloc[-2] if pd.notna(df[col_a].iloc[-2]) else None
    prev_b = df[col_b].iloc[-2] if pd.notna(df[col_b].iloc[-2]) else None

    if prev_a is None or prev_b is None:
        return "neutral"

    if prev_a <= prev_b and val_a > val_b:
        return "bullish"
    elif prev_a >= prev_b and val_a < val_b:
        return "bearish"
    return "neutral"


def _derive_supertrend_state(df: pd.DataFrame, last_row: pd.Series) -> Optional[Dict[str, Any]]:
    """Derive SuperTrend indicator state."""
    st_col = next((c for c in df.columns if "supertrend_direction" in c.lower()), None)
    if st_col and pd.notna(last_row[st_col]):
        return {"direction": str(last_row[st_col]).lower()}

    # Infer from SuperTrend vs price
    st_val_col = next((c for c in df.columns if "supertrend_supertrend" in c.lower()), None)
    close_col = "close" if "close" in df.columns else None

    if not st_val_col or not close_col:
        return None
    if not pd.notna(last_row[st_val_col]) or not pd.notna(last_row[close_col]):
        return None

    st_val = float(last_row[st_val_col])
    close_val = float(last_row[close_col])
    direction = "bullish" if close_val > st_val else "bearish"
    return {"direction": direction, "value": st_val}


def _derive_bollinger_state(df: pd.DataFrame, last_row: pd.Series) -> Optional[Dict[str, Any]]:
    """Derive Bollinger Bands indicator state."""
    bb_upper = next((c for c in df.columns if "bollinger_bb_upper" in c.lower()), None)
    bb_lower = next((c for c in df.columns if "bollinger_bb_lower" in c.lower()), None)
    close_col = "close" if "close" in df.columns else None

    if not bb_upper or not bb_lower or not close_col:
        return None
    if not all(pd.notna(last_row[c]) for c in [bb_upper, bb_lower, close_col]):
        return None

    upper = float(last_row[bb_upper])
    lower = float(last_row[bb_lower])
    close = float(last_row[close_col])

    if close <= lower:
        zone = "below_lower"
    elif close >= upper:
        zone = "above_upper"
    else:
        zone = "middle"
    return {"zone": zone, "upper": upper, "lower": lower}


def _derive_kdj_state(df: pd.DataFrame, last_row: pd.Series) -> Optional[Dict[str, Any]]:
    """Derive KDJ indicator state."""
    k_col = next((c for c in df.columns if c.lower().startswith("kdj_k")), None)
    d_col = next((c for c in df.columns if c.lower().startswith("kdj_d")), None)

    if not k_col or not d_col:
        return None
    if not pd.notna(last_row[k_col]) or not pd.notna(last_row[d_col]):
        return None

    k_val = float(last_row[k_col])
    d_val = float(last_row[d_col])
    zone = "oversold" if k_val < 20 else "overbought" if k_val > 80 else "neutral"
    cross = _detect_line_cross(df, k_col, d_col, k_val, d_val)
    return {"k": k_val, "d": d_val, "zone": zone, "cross": cross}


def _derive_adx_state(df: pd.DataFrame, last_row: pd.Series) -> Optional[Dict[str, Any]]:
    """Derive ADX indicator state."""
    adx_col = next((c for c in df.columns if c.lower().startswith("adx_adx")), None)
    di_plus_col = next(
        (c for c in df.columns if "di_plus" in c.lower() or "di_p" in c.lower()), None
    )
    di_minus_col = next(
        (c for c in df.columns if "di_minus" in c.lower() or "di_m" in c.lower()), None
    )

    if not di_plus_col or not di_minus_col:
        return None
    if not pd.notna(last_row[di_plus_col]) or not pd.notna(last_row[di_minus_col]):
        return None

    di_plus = float(last_row[di_plus_col])
    di_minus = float(last_row[di_minus_col])
    adx_val = float(last_row[adx_col]) if adx_col and pd.notna(last_row[adx_col]) else 0
    return {"adx": adx_val, "di_plus": di_plus, "di_minus": di_minus}


def derive_indicator_states(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Derive semantic indicator states from DataFrame's last row values.

    Translates raw indicator values into the state format expected by
    CrossIndicatorAnalyzer (e.g., RSI=35 -> {'zone': 'oversold', 'value': 35}).

    Args:
        df: DataFrame with indicator columns

    Returns:
        Dict mapping indicator names to state dicts
    """
    if df.empty:
        return {}

    last_row = df.iloc[-1]
    states: Dict[str, Dict[str, Any]] = {}

    # Derive each indicator state using helper functions
    derivers = [
        ("rsi", _derive_rsi_state),
        ("macd", _derive_macd_state),
        ("supertrend", _derive_supertrend_state),
        ("bollinger", _derive_bollinger_state),
        ("kdj", _derive_kdj_state),
        ("adx", _derive_adx_state),
    ]

    for name, deriver in derivers:
        state = deriver(df, last_row)
        if state:
            states[name] = state

    return states


def calculate_confluence(data: Dict[Tuple[str, str], pd.DataFrame]) -> Dict[str, ConfluenceScore]:
    """
    Calculate confluence scores for all symbol/timeframe combinations.

    Args:
        data: Dict mapping (symbol, timeframe) to DataFrame

    Returns:
        Dict mapping "symbol_timeframe" to ConfluenceScore
    """
    analyzer = CrossIndicatorAnalyzer()
    confluence_scores: Dict[str, ConfluenceScore] = {}

    for (symbol, timeframe), df in data.items():
        key = f"{symbol}_{timeframe}"
        indicator_states = derive_indicator_states(df)

        if indicator_states:
            score = analyzer.analyze(symbol, timeframe, indicator_states)
            confluence_scores[key] = score
            logger.debug(
                f"Confluence calculated for {key}",
                extra={
                    "alignment_score": score.alignment_score,
                    "bullish": score.bullish_count,
                    "bearish": score.bearish_count,
                    "indicators_analyzed": list(indicator_states.keys()),
                },
            )

    return confluence_scores


def calculate_mtf_confluence(
    data: Dict[Tuple[str, str], pd.DataFrame],
    symbol: str,
    timeframes: Tuple[str, ...] = ("1h", "4h", "1d"),
) -> Dict[str, Any]:
    """
    Calculate multi-timeframe confluence for a single symbol.

    Determines alignment across timeframes (1h, 4h, 1d) based on indicator states.
    Daily is the "slow truth" (primary), intraday provides early warning.

    Args:
        data: Dict mapping (symbol, timeframe) to DataFrame
        symbol: Symbol to analyze
        timeframes: Timeframes to check alignment across

    Returns:
        Dict with MTF confluence info:
            - timeframes: List of timeframes with data
            - bullish_count: Total bullish signals across timeframes
            - bearish_count: Total bearish signals across timeframes
            - aligned: True if all timeframes agree
            - alignment_score: -100 to +100 (-100=all bearish, +100=all bullish)
            - primary_direction: Direction from daily (slow truth)
            - confidence: 0-1 based on timeframe agreement
    """
    from typing import List

    states_by_tf: Dict[str, Dict[str, Dict[str, Any]]] = {}
    bullish_by_tf: Dict[str, int] = {}
    bearish_by_tf: Dict[str, int] = {}

    # Derive indicator states for each timeframe
    for tf in timeframes:
        if (symbol, tf) in data:
            df = data[(symbol, tf)]
            states = derive_indicator_states(df)
            if states:
                states_by_tf[tf] = states

                # Count bullish/bearish for this timeframe
                bullish = 0
                bearish = 0
                for ind_name, state in states.items():
                    direction = _get_indicator_direction(ind_name, state)
                    if direction == "bullish":
                        bullish += 1
                    elif direction == "bearish":
                        bearish += 1

                bullish_by_tf[tf] = bullish
                bearish_by_tf[tf] = bearish

    if len(states_by_tf) < 2:
        return {
            "timeframes": list(states_by_tf.keys()),
            "bullish_count": 0,
            "bearish_count": 0,
            "aligned": False,
            "alignment_score": 0,
            "primary_direction": "neutral",
            "confidence": 0.0,
            "message": "Insufficient timeframes for MTF analysis",
        }

    # Total counts
    total_bullish = sum(bullish_by_tf.values())
    total_bearish = sum(bearish_by_tf.values())
    total = total_bullish + total_bearish

    # Determine direction per timeframe
    tf_directions: List[str] = []
    for tf in timeframes:
        if tf in bullish_by_tf:
            if bullish_by_tf[tf] > bearish_by_tf[tf]:
                tf_directions.append("bullish")
            elif bearish_by_tf[tf] > bullish_by_tf[tf]:
                tf_directions.append("bearish")
            else:
                tf_directions.append("neutral")

    # Check alignment
    unique_directions = set(d for d in tf_directions if d != "neutral")
    aligned = len(unique_directions) <= 1 and len(unique_directions) > 0

    # Confidence based on agreement
    if len(tf_directions) == 0:
        confidence = 0.0
    elif len(unique_directions) == 0:
        confidence = 0.5  # All neutral
    elif len(unique_directions) == 1:
        confidence = 1.0  # All agree
    elif len(unique_directions) == 2:
        confidence = 0.5  # Mixed
    else:
        confidence = 0.3

    # Primary direction from daily (or longest available timeframe)
    primary_direction = "neutral"
    for tf in reversed(timeframes):  # Check longest TF first (1d)
        if tf in bullish_by_tf:
            if bullish_by_tf[tf] > bearish_by_tf[tf]:
                primary_direction = "bullish"
            elif bearish_by_tf[tf] > bullish_by_tf[tf]:
                primary_direction = "bearish"
            break

    # Alignment score: -100 to +100
    alignment_score = ((total_bullish - total_bearish) / max(total, 1)) * 100

    return {
        "timeframes": list(states_by_tf.keys()),
        "bullish_count": total_bullish,
        "bearish_count": total_bearish,
        "aligned": aligned,
        "alignment_score": round(alignment_score, 1),
        "primary_direction": primary_direction,
        "confidence": round(confidence, 2),
        "by_timeframe": {
            tf: {"bullish": bullish_by_tf.get(tf, 0), "bearish": bearish_by_tf.get(tf, 0)}
            for tf in states_by_tf.keys()
        },
    }


def _get_indicator_direction(indicator: str, state: Dict[str, Any]) -> str:
    """
    Determine bullish/bearish/neutral direction from indicator state.

    Args:
        indicator: Indicator name
        state: Indicator state dict

    Returns:
        "bullish", "bearish", or "neutral"
    """
    # Check explicit direction/cross fields
    if "direction" in state:
        d = str(state["direction"]).lower()
        if d in ("bullish", "up", "long"):
            return "bullish"
        elif d in ("bearish", "down", "short"):
            return "bearish"

    if "cross" in state:
        c = str(state["cross"]).lower()
        if c == "bullish":
            return "bullish"
        elif c == "bearish":
            return "bearish"

    # RSI/KDJ zone interpretation
    if "zone" in state:
        z = str(state["zone"]).lower()
        if z == "oversold":
            return "bullish"  # Oversold = potential reversal up
        elif z == "overbought":
            return "bearish"  # Overbought = potential reversal down
        elif z in ("below_lower",):
            return "bullish"  # Below lower BB = oversold
        elif z in ("above_upper",):
            return "bearish"  # Above upper BB = overbought

    # MACD histogram
    if indicator == "macd" and "histogram" in state:
        if state["histogram"] > 0:
            return "bullish"
        elif state["histogram"] < 0:
            return "bearish"

    # ADX DI comparison
    if indicator == "adx" and "di_plus" in state and "di_minus" in state:
        if state["di_plus"] > state["di_minus"]:
            return "bullish"
        elif state["di_minus"] > state["di_plus"]:
            return "bearish"

    return "neutral"
