"""
Auto-generate descriptions for indicators and rules from their metadata.

No additional fields required - descriptions are built from existing
indicator properties and rule configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from ..indicators.base import Indicator
    from ..models import SignalRule

from ..models import ConditionType

# Template descriptions for known indicators
INDICATOR_TEMPLATES: Dict[str, str] = {
    "rsi": "Relative Strength Index - momentum oscillator (0-100)",
    "macd": "Moving Average Convergence Divergence - trend momentum",
    "bollinger": "Bollinger Bands - volatility-based price envelope",
    "supertrend": "SuperTrend - trend-following with ATR bands",
    "atr": "Average True Range - volatility measurement",
    "adx": "Average Directional Index - trend strength (0-100)",
    "stochastic": "Stochastic Oscillator - momentum with K/D lines",
    "kdj": "KDJ Indicator - stochastic variant with J line",
    "cci": "Commodity Channel Index - mean reversion oscillator",
    "williams_r": "Williams %R - momentum oscillator (-100 to 0)",
    "roc": "Rate of Change - momentum as percentage",
    "momentum": "Momentum - price change over N periods",
    "mfi": "Money Flow Index - volume-weighted RSI",
    "obv": "On Balance Volume - cumulative volume flow",
    "vwap": "Volume Weighted Average Price",
    "cmf": "Chaikin Money Flow - volume-based momentum",
    "ema": "Exponential Moving Average",
    "sma": "Simple Moving Average",
    "wma": "Weighted Moving Average",
    "dema": "Double Exponential Moving Average",
    "tema": "Triple Exponential Moving Average",
    "keltner": "Keltner Channels - ATR-based bands",
    "donchian": "Donchian Channels - highest high/lowest low",
    "ichimoku": "Ichimoku Cloud - multi-component trend system",
    "parabolic_sar": "Parabolic SAR - trend reversal indicator",
    "pivot_points": "Pivot Points - support/resistance levels",
    "doji": "Doji Pattern - indecision candle",
    "hammer": "Hammer Pattern - potential reversal",
    "engulfing": "Engulfing Pattern - bullish/bearish reversal",
    "morning_star": "Morning Star - bullish reversal pattern",
    "evening_star": "Evening Star - bearish reversal pattern",
    "three_white_soldiers": "Three White Soldiers - strong bullish",
    "three_black_crows": "Three Black Crows - strong bearish",
    "harami": "Harami Pattern - potential reversal",
    "vwma": "Volume Weighted Moving Average",
    "pvo": "Percentage Volume Oscillator",
    "volume_profile": "Volume Profile - price-volume distribution",
    "force_index": "Force Index - price-volume momentum",
    "ease_of_movement": "Ease of Movement - volume-adjusted price change",
    "chaikin_volatility": "Chaikin Volatility - EMA of high-low range",
}


def generate_indicator_description(indicator: "Indicator") -> str:
    """
    Auto-generate description from indicator metadata.

    Uses template descriptions for known indicators, falls back to
    generic description with parameters for unknown ones.

    Args:
        indicator: Indicator instance with name, category, and default_params

    Returns:
        Human-readable description string

    Example:
        RSI (momentum): Relative Strength Index - momentum oscillator (0-100).
        Params: period=14, overbought=70, oversold=30
    """
    name = indicator.name.lower()
    indicator.category.value

    # Get base description
    base = INDICATOR_TEMPLATES.get(name, f"{indicator.name.upper()} indicator")

    # Format parameters
    params = indicator.default_params
    if params:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{base}. Params: {param_str}"

    return base


def generate_rule_description(rule: "SignalRule") -> str:
    """
    Auto-generate description from rule structure.

    Builds human-readable description from condition type and configuration.

    Args:
        rule: SignalRule with condition_type and condition_config

    Returns:
        Human-readable description string

    Examples:
        rsi_oversold_exit: BUY when RSI zone: oversold → neutral
        macd_bullish_cross: BUY when MACD crosses above signal
        rsi_overbought_threshold: SELL when RSI value > 70
    """
    direction = rule.direction.value.upper()
    indicator = rule.indicator
    config = rule.condition_config

    condition_text = _format_condition(rule.condition_type, config)

    return f"{direction} when {indicator} {condition_text}"


def _format_condition(condition_type: ConditionType, config: Dict[str, Any]) -> str:
    """Format condition configuration as readable text."""
    if condition_type == ConditionType.STATE_CHANGE:
        field = config.get("field", "state")
        from_states = config.get("from", [])
        to_states = config.get("to", [])
        from_str = "/".join(str(s) for s in from_states) if from_states else "any"
        to_str = "/".join(str(s) for s in to_states) if to_states else "any"
        return f"{field}: {from_str} → {to_str}"

    elif condition_type == ConditionType.THRESHOLD_CROSS_UP:
        field = config.get("field", "value")
        threshold = config.get("threshold", "?")
        return f"{field} > {threshold}"

    elif condition_type == ConditionType.THRESHOLD_CROSS_DOWN:
        field = config.get("field", "value")
        threshold = config.get("threshold", "?")
        return f"{field} < {threshold}"

    elif condition_type == ConditionType.CROSS_UP:
        line_a = config.get("line_a", "line1")
        line_b = config.get("line_b", "line2")
        return f"{line_a} crosses above {line_b}"

    elif condition_type == ConditionType.CROSS_DOWN:
        line_a = config.get("line_a", "line1")
        line_b = config.get("line_b", "line2")
        return f"{line_a} crosses below {line_b}"

    elif condition_type == ConditionType.RANGE_ENTRY:
        low = config.get("low", "?")
        high = config.get("high", "?")
        return f"enters range [{low}, {high}]"

    elif condition_type == ConditionType.RANGE_EXIT:
        low = config.get("low", "?")
        high = config.get("high", "?")
        return f"exits range [{low}, {high}]"

    elif condition_type == ConditionType.CUSTOM:
        return config.get("description", "custom condition")

    return str(config)
