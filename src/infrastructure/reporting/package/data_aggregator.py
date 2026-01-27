"""
Data Aggregator - DataFrame to JSON conversion for chart data.

Handles transformation of pandas DataFrames to chart-ready JSON format.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from .constants import BOUNDED_OSCILLATORS, OVERLAY_INDICATORS, UNBOUNDED_OSCILLATORS


def df_to_chart_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert DataFrame to chart-ready JSON format.

    Structures indicator data by category for multi-subplot rendering:
    - overlays: Bollinger Bands, SuperTrend, etc (same Y-axis as price)
    - rsi: RSI indicator (0-100 scale)
    - macd: MACD, Signal, Histogram (unbounded scale)
    - oscillators: Other bounded oscillators
    - volume_ind: Volume indicators

    Args:
        df: DataFrame with OHLCV + indicator columns

    Returns:
        Dict with chart data structured by subplot category
    """
    # Convert index to ISO strings
    timestamps = [ts.isoformat() if hasattr(ts, "isoformat") else str(ts) for ts in df.index]

    # OHLCV data
    chart_data: Dict[str, Any] = {
        "timestamps": timestamps,
        "open": df["open"].tolist() if "open" in df else [],
        "high": df["high"].tolist() if "high" in df else [],
        "low": df["low"].tolist() if "low" in df else [],
        "close": df["close"].tolist() if "close" in df else [],
        "volume": df["volume"].tolist() if "volume" in df else [],
        # Structured indicator data for chart subplots
        "overlays": {},
        "rsi": {},
        "macd": {},
        "dual_macd": {},  # DualMACD (55/89 + 13/21) overlapping histograms
        "oscillators": {},
        "volume_ind": {},
        "price_levels": {},  # Fibonacci, S/R, Pivots (price values, not signals)
    }

    # Price-level indicators (these show actual price levels, not signals)
    # Using full indicator names as they appear in column prefixes
    price_level_indicators = {"fibonacci", "support", "pivot"}

    # Categorize indicator columns (same logic as SignalReportGenerator)
    ohlcv_cols = {"open", "high", "low", "close", "volume", "timestamp"}
    oscillator_names = BOUNDED_OSCILLATORS | UNBOUNDED_OSCILLATORS

    for col in df.columns:
        if col.lower() in ohlcv_cols:
            continue

        # Convert to list, handling NaN
        values = df[col].tolist()
        values = [
            None if pd.isna(v) else float(v) if isinstance(v, (int, float)) else str(v)
            for v in values
        ]

        # Parse indicator name from prefixed column (e.g., "macd_histogram" -> "macd")
        parts = col.split("_")
        ind_name = parts[0].lower() if parts else col.lower()

        # Route to appropriate subplot bucket
        if ind_name in OVERLAY_INDICATORS:
            chart_data["overlays"][col] = values
        elif ind_name == "rsi":
            chart_data["rsi"][col] = values
        elif ind_name == "dual" and col.startswith("dual_macd"):
            # DualMACD indicator (dual_macd_long_histogram, dual_macd_short_histogram, etc.)
            chart_data["dual_macd"][col] = values
        elif ind_name == "macd":
            chart_data["macd"][col] = values
        elif ind_name in price_level_indicators:
            chart_data["price_levels"][col] = values
        elif ind_name in oscillator_names:
            chart_data["oscillators"][col] = values
        else:
            # Default to oscillators bucket
            chart_data["oscillators"][col] = values

    return chart_data


def build_indicators_data(
    indicators: List[Any],
    rules: List[Any],
) -> Dict[str, Any]:
    """
    Build indicators.json data structure.

    Args:
        indicators: List of Indicator objects
        rules: List of SignalRule objects

    Returns:
        Dict with categories and indicator information
    """
    # Group indicators by category
    categories: Dict[str, List[Dict[str, Any]]] = {}

    # Build rule lookup by indicator
    rule_lookup: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        ind_name = rule.indicator.lower()
        if ind_name not in rule_lookup:
            rule_lookup[ind_name] = []
        # Extract direction value from enum if needed
        direction_str: str = (
            rule.direction.value if hasattr(rule.direction, "value") else str(rule.direction)
        )
        rule_lookup[ind_name].append(
            {
                "id": rule.name,  # SignalRule uses 'name' not 'id'
                "direction": direction_str,
                "description": getattr(rule, "message_template", None)
                or f"{direction_str.upper()} when {rule.indicator} {rule.condition_type.value}",
            }
        )

    for indicator in indicators:
        category = indicator.category.value if hasattr(indicator, "category") else "other"
        if category not in categories:
            categories[category] = []

        ind_name = indicator.name.lower()
        params = {}
        if hasattr(indicator, "get_params"):
            params = indicator.get_params()

        categories[category].append(
            {
                "name": indicator.name.upper(),
                "description": getattr(indicator, "description", f"{indicator.name} indicator"),
                "params": params,
                "rules": rule_lookup.get(ind_name, []),
            }
        )

    # Sort categories
    category_order = ["momentum", "trend", "volatility", "volume", "pattern", "other"]
    sorted_categories = []
    for cat in category_order:
        if cat in categories:
            sorted_categories.append(
                {
                    "name": cat.title() + " Indicators",
                    "indicators": categories[cat],
                }
            )

    return {
        "categories": sorted_categories,
        "total_indicators": len(indicators),
        "total_rules": len(rules),
    }
