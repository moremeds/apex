"""
Regime Parameter Store.

Stores optimized regime detection parameters per symbol/symbol-group.
Parameters can be updated via backtesting optimization pipeline.

Usage:
    from src.domain.services.regime.params_store import get_regime_params, REGIME_PARAMS

    # Get params for a symbol (falls back to DEFAULT if not found)
    params = get_regime_params("NVDA")

    # Use in regime calculation
    result = regime_detector.calculate(data, params)
"""

from __future__ import annotations

from typing import Any, Dict

# Default parameters (conservative baseline)
# These are used when no symbol-specific optimization has been performed
DEFAULT_PARAMS: Dict[str, Any] = {
    # MAs
    "ma50_period": 50,
    "ma200_period": 200,
    "ma20_period": 20,
    "slope_lookback": 20,
    # Volatility (dual-window)
    "atr_period": 20,
    "atr_pct_short_window": 63,
    "atr_pct_long_window": 252,
    "vol_high_short_pct": 80,
    "vol_high_long_pct": 85,
    "vol_low_pct": 20,
    # IV (market level only)
    "iv_pct_window": 63,
    "iv_high_pct": 75,
    "iv_elevated_pct": 50,
    "iv_low_pct": 25,
    # Choppiness
    "chop_period": 14,
    "chop_pct_window": 252,
    "ma20_cross_lookback": 10,
    "chop_high_pct": 70,
    "chop_low_pct": 30,
    "chop_cross_high": 4,
    "chop_cross_low": 1,
    # Extension
    "ext_overbought": 2.0,
    "ext_oversold": -2.0,
    "ext_slightly_high": 1.5,
    "ext_slightly_low": -1.5,
}

# Symbol-specific optimized parameters
# Format: symbol -> param overrides (merged with DEFAULT_PARAMS)
#
# These are populated via the regime optimization pipeline:
#   python -m src.runners.systematic_backtest_runner --spec config/backtest/regime/market_regime_opt.yaml
#
# After optimization, update these values with the validated results.
REGIME_PARAMS: Dict[str, Dict[str, Any]] = {
    # Market benchmarks (conservative - avoid missing R2)
    "QQQ": {
        "vol_high_short_pct": 80,
        "vol_high_long_pct": 85,
        "chop_high_pct": 70,
        "ext_overbought": 2.0,
        "ext_oversold": -2.0,
    },
    "SPY": {
        "vol_high_short_pct": 80,
        "vol_high_long_pct": 85,
        "chop_high_pct": 70,
        "ext_overbought": 2.0,
        "ext_oversold": -2.0,
    },
    # Semiconductors (higher volatility tolerance)
    "SMH": {
        "vol_high_short_pct": 85,
        "vol_high_long_pct": 90,
        "chop_high_pct": 75,
        "ext_overbought": 2.5,
        "ext_oversold": -2.5,
    },
    # Individual stocks (even more lenient for high-vol names)
    "NVDA": {
        "vol_high_short_pct": 88,
        "vol_high_long_pct": 92,
        "chop_high_pct": 78,
        "ext_overbought": 2.8,
        "ext_oversold": -2.8,
    },
    "TSLA": {
        "vol_high_short_pct": 90,
        "vol_high_long_pct": 95,
        "chop_high_pct": 80,
        "ext_overbought": 3.0,
        "ext_oversold": -3.0,
    },
    "AMD": {
        "vol_high_short_pct": 88,
        "vol_high_long_pct": 92,
        "chop_high_pct": 78,
        "ext_overbought": 2.8,
        "ext_oversold": -2.8,
    },
}

# Symbol group mappings (for symbols without specific params)
# Uses correlation-based grouping
SYMBOL_GROUPS: Dict[str, str] = {
    # Tech mega-caps -> similar to QQQ
    "AAPL": "QQQ",
    "MSFT": "QQQ",
    "GOOGL": "QQQ",
    "GOOG": "QQQ",
    "META": "QQQ",
    "AMZN": "QQQ",
    # Semiconductors -> SMH
    "AVGO": "SMH",
    "QCOM": "SMH",
    "INTC": "SMH",
    "MU": "SMH",
    "AMAT": "SMH",
    "LRCX": "SMH",
    "KLAC": "SMH",
    "TSM": "SMH",
}


def get_regime_params(symbol: str) -> Dict[str, Any]:
    """
    Get regime detection parameters for a symbol.

    Lookup order:
    1. Symbol-specific params (REGIME_PARAMS[symbol])
    2. Group params (REGIME_PARAMS[SYMBOL_GROUPS[symbol]])
    3. Default params (DEFAULT_PARAMS)

    Args:
        symbol: Stock/ETF symbol

    Returns:
        Complete parameter dict (defaults merged with symbol-specific overrides)
    """
    # Start with defaults
    params = DEFAULT_PARAMS.copy()

    # Check for symbol-specific params
    if symbol in REGIME_PARAMS:
        params.update(REGIME_PARAMS[symbol])
        return params

    # Check for group params
    group = SYMBOL_GROUPS.get(symbol)
    if group and group in REGIME_PARAMS:
        params.update(REGIME_PARAMS[group])
        return params

    # Use defaults
    return params


def update_regime_params(symbol: str, optimized_params: Dict[str, Any]) -> None:
    """
    Update optimized parameters for a symbol.

    Called by the optimization pipeline after validation passes.

    Args:
        symbol: Symbol to update
        optimized_params: New parameter values (will be merged with defaults)
    """
    REGIME_PARAMS[symbol] = optimized_params


def validate_params(params: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate regime parameters for sanity.

    Args:
        params: Parameters to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Volatility thresholds must be ordered
    if params.get("vol_high_short_pct", 0) >= params.get("vol_high_long_pct", 100):
        return False, "vol_high_short_pct must be < vol_high_long_pct"

    # Chop thresholds must be ordered
    if params.get("chop_low_pct", 0) >= params.get("chop_high_pct", 100):
        return False, "chop_low_pct must be < chop_high_pct"

    # Extension thresholds must be ordered
    if params.get("ext_oversold", 0) >= params.get("ext_overbought", 0):
        return False, "ext_oversold must be < ext_overbought"

    # Positive periods
    for key in ["ma50_period", "ma200_period", "ma20_period", "atr_period", "chop_period"]:
        if params.get(key, 1) <= 0:
            return False, f"{key} must be positive"

    return True, ""


def get_all_symbols_with_params() -> list[str]:
    """Get list of all symbols with specific parameter sets."""
    return list(REGIME_PARAMS.keys())


def get_optimization_status() -> Dict[str, Dict[str, Any]]:
    """
    Get optimization status for all configured symbols.

    Returns:
        Dict with symbol -> {has_custom_params, param_count, last_optimized}
    """
    status = {}
    for symbol in REGIME_PARAMS:
        status[symbol] = {
            "has_custom_params": True,
            "param_count": len(REGIME_PARAMS[symbol]),
            "last_optimized": None,  # Would need to track this separately
        }
    return status
