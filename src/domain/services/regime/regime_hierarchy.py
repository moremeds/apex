"""
Regime Hierarchy - Level Synthesis Logic.

Handles:
- QQQ vs SPY disagreement resolution
- Sector mapping (static and dynamic)
- Multi-level regime synthesis
- Weekly veto application
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.domain.signals.indicators.regime import MarketRegime

from .models import (
    ACTION_MAPS,
    MARKET_BENCHMARKS,
    SECTOR_ETFS,
    STOCK_TO_SECTOR,
    AccountType,
    TradingAction,
)


def resolve_market_action(
    qqq_regime: MarketRegime,
    spy_regime: MarketRegime,
    account_type: AccountType = AccountType.SHORT_PUT,
) -> Tuple[TradingAction, MarketRegime]:
    """
    Resolve market action when QQQ and SPY disagree.

    When QQQ and SPY disagree, resolve to the MORE CONSERVATIVE ACTION.
    This avoids the problem where R1 vs R3 ordering is account-dependent.

    Args:
        qqq_regime: QQQ regime classification
        spy_regime: SPY regime classification
        account_type: Account type for action mapping

    Returns:
        Tuple of (resolved_action, effective_regime)
    """
    action_map = ACTION_MAPS[account_type]

    qqq_action = action_map[qqq_regime]
    spy_action = action_map[spy_regime]

    # Take more restrictive action (higher value = more restrictive)
    if qqq_action.value >= spy_action.value:
        return qqq_action, qqq_regime
    else:
        return spy_action, spy_regime


def get_sector_for_symbol(symbol: str) -> Optional[str]:
    """
    Get the sector ETF for a given stock symbol.

    Args:
        symbol: Stock symbol (e.g., "NVDA")

    Returns:
        Sector ETF symbol (e.g., "SMH") or None if not mapped
    """
    return STOCK_TO_SECTOR.get(symbol.upper())


def get_dynamic_sector(
    symbol: str,
    symbol_returns: pd.Series,
    sector_returns: Dict[str, pd.Series],
    correlation_window: int = 60,
) -> Optional[str]:
    """
    Compute rolling correlation with sector ETFs to find best match.

    This handles cases where a stock's sector attribution drifts
    (e.g., TSLA sometimes trades like tech, sometimes like consumer).

    Args:
        symbol: Stock symbol
        symbol_returns: Daily returns for the symbol
        sector_returns: Dict of sector ETF -> daily returns
        correlation_window: Rolling correlation window (default 60 days)

    Returns:
        Sector ETF with highest correlation, or None if insufficient data
    """
    if len(symbol_returns) < correlation_window:
        # Fall back to static mapping
        return get_sector_for_symbol(symbol)

    correlations = {}
    for sector_etf, sector_ret in sector_returns.items():
        if len(sector_ret) >= correlation_window:
            # Align indexes
            aligned = pd.concat([symbol_returns, sector_ret], axis=1, join="inner")
            if len(aligned) >= correlation_window:
                corr = aligned.iloc[-correlation_window:].corr().iloc[0, 1]
                if not np.isnan(corr):
                    correlations[sector_etf] = corr

    if not correlations:
        return get_sector_for_symbol(symbol)

    return max(correlations, key=lambda k: correlations[k])


def synthesize_regimes(
    market_regime: MarketRegime,
    sector_regime: Optional[MarketRegime],
    stock_regime: Optional[MarketRegime],
    account_type: AccountType = AccountType.SHORT_PUT,
) -> TradingAction:
    """
    Synthesize action from all hierarchy levels.

    Decision logic:
    1. Market regime provides the base gate
    2. Sector can only tighten (never loosen) the action
    3. Stock can only tighten (never loosen) the action

    Args:
        market_regime: Market-level regime (QQQ/SPY)
        sector_regime: Sector-level regime (SMH/XLV/etc.)
        stock_regime: Stock-level regime
        account_type: Account type for action mapping

    Returns:
        Synthesized TradingAction
    """
    action_map = ACTION_MAPS[account_type]

    # Start with market action
    action = action_map[market_regime]

    # Sector can only tighten
    if sector_regime is not None:
        sector_action = action_map[sector_regime]
        if sector_action.value > action.value:
            action = sector_action

    # Stock can only tighten
    if stock_regime is not None:
        stock_action = action_map[stock_regime]
        if stock_action.value > action.value:
            action = stock_action

    return action


def apply_weekly_veto(
    daily_regime: MarketRegime,
    weekly_trend_state: str,  # "trend_up", "trend_down", "neutral"
    weekly_vol_state: str,  # "vol_high", "vol_normal", "vol_low"
    veto_active: bool,
    bars_since_veto: int,
    min_veto_bars: int = 5,
) -> Tuple[MarketRegime, bool]:
    """
    Apply weekly veto logic with release conditions.

    Weekly regime can upgrade severity but never downgrade.

    IMPORTANT REFINEMENTS:
    1. Weekly veto only affects R0/R1, NOT R3 (allow rebound window trades)
    2. Explicit release conditions to avoid permanent bearish bias

    Args:
        daily_regime: Daily regime classification
        weekly_trend_state: Weekly trend state
        weekly_vol_state: Weekly volatility state
        veto_active: Whether veto is currently active
        bars_since_veto: Bars since veto was activated
        min_veto_bars: Minimum bars before veto can release (default 5)

    Returns:
        Tuple of (effective_regime, veto_still_active)
    """
    weekly_trend_down = weekly_trend_state == "trend_down"
    weekly_high_vol = weekly_vol_state == "vol_high"
    weekly_trend_up = weekly_trend_state == "trend_up"
    weekly_vol_normal = weekly_vol_state in ("vol_normal", "vol_low")

    # === TRIGGER CONDITIONS ===
    if weekly_trend_down:
        # Force R2 regardless of daily
        return MarketRegime.R2_RISK_OFF, True  # veto_active = True

    if weekly_high_vol and daily_regime == MarketRegime.R0_HEALTHY_UPTREND:
        # Downgrade R0 to R1 when weekly vol is elevated
        return MarketRegime.R1_CHOPPY_EXTENDED, True

    # === RELEASE CONDITIONS ===
    if veto_active:
        release_condition = (
            weekly_trend_up and weekly_vol_normal and bars_since_veto >= min_veto_bars
        )
        if release_condition:
            return daily_regime, False  # veto_active = False

        # Keep veto active but don't override R3 (rebound window)
        if daily_regime == MarketRegime.R3_REBOUND_WINDOW:
            # Allow R3 trades even under weekly veto (small defined-risk)
            return daily_regime, veto_active

    return daily_regime, veto_active


def is_market_benchmark(symbol: str) -> bool:
    """Check if symbol is a market benchmark."""
    return symbol.upper() in MARKET_BENCHMARKS


def is_sector_etf(symbol: str) -> bool:
    """Check if symbol is a sector ETF."""
    return symbol.upper() in SECTOR_ETFS


def get_hierarchy_level(symbol: str) -> str:
    """
    Determine hierarchy level for a symbol.

    Args:
        symbol: Stock/ETF symbol

    Returns:
        "market", "sector", or "stock"
    """
    symbol = symbol.upper()
    if symbol in MARKET_BENCHMARKS:
        return "market"
    if symbol in SECTOR_ETFS:
        return "sector"
    return "stock"


def get_4h_alerts(
    current_state: Dict[str, Any],
    previous_state: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Generate 4H early warning alerts based on rate-of-change.

    Key insight: Rate of change (delta percentile) is more predictive
    than absolute level - catches "regime is changing" before it fully changes.

    Args:
        current_state: Current indicator state
        previous_state: State from ~5 bars ago (if available)

    Returns:
        List of alert messages
    """
    alerts: List[str] = []

    if previous_state is None:
        return alerts

    # Get current and previous percentiles
    curr_atr_pct = current_state.get("atr_pct_63", 50)
    prev_atr_pct = previous_state.get("atr_pct_63", 50)

    curr_ma50_slope = current_state.get("ma50_slope", 0)
    prev_ma50_slope = previous_state.get("ma50_slope", 0)

    close = current_state.get("close", 0)
    ma20 = current_state.get("ma20", 0)

    # Vol acceleration: Δpercentile > 20
    if curr_atr_pct - prev_atr_pct > 20:
        alerts.append(
            f"Volatility accelerating (+{curr_atr_pct - prev_atr_pct:.0f} percentile points) - "
            "prepare for regime change"
        )

    # Vol spike with breakdown
    if curr_atr_pct > 90 and close < ma20:
        alerts.append("Volatility spike with MA20 breakdown - high risk environment")

    # Trend weakening: slope turning negative
    if prev_ma50_slope > 0 and curr_ma50_slope < 0:
        alerts.append("MA50 slope turning negative - trend weakening")

    # Trend strengthening after weakness
    if prev_ma50_slope < 0 and curr_ma50_slope > 0.01:
        alerts.append("MA50 slope turning positive - potential trend recovery")

    return alerts
