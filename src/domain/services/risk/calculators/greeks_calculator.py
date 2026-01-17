"""
Greeks Calculator - Pure functions for position Greeks scaling.

Scales raw Greeks (from IBKR) by position size and handles asset-type-specific
logic (stocks have delta=1, options use market Greeks).

Usage:
    result = calculate_position_greeks(
        raw_delta=0.5,
        raw_gamma=0.02,
        raw_vega=0.25,
        raw_theta=-0.15,
        quantity=10,
        multiplier=100,
        asset_type=AssetType.OPTION,
    )
    print(result.delta)  # 500.0 (0.5 * 10 * 100)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.models.position import AssetType


@dataclass(frozen=True, slots=True)
class GreeksResult:
    """
    Immutable Greeks calculation result.

    All Greeks are position-adjusted (multiplied by qty * multiplier).
    Thread-safe due to frozen=True.

    Attributes:
        delta: Position delta exposure.
        gamma: Position gamma (rate of delta change).
        vega: Position vega (volatility sensitivity).
        theta: Position theta (time decay per day).
        has_greeks: True if real Greeks were available (not synthetic).
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    has_greeks: bool


def calculate_position_greeks(
    raw_delta: Optional[float],
    raw_gamma: Optional[float],
    raw_vega: Optional[float],
    raw_theta: Optional[float],
    quantity: float,
    multiplier: int,
    asset_type: AssetType,
) -> GreeksResult:
    """
    Scale raw Greeks by position size and handle asset type logic.

    Pure function: no side effects, deterministic output.

    For stocks, Greeks are synthetic:
    - delta = quantity * multiplier (1:1 exposure)
    - gamma, vega, theta = 0 (no optionality)

    For options, Greeks come from market data (IBKR model Greeks).

    Args:
        raw_delta: Per-contract delta (None if unavailable).
        raw_gamma: Per-contract gamma (None if unavailable).
        raw_vega: Per-contract vega (None if unavailable).
        raw_theta: Per-contract theta (None if unavailable).
        quantity: Position size (positive for long, negative for short).
        multiplier: Contract multiplier (1 for stocks, 100 for options).
        asset_type: Type of asset (STOCK, OPTION, etc.).

    Returns:
        GreeksResult with position-scaled Greeks.

    Example:
        >>> result = calculate_position_greeks(
        ...     raw_delta=0.5,
        ...     raw_gamma=0.02,
        ...     raw_vega=0.25,
        ...     raw_theta=-0.15,
        ...     quantity=10,
        ...     multiplier=100,
        ...     asset_type=AssetType.OPTION,
        ... )
        >>> result.delta
        500.0
    """
    qty_mult = quantity * multiplier

    # Stocks: synthetic delta (1:1 exposure), no other Greeks
    if asset_type == AssetType.STOCK:
        return GreeksResult(
            delta=qty_mult,  # Stock delta = 1.0 per share
            gamma=0.0,
            vega=0.0,
            theta=0.0,
            has_greeks=False,  # These are synthetic, not market Greeks
        )

    # Options/Futures: use market Greeks from IBKR
    # Missing Greeks are treated as zero (off-hours, connection issues)
    has_any_greeks = any(g is not None for g in (raw_delta, raw_gamma, raw_vega, raw_theta))

    return GreeksResult(
        delta=(raw_delta or 0.0) * qty_mult,
        gamma=(raw_gamma or 0.0) * qty_mult,
        vega=(raw_vega or 0.0) * qty_mult,
        theta=(raw_theta or 0.0) * qty_mult,
        has_greeks=has_any_greeks,
    )


def calculate_near_term_greeks(
    gamma: float,
    vega: float,
    mark_price: float,
    quantity: float,
    multiplier: int,
    days_to_expiry: Optional[int],
    near_term_gamma_dte: int = 7,
    near_term_vega_dte: int = 30,
) -> tuple[float, float]:
    """
    Calculate near-term Greeks concentration metrics.

    These identify concentrated risk in short-dated options where
    gamma and vega can cause large P&L swings.

    Args:
        gamma: Per-contract gamma.
        vega: Per-contract vega.
        mark_price: Current mark price.
        quantity: Position size.
        multiplier: Contract multiplier.
        days_to_expiry: Days until expiration (None for non-expiring).
        near_term_gamma_dte: DTE threshold for gamma tracking (default 7).
        near_term_vega_dte: DTE threshold for vega tracking (default 30).

    Returns:
        Tuple of (gamma_notional_near_term, vega_notional_near_term).
        Both are zero if position is beyond the DTE thresholds.
    """
    gamma_notional = 0.0
    vega_notional = 0.0

    if days_to_expiry is None:
        return (0.0, 0.0)

    qty_mult = quantity * multiplier
    gamma_factor = 0.01  # Gamma notional scaling factor

    # Near-term gamma (0-7 DTE by default)
    if days_to_expiry <= near_term_gamma_dte:
        gamma_notional = abs((gamma or 0.0) * (mark_price**2) * gamma_factor * qty_mult)

    # Near-term vega (0-30 DTE by default)
    if days_to_expiry <= near_term_vega_dte:
        vega_notional = abs((vega or 0.0) * qty_mult)

    return (gamma_notional, vega_notional)
