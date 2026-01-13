"""
Notional Calculator - Pure functions for notional exposure calculations.

Provides position notional, market value, and concentration metrics.

Usage:
    result = calculate_notional(
        mark_price=155.0,
        quantity=100,
        multiplier=1,
    )
    print(result.notional)  # 15500.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class NotionalResult:
    """
    Immutable notional calculation result.

    Thread-safe due to frozen=True.

    Attributes:
        notional: Position notional value (mark * qty * mult).
        gross_notional: Absolute value of notional (for exposure tracking).
    """

    notional: float
    gross_notional: float


def calculate_notional(
    mark_price: float,
    quantity: float,
    multiplier: int,
) -> NotionalResult:
    """
    Calculate position notional exposure.

    Pure function: no side effects, deterministic output.

    Args:
        mark_price: Current mark price.
        quantity: Position size (positive for long, negative for short).
        multiplier: Contract multiplier (1 for stocks, 100 for options).

    Returns:
        NotionalResult with signed notional and gross notional.

    Example:
        >>> result = calculate_notional(155.0, 100, 1)
        >>> result.notional
        15500.0
        >>> result.gross_notional
        15500.0

        >>> result = calculate_notional(155.0, -100, 1)
        >>> result.notional
        -15500.0
        >>> result.gross_notional
        15500.0
    """
    notional = mark_price * quantity * multiplier
    return NotionalResult(
        notional=notional,
        gross_notional=abs(notional),
    )


def calculate_delta_dollars(
    delta: float,
    underlying_price: float,
    quantity: float,
    multiplier: int,
    beta: Optional[float] = None,
) -> tuple[float, float]:
    """
    Calculate delta-dollars and beta-adjusted delta.

    Delta-dollars represents the equivalent stock exposure in dollar terms.
    Beta-adjusted delta normalizes to SPY-equivalent exposure.

    Args:
        delta: Per-contract delta.
        underlying_price: Price of the underlying asset.
        quantity: Position size.
        multiplier: Contract multiplier.
        beta: Asset beta vs SPY (None uses 1.0).

    Returns:
        Tuple of (delta_dollars, beta_adjusted_delta).

    Example:
        >>> delta_dollars, beta_adj = calculate_delta_dollars(
        ...     delta=0.5,
        ...     underlying_price=175.0,
        ...     quantity=10,
        ...     multiplier=100,
        ...     beta=1.2,
        ... )
        >>> delta_dollars
        87500.0  # 0.5 * 175 * 10 * 100
        >>> beta_adj
        600.0    # 0.5 * 10 * 100 * 1.2
    """
    qty_mult = quantity * multiplier
    effective_beta = beta if beta is not None else 1.0

    delta_dollars = delta * underlying_price * qty_mult
    beta_adjusted = delta * qty_mult * effective_beta

    return (delta_dollars, beta_adjusted)


def calculate_concentration(
    notional_by_underlying: dict[str, float],
    total_gross_notional: float,
) -> tuple[str, float, float]:
    """
    Calculate portfolio concentration metrics.

    Identifies the largest underlying exposure and concentration percentage.

    Args:
        notional_by_underlying: Dict mapping underlying symbol to net notional.
        total_gross_notional: Sum of absolute notionals across portfolio.

    Returns:
        Tuple of (max_underlying_symbol, max_underlying_notional, concentration_pct).

    Example:
        >>> symbol, notional, pct = calculate_concentration(
        ...     {"AAPL": 25000, "SPY": -15000, "TSLA": 10000},
        ...     50000,
        ... )
        >>> symbol
        "AAPL"
        >>> notional
        25000.0
        >>> pct
        0.5
    """
    if not notional_by_underlying or total_gross_notional <= 0:
        return ("", 0.0, 0.0)

    # Find underlying with largest absolute notional
    max_underlying = max(
        notional_by_underlying.items(),
        key=lambda x: abs(x[1]),
    )

    max_symbol = max_underlying[0]
    max_notional = abs(max_underlying[1])
    concentration_pct = max_notional / total_gross_notional

    return (max_symbol, max_notional, concentration_pct)
