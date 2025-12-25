"""
ATR ViewModel - framework agnostic.

Extracts ATR risk calculations from the widget layer.
All business logic (R-multiples, trailing stops, exit plans) lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ATRLevels:
    """Computed ATR levels for display."""

    # Core values
    symbol: str
    price: float
    atr: float
    atr_pct: float
    timeframe: str
    period: int

    # Stop loss levels
    sl_2x: float
    sl_1_5x: float
    sma21: float

    # Risk and R-levels
    risk: float  # 1.5x ATR
    r1: float
    r2: float
    r3: float
    r4: float
    r8: float

    # Trailing stop
    trail: float  # 2x ATR
    r2_stop: float
    r4_stop: float

    # Position info (optional)
    quantity: float = 0
    cost_basis: float = 0

    # Exit plan values (only if position)
    q3: int = 0  # 1/3 of quantity
    runners: int = 0  # remaining after 2/3 sold
    profit_2r: float = 0
    profit_3r: float = 0
    pct_2r: float = 0
    pct_3r: float = 0

    # Summary
    risk_pct: float = 0
    max_gain: float = 0
    max_pct: float = 0


class ATRViewModel:
    """
    Framework-agnostic ViewModel for ATR calculations.

    All R-multiple math, trailing stop logic, and exit plan
    calculations are performed here, not in the widget.
    """

    def compute_levels(
        self,
        atr_data: Any,
        position: Optional[Any] = None,
        timeframe: str = "Daily",
        period: int = 14,
    ) -> Optional[ATRLevels]:
        """
        Compute all ATR levels from raw ATR data.

        Args:
            atr_data: ATR service response with atr_value/atr, current_price/spot
            position: Optional position with quantity and avg_price
            timeframe: Display timeframe label
            period: ATR period

        Returns:
            ATRLevels dataclass with all computed values, or None if invalid data
        """
        if atr_data is None:
            return None

        # Extract core values
        atr = getattr(atr_data, "atr_value", getattr(atr_data, "atr", 0))
        price = getattr(atr_data, "current_price", getattr(atr_data, "spot", 0))
        symbol = getattr(atr_data, "symbol", "?")
        atr_pct = getattr(atr_data, "atr_percent", getattr(atr_data, "atr_pct", 0))

        if price <= 0:
            return None

        # Core calculations
        risk = atr * 1.5
        sma21 = price * 0.97  # Approximate SMA21 as 3% below price
        sl_2x = price - (atr * 2)
        sl_1_5x = price - (atr * 1.5)
        trail = atr * 2

        # R-levels (based on 1R = 1.5x ATR)
        r1 = price + risk
        r2 = price + (risk * 2)
        r3 = price + (risk * 3)
        r4 = price + (risk * 4)
        r8 = price + (risk * 8)

        # Trailing stop levels
        r2_stop = r2 - trail
        r4_stop = r4 - trail

        # Position info
        qty = 0.0
        cost_basis = price
        if position:
            qty = float(getattr(position, "quantity", 0))
            if hasattr(position, "position"):
                avg_price = getattr(position.position, "avg_price", None)
                if avg_price:
                    cost_basis = avg_price

        # Exit plan (only if position)
        q3 = 0
        runners = 0
        profit_2r = 0.0
        profit_3r = 0.0
        pct_2r = 0.0
        pct_3r = 0.0

        if qty > 0:
            q3 = int(qty // 3)
            runners = int(qty - 2 * q3)
            profit_2r = (r2 - cost_basis) * q3
            profit_3r = (r3 - cost_basis) * q3
            if cost_basis > 0:
                pct_2r = (r2 - cost_basis) / cost_basis * 100
                pct_3r = (r3 - cost_basis) / cost_basis * 100

        # Summary
        risk_pct = (risk / price * 100) if price > 0 else 0
        max_gain = r8 - price
        max_pct = (max_gain / price * 100) if price > 0 else 0

        return ATRLevels(
            symbol=symbol,
            price=price,
            atr=atr,
            atr_pct=atr_pct,
            timeframe=timeframe,
            period=period,
            sl_2x=sl_2x,
            sl_1_5x=sl_1_5x,
            sma21=sma21,
            risk=risk,
            r1=r1,
            r2=r2,
            r3=r3,
            r4=r4,
            r8=r8,
            trail=trail,
            r2_stop=r2_stop,
            r4_stop=r4_stop,
            quantity=qty,
            cost_basis=cost_basis,
            q3=q3,
            runners=runners,
            profit_2r=profit_2r,
            profit_3r=profit_3r,
            pct_2r=pct_2r,
            pct_3r=pct_3r,
            risk_pct=risk_pct,
            max_gain=max_gain,
            max_pct=max_pct,
        )
