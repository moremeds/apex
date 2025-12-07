"""
Strategy classification engine for options trading.

Implements rule-based pattern detection for common options strategies:
- Vertical spreads (bull/bear call/put spreads)
- Volatility plays (straddles, strangles)
- Calendar/diagonal spreads
- Income strategies (covered calls, protective puts)
- Multi-leg complex (iron condors, iron butterflies)

Classification uses time-window clustering to group related trades.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import hashlib
import logging

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Options strategy types."""

    # Single-leg
    DIRECTIONAL = "directional"

    # Vertical spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Volatility plays
    LONG_STRADDLE = "long_straddle"
    SHORT_STRADDLE = "short_straddle"
    LONG_STRANGLE = "long_strangle"
    SHORT_STRANGLE = "short_strangle"

    # Calendar/Diagonal
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"

    # Income strategies
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    CASH_SECURED_PUT = "cash_secured_put"

    # Multi-leg complex
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"

    # Stock-only
    STOCK_POSITION = "stock_position"

    # Fallback
    UNKNOWN = "unknown"


@dataclass
class LegInfo:
    """Information about a single leg of a strategy."""

    order_uid: str
    symbol: str
    underlying: str
    instrument_type: str  # STOCK, OPTION, FUTURE
    side: str  # BUY, SELL
    qty: float
    strike: Optional[float] = None
    expiry: Optional[str] = None  # YYYYMMDD format
    option_right: Optional[str] = None  # CALL, PUT
    trade_time: Optional[datetime] = None
    price: Optional[float] = None
    commission: Optional[float] = None

    @classmethod
    def from_order_dict(cls, order: Dict[str, Any]) -> "LegInfo":
        """Create LegInfo from an order dictionary (from DB)."""
        return cls(
            order_uid=f"{order.get('source')}_{order.get('account_id')}_{order.get('order_id')}",
            symbol=order.get('symbol', ''),
            underlying=order.get('underlying', order.get('symbol', '')),
            instrument_type=order.get('asset_type', 'STOCK'),
            side=order.get('side', 'BUY'),
            qty=float(order.get('filled_quantity') or order.get('quantity') or 0),
            strike=order.get('strike'),
            expiry=order.get('expiry'),
            option_right=order.get('option_right'),
            trade_time=order.get('filled_time') or order.get('created_time'),
            price=order.get('avg_fill_price') or order.get('limit_price'),
            commission=order.get('commission'),
        )


@dataclass
class StrategyResult:
    """Result of strategy classification."""

    strategy_type: StrategyType
    strategy_id: str
    confidence: float
    legs: List[LegInfo]
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "strategy_type": self.strategy_type.value,
            "strategy_id": self.strategy_id,
            "confidence": self.confidence,
            "name": self.name,
            "legs": [
                {
                    "order_uid": leg.order_uid,
                    "symbol": leg.symbol,
                    "side": leg.side,
                    "qty": leg.qty,
                    "strike": leg.strike,
                    "expiry": leg.expiry,
                    "option_right": leg.option_right,
                }
                for leg in self.legs
            ],
            "metadata": self.metadata,
        }


class StrategyClassifierV1:
    """
    Rule-based strategy classifier using time-window clustering.

    Classification confidence levels:
    - 0.95+: Very high confidence (exact pattern match)
    - 0.80-0.94: High confidence (pattern match with minor variations)
    - 0.50-0.79: Medium confidence (partial pattern match)
    - <0.50: Low confidence (fallback to unknown)
    """

    TIME_WINDOW_SECONDS = 5  # Legs within 5s considered same strategy

    def classify(self, legs: List[LegInfo]) -> StrategyResult:
        """
        Main classification entry point.

        Args:
            legs: List of legs grouped by (underlying, time_window)

        Returns:
            Strategy classification with confidence
        """
        if len(legs) == 0:
            return self._unknown(legs, 0.0)

        # Single leg
        if len(legs) == 1:
            return self._classify_single(legs[0])

        # Multi-leg: separate by instrument type
        options = [leg for leg in legs if leg.instrument_type == 'OPTION']
        stocks = [leg for leg in legs if leg.instrument_type == 'STOCK']

        # Stock + Option combinations
        if len(stocks) == 1 and len(options) == 1:
            return self._classify_stock_option(stocks[0], options[0])

        # Stock + Multiple Options (covered call with multiple strikes, etc.)
        if len(stocks) == 1 and len(options) > 1:
            return self._classify_stock_multi_option(stocks[0], options)

        # Pure option combinations
        if len(stocks) == 0 and len(options) >= 2:
            return self._classify_option_spread(options)

        # Multiple stocks (unusual)
        if len(stocks) > 1 and len(options) == 0:
            return StrategyResult(
                strategy_type=StrategyType.STOCK_POSITION,
                strategy_id=self._generate_id(legs),
                confidence=0.60,
                legs=legs,
                name="Multi-stock position"
            )

        return self._unknown(legs, 0.2)

    def _classify_single(self, leg: LegInfo) -> StrategyResult:
        """Single-leg position classification."""
        direction = "Long" if leg.side == "BUY" else "Short"

        if leg.instrument_type == "STOCK":
            return StrategyResult(
                strategy_type=StrategyType.STOCK_POSITION,
                strategy_id=self._generate_id([leg]),
                confidence=0.95,
                legs=[leg],
                name=f"{direction} {leg.symbol}"
            )

        if leg.instrument_type == "OPTION":
            # Check for cash-secured put (selling puts)
            if leg.side == "SELL" and leg.option_right == "PUT":
                return StrategyResult(
                    strategy_type=StrategyType.CASH_SECURED_PUT,
                    strategy_id=self._generate_id([leg]),
                    confidence=0.85,
                    legs=[leg],
                    name=f"Cash Secured Put {leg.underlying} {leg.strike}"
                )

            return StrategyResult(
                strategy_type=StrategyType.DIRECTIONAL,
                strategy_id=self._generate_id([leg]),
                confidence=0.95,
                legs=[leg],
                name=f"{direction} {leg.symbol}"
            )

        return StrategyResult(
            strategy_type=StrategyType.DIRECTIONAL,
            strategy_id=self._generate_id([leg]),
            confidence=0.90,
            legs=[leg],
            name=f"{direction} {leg.symbol}"
        )

    def _classify_stock_option(self, stock: LegInfo, opt: LegInfo) -> StrategyResult:
        """Stock + Option combinations."""

        # Covered Call: Long stock + Short call
        if stock.side == "BUY" and opt.side == "SELL" and opt.option_right == "CALL":
            return StrategyResult(
                strategy_type=StrategyType.COVERED_CALL,
                strategy_id=self._generate_id([stock, opt]),
                confidence=0.90,
                legs=[stock, opt],
                name=f"Covered Call {stock.underlying} @ {opt.strike}"
            )

        # Protective Put: Long stock + Long put
        if stock.side == "BUY" and opt.side == "BUY" and opt.option_right == "PUT":
            return StrategyResult(
                strategy_type=StrategyType.PROTECTIVE_PUT,
                strategy_id=self._generate_id([stock, opt]),
                confidence=0.90,
                legs=[stock, opt],
                name=f"Protective Put {stock.underlying} @ {opt.strike}"
            )

        # Short stock + short put (reverse conversion component)
        if stock.side == "SELL" and opt.side == "SELL" and opt.option_right == "PUT":
            return StrategyResult(
                strategy_type=StrategyType.UNKNOWN,
                strategy_id=self._generate_id([stock, opt]),
                confidence=0.50,
                legs=[stock, opt],
                name=f"Short Stock + Short Put {stock.underlying}",
                metadata={"possible_strategy": "reverse_conversion_partial"}
            )

        return self._unknown([stock, opt], 0.3)

    def _classify_stock_multi_option(
        self, stock: LegInfo, options: List[LegInfo]
    ) -> StrategyResult:
        """Stock + multiple options."""

        # Check if all options are short calls (covered call ladder)
        all_short_calls = all(
            opt.side == "SELL" and opt.option_right == "CALL"
            for opt in options
        )
        if stock.side == "BUY" and all_short_calls:
            strikes = sorted([opt.strike for opt in options if opt.strike])
            return StrategyResult(
                strategy_type=StrategyType.COVERED_CALL,
                strategy_id=self._generate_id([stock] + options),
                confidence=0.85,
                legs=[stock] + options,
                name=f"Covered Call Ladder {stock.underlying} @ {strikes}",
                metadata={"variant": "ladder"}
            )

        return self._unknown([stock] + options, 0.3)

    def _classify_option_spread(self, options: List[LegInfo]) -> StrategyResult:
        """Pure option multi-leg strategies."""

        # Sort by strike for consistent analysis
        options = sorted(options, key=lambda x: (x.expiry or '', x.strike or 0))

        # 2-leg spreads
        if len(options) == 2:
            return self._classify_two_leg(options[0], options[1])

        # 3-leg spreads (butterfly, etc.)
        if len(options) == 3:
            return self._classify_three_leg(options)

        # 4-leg: Iron Condor / Iron Butterfly
        if len(options) == 4:
            return self._classify_four_leg(options)

        # More than 4 legs - likely a complex or custom strategy
        return self._unknown(options, 0.2)

    def _classify_two_leg(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Two-leg option spreads."""

        same_expiry = leg1.expiry == leg2.expiry
        same_right = leg1.option_right == leg2.option_right
        same_strike = leg1.strike == leg2.strike

        # Vertical spread: same expiry, same right, different strikes
        if same_expiry and same_right and not same_strike:
            return self._classify_vertical(leg1, leg2)

        # Straddle: same expiry, same strike, different right
        if same_expiry and same_strike and not same_right:
            return self._classify_straddle(leg1, leg2)

        # Strangle: same expiry, different strikes, different rights
        if same_expiry and not same_strike and not same_right:
            return self._classify_strangle(leg1, leg2)

        # Calendar: different expiry, same strike, same right
        if not same_expiry and same_strike and same_right:
            return StrategyResult(
                strategy_type=StrategyType.CALENDAR_SPREAD,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.80,
                legs=[leg1, leg2],
                name=f"Calendar {leg1.option_right} {leg1.strike}"
            )

        # Diagonal: different expiry, different strikes, same right
        if not same_expiry and not same_strike and same_right:
            return StrategyResult(
                strategy_type=StrategyType.DIAGONAL_SPREAD,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.75,
                legs=[leg1, leg2],
                name=f"Diagonal {leg1.option_right}"
            )

        return self._unknown([leg1, leg2], 0.3)

    def _classify_vertical(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Vertical spread classification."""

        # Ensure leg1 has lower strike
        if (leg1.strike or 0) > (leg2.strike or 0):
            leg1, leg2 = leg2, leg1

        is_call = leg1.option_right == "CALL"
        buy_lower = leg1.side == "BUY"

        if is_call:
            if buy_lower:
                # Buy lower call, sell higher call = Bull Call Spread
                strategy_type = StrategyType.BULL_CALL_SPREAD
                name = f"Bull Call {leg1.strike}/{leg2.strike}"
            else:
                # Sell lower call, buy higher call = Bear Call Spread
                strategy_type = StrategyType.BEAR_CALL_SPREAD
                name = f"Bear Call {leg1.strike}/{leg2.strike}"
        else:  # PUT
            if buy_lower:
                # Buy lower put, sell higher put = Bear Put Spread
                strategy_type = StrategyType.BEAR_PUT_SPREAD
                name = f"Bear Put {leg1.strike}/{leg2.strike}"
            else:
                # Sell lower put, buy higher put = Bull Put Spread
                strategy_type = StrategyType.BULL_PUT_SPREAD
                name = f"Bull Put {leg1.strike}/{leg2.strike}"

        return StrategyResult(
            strategy_type=strategy_type,
            strategy_id=self._generate_id([leg1, leg2]),
            confidence=0.90,
            legs=[leg1, leg2],
            name=name
        )

    def _classify_straddle(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Straddle classification."""

        both_buy = leg1.side == "BUY" and leg2.side == "BUY"
        both_sell = leg1.side == "SELL" and leg2.side == "SELL"

        if both_buy:
            return StrategyResult(
                strategy_type=StrategyType.LONG_STRADDLE,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.90,
                legs=[leg1, leg2],
                name=f"Long Straddle {leg1.strike}"
            )
        elif both_sell:
            return StrategyResult(
                strategy_type=StrategyType.SHORT_STRADDLE,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.90,
                legs=[leg1, leg2],
                name=f"Short Straddle {leg1.strike}"
            )

        return self._unknown([leg1, leg2], 0.4)

    def _classify_strangle(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Strangle classification."""

        both_buy = leg1.side == "BUY" and leg2.side == "BUY"
        both_sell = leg1.side == "SELL" and leg2.side == "SELL"

        # Ensure put has lower strike
        put_leg = leg1 if leg1.option_right == "PUT" else leg2
        call_leg = leg2 if leg1.option_right == "PUT" else leg1

        if both_buy:
            return StrategyResult(
                strategy_type=StrategyType.LONG_STRANGLE,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.85,
                legs=[put_leg, call_leg],
                name=f"Long Strangle {put_leg.strike}/{call_leg.strike}"
            )
        elif both_sell:
            return StrategyResult(
                strategy_type=StrategyType.SHORT_STRANGLE,
                strategy_id=self._generate_id([leg1, leg2]),
                confidence=0.85,
                legs=[put_leg, call_leg],
                name=f"Short Strangle {put_leg.strike}/{call_leg.strike}"
            )

        return self._unknown([leg1, leg2], 0.4)

    def _classify_three_leg(self, options: List[LegInfo]) -> StrategyResult:
        """Three-leg strategy classification (butterfly components, etc.)."""
        # Most 3-leg strategies are partial butterflies or incomplete spreads
        return self._unknown(options, 0.3)

    def _classify_four_leg(self, options: List[LegInfo]) -> StrategyResult:
        """Iron Condor / Iron Butterfly classification."""

        puts = sorted(
            [o for o in options if o.option_right == "PUT"],
            key=lambda x: x.strike or 0
        )
        calls = sorted(
            [o for o in options if o.option_right == "CALL"],
            key=lambda x: x.strike or 0
        )

        if len(puts) != 2 or len(calls) != 2:
            return self._unknown(options, 0.3)

        # Iron Condor: OTM put spread + OTM call spread
        # Structure: Buy low put, Sell mid-low put, Sell mid-high call, Buy high call
        if (puts[0].side == "BUY" and puts[1].side == "SELL" and
            calls[0].side == "SELL" and calls[1].side == "BUY"):

            return StrategyResult(
                strategy_type=StrategyType.IRON_CONDOR,
                strategy_id=self._generate_id(options),
                confidence=0.85,
                legs=options,
                name=f"Iron Condor {puts[0].strike}/{puts[1].strike}/{calls[0].strike}/{calls[1].strike}"
            )

        # Short Iron Condor (reverse)
        if (puts[0].side == "SELL" and puts[1].side == "BUY" and
            calls[0].side == "BUY" and calls[1].side == "SELL"):

            return StrategyResult(
                strategy_type=StrategyType.IRON_CONDOR,
                strategy_id=self._generate_id(options),
                confidence=0.85,
                legs=options,
                name=f"Short Iron Condor {puts[0].strike}/{puts[1].strike}/{calls[0].strike}/{calls[1].strike}",
                metadata={"direction": "short"}
            )

        # Iron Butterfly: ATM straddle + OTM wings
        if puts[1].strike == calls[0].strike:  # Middle strikes match
            return StrategyResult(
                strategy_type=StrategyType.IRON_BUTTERFLY,
                strategy_id=self._generate_id(options),
                confidence=0.80,
                legs=options,
                name=f"Iron Butterfly @ {puts[1].strike}"
            )

        return self._unknown(options, 0.3)

    def _unknown(self, legs: List[LegInfo], confidence: float) -> StrategyResult:
        """Fallback for unrecognized patterns."""
        return StrategyResult(
            strategy_type=StrategyType.UNKNOWN,
            strategy_id=self._generate_id(legs),
            confidence=confidence,
            legs=legs,
            name="Unknown/Custom Strategy"
        )

    @staticmethod
    def _generate_id(legs: List[LegInfo]) -> str:
        """Generate deterministic strategy ID from legs."""
        leg_strs = sorted([
            f"{leg.symbol}|{leg.side}|{leg.strike}|{leg.expiry}|{leg.option_right}"
            for leg in legs
        ])
        return hashlib.sha256("|".join(leg_strs).encode()).hexdigest()[:16]


def group_trades_by_strategy(
    trades: List[LegInfo],
    window_seconds: int = 5
) -> List[List[LegInfo]]:
    """
    Group trades into potential strategy legs based on:
    1. Same underlying
    2. Trade times within window_seconds of each other

    Args:
        trades: List of LegInfo objects
        window_seconds: Time window for grouping (default 5 seconds)

    Returns:
        List of trade groups, each group is a potential strategy
    """
    if not trades:
        return []

    # Filter trades with valid trade_time
    trades_with_time = [t for t in trades if t.trade_time is not None]
    trades_without_time = [t for t in trades if t.trade_time is None]

    if not trades_with_time:
        # If no trades have time, group all by underlying
        return _group_by_underlying_only(trades)

    # Sort by (underlying, trade_time)
    sorted_trades = sorted(
        trades_with_time,
        key=lambda t: (t.underlying, t.trade_time)
    )

    groups: List[List[LegInfo]] = []
    current_group: List[LegInfo] = [sorted_trades[0]]

    for trade in sorted_trades[1:]:
        prev = current_group[-1]

        # Same underlying and within time window
        time_diff = (trade.trade_time - prev.trade_time).total_seconds()
        if trade.underlying == prev.underlying and time_diff <= window_seconds:
            current_group.append(trade)
        else:
            groups.append(current_group)
            current_group = [trade]

    groups.append(current_group)

    # Handle trades without time (put each in its own group)
    for trade in trades_without_time:
        groups.append([trade])

    return groups


def _group_by_underlying_only(trades: List[LegInfo]) -> List[List[LegInfo]]:
    """Fallback grouping by underlying only (when no timestamps available)."""
    by_underlying: Dict[str, List[LegInfo]] = {}
    for trade in trades:
        if trade.underlying not in by_underlying:
            by_underlying[trade.underlying] = []
        by_underlying[trade.underlying].append(trade)
    return list(by_underlying.values())
