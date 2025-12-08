"""
Strategy classifier for grouping and identifying option strategies.

Uses time-window clustering and rule-based detection to classify trades
into strategy types (spreads, straddles, condors, etc.).
"""

from __future__ import annotations
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class StrategyType(Enum):
    """Strategy pattern types (what kind of strategy)."""

    # Single-leg stock
    LONG_STOCK = "long_stock"
    SHORT_STOCK = "short_stock"

    # Single-leg options
    LONG_CALL = "long_call"
    SHORT_CALL = "short_call"
    LONG_PUT = "long_put"
    SHORT_PUT = "short_put"

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
    BUTTERFLY = "butterfly"

    # Fallback
    UNKNOWN = "unknown"


class StrategyOutcome(Enum):
    """Strategy outcome (what happened to the strategy)."""
    OPEN = "open"                # Position still open
    TAKE_PROFIT = "take_profit"  # Closed with profit
    STOP_LOSS = "stop_loss"      # Closed with loss
    CLOSE_FLAT = "close_flat"    # Closed at breakeven


@dataclass
class LegInfo:
    """Information about a single leg of a strategy."""

    order_uid: str
    trade_uid: Optional[str]
    symbol: str
    underlying: str
    instrument_type: str  # STOCK, OPTION
    side: str  # BUY, SELL
    qty: float
    strike: Optional[float]
    expiry: Optional[str]  # YYYYMMDD or date
    option_right: Optional[str]  # CALL, PUT
    trade_time: datetime
    broker: str
    account_id: str
    # Fields for position effect and PnL
    position_effect: Optional[str] = None  # OPEN, CLOSE
    realized_pnl: Optional[float] = None   # Calculated from entry vs exit
    updated_time: Optional[datetime] = None  # Order's updated_time
    price: Optional[float] = None  # Execution price for P&L calculation
    # Matched entry info for closing trades (populated by match_trades)
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None


class TradeDuration(Enum):
    """Trade duration classification."""
    INTRADAY = "intraday"  # Open and close on same day
    SWING = "swing"        # Open and close on different days
    UNKNOWN = "unknown"    # Cannot determine


@dataclass
class StrategyResult:
    """Result of strategy classification."""

    # Pattern classification
    strategy_type: StrategyType          # What kind of strategy (long_call, bull_put_spread, etc.)
    strategy_id: str                     # Hash for grouping
    strategy_name: str                   # Human readable name

    # Outcome
    strategy_outcome: StrategyOutcome    # Result (open, take_profit, stop_loss, close_flat)
    is_closed: bool                      # Whether strategy is fully closed

    # Legs and orders
    legs: List[LegInfo]
    involved_orders: List[Dict[str, Any]]  # All orders involved (open + close)
    opening_order_uid: str               # The opening order ID (primary identifier)

    # Timing
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    trade_duration: Optional[TradeDuration] = None

    # Metadata
    confidence: float = 0.0
    calculated_pnl: Optional[float] = None
    classify_version: str = "v1"


class StrategyClassifier:
    """
    Rule-based strategy classifier using time-window clustering.

    Groups trades by underlying and time proximity, then applies
    pattern matching rules to identify strategy types.
    """

    VERSION = "v1"
    DEFAULT_TIME_WINDOW_SECONDS = 5

    def __init__(self, time_window_seconds: int = DEFAULT_TIME_WINDOW_SECONDS):
        """
        Initialize classifier.

        Args:
            time_window_seconds: Trades within this window are considered
                                 part of the same strategy.
        """
        self.time_window_seconds = time_window_seconds

    def classify(self, legs: List[LegInfo]) -> StrategyResult:
        """
        Classify a group of legs into a strategy type.

        Args:
            legs: List of LegInfo objects representing strategy legs.

        Returns:
            StrategyResult with strategy type, confidence, and metadata.
        """
        if len(legs) == 0:
            return self._unknown(legs, 0.0)

        if len(legs) == 1:
            return self._classify_single(legs[0])

        # Separate by instrument type
        options = [l for l in legs if l.instrument_type == "OPTION"]
        stocks = [l for l in legs if l.instrument_type == "STOCK"]

        # Stock + Option combo
        if len(stocks) == 1 and len(options) == 1:
            return self._classify_stock_option(stocks[0], options[0])

        # All options
        if len(stocks) == 0 and len(options) >= 2:
            return self._classify_option_spread(options)

        # Mixed or unrecognized
        return self._unknown(legs, 0.2)

    def classify_batch(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[StrategyResult]:
        """
        Classify multiple trades by grouping them first.

        This method:
        1. Converts trades to LegInfo objects
        2. Matches opening and closing trades to calculate P&L
        3. Determines trade duration (INTRADAY vs SWING)
        4. Groups and classifies strategies

        Args:
            trades: List of normalized trade dicts.

        Returns:
            List of StrategyResult for each identified strategy.
        """
        legs = [self._trade_to_leg(t) for t in trades]
        legs = [l for l in legs if l is not None]

        if not legs:
            return []

        # Match opening and closing trades to calculate P&L and duration
        legs = self._match_trades_for_pnl(legs)

        groups = self.group_trades_by_strategy(legs)
        results = []
        for g in groups:
            result = self.classify(g)
            # Determine trade duration for the group
            result.trade_duration = self._determine_group_duration(g)
            results.append(result)
        return results

    def _match_trades_for_pnl(self, legs: List[LegInfo]) -> List[LegInfo]:
        """
        Match opening and closing trades to calculate P&L.

        For each closing trade, finds matching opening trades on the same
        underlying/symbol and calculates realized P&L.

        Args:
            legs: List of LegInfo objects.

        Returns:
            Updated list with calculated P&L for closing trades.
        """
        # Group by underlying for matching
        by_underlying: Dict[str, List[LegInfo]] = {}
        for leg in legs:
            key = (leg.underlying, leg.symbol, leg.strike, leg.expiry, leg.option_right)
            if key not in by_underlying:
                by_underlying[key] = []
            by_underlying[key].append(leg)

        # For each group, match OPEN and CLOSE trades
        for key, group in by_underlying.items():
            opens = sorted(
                [l for l in group if l.position_effect == "OPEN"],
                key=lambda x: x.trade_time
            )
            closes = sorted(
                [l for l in group if l.position_effect == "CLOSE"],
                key=lambda x: x.trade_time
            )

            # Simple FIFO matching
            for close_leg in closes:
                if not opens:
                    continue

                # Find matching open (by side - BUY close matches SELL open, etc.)
                close_is_buy = close_leg.side == "BUY"
                matching_opens = [o for o in opens if (o.side == "SELL") == close_is_buy]

                if matching_opens:
                    open_leg = matching_opens[0]
                    opens.remove(open_leg)

                    # Calculate P&L
                    if close_leg.price is not None and open_leg.price is not None:
                        if close_is_buy:
                            # Buy to close a short position
                            # P&L = (entry_price - exit_price) * qty
                            pnl = (open_leg.price - close_leg.price) * close_leg.qty
                        else:
                            # Sell to close a long position
                            # P&L = (exit_price - entry_price) * qty
                            pnl = (close_leg.price - open_leg.price) * close_leg.qty

                        # For options, multiply by contract multiplier (typically 100)
                        if close_leg.instrument_type == "OPTION":
                            pnl *= 100

                        close_leg.realized_pnl = pnl
                        close_leg.entry_price = open_leg.price
                        close_leg.entry_time = open_leg.trade_time

        return legs

    def _determine_group_duration(self, legs: List[LegInfo]) -> TradeDuration:
        """
        Determine trade duration for a group of legs.

        If any leg has matched entry/exit on same day -> INTRADAY
        If legs have matched entry/exit on different days -> SWING
        Otherwise -> UNKNOWN

        Args:
            legs: List of LegInfo in a strategy group.

        Returns:
            TradeDuration enum value.
        """
        for leg in legs:
            if leg.entry_time is not None:
                entry_date = leg.entry_time.date()
                exit_date = leg.trade_time.date()

                if entry_date == exit_date:
                    return TradeDuration.INTRADAY
                else:
                    return TradeDuration.SWING

        # Check if all legs are from the same day (potential intraday)
        if len(legs) > 1:
            dates = {l.trade_time.date() for l in legs}
            if len(dates) == 1:
                # All trades on same day, could be intraday
                opens = [l for l in legs if l.position_effect == "OPEN"]
                closes = [l for l in legs if l.position_effect == "CLOSE"]
                if opens and closes:
                    return TradeDuration.INTRADAY

        return TradeDuration.UNKNOWN

    def group_trades_by_strategy(
        self,
        legs: List[LegInfo],
    ) -> List[List[LegInfo]]:
        """
        Group trades into potential strategy legs based on:
        1. Same underlying
        2. Trade times within time_window_seconds of each other

        Args:
            legs: List of LegInfo objects.

        Returns:
            List of leg groups, each potentially forming a strategy.
        """
        if not legs:
            return []

        # Sort by underlying, then by time
        legs = sorted(legs, key=lambda t: (t.underlying, t.trade_time))

        groups = []
        current_group = [legs[0]]

        for leg in legs[1:]:
            prev = current_group[-1]

            # Check if same underlying and within time window
            time_diff = abs((leg.trade_time - prev.trade_time).total_seconds())

            if (
                leg.underlying == prev.underlying
                and time_diff <= self.time_window_seconds
            ):
                current_group.append(leg)
            else:
                groups.append(current_group)
                current_group = [leg]

        groups.append(current_group)
        return groups

    def _trade_to_leg(self, trade: Dict[str, Any]) -> Optional[LegInfo]:
        """Convert a normalized trade dict to LegInfo."""
        try:
            trade_time = trade.get("trade_time_utc")
            if isinstance(trade_time, str):
                trade_time = datetime.fromisoformat(trade_time.replace("Z", "+00:00"))
            if not trade_time:
                trade_time = datetime.utcnow()

            updated_time = trade.get("update_time_utc") or trade.get("updated_time")
            if isinstance(updated_time, str):
                updated_time = datetime.fromisoformat(updated_time.replace("Z", "+00:00"))

            expiry = trade.get("expiry")
            if hasattr(expiry, "strftime"):
                expiry = expiry.strftime("%Y%m%d")

            # Convert Decimal to float for strike (PostgreSQL returns Decimal)
            strike = trade.get("strike")
            if strike is not None:
                strike = float(strike)

            # Get execution price for P&L calculation
            price = trade.get("price")
            if price is not None:
                price = float(price)

            # Get realized PnL if available (will be recalculated if we can match trades)
            realized_pnl = trade.get("realized_pnl")
            if realized_pnl is not None:
                realized_pnl = float(realized_pnl)

            # Determine position effect (OPEN or CLOSE)
            position_effect = self._determine_position_effect(trade)

            return LegInfo(
                order_uid=trade.get("order_uid", ""),
                trade_uid=trade.get("trade_uid"),
                symbol=trade.get("symbol", ""),
                underlying=trade.get("underlying") or trade.get("symbol", ""),
                instrument_type=trade.get("instrument_type", "STOCK"),
                side=trade.get("side", ""),
                qty=float(trade.get("qty", 0)),
                strike=strike,
                expiry=expiry,
                option_right=trade.get("option_right"),
                trade_time=trade_time,
                broker=trade.get("broker", ""),
                account_id=trade.get("account_id", ""),
                position_effect=position_effect,
                realized_pnl=realized_pnl,
                updated_time=updated_time,
                price=price,
            )
        except Exception:
            return None

    def _determine_position_effect(self, trade: Dict[str, Any]) -> str:
        """
        Determine if trade is opening or closing a position.

        Uses trd_side from Futu or position_effect field if available.
        Heuristic: BUY_BACK/SELL_SHORT are closing, BUY/SELL are opening.
        """
        # Check explicit position_effect field
        effect = trade.get("position_effect")
        if effect:
            return effect.upper()

        # Check Futu-specific trd_side values
        trd_side = str(trade.get("trd_side", "")).upper()
        if trd_side in ("BUY_BACK", "SELL_SHORT"):
            return "CLOSE"

        # Check for closing indicators in remark/notes
        remark = str(trade.get("remark", "")).lower()
        if "close" in remark or "closing" in remark:
            return "CLOSE"

        # Default to OPEN for regular BUY/SELL
        return "OPEN"

    def _classify_single(self, leg: LegInfo) -> StrategyResult:
        """Classify a single-leg position."""
        # Determine pattern type based on instrument and side
        if leg.instrument_type == "OPTION":
            if leg.side == "BUY":
                if leg.option_right == "CALL":
                    strategy_type = StrategyType.LONG_CALL
                else:
                    strategy_type = StrategyType.LONG_PUT
            else:
                if leg.option_right == "CALL":
                    strategy_type = StrategyType.SHORT_CALL
                else:
                    strategy_type = StrategyType.SHORT_PUT
        else:  # STOCK
            strategy_type = StrategyType.LONG_STOCK if leg.side == "BUY" else StrategyType.SHORT_STOCK

        # Build human-readable name
        name = self._build_strategy_name(strategy_type, leg)

        # Determine outcome and closed status
        is_closed = leg.position_effect == "CLOSE"
        if is_closed:
            outcome = self._determine_outcome(leg.realized_pnl)
        else:
            outcome = StrategyOutcome.OPEN

        # Build involved orders list
        involved_orders = [{
            "order_uid": leg.order_uid,
            "side": leg.side,
            "position_effect": leg.position_effect,
            "price": leg.price,
            "qty": leg.qty,
        }]

        return StrategyResult(
            strategy_type=strategy_type,
            strategy_id=self._generate_id([leg]),
            strategy_name=name,
            strategy_outcome=outcome,
            is_closed=is_closed,
            legs=[leg],
            involved_orders=involved_orders,
            opening_order_uid=leg.order_uid,
            open_time=leg.entry_time or leg.trade_time,
            close_time=leg.trade_time if is_closed else None,
            confidence=0.95,
            calculated_pnl=leg.realized_pnl,
        )

    def _build_strategy_name(self, strategy_type: StrategyType, leg: LegInfo) -> str:
        """Build human-readable strategy name."""
        type_names = {
            StrategyType.LONG_STOCK: "Long Stock",
            StrategyType.SHORT_STOCK: "Short Stock",
            StrategyType.LONG_CALL: "Long Call",
            StrategyType.SHORT_CALL: "Short Call",
            StrategyType.LONG_PUT: "Long Put",
            StrategyType.SHORT_PUT: "Short Put",
        }
        base_name = type_names.get(strategy_type, strategy_type.value.replace("_", " ").title())

        if leg.instrument_type == "OPTION":
            return f"{base_name} {leg.underlying} {leg.strike} {leg.expiry}"
        else:
            return f"{base_name} {leg.symbol}"

    def _determine_outcome(self, pnl: Optional[float]) -> StrategyOutcome:
        """Determine strategy outcome based on P&L."""
        if pnl is None:
            return StrategyOutcome.OPEN  # Can't determine, assume still open
        if pnl > 0:
            return StrategyOutcome.TAKE_PROFIT
        elif pnl < 0:
            return StrategyOutcome.STOP_LOSS
        else:
            return StrategyOutcome.CLOSE_FLAT

    def _make_result(
        self,
        strategy_type: StrategyType,
        legs: List[LegInfo],
        name: str,
        confidence: float = 0.90,
    ) -> StrategyResult:
        """Helper to create a StrategyResult with common logic."""
        # Find opening legs and closing legs
        opens = [l for l in legs if l.position_effect == "OPEN"]
        closes = [l for l in legs if l.position_effect == "CLOSE"]

        # Determine if strategy is closed
        is_closed = len(closes) > 0 and len(opens) > 0

        # Calculate total P&L from all legs
        total_pnl = sum(l.realized_pnl or 0 for l in legs if l.realized_pnl)
        if not any(l.realized_pnl for l in legs):
            total_pnl = None

        # Determine outcome
        if is_closed:
            outcome = self._determine_outcome(total_pnl)
        else:
            outcome = StrategyOutcome.OPEN

        # Build involved orders list
        involved_orders = [{
            "order_uid": l.order_uid,
            "side": l.side,
            "position_effect": l.position_effect,
            "price": l.price,
            "qty": l.qty,
        } for l in legs]

        # Use first opening order as the primary identifier, or first leg
        opening_order = opens[0] if opens else legs[0]

        # Get timing
        open_time = min((l.trade_time for l in opens), default=legs[0].trade_time) if opens else legs[0].trade_time
        close_time = max((l.trade_time for l in closes), default=None) if closes else None

        return StrategyResult(
            strategy_type=strategy_type,
            strategy_id=self._generate_id(legs),
            strategy_name=name,
            strategy_outcome=outcome,
            is_closed=is_closed,
            legs=legs,
            involved_orders=involved_orders,
            opening_order_uid=opening_order.order_uid,
            open_time=open_time,
            close_time=close_time,
            confidence=confidence,
            calculated_pnl=total_pnl,
        )

    def _classify_stock_option(
        self,
        stock: LegInfo,
        opt: LegInfo,
    ) -> StrategyResult:
        """Classify stock + option combo."""
        legs = [stock, opt]

        # Covered call: long stock + short call
        if stock.side == "BUY" and opt.side == "SELL" and opt.option_right == "CALL":
            return self._make_result(
                StrategyType.COVERED_CALL, legs,
                f"Covered Call {stock.underlying} {opt.strike}",
            )

        # Protective put: long stock + long put
        if stock.side == "BUY" and opt.side == "BUY" and opt.option_right == "PUT":
            return self._make_result(
                StrategyType.PROTECTIVE_PUT, legs,
                f"Protective Put {stock.underlying} {opt.strike}",
            )

        # Cash-secured put: short put (stock is synthetic or for assignment)
        if opt.side == "SELL" and opt.option_right == "PUT":
            return self._make_result(
                StrategyType.CASH_SECURED_PUT, legs,
                f"Cash-Secured Put {opt.underlying} {opt.strike}",
                confidence=0.80,
            )

        return self._unknown(legs, 0.3)

    def _classify_option_spread(self, options: List[LegInfo]) -> StrategyResult:
        """Classify multi-leg option strategies."""
        # Sort by expiry, then strike
        options = sorted(
            options,
            key=lambda x: (x.expiry or "", x.strike or 0),
        )

        if len(options) == 2:
            return self._classify_two_leg(options[0], options[1])

        if len(options) == 3:
            return self._classify_three_leg(options)

        if len(options) == 4:
            return self._classify_four_leg(options)

        return self._unknown(options, 0.2)

    def _classify_two_leg(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Classify two-leg option strategies."""
        legs = [leg1, leg2]
        same_expiry = leg1.expiry == leg2.expiry
        same_right = leg1.option_right == leg2.option_right
        same_strike = leg1.strike == leg2.strike

        # Vertical spread: same expiry, same right, different strikes
        if same_expiry and same_right and not same_strike:
            return self._classify_vertical(leg1, leg2)

        # Straddle: same expiry, same strike, different rights
        if same_expiry and same_strike and not same_right:
            return self._classify_straddle(leg1, leg2)

        # Strangle: same expiry, different strikes, different rights
        if same_expiry and not same_strike and not same_right:
            return self._classify_strangle(leg1, leg2)

        # Calendar: different expiry, same strike, same right
        if not same_expiry and same_strike and same_right:
            return self._make_result(
                StrategyType.CALENDAR_SPREAD, legs,
                f"Calendar {leg1.option_right} {leg1.underlying} {leg1.strike}",
                confidence=0.80,
            )

        # Diagonal: different expiry, different strike, same right
        if not same_expiry and not same_strike and same_right:
            return self._make_result(
                StrategyType.DIAGONAL_SPREAD, legs,
                f"Diagonal {leg1.option_right} {leg1.underlying}",
                confidence=0.75,
            )

        return self._unknown(legs, 0.3)

    def _classify_vertical(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Classify vertical spreads."""
        # Ensure leg1 has lower strike
        if (leg1.strike or 0) > (leg2.strike or 0):
            leg1, leg2 = leg2, leg1

        legs = [leg1, leg2]
        is_call = leg1.option_right == "CALL"
        buy_lower = leg1.side == "BUY"

        if is_call:
            if buy_lower:
                strategy_type = StrategyType.BULL_CALL_SPREAD
                name = f"Bull Call Spread {leg1.underlying} {leg1.strike}/{leg2.strike}"
            else:
                strategy_type = StrategyType.BEAR_CALL_SPREAD
                name = f"Bear Call Spread {leg1.underlying} {leg1.strike}/{leg2.strike}"
        else:
            if buy_lower:
                strategy_type = StrategyType.BEAR_PUT_SPREAD
                name = f"Bear Put Spread {leg1.underlying} {leg1.strike}/{leg2.strike}"
            else:
                strategy_type = StrategyType.BULL_PUT_SPREAD
                name = f"Bull Put Spread {leg1.underlying} {leg1.strike}/{leg2.strike}"

        return self._make_result(strategy_type, legs, name)

    def _classify_straddle(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Classify straddles."""
        legs = [leg1, leg2]
        both_buy = leg1.side == "BUY" and leg2.side == "BUY"
        both_sell = leg1.side == "SELL" and leg2.side == "SELL"

        if both_buy:
            return self._make_result(
                StrategyType.LONG_STRADDLE, legs,
                f"Long Straddle {leg1.underlying} {leg1.strike}",
            )
        elif both_sell:
            return self._make_result(
                StrategyType.SHORT_STRADDLE, legs,
                f"Short Straddle {leg1.underlying} {leg1.strike}",
            )

        return self._unknown(legs, 0.4)

    def _classify_strangle(self, leg1: LegInfo, leg2: LegInfo) -> StrategyResult:
        """Classify strangles."""
        both_buy = leg1.side == "BUY" and leg2.side == "BUY"
        both_sell = leg1.side == "SELL" and leg2.side == "SELL"

        # Identify put and call legs
        put_leg = leg1 if leg1.option_right == "PUT" else leg2
        call_leg = leg2 if leg1.option_right == "PUT" else leg1
        legs = [put_leg, call_leg]

        if both_buy:
            return self._make_result(
                StrategyType.LONG_STRANGLE, legs,
                f"Long Strangle {put_leg.underlying} {put_leg.strike}/{call_leg.strike}",
                confidence=0.85,
            )
        elif both_sell:
            return self._make_result(
                StrategyType.SHORT_STRANGLE, legs,
                f"Short Strangle {put_leg.underlying} {put_leg.strike}/{call_leg.strike}",
                confidence=0.85,
            )

        return self._unknown(legs, 0.4)

    def _classify_three_leg(self, options: List[LegInfo]) -> StrategyResult:
        """Classify three-leg strategies (butterflies)."""
        # Sort by strike
        options = sorted(options, key=lambda x: x.strike or 0)

        # Check for butterfly pattern: low strike, 2x middle strike, high strike
        if len(options) == 3:
            low, mid, high = options[0], options[1], options[2]

            # All same right and expiry for butterfly
            same_right = low.option_right == mid.option_right == high.option_right
            same_expiry = low.expiry == mid.expiry == high.expiry

            if same_right and same_expiry:
                # Check for butterfly structure
                strikes = [low.strike, mid.strike, high.strike]
                if strikes[1] is not None:
                    # Check if middle is equidistant
                    if abs((strikes[2] - strikes[1]) - (strikes[1] - strikes[0])) < 0.01:
                        return self._make_result(
                            StrategyType.BUTTERFLY, options,
                            f"Butterfly {low.underlying} {low.strike}/{mid.strike}/{high.strike}",
                            confidence=0.80,
                        )

        return self._unknown(options, 0.3)

    def _classify_four_leg(self, options: List[LegInfo]) -> StrategyResult:
        """Classify four-leg strategies (iron condors, iron butterflies)."""
        puts = sorted(
            [o for o in options if o.option_right == "PUT"],
            key=lambda x: x.strike or 0,
        )
        calls = sorted(
            [o for o in options if o.option_right == "CALL"],
            key=lambda x: x.strike or 0,
        )

        if len(puts) != 2 or len(calls) != 2:
            return self._unknown(options, 0.3)

        underlying = options[0].underlying

        # Iron Condor: buy lower put, sell higher put, sell lower call, buy higher call
        # Pattern: +P1 -P2 -C1 +C2 where P1 < P2 < C1 < C2
        if (
            puts[0].side == "BUY"
            and puts[1].side == "SELL"
            and calls[0].side == "SELL"
            and calls[1].side == "BUY"
        ):
            return self._make_result(
                StrategyType.IRON_CONDOR, options,
                f"Iron Condor {underlying} {puts[0].strike}/{puts[1].strike}/{calls[0].strike}/{calls[1].strike}",
                confidence=0.85,
            )

        # Iron Butterfly: put and call sold at same strike
        if puts[1].strike == calls[0].strike:
            return self._make_result(
                StrategyType.IRON_BUTTERFLY, options,
                f"Iron Butterfly {underlying} @ {puts[1].strike}",
                confidence=0.80,
            )

        return self._unknown(options, 0.3)

    def _unknown(self, legs: List[LegInfo], confidence: float) -> StrategyResult:
        """Return unknown strategy result."""
        return self._make_result(
            StrategyType.UNKNOWN, legs,
            "Unknown/Custom Strategy",
            confidence=confidence,
        )

    def _generate_id(self, legs: List[LegInfo]) -> str:
        """Generate deterministic strategy ID from legs."""
        leg_strs = sorted([
            f"{l.symbol}|{l.side}|{l.strike}|{l.expiry}|{l.option_right}"
            for l in legs
        ])
        return hashlib.sha256("|".join(leg_strs).encode()).hexdigest()[:16]

    def result_to_mapping(
        self,
        result: StrategyResult,
    ) -> Dict[str, Any]:
        """
        Convert StrategyResult to a single apex_strategy_analysis record.

        Each strategy produces ONE record, keyed by the opening order's order_uid.

        Args:
            result: Classification result.

        Returns:
            Mapping dict for database insertion.
        """
        # Convert legs to JSON-serializable format
        legs_data = [
            {
                "order_uid": l.order_uid,
                "trade_uid": l.trade_uid,
                "symbol": l.symbol,
                "side": l.side,
                "strike": float(l.strike) if l.strike is not None else None,
                "expiry": l.expiry,
                "option_right": l.option_right,
                "position_effect": l.position_effect,
                "price": float(l.price) if l.price is not None else None,
                "qty": float(l.qty) if l.qty else None,
                "realized_pnl": float(l.realized_pnl) if l.realized_pnl is not None else None,
            }
            for l in result.legs
        ]

        # Convert involved_orders to JSON-serializable format
        involved_orders = [
            {
                "order_uid": o.get("order_uid"),
                "side": o.get("side"),
                "position_effect": o.get("position_effect"),
                "price": float(o.get("price")) if o.get("price") is not None else None,
                "qty": float(o.get("qty")) if o.get("qty") is not None else None,
            }
            for o in result.involved_orders
        ]

        # Get broker/account from first leg
        first_leg = result.legs[0] if result.legs else None
        broker = first_leg.broker if first_leg else ""
        account_id = first_leg.account_id if first_leg else ""

        # Get latest updated_time from all legs
        updated_times = [l.updated_time for l in result.legs if l.updated_time]
        updated_time = max(updated_times) if updated_times else None

        return {
            "broker": broker,
            "account_id": account_id,
            "order_uid": result.opening_order_uid,
            "strategy_id": result.strategy_id,
            "strategy_type": result.strategy_type.value,
            "strategy_name": result.strategy_name,
            "strategy_outcome": result.strategy_outcome.value,
            "is_closed": result.is_closed,
            "trade_duration": result.trade_duration.value if result.trade_duration else None,
            "involved_orders": involved_orders,
            "confidence": float(result.confidence) if result.confidence is not None else None,
            "legs": legs_data,
            "open_time": result.open_time,
            "close_time": result.close_time,
            "updated_time": updated_time,
            "classify_version": result.classify_version,
        }

    # Backward compatibility alias
    def result_to_mappings(self, result: StrategyResult) -> List[Dict[str, Any]]:
        """Backward compatible wrapper - returns single-item list."""
        return [self.result_to_mapping(result)]
