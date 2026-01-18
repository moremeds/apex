"""
Strategy Detector - Identify multi-leg option strategies.

Automatically detects common option strategies:
- Vertical Spreads (Call/Put Spreads)
- Diagonal Spreads
- Calendar Spreads
- Iron Condors
- Covered Calls/Puts
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ...models.position import AssetType, Position
from ...utils.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class DetectedStrategy:
    """
    Represents a detected multi-leg strategy.
    """

    strategy_type: str  # "VERTICAL_SPREAD", "DIAGONAL", "IRON_CONDOR", etc.
    underlying: str
    positions: List[Position]

    # Strategy metadata
    net_quantity: float = 0.0  # Net position quantity
    is_credit: bool = False  # Credit spread vs debit spread
    max_risk: Optional[float] = None  # Maximum loss
    max_profit: Optional[float] = None  # Maximum profit

    # Greeks (aggregated)
    net_delta: float = 0.0
    net_gamma: float = 0.0
    net_vega: float = 0.0

    metadata: Dict = field(default_factory=dict)

    def strategy_id(self) -> str:
        """Generate unique strategy ID."""
        symbols = "_".join(sorted(pos.symbol for pos in self.positions))
        return f"{self.strategy_type}:{self.underlying}:{symbols}"


class StrategyDetector:
    """
    Detects multi-leg option strategies from positions.

    Supports:
    - Vertical spreads (same expiry, different strikes)
    - Diagonal spreads (different expiries, same type)
    - Calendar spreads (different expiries, same strike)
    - Iron condors (4-leg structure)
    - Covered calls/puts (stock + option)
    """

    def __init__(self) -> None:
        """Initialize strategy detector."""
        self._detection_stats: Dict[str, int] = {
            "total_positions": 0,
            "strategies_detected": 0,
            "vertical_spreads": 0,
            "diagonals": 0,
            "calendars": 0,
            "iron_condors": 0,
            "covered_positions": 0,
        }

    def detect(self, positions: List[Position]) -> List[DetectedStrategy]:
        """
        Detect strategies from list of positions.

        Args:
            positions: List of all positions

        Returns:
            List of detected strategies
        """
        self._detection_stats["total_positions"] = len(positions)
        strategies = []

        # Group positions by underlying
        by_underlying = self._group_by_underlying(positions)

        for underlying, underlying_positions in by_underlying.items():
            # Separate stocks and options
            stock_positions = [p for p in underlying_positions if p.asset_type == AssetType.STOCK]
            option_positions = [p for p in underlying_positions if p.asset_type == AssetType.OPTION]

            # Detect covered calls/puts
            covered_strategies = self._detect_covered_positions(stock_positions, option_positions)
            strategies.extend(covered_strategies)

            # Detect option-only strategies
            option_strategies = self._detect_option_strategies(option_positions, underlying)
            strategies.extend(option_strategies)

        self._detection_stats["strategies_detected"] = len(strategies)
        logger.info(
            f"Detected {len(strategies)} strategies from {len(positions)} positions: "
            f"{self._detection_stats}"
        )

        return strategies

    def _group_by_underlying(self, positions: List[Position]) -> Dict[str, List[Position]]:
        """Group positions by underlying symbol."""
        by_underlying = defaultdict(list)
        for pos in positions:
            by_underlying[pos.underlying].append(pos)
        return dict(by_underlying)

    def _detect_covered_positions(
        self,
        stock_positions: List[Position],
        option_positions: List[Position],
    ) -> List[DetectedStrategy]:
        """
        Detect covered calls and covered puts.

        Covered call: Long stock + Short call
        Covered put: Short stock + Short put
        """
        strategies = []

        for stock in stock_positions:
            underlying = stock.underlying

            # Covered call: long stock + short call
            if stock.quantity > 0:
                short_calls = [
                    opt for opt in option_positions if opt.right == "C" and opt.quantity < 0
                ]

                for call in short_calls:
                    # Check if quantities match (roughly)
                    stock_contracts = stock.quantity / 100  # Convert shares to contracts
                    call_contracts = abs(call.quantity)

                    if abs(stock_contracts - call_contracts) < 0.1:  # Allow small mismatch
                        strategies.append(
                            DetectedStrategy(
                                strategy_type="COVERED_CALL",
                                underlying=underlying,
                                positions=[stock, call],
                                is_credit=True,
                                metadata={
                                    "stock_qty": stock.quantity,
                                    "call_strike": call.strike,
                                    "call_expiry": call.expiry,
                                },
                            )
                        )
                        self._detection_stats["covered_positions"] += 1

            # Covered put: short stock + short put
            elif stock.quantity < 0:
                short_puts = [
                    opt for opt in option_positions if opt.right == "P" and opt.quantity < 0
                ]

                for put in short_puts:
                    stock_contracts = abs(stock.quantity) / 100
                    put_contracts = abs(put.quantity)

                    if abs(stock_contracts - put_contracts) < 0.1:
                        strategies.append(
                            DetectedStrategy(
                                strategy_type="COVERED_PUT",
                                underlying=underlying,
                                positions=[stock, put],
                                is_credit=True,
                                metadata={
                                    "stock_qty": stock.quantity,
                                    "put_strike": put.strike,
                                    "put_expiry": put.expiry,
                                },
                            )
                        )
                        self._detection_stats["covered_positions"] += 1

        return strategies

    def _detect_option_strategies(
        self,
        option_positions: List[Position],
        underlying: str,
    ) -> List[DetectedStrategy]:
        """
        Detect option-only strategies (spreads, condors, etc.).
        """
        strategies: List[DetectedStrategy] = []

        # Need at least 2 legs for a spread
        if len(option_positions) < 2:
            return strategies

        # Detect iron condors (4 legs)
        iron_condors = self._detect_iron_condors(option_positions, underlying)
        strategies.extend(iron_condors)

        # Detect vertical spreads (2 legs, same expiry, different strikes)
        vertical_spreads = self._detect_vertical_spreads(option_positions, underlying)
        strategies.extend(vertical_spreads)

        # Detect diagonal/calendar spreads (2 legs, different expiries)
        diagonal_spreads = self._detect_diagonal_spreads(option_positions, underlying)
        strategies.extend(diagonal_spreads)

        return strategies

    def _detect_vertical_spreads(
        self,
        option_positions: List[Position],
        underlying: str,
    ) -> List[DetectedStrategy]:
        """
        Detect vertical spreads (call/put spreads).

        Criteria:
        - Same expiry
        - Different strikes
        - Opposite quantities (one long, one short)
        - Same option type (both calls or both puts)
        """
        strategies = []
        used_positions = set()

        for i, pos1 in enumerate(option_positions):
            if i in used_positions:
                continue

            for j, pos2 in enumerate(option_positions[i + 1 :], start=i + 1):
                if j in used_positions:
                    continue

                # Check if it's a vertical spread
                if (
                    pos1.expiry == pos2.expiry  # Same expiry
                    and pos1.right == pos2.right  # Same type (C or P)
                    and pos1.strike != pos2.strike  # Different strikes
                    and pos1.quantity * pos2.quantity < 0  # Opposite signs
                    and abs(pos1.quantity) == abs(pos2.quantity)  # Same size
                ):
                    # Determine if credit or debit
                    long_leg = pos1 if pos1.quantity > 0 else pos2
                    short_leg = pos2 if pos1.quantity > 0 else pos1

                    # Credit spread: short leg closer to ATM (higher strike for calls, lower for puts)
                    # Strike is guaranteed non-None for options, but need to guard for type checker
                    short_strike = short_leg.strike or 0.0
                    long_strike = long_leg.strike or 0.0
                    is_credit = (pos1.right == "C" and short_strike < long_strike) or (
                        pos1.right == "P" and short_strike > long_strike
                    )

                    strategy_type = (
                        f"VERTICAL_{'CALL' if pos1.right == 'C' else 'PUT'}_"
                        f"{'CREDIT' if is_credit else 'DEBIT'}"
                    )

                    strategies.append(
                        DetectedStrategy(
                            strategy_type=strategy_type,
                            underlying=underlying,
                            positions=[pos1, pos2],
                            is_credit=is_credit,
                            metadata={
                                "expiry": pos1.expiry,
                                "long_strike": long_leg.strike,
                                "short_strike": short_leg.strike,
                                "width": abs((pos1.strike or 0.0) - (pos2.strike or 0.0)),
                            },
                        )
                    )

                    used_positions.add(i)
                    used_positions.add(j)
                    self._detection_stats["vertical_spreads"] += 1
                    break

        return strategies

    def _detect_diagonal_spreads(
        self,
        option_positions: List[Position],
        underlying: str,
    ) -> List[DetectedStrategy]:
        """
        Detect diagonal and calendar spreads.

        Diagonal: Different expiries, different strikes, same type
        Calendar: Different expiries, same strike, same type
        """
        strategies = []
        used_positions = set()

        for i, pos1 in enumerate(option_positions):
            if i in used_positions:
                continue

            for j, pos2 in enumerate(option_positions[i + 1 :], start=i + 1):
                if j in used_positions:
                    continue

                # Check for diagonal or calendar
                if (
                    pos1.expiry != pos2.expiry  # Different expiries
                    and pos1.right == pos2.right  # Same type
                    and pos1.quantity * pos2.quantity < 0  # Opposite signs
                    and abs(pos1.quantity) == abs(pos2.quantity)  # Same size
                ):
                    # Determine type
                    is_calendar = pos1.strike == pos2.strike
                    strategy_type = "CALENDAR_SPREAD" if is_calendar else "DIAGONAL_SPREAD"

                    long_leg = pos1 if pos1.quantity > 0 else pos2
                    short_leg = pos2 if pos1.quantity > 0 else pos1

                    strategies.append(
                        DetectedStrategy(
                            strategy_type=strategy_type,
                            underlying=underlying,
                            positions=[pos1, pos2],
                            is_credit=False,  # Typically debit strategies
                            metadata={
                                "long_expiry": long_leg.expiry,
                                "short_expiry": short_leg.expiry,
                                "long_strike": long_leg.strike,
                                "short_strike": short_leg.strike,
                                "is_calendar": is_calendar,
                            },
                        )
                    )

                    used_positions.add(i)
                    used_positions.add(j)

                    if is_calendar:
                        self._detection_stats["calendars"] += 1
                    else:
                        self._detection_stats["diagonals"] += 1
                    break

        return strategies

    def _detect_iron_condors(
        self,
        option_positions: List[Position],
        underlying: str,
    ) -> List[DetectedStrategy]:
        """
        Detect iron condors.

        Structure:
        - 4 legs, same expiry
        - Short call spread + Short put spread
        - Example: Sell 100C, Buy 105C, Sell 95P, Buy 90P
        """
        strategies: List[DetectedStrategy] = []

        # Need exactly 4 legs
        if len(option_positions) != 4:
            return strategies

        # Check if all same expiry
        expiries = set(pos.expiry for pos in option_positions)
        if len(expiries) != 1:
            return strategies

        # Separate calls and puts
        calls = [pos for pos in option_positions if pos.right == "C"]
        puts = [pos for pos in option_positions if pos.right == "P"]

        # Need 2 calls and 2 puts
        if len(calls) != 2 or len(puts) != 2:
            return strategies

        # Check if both are vertical spreads
        call_is_spread = calls[0].quantity * calls[1].quantity < 0 and abs(
            calls[0].quantity
        ) == abs(calls[1].quantity)
        put_is_spread = puts[0].quantity * puts[1].quantity < 0 and abs(puts[0].quantity) == abs(
            puts[1].quantity
        )

        if call_is_spread and put_is_spread:
            strategies.append(
                DetectedStrategy(
                    strategy_type="IRON_CONDOR",
                    underlying=underlying,
                    positions=option_positions,
                    is_credit=True,
                    metadata={
                        "expiry": list(expiries)[0],
                        "call_strikes": sorted([c.strike for c in calls]),
                        "put_strikes": sorted([p.strike for p in puts]),
                    },
                )
            )
            self._detection_stats["iron_condors"] += 1

        return strategies

    def get_stats(self) -> Dict[str, int]:
        """Get detection statistics."""
        return self._detection_stats.copy()

    def __repr__(self) -> str:
        """Debug representation."""
        return f"StrategyDetector(detected={self._detection_stats['strategies_detected']})"
