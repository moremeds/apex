"""
Trade cost estimation for strategy evaluation.

CostEstimator provides pre-trade cost estimates including:
- Commission/fees
- Slippage (bid-ask spread + market impact)
- Borrowing costs (for shorts)

Strategies use this to evaluate trade profitability before execution.

Usage:
    estimator = CostEstimator(fee_model=IBFeeModel(), slippage_bps=5.0)

    # Estimate cost for a trade
    cost = estimator.estimate(order, context)
    print(f"Estimated cost: ${cost.total_cost:.2f}")

    # Check if trade is profitable after costs
    if expected_profit > cost.total_cost:
        execute(order)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Protocol

from ..interfaces.execution_provider import OrderRequest
from .base import StrategyContext

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset class for fee calculation."""

    STOCK = "stock"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"
    CRYPTO = "crypto"


@dataclass
class CostEstimate:
    """Estimated costs for a trade."""

    # Core costs
    commission: float = 0.0  # Broker commission
    exchange_fee: float = 0.0  # Exchange/clearing fees
    regulatory_fee: float = 0.0  # SEC/FINRA fees

    # Slippage
    spread_cost: float = 0.0  # Bid-ask spread
    market_impact: float = 0.0  # Price impact from order

    # Other costs
    borrow_cost: float = 0.0  # Stock borrow for shorts
    financing_cost: float = 0.0  # Margin interest

    # Metadata
    symbol: str = ""
    quantity: float = 0.0
    notional: float = 0.0
    side: str = ""

    @property
    def total_commission(self) -> float:
        """Total commission and fees."""
        return self.commission + self.exchange_fee + self.regulatory_fee

    @property
    def total_slippage(self) -> float:
        """Total slippage cost."""
        return self.spread_cost + self.market_impact

    @property
    def total_cost(self) -> float:
        """Total estimated cost."""
        return self.total_commission + self.total_slippage + self.borrow_cost + self.financing_cost

    @property
    def cost_bps(self) -> float:
        """Total cost in basis points of notional."""
        if self.notional == 0:
            return 0.0
        return (self.total_cost / self.notional) * 10000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "notional": self.notional,
            "commission": self.commission,
            "exchange_fee": self.exchange_fee,
            "regulatory_fee": self.regulatory_fee,
            "spread_cost": self.spread_cost,
            "market_impact": self.market_impact,
            "borrow_cost": self.borrow_cost,
            "financing_cost": self.financing_cost,
            "total_cost": self.total_cost,
            "cost_bps": self.cost_bps,
        }


class FeeSchedule(Protocol):
    """Protocol for fee calculation."""

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_class: AssetClass,
    ) -> float:
        """Calculate commission for a trade."""
        ...


@dataclass
class SimpleFeeSchedule:
    """Simple per-share/per-contract fee schedule."""

    per_share: float = 0.005  # Per share for stocks
    per_contract: float = 0.65  # Per contract for options
    minimum: float = 1.0  # Minimum commission

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_class: AssetClass,
    ) -> float:
        """Calculate commission."""
        if asset_class == AssetClass.OPTION:
            commission = quantity * self.per_contract
        else:
            commission = quantity * self.per_share

        return max(commission, self.minimum)


@dataclass
class IBFeeSchedule:
    """Interactive Brokers fee schedule (tiered pricing)."""

    # Stock pricing (tiered)
    stock_per_share: float = 0.005
    stock_minimum: float = 1.0
    stock_maximum_pct: float = 0.5  # Max 0.5% of trade value

    # Option pricing
    option_per_contract: float = 0.65
    option_minimum: float = 1.0

    # Exchange fees (approximate)
    exchange_fee_per_share: float = 0.0002
    option_exchange_fee: float = 0.05

    # Regulatory fees
    sec_fee_per_million: float = 22.90  # SEC fee per $1M sold
    finra_taf_per_share: float = 0.000119  # FINRA TAF

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_class: AssetClass,
    ) -> float:
        """Calculate IB commission."""
        notional = quantity * price

        if asset_class == AssetClass.OPTION:
            commission = max(quantity * self.option_per_contract, self.option_minimum)
        else:
            commission = quantity * self.stock_per_share
            commission = max(commission, self.stock_minimum)
            # Cap at max percentage
            max_comm = notional * (self.stock_maximum_pct / 100)
            commission = min(commission, max_comm)

        return commission

    def calculate_exchange_fees(
        self,
        quantity: float,
        asset_class: AssetClass,
    ) -> float:
        """Calculate exchange fees."""
        if asset_class == AssetClass.OPTION:
            return quantity * self.option_exchange_fee
        return quantity * self.exchange_fee_per_share

    def calculate_regulatory_fees(
        self,
        quantity: float,
        price: float,
        side: str,
    ) -> float:
        """Calculate regulatory fees (SEC + FINRA)."""
        fees = 0.0

        # SEC fee (sells only)
        if side == "SELL":
            notional = quantity * price
            fees += (notional / 1_000_000) * self.sec_fee_per_million

        # FINRA TAF (sells only)
        if side == "SELL":
            fees += quantity * self.finra_taf_per_share

        return fees


@dataclass
class FutuFeeSchedule:
    """Futu/Moomoo fee schedule."""

    # US stocks
    us_per_share: float = 0.0049
    us_minimum: float = 0.99
    us_platform_fee: float = 0.005  # Per share

    # HK stocks (% of trade value)
    hk_commission_pct: float = 0.03  # 0.03% (3 bps)
    hk_minimum: float = 3.0  # HKD
    hk_platform_fee_pct: float = 0.015  # 0.015%

    # Options
    option_per_contract: float = 0.65
    option_minimum: float = 1.99

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_class: AssetClass,
        market: str = "US",
    ) -> float:
        """Calculate Futu commission."""
        notional = quantity * price

        if asset_class == AssetClass.OPTION:
            return max(quantity * self.option_per_contract, self.option_minimum)

        if market == "HK":
            commission = notional * (self.hk_commission_pct / 100)
            platform = notional * (self.hk_platform_fee_pct / 100)
            return max(commission + platform, self.hk_minimum)

        # US stocks
        commission = quantity * self.us_per_share
        platform = quantity * self.us_platform_fee
        return max(commission + platform, self.us_minimum)


class CostEstimator:
    """
    Trade cost estimator for pre-trade analysis.

    Estimates total cost of a trade including:
    - Commissions and fees
    - Slippage (spread + impact)
    - Other costs (borrow, financing)
    """

    def __init__(
        self,
        fee_schedule: Optional[FeeSchedule] = None,
        slippage_bps: float = 5.0,
        market_impact_bps: float = 2.0,
        borrow_rate_pct: float = 0.0,
    ):
        """
        Initialize CostEstimator.

        Args:
            fee_schedule: Fee schedule for commission calculation.
            slippage_bps: Expected slippage in basis points.
            market_impact_bps: Market impact in basis points.
            borrow_rate_pct: Annual stock borrow rate (for shorts).
        """
        self._fee_schedule = fee_schedule or SimpleFeeSchedule()
        self._slippage_bps = slippage_bps
        self._market_impact_bps = market_impact_bps
        self._borrow_rate_pct = borrow_rate_pct

    def estimate(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> CostEstimate:
        """
        Estimate costs for an order.

        Args:
            order: Order to estimate costs for.
            context: Strategy context with market data.

        Returns:
            CostEstimate with breakdown of costs.
        """
        # Get price
        price = self._get_price(order, context)
        if not price or price <= 0:
            return CostEstimate(symbol=order.symbol, side=order.side)

        notional = order.quantity * price
        asset_class = self._get_asset_class(order)

        # Calculate commission
        commission = self._fee_schedule.calculate(
            symbol=order.symbol,
            quantity=order.quantity,
            price=price,
            side=order.side,
            asset_class=asset_class,
        )

        # Calculate exchange fees
        exchange_fee = 0.0
        if isinstance(self._fee_schedule, IBFeeSchedule):
            exchange_fee = self._fee_schedule.calculate_exchange_fees(order.quantity, asset_class)

        # Calculate regulatory fees
        regulatory_fee = 0.0
        if isinstance(self._fee_schedule, IBFeeSchedule):
            regulatory_fee = self._fee_schedule.calculate_regulatory_fees(
                order.quantity, price, order.side
            )

        # Calculate slippage from spread
        spread_cost = self._estimate_spread_cost(order, context, notional)

        # Calculate market impact
        market_impact = notional * (self._market_impact_bps / 10000)

        # Calculate borrow cost (for shorts)
        borrow_cost = 0.0
        if order.side == "SELL":
            current_pos = context.get_position_quantity(order.symbol)
            if current_pos <= 0:  # Opening or adding to short
                # Estimate 1-day borrow cost
                borrow_cost = notional * (self._borrow_rate_pct / 100 / 365)

        return CostEstimate(
            commission=commission,
            exchange_fee=exchange_fee,
            regulatory_fee=regulatory_fee,
            spread_cost=spread_cost,
            market_impact=market_impact,
            borrow_cost=borrow_cost,
            symbol=order.symbol,
            quantity=order.quantity,
            notional=notional,
            side=order.side,
        )

    def estimate_round_trip(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> CostEstimate:
        """
        Estimate costs for a round-trip trade (entry + exit).

        Args:
            order: Entry order.
            context: Strategy context.

        Returns:
            CostEstimate for the round trip.
        """
        # Estimate entry cost
        entry_cost = self.estimate(order, context)

        # Create exit order
        exit_order = OrderRequest(
            symbol=order.symbol,
            side="SELL" if order.side == "BUY" else "BUY",
            quantity=order.quantity,
            order_type="MARKET",
        )
        exit_cost = self.estimate(exit_order, context)

        # Combine costs
        return CostEstimate(
            commission=entry_cost.commission + exit_cost.commission,
            exchange_fee=entry_cost.exchange_fee + exit_cost.exchange_fee,
            regulatory_fee=entry_cost.regulatory_fee + exit_cost.regulatory_fee,
            spread_cost=entry_cost.spread_cost + exit_cost.spread_cost,
            market_impact=entry_cost.market_impact + exit_cost.market_impact,
            borrow_cost=entry_cost.borrow_cost,
            symbol=order.symbol,
            quantity=order.quantity,
            notional=entry_cost.notional,
            side=f"ROUND_TRIP_{order.side}",
        )

    def break_even_move(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> float:
        """
        Calculate price move needed to break even after costs.

        Args:
            order: Order to analyze.
            context: Strategy context.

        Returns:
            Price move in dollars needed to break even.
        """
        cost = self.estimate_round_trip(order, context)
        if order.quantity == 0:
            return 0.0
        return cost.total_cost / order.quantity

    def break_even_pct(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> float:
        """
        Calculate percentage move needed to break even.

        Args:
            order: Order to analyze.
            context: Strategy context.

        Returns:
            Percentage move needed to break even.
        """
        cost = self.estimate_round_trip(order, context)
        if cost.notional == 0:
            return 0.0
        return (cost.total_cost / cost.notional) * 100

    def _get_price(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> Optional[float]:
        """Get price for estimation."""
        if order.limit_price:
            return order.limit_price

        quote = context.get_quote(order.symbol)
        if quote:
            return quote.mid or quote.last

        return None

    def _get_asset_class(self, order: OrderRequest) -> AssetClass:
        """Determine asset class from order."""
        asset_type = order.asset_type or "STK"

        if asset_type in ("OPT", "OPTION"):
            return AssetClass.OPTION
        elif asset_type in ("FUT", "FUTURE"):
            return AssetClass.FUTURE
        elif asset_type in ("CASH", "FOREX"):
            return AssetClass.FOREX
        elif asset_type in ("CRYPTO",):
            return AssetClass.CRYPTO

        return AssetClass.STOCK

    def _estimate_spread_cost(
        self,
        order: OrderRequest,
        context: StrategyContext,
        notional: float,
    ) -> float:
        """Estimate cost from crossing the bid-ask spread."""
        quote = context.get_quote(order.symbol)
        if not quote or not quote.bid or not quote.ask:
            # Use default slippage
            return notional * (self._slippage_bps / 10000)

        # Calculate half-spread (cost of crossing)
        spread = quote.ask - quote.bid
        mid = (quote.ask + quote.bid) / 2
        spread_pct = (spread / mid) / 2 if mid > 0 else 0

        return notional * spread_pct


# Convenience factory functions
def create_ib_cost_estimator(
    slippage_bps: float = 5.0,
    market_impact_bps: float = 2.0,
) -> CostEstimator:
    """Create cost estimator with IB fee schedule."""
    return CostEstimator(
        fee_schedule=IBFeeSchedule(),
        slippage_bps=slippage_bps,
        market_impact_bps=market_impact_bps,
    )


def create_futu_cost_estimator(
    slippage_bps: float = 5.0,
    market_impact_bps: float = 2.0,
) -> CostEstimator:
    """Create cost estimator with Futu fee schedule."""
    return CostEstimator(
        fee_schedule=FutuFeeSchedule(),
        slippage_bps=slippage_bps,
        market_impact_bps=market_impact_bps,
    )


def create_zero_cost_estimator() -> CostEstimator:
    """Create cost estimator with zero costs (for testing)."""
    return CostEstimator(
        fee_schedule=SimpleFeeSchedule(per_share=0, per_contract=0, minimum=0),
        slippage_bps=0,
        market_impact_bps=0,
    )
