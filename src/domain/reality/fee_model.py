"""
Fee models for transaction cost simulation.

Provides commission and fee calculation for different brokers and asset types.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from enum import Enum


class AssetType(Enum):
    """Asset type for fee calculation."""
    STOCK = "stock"
    OPTION = "option"
    FUTURE = "future"
    FOREX = "forex"


@dataclass
class FeeBreakdown:
    """Detailed fee breakdown."""
    commission: float = 0.0
    exchange_fee: float = 0.0
    clearing_fee: float = 0.0
    regulatory_fee: float = 0.0  # SEC, FINRA fees
    platform_fee: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.commission
            + self.exchange_fee
            + self.clearing_fee
            + self.regulatory_fee
            + self.platform_fee
        )


class FeeModel(ABC):
    """
    Abstract base class for fee calculation.

    Fee models calculate transaction costs including:
    - Broker commission
    - Exchange fees
    - Regulatory fees (SEC, FINRA)
    - Clearing fees
    """

    @abstractmethod
    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_type: AssetType = AssetType.STOCK,
    ) -> FeeBreakdown:
        """
        Calculate fees for a trade.

        Args:
            symbol: Symbol being traded.
            quantity: Number of shares/contracts.
            price: Execution price.
            side: "BUY" or "SELL".
            asset_type: Type of asset.

        Returns:
            FeeBreakdown with detailed costs.
        """
        ...


class ZeroFeeModel(FeeModel):
    """Zero fees - for testing."""

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_type: AssetType = AssetType.STOCK,
    ) -> FeeBreakdown:
        return FeeBreakdown()


class ConstantFeeModel(FeeModel):
    """Constant fee per trade."""

    def __init__(self, fee_per_trade: float = 5.0):
        self.fee_per_trade = fee_per_trade

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_type: AssetType = AssetType.STOCK,
    ) -> FeeBreakdown:
        return FeeBreakdown(commission=self.fee_per_trade)


class PerShareFeeModel(FeeModel):
    """Per-share/per-contract fee model."""

    def __init__(
        self,
        per_share: float = 0.005,
        per_contract: float = 0.65,
        minimum: float = 1.0,
    ):
        self.per_share = per_share
        self.per_contract = per_contract
        self.minimum = minimum

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_type: AssetType = AssetType.STOCK,
    ) -> FeeBreakdown:
        if asset_type == AssetType.OPTION:
            commission = max(quantity * self.per_contract, self.minimum)
        else:
            commission = max(quantity * self.per_share, self.minimum)

        return FeeBreakdown(commission=commission)


class IBFeeModel(FeeModel):
    """
    Interactive Brokers fee model - IBKR Pro Fixed pricing (US).

    Based on:
    - Stocks: https://www.interactivebrokers.com/en/pricing/commissions-stocks.php
    - Options: https://www.interactivebrokers.com/en/pricing/commissions-options.php

    STOCKS:
    - Commission: USD 0.005 per share
    - Minimum: USD 1.00 per order
    - Maximum: 1.0% of trade value
    - FINRA TAF: USD 0.000166/share sold, capped at USD 8.30/trade
    - Clearing (NSCC/DTC): USD 0.0002/share, capped at 0.5% of trade value

    OPTIONS (≤10,000 contracts/month tier, premium-based):
    - Premium < $0.05: USD 0.25 per contract
    - Premium $0.05-$0.10: USD 0.50 per contract
    - Premium ≥ $0.10: USD 0.65 per contract
    - Minimum: USD 1.00 per order
    - ORF (Options Regulatory Fee): USD 0.0141 per contract
    - Exchange fees: ~USD 0.05-0.30 per contract (varies by exchange)
    """

    def __init__(
        self,
        # Stock pricing (IBKR Pro Fixed)
        stock_per_share: float = 0.005,
        stock_minimum: float = 1.0,
        stock_maximum_pct: float = 1.0,  # 1% of trade value
        # Option pricing (≤10,000 contracts/month tier, premium-based)
        option_rate_low: float = 0.25,  # Premium < $0.05
        option_rate_mid: float = 0.50,  # Premium $0.05-$0.10
        option_rate_high: float = 0.65,  # Premium ≥ $0.10
        option_minimum: float = 1.0,
        option_orf: float = 0.0141,  # Options Regulatory Fee per contract
        option_exchange: float = 0.05,  # Exchange fee (varies, use conservative estimate)
        # Exchange fees for stocks (pass-through, varies by venue)
        exchange_per_share: float = 0.0003,
        # Regulatory fees for stocks (sells only)
        sec_per_million: float = 0.0,  # Currently ~$0, fluctuates
        finra_taf_per_share: float = 0.000166,
        finra_taf_max: float = 8.30,  # Cap per trade
        # Clearing fees for stocks
        clearing_per_share: float = 0.0002,  # NSCC/DTC
        clearing_max_pct: float = 0.5,  # 0.5% of trade value cap
    ):
        self.stock_per_share = stock_per_share
        self.stock_minimum = stock_minimum
        self.stock_maximum_pct = stock_maximum_pct
        self.option_rate_low = option_rate_low
        self.option_rate_mid = option_rate_mid
        self.option_rate_high = option_rate_high
        self.option_minimum = option_minimum
        self.option_orf = option_orf
        self.option_exchange = option_exchange
        self.exchange_per_share = exchange_per_share
        self.sec_per_million = sec_per_million
        self.finra_taf_per_share = finra_taf_per_share
        self.finra_taf_max = finra_taf_max
        self.clearing_per_share = clearing_per_share
        self.clearing_max_pct = clearing_max_pct

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_type: AssetType = AssetType.STOCK,
    ) -> FeeBreakdown:
        notional = quantity * price

        # Option fees
        if asset_type == AssetType.OPTION:
            # Commission based on premium (price = premium per share, multiply by 100 for contract)
            # price parameter is the option premium per share
            premium = price
            if premium < 0.05:
                rate = self.option_rate_low  # $0.25/contract
            elif premium < 0.10:
                rate = self.option_rate_mid  # $0.50/contract
            else:
                rate = self.option_rate_high  # $0.65/contract

            commission = max(quantity * rate, self.option_minimum)
            # Exchange fees (varies by exchange, use estimate)
            exchange_fee = quantity * self.option_exchange
            # Regulatory: ORF applies to all option trades
            regulatory_fee = quantity * self.option_orf
            # OCC clearing is $0.00 per contract
            clearing_fee = 0.0

            return FeeBreakdown(
                commission=commission,
                exchange_fee=exchange_fee,
                clearing_fee=clearing_fee,
                regulatory_fee=regulatory_fee,
            )

        # Stock fees
        commission = quantity * self.stock_per_share
        commission = max(commission, self.stock_minimum)
        # Cap at max percentage of trade value
        max_comm = notional * (self.stock_maximum_pct / 100)
        commission = min(commission, max_comm)

        # Exchange fees (pass-through)
        exchange_fee = quantity * self.exchange_per_share

        # Clearing fees (NSCC/DTC)
        clearing_fee = quantity * self.clearing_per_share
        clearing_cap = notional * (self.clearing_max_pct / 100)
        clearing_fee = min(clearing_fee, clearing_cap)

        # Regulatory fees (sells only for stocks)
        regulatory_fee = 0.0
        if side == "SELL":
            # SEC fee
            regulatory_fee = (notional / 1_000_000) * self.sec_per_million
            # FINRA TAF with cap
            finra_fee = min(quantity * self.finra_taf_per_share, self.finra_taf_max)
            regulatory_fee += finra_fee

        return FeeBreakdown(
            commission=commission,
            exchange_fee=exchange_fee,
            clearing_fee=clearing_fee,
            regulatory_fee=regulatory_fee,
        )


class FutuFeeModel(FeeModel):
    """
    Futu/Moomoo fee model.

    Different rates for US and HK markets.
    """

    def __init__(
        self,
        # US stocks
        us_per_share: float = 0.0049,
        us_minimum: float = 0.99,
        us_platform: float = 0.005,
        # HK stocks
        hk_commission_pct: float = 0.03,  # 0.03%
        hk_minimum: float = 3.0,
        hk_platform_pct: float = 0.015,
        # Options
        option_per_contract: float = 0.65,
        option_minimum: float = 1.99,
        # Market detection
        default_market: str = "US",
    ):
        self.us_per_share = us_per_share
        self.us_minimum = us_minimum
        self.us_platform = us_platform
        self.hk_commission_pct = hk_commission_pct
        self.hk_minimum = hk_minimum
        self.hk_platform_pct = hk_platform_pct
        self.option_per_contract = option_per_contract
        self.option_minimum = option_minimum
        self.default_market = default_market

    def calculate(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str,
        asset_type: AssetType = AssetType.STOCK,
    ) -> FeeBreakdown:
        notional = quantity * price

        # Detect market from symbol (simple heuristic)
        market = self._detect_market(symbol)

        if asset_type == AssetType.OPTION:
            commission = max(quantity * self.option_per_contract, self.option_minimum)
            return FeeBreakdown(commission=commission)

        if market == "HK":
            commission = notional * (self.hk_commission_pct / 100)
            platform = notional * (self.hk_platform_pct / 100)
            total = max(commission + platform, self.hk_minimum)
            return FeeBreakdown(commission=commission, platform_fee=platform)

        # US market
        commission = quantity * self.us_per_share
        platform = quantity * self.us_platform
        total = commission + platform
        total = max(total, self.us_minimum)

        return FeeBreakdown(commission=commission, platform_fee=platform)

    def _detect_market(self, symbol: str) -> str:
        """Detect market from symbol."""
        # HK stocks typically have .HK suffix or are numeric
        if symbol.endswith(".HK") or symbol.isdigit():
            return "HK"
        return self.default_market
