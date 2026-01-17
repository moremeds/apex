"""
Admin fee models for management fees and financing cost simulation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class AdminFeeResult:
    """Result of admin fee calculation."""

    amount: float  # Absolute fee amount (always positive for costs)
    description: str  # Fee description
    fee_type: str  # e.g., "mgmt_fee", "margin_interest", "borrow_fee"


class AdminFeeModel(ABC):
    """
    Abstract base class for administrative and financing fees.

    Models time-based costs such as:
    - Management fees (accrued daily)
    - Margin interest (on negative cash balances)
    - Borrow fees (for short positions)
    """

    @abstractmethod
    def calculate_daily_fees(
        self,
        timestamp: datetime,
        cash: float,
        position_value: float,
        net_asset_value: float,
    ) -> list[AdminFeeResult]:
        """
        Calculate daily administrative fees.

        Args:
            timestamp: Current timestamp.
            cash: Current cash balance.
            position_value: Total value of all positions.
            net_asset_value: Total account value (cash + position_value).

        Returns:
            List of AdminFeeResult objects.
        """
        ...


class ZeroAdminFeeModel(AdminFeeModel):
    """Zero admin fees."""

    def calculate_daily_fees(
        self,
        timestamp: datetime,
        cash: float,
        position_value: float,
        net_asset_value: float,
    ) -> list[AdminFeeResult]:
        return []


class ConstantAdminFeeModel(AdminFeeModel):
    """
    Constant daily administrative fees.

    Useful for mgmt fees and basic financing costs.
    """

    def __init__(
        self,
        mgmt_fee_annual_pct: float = 0.0,
        margin_interest_annual_pct: float = 0.0,
    ):
        """
        Initialize constant admin fee model.

        Args:
            mgmt_fee_annual_pct: Annual management fee as percentage (e.g., 2.0 = 2%).
            margin_interest_annual_pct: Annual margin interest on negative cash (e.g., 5.0 = 5%).
        """
        self.mgmt_fee_rate = mgmt_fee_annual_pct / 100.0 / 365.0
        self.margin_rate = margin_interest_annual_pct / 100.0 / 365.0

    def calculate_daily_fees(
        self,
        timestamp: datetime,
        cash: float,
        position_value: float,
        net_asset_value: float,
    ) -> list[AdminFeeResult]:
        results = []

        # 1. Management fee (accrued on NAV)
        if self.mgmt_fee_rate > 0:
            amount = net_asset_value * self.mgmt_fee_rate
            results.append(
                AdminFeeResult(
                    amount=amount,
                    description=f"Management fee ({self.mgmt_fee_rate*365*100:.2f}% p.a.)",
                    fee_type="mgmt_fee",
                )
            )

        # 2. Margin interest (on negative cash)
        if self.margin_rate > 0 and cash < 0:
            amount = abs(cash) * self.margin_rate
            results.append(
                AdminFeeResult(
                    amount=amount,
                    description=f"Margin interest ({self.margin_rate*365*100:.2f}% p.a.)",
                    fee_type="margin_interest",
                )
            )

        return results
