"""
Pre-trade risk validation gate.

RiskGate validates orders before they are submitted for execution.
It checks position limits, order size limits, and other risk constraints.

Usage:
    risk_gate = RiskGate(config={
        'max_position_size': 10000,
        'max_order_size': 1000,
        'max_notional_per_order': 100000,
    })

    # Validate an order
    result = risk_gate.validate(order, context)
    if result.approved:
        execute(order)
    else:
        log(f"Order rejected: {result.reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Protocol
from enum import Enum
import logging

from ..interfaces.execution_provider import OrderRequest
from .base import StrategyContext

logger = logging.getLogger(__name__)


class RejectionReason(Enum):
    """Reasons for order rejection."""
    NONE = "none"
    MAX_POSITION_SIZE = "max_position_size"
    MAX_ORDER_SIZE = "max_order_size"
    MAX_NOTIONAL = "max_notional"
    MAX_CONCENTRATION = "max_concentration"
    MARGIN_INSUFFICIENT = "margin_insufficient"
    SYMBOL_NOT_ALLOWED = "symbol_not_allowed"
    TRADING_HALTED = "trading_halted"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    CUSTOM = "custom"


@dataclass
class ValidationResult:
    """Result of order validation."""
    approved: bool
    reason: RejectionReason = RejectionReason.NONE
    message: str = ""
    adjusted_order: Optional[OrderRequest] = None  # If order was modified
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def approve(cls, message: str = "Order approved") -> "ValidationResult":
        """Create an approval result."""
        return cls(approved=True, message=message)

    @classmethod
    def reject(
        cls,
        reason: RejectionReason,
        message: str,
        **metadata
    ) -> "ValidationResult":
        """Create a rejection result."""
        return cls(
            approved=False,
            reason=reason,
            message=message,
            metadata=metadata,
        )


class RiskValidator(Protocol):
    """Protocol for custom risk validators."""

    def validate(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> ValidationResult:
        """Validate an order. Return ValidationResult."""
        ...


@dataclass
class RiskGateConfig:
    """Configuration for RiskGate."""

    # Position limits
    max_position_size: float = 10000  # Max shares/contracts per symbol
    max_position_notional: float = 1_000_000  # Max $ value per position

    # Order limits
    max_order_size: float = 1000  # Max shares/contracts per order
    max_order_notional: float = 100_000  # Max $ value per order

    # Portfolio limits
    max_portfolio_notional: float = 10_000_000  # Max total portfolio value
    max_concentration_pct: float = 20.0  # Max % of portfolio in one symbol
    max_open_orders: int = 100  # Max concurrent open orders

    # Symbol restrictions
    allowed_symbols: Optional[List[str]] = None  # If set, only these symbols
    blocked_symbols: List[str] = field(default_factory=list)

    # Trading halts
    halt_all_trading: bool = False
    halted_symbols: List[str] = field(default_factory=list)

    # Soft mode (warn but don't reject)
    soft_mode: bool = False

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RiskGateConfig":
        """Create config from dictionary."""
        return cls(
            max_position_size=config.get("max_position_size", 10000),
            max_position_notional=config.get("max_position_notional", 1_000_000),
            max_order_size=config.get("max_order_size", 1000),
            max_order_notional=config.get("max_order_notional", 100_000),
            max_portfolio_notional=config.get("max_portfolio_notional", 10_000_000),
            max_concentration_pct=config.get("max_concentration_pct", 20.0),
            max_open_orders=config.get("max_open_orders", 100),
            allowed_symbols=config.get("allowed_symbols"),
            blocked_symbols=config.get("blocked_symbols", []),
            halt_all_trading=config.get("halt_all_trading", False),
            halted_symbols=config.get("halted_symbols", []),
            soft_mode=config.get("soft_mode", False),
        )


class RiskGate:
    """
    Pre-trade risk validation gate.

    Validates orders against configurable risk limits before submission.
    Can reject, modify, or approve orders.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        custom_validators: Optional[List[RiskValidator]] = None,
    ):
        """
        Initialize RiskGate.

        Args:
            config: Risk configuration dictionary.
            custom_validators: Additional custom validators.
        """
        self._config = RiskGateConfig.from_dict(config or {})
        self._custom_validators = custom_validators or []
        self._open_orders: Dict[str, OrderRequest] = {}
        self._rejection_counts: Dict[RejectionReason, int] = {}

    @property
    def config(self) -> RiskGateConfig:
        """Get current configuration."""
        return self._config

    def validate(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> ValidationResult:
        """
        Validate an order against risk limits.

        Args:
            order: Order request to validate.
            context: Strategy context with positions and market data.

        Returns:
            ValidationResult indicating approval or rejection.
        """
        # Check trading halts first
        result = self._check_trading_halts(order)
        if not result.approved:
            return self._handle_rejection(result)

        # Check symbol restrictions
        result = self._check_symbol_restrictions(order)
        if not result.approved:
            return self._handle_rejection(result)

        # Check order size limits
        result = self._check_order_limits(order, context)
        if not result.approved:
            return self._handle_rejection(result)

        # Check position limits
        result = self._check_position_limits(order, context)
        if not result.approved:
            return self._handle_rejection(result)

        # Check portfolio concentration
        result = self._check_concentration(order, context)
        if not result.approved:
            return self._handle_rejection(result)

        # Run custom validators
        for validator in self._custom_validators:
            result = validator.validate(order, context)
            if not result.approved:
                return self._handle_rejection(result)

        logger.debug(f"Order approved: {order.symbol} {order.side} {order.quantity}")
        return ValidationResult.approve()

    def _check_trading_halts(self, order: OrderRequest) -> ValidationResult:
        """Check if trading is halted."""
        if self._config.halt_all_trading:
            return ValidationResult.reject(
                RejectionReason.TRADING_HALTED,
                "All trading is halted",
            )

        if order.symbol in self._config.halted_symbols:
            return ValidationResult.reject(
                RejectionReason.TRADING_HALTED,
                f"Trading halted for {order.symbol}",
            )

        return ValidationResult.approve()

    def _check_symbol_restrictions(self, order: OrderRequest) -> ValidationResult:
        """Check symbol restrictions."""
        if order.symbol in self._config.blocked_symbols:
            return ValidationResult.reject(
                RejectionReason.SYMBOL_NOT_ALLOWED,
                f"Symbol {order.symbol} is blocked",
            )

        if self._config.allowed_symbols is not None:
            if order.symbol not in self._config.allowed_symbols:
                return ValidationResult.reject(
                    RejectionReason.SYMBOL_NOT_ALLOWED,
                    f"Symbol {order.symbol} not in allowed list",
                )

        return ValidationResult.approve()

    def _check_order_limits(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> ValidationResult:
        """Check order size limits."""
        # Check max order size
        if order.quantity > self._config.max_order_size:
            return ValidationResult.reject(
                RejectionReason.MAX_ORDER_SIZE,
                f"Order size {order.quantity} exceeds max {self._config.max_order_size}",
                requested=order.quantity,
                max_allowed=self._config.max_order_size,
            )

        # Check max order notional
        price = self._get_price(order, context)
        if price:
            notional = order.quantity * price
            if notional > self._config.max_order_notional:
                return ValidationResult.reject(
                    RejectionReason.MAX_NOTIONAL,
                    f"Order notional ${notional:,.0f} exceeds max ${self._config.max_order_notional:,.0f}",
                    notional=notional,
                    max_allowed=self._config.max_order_notional,
                )

        # Check max open orders
        if len(self._open_orders) >= self._config.max_open_orders:
            return ValidationResult.reject(
                RejectionReason.RISK_LIMIT_BREACH,
                f"Max open orders ({self._config.max_open_orders}) reached",
            )

        return ValidationResult.approve()

    def _check_position_limits(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> ValidationResult:
        """Check position limits."""
        current_qty = context.get_position_quantity(order.symbol)

        # Calculate resulting position
        if order.side == "BUY":
            new_qty = current_qty + order.quantity
        else:
            new_qty = current_qty - order.quantity

        # Check max position size
        if abs(new_qty) > self._config.max_position_size:
            return ValidationResult.reject(
                RejectionReason.MAX_POSITION_SIZE,
                f"Resulting position {new_qty} exceeds max {self._config.max_position_size}",
                current_position=current_qty,
                order_quantity=order.quantity,
                resulting_position=new_qty,
            )

        # Check max position notional
        price = self._get_price(order, context)
        if price:
            notional = abs(new_qty) * price
            if notional > self._config.max_position_notional:
                return ValidationResult.reject(
                    RejectionReason.MAX_NOTIONAL,
                    f"Position notional ${notional:,.0f} exceeds max ${self._config.max_position_notional:,.0f}",
                    position_notional=notional,
                    max_allowed=self._config.max_position_notional,
                )

        return ValidationResult.approve()

    def _check_concentration(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> ValidationResult:
        """Check portfolio concentration limits."""
        # Calculate total portfolio value
        total_value = 0.0
        for symbol, pos in context.positions.items():
            price = context.get_mid_price(symbol)
            if price and pos.quantity:
                total_value += abs(pos.quantity * price)

        if total_value == 0:
            return ValidationResult.approve()

        # Calculate position value after order
        current_qty = context.get_position_quantity(order.symbol)
        if order.side == "BUY":
            new_qty = current_qty + order.quantity
        else:
            new_qty = current_qty - order.quantity

        price = self._get_price(order, context)
        if not price:
            return ValidationResult.approve()

        position_value = abs(new_qty) * price
        concentration_pct = (position_value / total_value) * 100

        if concentration_pct > self._config.max_concentration_pct:
            return ValidationResult.reject(
                RejectionReason.MAX_CONCENTRATION,
                f"Position concentration {concentration_pct:.1f}% exceeds max {self._config.max_concentration_pct}%",
                concentration=concentration_pct,
                max_allowed=self._config.max_concentration_pct,
            )

        return ValidationResult.approve()

    def _get_price(
        self,
        order: OrderRequest,
        context: StrategyContext,
    ) -> Optional[float]:
        """Get price for order (limit price or market price)."""
        if order.limit_price:
            return order.limit_price

        quote = context.get_quote(order.symbol)
        if quote:
            if order.side == "BUY":
                return quote.ask or quote.last
            else:
                return quote.bid or quote.last

        return None

    def _handle_rejection(self, result: ValidationResult) -> ValidationResult:
        """Handle rejection (count and potentially downgrade to warning)."""
        self._rejection_counts[result.reason] = (
            self._rejection_counts.get(result.reason, 0) + 1
        )

        if self._config.soft_mode:
            logger.warning(f"RiskGate SOFT REJECT: {result.message}")
            return ValidationResult.approve(f"Soft mode: {result.message}")

        logger.warning(f"RiskGate REJECT: {result.message}")
        return result

    def add_validator(self, validator: RiskValidator) -> None:
        """Add a custom validator."""
        self._custom_validators.append(validator)

    def track_order(self, order_id: str, order: OrderRequest) -> None:
        """Track an open order."""
        self._open_orders[order_id] = order

    def remove_order(self, order_id: str) -> None:
        """Remove a tracked order."""
        self._open_orders.pop(order_id, None)

    def halt_symbol(self, symbol: str) -> None:
        """Halt trading for a symbol."""
        if symbol not in self._config.halted_symbols:
            self._config.halted_symbols.append(symbol)
            logger.info(f"Trading halted for {symbol}")

    def resume_symbol(self, symbol: str) -> None:
        """Resume trading for a symbol."""
        if symbol in self._config.halted_symbols:
            self._config.halted_symbols.remove(symbol)
            logger.info(f"Trading resumed for {symbol}")

    def halt_all(self) -> None:
        """Halt all trading."""
        self._config.halt_all_trading = True
        logger.warning("All trading halted")

    def resume_all(self) -> None:
        """Resume all trading."""
        self._config.halt_all_trading = False
        logger.info("All trading resumed")

    def get_rejection_stats(self) -> Dict[str, int]:
        """Get rejection statistics."""
        return {r.value: c for r, c in self._rejection_counts.items()}

    def reset_stats(self) -> None:
        """Reset rejection statistics."""
        self._rejection_counts.clear()
