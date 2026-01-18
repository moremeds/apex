"""
Action Resolver - Decision Table to Action Mapping.

Resolves hierarchical regimes to specific trading actions with context.
Uses decision tables per account type.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, cast

from src.domain.signals.indicators.regime import MarketRegime

from .models import (
    DECISION_TABLE_SHORT_PUT,
    DEFAULT_CONTEXTS,
    AccountType,
    ActionContext,
    TradingAction,
)
from .regime_hierarchy import synthesize_regimes


def resolve_action(
    market_regime: MarketRegime,
    sector_regime: Optional[MarketRegime],
    stock_regime: Optional[MarketRegime],
    account_type: AccountType = AccountType.SHORT_PUT,
) -> Tuple[TradingAction, ActionContext]:
    """
    Resolve trading action from hierarchical regimes.

    Looks up the decision table for specific regime combinations,
    falling back to synthesized action with default context.

    Args:
        market_regime: Market-level regime
        sector_regime: Sector-level regime (optional)
        stock_regime: Stock-level regime (optional)
        account_type: Account type for decision table

    Returns:
        Tuple of (TradingAction, ActionContext)
    """
    # Try to find specific decision table entry
    if account_type == AccountType.SHORT_PUT:
        context = _lookup_decision_table(
            market_regime, sector_regime, stock_regime, DECISION_TABLE_SHORT_PUT
        )
        if context is not None:
            # Determine action from context
            if context.size_factor == 0:
                if market_regime == MarketRegime.R2_RISK_OFF:
                    action = TradingAction.HARD_NO
                else:
                    action = TradingAction.NO_GO
            elif context.size_factor < 0.5:
                action = TradingAction.GO_SMALL
            else:
                action = TradingAction.GO
            return action, context

    # Fall back to synthesized action with default context
    action = synthesize_regimes(market_regime, sector_regime, stock_regime, account_type)
    context = DEFAULT_CONTEXTS.get(action, ActionContext())

    return action, context


def _lookup_decision_table(
    market_regime: MarketRegime,
    sector_regime: Optional[MarketRegime],
    stock_regime: Optional[MarketRegime],
    table: Dict,
) -> Optional[ActionContext]:
    """
    Look up action context from decision table.

    Tries progressively less specific keys:
    1. (market, sector, stock)
    2. (market, sector, None)
    3. (market, None, stock)
    4. (market, None, None)

    Args:
        market_regime: Market regime
        sector_regime: Sector regime (or None)
        stock_regime: Stock regime (or None)
        table: Decision table dict

    Returns:
        ActionContext if found, None otherwise
    """
    m = market_regime.value
    s = sector_regime.value if sector_regime else None
    st = stock_regime.value if stock_regime else None

    # Try specific to general
    keys_to_try = [
        (m, s, st),
        (m, s, None),
        (m, None, st),
        (m, None, None),
    ]

    for key in keys_to_try:
        if key in table:
            return cast(ActionContext, table[key])

    return None


def get_position_sizing(
    action: TradingAction,
    context: ActionContext,
    base_position_size: float,
) -> float:
    """
    Calculate position size based on action and context.

    Args:
        action: Resolved trading action
        context: Action context with size_factor
        base_position_size: Normal position size

    Returns:
        Adjusted position size
    """
    return base_position_size * context.size_factor


def get_action_summary(
    action: TradingAction,
    context: ActionContext,
    symbol: str,
) -> str:
    """
    Generate human-readable action summary.

    Args:
        action: Trading action
        context: Action context
        symbol: Symbol being analyzed

    Returns:
        Summary string
    """
    if action == TradingAction.HARD_NO:
        return f"{symbol}: HARD NO - {context.rationale}"
    elif action == TradingAction.NO_GO:
        return f"{symbol}: NO GO - {context.rationale}"
    elif action == TradingAction.GO_SMALL:
        type_str = context.position_type.replace("_", " ").title()
        delta_str = ""
        if context.delta_min and context.delta_max:
            delta_str = f", delta {context.delta_min:.2f}-{context.delta_max:.2f}"
        dte_str = ""
        if context.dte_min and context.dte_max:
            dte_str = f", DTE {context.dte_min}-{context.dte_max}"
        return f"{symbol}: GO SMALL ({int(context.size_factor * 100)}% size) - {type_str}{delta_str}{dte_str}"
    else:  # GO
        type_str = context.position_type.replace("_", " ").title()
        delta_str = ""
        if context.delta_min and context.delta_max:
            delta_str = f", delta {context.delta_min:.2f}-{context.delta_max:.2f}"
        dte_str = ""
        if context.dte_min and context.dte_max:
            dte_str = f", DTE {context.dte_min}-{context.dte_max}"
        return f"{symbol}: GO - {type_str}{delta_str}{dte_str}"


def should_reduce_exposure(
    action: TradingAction,
    current_exposure: float,
    max_exposure: float,
) -> bool:
    """
    Determine if existing exposure should be reduced.

    Args:
        action: Current trading action
        current_exposure: Current position exposure
        max_exposure: Maximum allowed exposure

    Returns:
        True if exposure should be reduced
    """
    if action == TradingAction.HARD_NO:
        # Consider reducing all exposure
        return True
    elif action == TradingAction.NO_GO:
        # Reduce if above threshold
        return current_exposure > max_exposure * 0.8
    return False


def get_defensive_actions(
    action: TradingAction,
    context: ActionContext,
) -> list[str]:
    """
    Get list of defensive actions to consider.

    Args:
        action: Current trading action
        context: Action context

    Returns:
        List of defensive action suggestions
    """
    actions = []

    if action == TradingAction.HARD_NO:
        actions.extend(
            [
                "Close short puts in profit",
                "Roll losing positions to defined-risk spreads",
                "Consider protective puts on long positions",
                "Reduce overall delta exposure",
            ]
        )
    elif action == TradingAction.NO_GO:
        actions.extend(
            [
                "No new short puts",
                "Let existing positions decay or take profit",
                "Consider tightening stops on swing positions",
            ]
        )

    actions.extend(context.warnings)

    return actions
