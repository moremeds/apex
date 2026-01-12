"""
Tick Processor - Processes market data ticks into position deltas.

This is the core of the streaming risk calculation hot path. Each tick
is processed immediately to produce a PositionDelta for direct TUI consumption.

Flow:
    MarketDataTickEvent
        → TickProcessor.process_tick()
        → PositionDelta (if position exists and tick is valid)
        → Published to TUI via POSITION_DELTA event

Performance:
    - Single tick processing: ~2-5ms
    - Includes P&L calc, Greeks calc, notional calc
    - Bad ticks filtered before calculation

Thread Safety:
    - TickProcessor is stateless (pure function orchestrator)
    - All state lives in PortfolioState (thread-safe)
    - Calculator functions are pure (no side effects)

Known Limitations (Phase 3):
    - session_open is not populated by market data adapters, so intraday P&L
      is always zero. To enable intraday P&L, market_data_manager.py needs to
      capture session open prices (first tick after market open).

Usage:
    processor = TickProcessor()
    delta = processor.process_tick(tick, position, current_state)
    if delta:
        portfolio_state.apply_delta(delta)
        event_bus.publish(POSITION_DELTA, delta.to_event())
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ..calculators.pnl_calculator import calculate_pnl, DataQuality
from ..calculators.greeks_calculator import calculate_position_greeks
from ..calculators.notional_calculator import calculate_notional, calculate_delta_dollars
from ..state.position_state import PositionState, PositionDelta
from src.utils.timezone import now_utc

if TYPE_CHECKING:
    from src.domain.events.domain_events import MarketDataTickEvent
    from src.models.position import Position


class TickProcessor:
    """
    Processes market data ticks into position deltas.

    Stateless processor that orchestrates calculator calls to produce
    incremental updates for streaming TUI display.

    Design:
        - No internal state (all state in PortfolioState)
        - Filters bad ticks before processing
        - Returns None if tick should be ignored
    """

    # Data quality values that should not emit deltas
    BAD_QUALITY = frozenset({"stale", "suspicious", "zero_quote"})

    # Wide spread threshold (percentage) for suspicious tick filtering
    WIDE_SPREAD_PCT = 5.0

    def process_tick(
        self,
        tick: MarketDataTickEvent,
        position: Position,
        current_state: PositionState,
    ) -> Optional[PositionDelta]:
        """
        Process a single tick and return delta if valid.

        Args:
            tick: Market data tick event with prices and Greeks.
            position: Position model with quantity, avg_price, etc.
            current_state: Current position state from PortfolioState.

        Returns:
            PositionDelta to apply, or None if tick should be ignored.
        """
        # Filter bad quality ticks
        if not self._should_emit_delta(tick):
            return None

        # Resolve mark price (mid preferred, then last)
        mark = self._resolve_mark_price(tick)
        if mark is None or mark <= 0:
            return None

        # Map tick quality string to DataQuality enum
        data_quality = self._map_quality(tick.quality)

        # Use tick values with fallback to current state (tick may not have reference prices)
        yesterday_close = tick.yesterday_close or current_state.yesterday_close
        session_open = tick.session_open or current_state.session_open

        # Calculate new P&L
        new_pnl = calculate_pnl(
            mark=mark,
            avg_cost=position.avg_price,
            yesterday_close=yesterday_close if yesterday_close > 0 else None,
            session_open=session_open if session_open > 0 else None,
            quantity=position.quantity,
            multiplier=position.multiplier,
            data_quality=data_quality,
        )

        # Calculate new Greeks
        new_greeks = calculate_position_greeks(
            raw_delta=tick.delta,
            raw_gamma=tick.gamma,
            raw_vega=tick.vega,
            raw_theta=tick.theta,
            quantity=position.quantity,
            multiplier=position.multiplier,
            asset_type=position.asset_type,
        )

        # Calculate new notional
        new_notional = calculate_notional(
            mark_price=mark,
            quantity=position.quantity,
            multiplier=position.multiplier,
        )

        # Resolve underlying price for delta dollars calculation
        # For stocks: underlying price = mark price
        # For options: use tick.underlying_price if available, else mark_price
        underlying_price = tick.underlying_price or mark

        # Calculate new delta dollars
        new_delta_dollars, _ = calculate_delta_dollars(
            delta=new_greeks.delta / (position.quantity * position.multiplier)
            if position.quantity * position.multiplier != 0
            else 0.0,  # Per-contract delta
            underlying_price=underlying_price,
            quantity=position.quantity,
            multiplier=position.multiplier,
        )

        # Build delta (changes from current state)
        return PositionDelta(
            symbol=position.symbol,
            underlying=position.underlying,
            timestamp=tick.timestamp,
            new_mark_price=mark,
            pnl_change=new_pnl.unrealized - current_state.unrealized_pnl,
            daily_pnl_change=new_pnl.daily - current_state.daily_pnl,
            delta_change=new_greeks.delta - current_state.delta,
            gamma_change=new_greeks.gamma - current_state.gamma,
            vega_change=new_greeks.vega - current_state.vega,
            theta_change=new_greeks.theta - current_state.theta,
            notional_change=new_notional.notional - current_state.notional,
            delta_dollars_change=new_delta_dollars - current_state.delta_dollars,
            underlying_price=underlying_price,
            is_reliable=new_pnl.is_reliable,
            has_greeks=new_greeks.has_greeks,
        )

    def _should_emit_delta(self, tick: MarketDataTickEvent) -> bool:
        """
        Filter bad ticks before emitting deltas.

        Checks:
            - Quality flag not in BAD_QUALITY set
            - Spread not suspiciously wide (>5%)

        Args:
            tick: Market data tick to validate.

        Returns:
            True if tick is valid for delta emission.
        """
        # Check quality flag (normalize to lowercase for case-insensitive comparison)
        quality_lower = tick.quality.lower() if tick.quality else "good"
        if quality_lower in self.BAD_QUALITY:
            return False

        # Check for suspiciously wide spread or crossed market
        if tick.bid is not None and tick.ask is not None and tick.bid > 0:
            mid = (tick.bid + tick.ask) / 2
            if mid > 0:
                spread = abs(tick.ask - tick.bid)  # Use abs() to catch crossed markets
                spread_pct = (spread / mid) * 100
                if spread_pct > self.WIDE_SPREAD_PCT:
                    return False

        return True

    def _resolve_mark_price(self, tick: MarketDataTickEvent) -> Optional[float]:
        """
        Resolve mark price from tick data.

        Priority:
            1. Pre-calculated mid price
            2. Calculate mid from bid/ask
            3. Last price as fallback

        Args:
            tick: Market data tick.

        Returns:
            Mark price, or None if no valid price available.
        """
        # Use pre-calculated mid if available
        if tick.mid is not None and tick.mid > 0:
            return tick.mid

        # Calculate mid from bid/ask
        if tick.bid is not None and tick.ask is not None:
            if tick.bid > 0 and tick.ask > 0:
                return (tick.bid + tick.ask) / 2

        # Fall back to last price
        if tick.last is not None and tick.last > 0:
            return tick.last

        return None

    def _map_quality(self, quality: str) -> DataQuality:
        """Map tick quality string to DataQuality enum (case-insensitive)."""
        quality_lower = quality.lower() if quality else "good"
        quality_map = {
            "good": DataQuality.GOOD,
            "stale": DataQuality.STALE,
            "suspicious": DataQuality.SUSPICIOUS,
            "zero_quote": DataQuality.ZERO_QUOTE,
            "missing": DataQuality.STALE,  # Treat missing as stale
        }
        return quality_map.get(quality_lower, DataQuality.GOOD)


def create_initial_state(
    position: Position,
    tick: MarketDataTickEvent,
    strict_quality: bool = True,
) -> Optional[PositionState]:
    """
    Create initial PositionState from position and tick.

    Used when adding a new position to PortfolioState or on full resync.

    Args:
        position: Position model.
        tick: Initial market data tick.
        strict_quality: If True (default), reject bad quality ticks.
            Set to False when initializing from potentially stale data
            (e.g., on startup before fresh ticks arrive).

    Returns:
        PositionState, or None if tick is invalid or (when strict) bad quality.
    """
    processor = TickProcessor()

    # Apply quality filtering for initial state (unless explicitly disabled)
    if strict_quality and not processor._should_emit_delta(tick):
        return None

    # Resolve mark price
    mark = processor._resolve_mark_price(tick)
    if mark is None or mark <= 0:
        return None

    # Map quality
    data_quality = processor._map_quality(tick.quality)

    # Calculate initial P&L
    pnl = calculate_pnl(
        mark=mark,
        avg_cost=position.avg_price,
        yesterday_close=tick.yesterday_close,
        session_open=tick.session_open,
        quantity=position.quantity,
        multiplier=position.multiplier,
        data_quality=data_quality,
    )

    # Calculate initial Greeks
    greeks = calculate_position_greeks(
        raw_delta=tick.delta,
        raw_gamma=tick.gamma,
        raw_vega=tick.vega,
        raw_theta=tick.theta,
        quantity=position.quantity,
        multiplier=position.multiplier,
        asset_type=position.asset_type,
    )

    # Calculate initial notional
    notional = calculate_notional(
        mark_price=mark,
        quantity=position.quantity,
        multiplier=position.multiplier,
    )

    # Resolve underlying price for delta dollars calculation
    underlying_price = tick.underlying_price or mark

    # Calculate initial delta dollars
    delta_dollars, _ = calculate_delta_dollars(
        delta=greeks.delta / (position.quantity * position.multiplier)
        if position.quantity * position.multiplier != 0
        else 0.0,
        underlying_price=underlying_price,
        quantity=position.quantity,
        multiplier=position.multiplier,
    )

    return PositionState(
        symbol=position.symbol,
        underlying=position.underlying,
        quantity=position.quantity,
        multiplier=position.multiplier,
        avg_cost=position.avg_price,
        mark_price=mark,
        yesterday_close=tick.yesterday_close or 0.0,
        session_open=tick.session_open or 0.0,
        unrealized_pnl=pnl.unrealized,
        daily_pnl=pnl.daily,
        delta=greeks.delta,
        gamma=greeks.gamma,
        vega=greeks.vega,
        theta=greeks.theta,
        notional=notional.notional,
        delta_dollars=delta_dollars,
        underlying_price=underlying_price,
        is_reliable=pnl.is_reliable,
        has_greeks=greeks.has_greeks,
        last_update=tick.timestamp,
    )
