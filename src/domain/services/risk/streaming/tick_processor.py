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

from typing import TYPE_CHECKING, Optional

from src.utils.market_hours import MarketHours
from src.utils.timezone import now_utc

from ..calculators.greeks_calculator import calculate_position_greeks
from ..calculators.notional_calculator import calculate_delta_dollars, calculate_notional
from ..calculators.pnl_calculator import DataQuality, calculate_pnl
from ..state.position_state import PositionDelta, PositionState

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

        # Resolve mark price with fallback chain
        # Priority: tick mid → tick bid/ask → tick last → current mark → avg cost
        mark = self._resolve_mark_price(tick)
        if mark is None or mark <= 0:
            # Fallback to current state's mark (preserve last known good value)
            if current_state.mark_price > 0:
                mark = current_state.mark_price
            # Fallback to position entry price
            elif position.avg_price > 0:
                mark = position.avg_price
            else:
                return None

        # Map tick quality string to DataQuality enum
        data_quality = self._map_quality(tick.quality)

        # Use tick values with fallback to current state (tick may not have reference prices)
        yesterday_close = tick.yesterday_close or current_state.yesterday_close
        session_open = tick.session_open or current_state.session_open

        # Apply market hours logic for P&L calculation (matches RiskEngine behavior)
        # - Market OPEN: use live mark for all assets
        # - Market EXTENDED: use live mark for stocks only, yesterday_close for options
        # - Market CLOSED: use yesterday_close for all assets
        pnl_price = self._resolve_pnl_price(
            mark=mark,
            yesterday_close=yesterday_close,
            asset_type=position.asset_type.value,
        )

        # Calculate new P&L
        new_pnl = calculate_pnl(
            mark=pnl_price,
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
            delta=(
                new_greeks.delta / (position.quantity * position.multiplier)
                if position.quantity * position.multiplier != 0
                else 0.0
            ),  # Per-contract delta
            underlying_price=underlying_price,
            quantity=position.quantity,
            multiplier=position.multiplier,
        )

        # Calculate all changes
        pnl_change = new_pnl.unrealized - current_state.unrealized_pnl
        daily_pnl_change = new_pnl.daily - current_state.daily_pnl
        delta_change = new_greeks.delta - current_state.delta
        gamma_change = new_greeks.gamma - current_state.gamma
        vega_change = new_greeks.vega - current_state.vega
        theta_change = new_greeks.theta - current_state.theta
        notional_change = new_notional.notional - current_state.notional
        delta_dollars_change = new_delta_dollars - current_state.delta_dollars

        # Skip no-op deltas (common with fallback prices during extended hours)
        # This prevents flooding the event bus with empty updates
        if self._is_negligible_delta(
            pnl_change,
            daily_pnl_change,
            delta_change,
            gamma_change,
            vega_change,
            theta_change,
            notional_change,
            delta_dollars_change,
        ):
            return None

        # Build delta (changes from current state)
        return PositionDelta(
            symbol=position.symbol,
            underlying=position.underlying,
            timestamp=tick.timestamp,
            new_mark_price=mark,
            pnl_change=pnl_change,
            daily_pnl_change=daily_pnl_change,
            delta_change=delta_change,
            gamma_change=gamma_change,
            vega_change=vega_change,
            theta_change=theta_change,
            notional_change=notional_change,
            delta_dollars_change=delta_dollars_change,
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

    def _is_negligible_delta(
        self,
        pnl_change: float,
        daily_pnl_change: float,
        delta_change: float,
        gamma_change: float,
        vega_change: float,
        theta_change: float,
        notional_change: float,
        delta_dollars_change: float,
    ) -> bool:
        """
        Check if all delta changes are negligible (within tolerance).

        This filters out "empty" updates from fallback prices during extended hours,
        preventing event bus flooding with no-op deltas.

        Tolerances chosen to filter floating-point noise while preserving
        any real changes (even small price moves on large positions).
        """
        # Tolerances: $0.01 for P&L/notional, 0.001 for Greeks
        return (
            abs(pnl_change) < 0.01
            and abs(daily_pnl_change) < 0.01
            and abs(delta_change) < 0.001
            and abs(gamma_change) < 0.0001
            and abs(vega_change) < 0.001
            and abs(theta_change) < 0.001
            and abs(notional_change) < 0.01
            and abs(delta_dollars_change) < 0.01
        )

    def _resolve_pnl_price(
        self,
        mark: float,
        yesterday_close: float,
        asset_type: str,
    ) -> float:
        """
        Resolve price to use for P&L calculation based on market hours.

        This matches RiskEngine's behavior for extended hours:
        - Market OPEN: use live mark for all assets
        - Market EXTENDED: use live mark for stocks only, yesterday_close for options
        - Market CLOSED: use yesterday_close for all assets

        Args:
            mark: Current mark price from tick.
            yesterday_close: Yesterday's closing price.
            asset_type: Asset type string (e.g., "STOCK", "OPTION").

        Returns:
            Price to use for P&L calculation.
        """
        market_status = MarketHours.get_market_status()
        is_stock = asset_type == "STOCK"

        # Use live price when: market is OPEN, or (EXTENDED and is a stock)
        use_live_price = market_status == "OPEN" or (market_status == "EXTENDED" and is_stock)

        if use_live_price:
            return mark

        # Use yesterday_close for options during extended hours, or all assets when closed
        # Fall back to mark if yesterday_close is not available
        if yesterday_close and yesterday_close > 0:
            return yesterday_close
        return mark


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

    # Resolve mark price with fallback to entry price
    # Priority: tick mid → tick bid/ask → tick last → avg cost
    mark = processor._resolve_mark_price(tick)
    if mark is None or mark <= 0:
        # Fallback to position entry price (no current state for initial state)
        if position.avg_price > 0:
            mark = position.avg_price
        else:
            return None

    # Map quality
    data_quality = processor._map_quality(tick.quality)

    # Apply market hours logic for P&L calculation (matches RiskEngine behavior)
    yesterday_close = tick.yesterday_close or 0.0
    pnl_price = processor._resolve_pnl_price(
        mark=mark,
        yesterday_close=yesterday_close,
        asset_type=position.asset_type.value,
    )

    # Calculate initial P&L
    pnl = calculate_pnl(
        mark=pnl_price,
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
        delta=(
            greeks.delta / (position.quantity * position.multiplier)
            if position.quantity * position.multiplier != 0
            else 0.0
        ),
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
