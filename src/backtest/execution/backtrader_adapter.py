"""
Backtrader integration adapter.

Wraps Apex strategies to run within Backtrader's execution engine.
This provides production-grade backtesting with Backtrader's robust
fill simulation, slippage models, and analyzers.

Usage:
    import backtrader as bt
    from src.backtest.execution.backtrader_adapter import (
        ApexStrategyWrapper,
        run_backtest_with_backtrader,
    )
    from src.domain.strategy.playbook import TrendPulseStrategy

    # Create wrapper
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        ApexStrategyWrapper,
        apex_strategy_class=TrendPulseStrategy,
        apex_params={'zig_pct': 2.5},
    )

    # Add data and run
    cerebro.adddata(bt.feeds.YahooFinanceData(dataname='AAPL'))
    results = cerebro.run()

Note: Requires backtrader package: pip install backtrader
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type

# Check if backtrader is available
try:
    import backtrader as bt

    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None

from ...domain.clock import BacktraderClock
from ...domain.events.domain_events import BarData, PositionSnapshot
from ...domain.interfaces.execution_provider import OrderRequest
from ...domain.reality import RealityModelPack
from ...domain.strategy.base import Strategy, StrategyContext
from ...domain.strategy.scheduler import ScheduledAction, ScheduleFrequency, Scheduler

if TYPE_CHECKING:
    from datetime import time as time_type

logger = logging.getLogger(__name__)


class BacktraderScheduler(Scheduler):
    """
    Scheduler implementation for Backtrader integration.

    Maps Apex scheduler calls to Backtrader's execution model.
    Since Backtrader runs bar-by-bar synchronously, time-based actions
    are checked on each bar against the bar's timestamp.
    """

    def __init__(self, bt_strategy: Optional["bt.Strategy"] = None):
        """
        Initialize Backtrader scheduler.

        Args:
            bt_strategy: Backtrader strategy instance (optional).
        """
        self._bt = bt_strategy
        self._actions: Dict[str, ScheduledAction] = {}
        self._last_checked_date: Optional[datetime] = None

    def schedule_once(
        self,
        action_id: str,
        callback: Callable[[], None],
        at_time: datetime,
    ) -> None:
        """Schedule a one-time action at a specific datetime."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ONCE,
        )
        self._actions[action_id] = action
        # Store target time in metadata
        action.time_of_day = at_time.time()
        logger.debug(f"BT Scheduled once: {action_id} at {at_time}")

    def schedule_daily(
        self,
        action_id: str,
        callback: Callable[[], None],
        time_of_day: "time_type",
    ) -> None:
        """Schedule daily action."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.DAILY,
            time_of_day=time_of_day,
        )
        self._actions[action_id] = action
        logger.debug(f"BT Scheduled daily: {action_id} at {time_of_day}")

    def schedule_weekly(
        self,
        action_id: str,
        callback: Callable[[], None],
        day_of_week: int,
        time_of_day: "time_type",
    ) -> None:
        """Schedule weekly action."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.WEEKLY,
            day_of_week=day_of_week,
            time_of_day=time_of_day,
        )
        self._actions[action_id] = action
        logger.debug(f"BT Scheduled weekly: {action_id} day={day_of_week}")

    def schedule_on_bar_close(
        self,
        action_id: str,
        callback: Callable[[], None],
    ) -> None:
        """Schedule action for bar close."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ON_BAR,
        )
        self._actions[action_id] = action
        logger.debug(f"BT Scheduled on_bar_close: {action_id}")

    def schedule_before_close(
        self,
        action_id: str,
        callback: Callable[[], None],
        minutes_before: int = 5,
    ) -> None:
        """Schedule action before market close."""
        action = ScheduledAction(
            action_id=action_id,
            callback=callback,
            frequency=ScheduleFrequency.ON_SESSION_CLOSE,
            minutes_before_close=minutes_before,
        )
        self._actions[action_id] = action
        logger.debug(f"BT Scheduled before_close: {action_id}")

    def cancel(self, action_id: str) -> bool:
        """Cancel a scheduled action."""
        if action_id in self._actions:
            del self._actions[action_id]
            return True
        return False

    def get_scheduled_actions(self) -> List[ScheduledAction]:
        """Get all scheduled actions."""
        return list(self._actions.values())

    def check_triggers(self, bar_datetime: datetime) -> None:
        """
        Check and trigger actions based on bar datetime.

        Called on each bar by the strategy wrapper.

        Args:
            bar_datetime: Datetime of current bar.
        """
        bar_time = bar_datetime.time()
        bar_date = bar_datetime.date()

        for action in list(self._actions.values()):
            if not action.enabled:
                continue

            triggered = False

            if action.frequency == ScheduleFrequency.DAILY:
                # Trigger if bar time matches (or just passed) the scheduled time
                # and we haven't triggered today
                if action.time_of_day and bar_time >= action.time_of_day:
                    if action.last_triggered is None or action.last_triggered.date() < bar_date:
                        triggered = True

            elif action.frequency == ScheduleFrequency.WEEKLY:
                if (
                    action.day_of_week is not None
                    and action.time_of_day
                    and bar_datetime.weekday() == action.day_of_week
                    and bar_time >= action.time_of_day
                ):
                    if (
                        action.last_triggered is None
                        or (bar_date - action.last_triggered.date()).days >= 7
                    ):
                        triggered = True

            elif action.frequency == ScheduleFrequency.ON_SESSION_CLOSE:
                # Check if we're within minutes_before of market close (16:00 default)
                market_close_minutes = 16 * 60  # 4:00 PM
                bar_minutes = bar_time.hour * 60 + bar_time.minute
                if action.minutes_before_close:
                    trigger_minutes = market_close_minutes - action.minutes_before_close
                    if bar_minutes >= trigger_minutes:
                        if action.last_triggered is None or action.last_triggered.date() < bar_date:
                            triggered = True

            elif action.frequency == ScheduleFrequency.ONCE:
                # One-time action - check if time has passed
                if action.time_of_day and bar_time >= action.time_of_day:
                    if action.last_triggered is None:
                        triggered = True
                        # Remove one-time actions after triggering
                        self._actions.pop(action.action_id, None)

            if triggered:
                try:
                    action.callback()
                    action.last_triggered = bar_datetime
                    action.trigger_count += 1
                except Exception as e:
                    logger.error(f"BT Scheduler action {action.action_id} error: {e}")

    def trigger_bar_close_actions(self) -> None:
        """Trigger all ON_BAR actions."""
        for action in self._actions.values():
            if action.enabled and action.frequency == ScheduleFrequency.ON_BAR:
                try:
                    action.callback()
                    action.trigger_count += 1
                except Exception as e:
                    logger.error(f"BT Bar close action {action.action_id} error: {e}")


if BACKTRADER_AVAILABLE:

    class ApexStrategyWrapper(bt.Strategy):
        """
        Backtrader strategy that wraps an Apex strategy.

        This wrapper:
        1. Creates Apex strategy instance with Backtrader-compatible context
        2. Translates Backtrader events to Apex events
        3. Translates Apex orders to Backtrader orders
        4. Provides access to Backtrader's data and broker

        Usage:
            cerebro.addstrategy(
                ApexStrategyWrapper,
                apex_strategy_class=MyStrategy,
                apex_params={'param1': 10},
            )
        """

        params: Tuple[Tuple[str, Any], ...] = (
            ("apex_strategy_class", None),  # Required: Apex Strategy class
            ("apex_params", {}),  # Optional: Strategy parameters
            ("apex_strategy_id", "bt-apex"),  # Optional: Strategy ID
        )

        def __init__(self) -> None:
            """Initialize wrapper and create Apex strategy."""
            if self.p.apex_strategy_class is None:
                raise ValueError("apex_strategy_class parameter is required")

            # Create clock and context
            self._clock = BacktraderClock(self)
            self._scheduler = BacktraderScheduler(self)

            # Get symbols from data feeds
            symbols = [d._name for d in self.datas if d._name]
            if not symbols:
                symbols = [f"data{i}" for i in range(len(self.datas))]

            self._context = StrategyContext(
                clock=self._clock,
                scheduler=self._scheduler,
                positions={},
                account=None,
            )

            # Create Apex strategy
            self._apex = self.p.apex_strategy_class(
                strategy_id=self.p.apex_strategy_id,
                symbols=symbols,
                context=self._context,
                **self.p.apex_params,
            )

            # Wire order callback to Backtrader execution
            self._apex.on_order_callback(self._submit_order)

            # Track pending orders
            self._pending_orders: Dict[str, bt.Order] = {}

            logger.info(f"ApexStrategyWrapper initialized: {self._apex.__class__.__name__}")

        def start(self) -> None:
            """Called when strategy starts."""
            self._apex.start()

        def stop(self) -> None:
            """Called when strategy stops."""
            self._apex.stop()

        def next(self) -> None:
            """Called on each bar."""
            # Update positions in context
            self._update_positions()

            # Create bar/tick for each data feed
            for i, data in enumerate(self.datas):
                symbol = data._name or f"data{i}"

                # Create bar data
                bar = BarData(
                    symbol=symbol,
                    timeframe=self._get_timeframe(data),
                    open=data.open[0],
                    high=data.high[0],
                    low=data.low[0],
                    close=data.close[0],
                    volume=int(data.volume[0]) if data.volume[0] else 0,
                    bar_start=data.datetime.datetime(0),
                    bar_end=data.datetime.datetime(0),
                    timestamp=data.datetime.datetime(0),
                    source="backtrader",
                )

                # Feed to Apex strategy
                self._apex.on_bar(bar)

            # Check time-based scheduled actions
            bar_datetime = self.datas[0].datetime.datetime(0)
            self._scheduler.check_triggers(bar_datetime)

            # Trigger bar close actions
            self._scheduler.trigger_bar_close_actions()

        def notify_order(self, order: Any) -> None:
            """Called when order status changes."""
            if order.status == order.Completed:
                # Create fill event
                from ...domain.events.domain_events import TradeFill

                fill = TradeFill(
                    symbol=order.data._name or "unknown",
                    side="BUY" if order.isbuy() else "SELL",
                    quantity=order.executed.size,
                    price=order.executed.price,
                    commission=order.executed.comm,
                    exec_id=str(order.ref),
                    order_id=str(order.ref),
                    source="backtrader",
                    timestamp=self.data.datetime.datetime(0),
                )

                self._apex.on_fill(fill)

        def _submit_order(self, order_request: OrderRequest) -> None:
            """Submit order to Backtrader broker."""
            # Find data for symbol
            data = None
            for d in self.datas:
                if d._name == order_request.symbol:
                    data = d
                    break

            if data is None:
                data = self.datas[0]  # Default to first data

            # Create Backtrader order
            if order_request.order_type == "MARKET":
                if order_request.side == "BUY":
                    bt_order = self.buy(data=data, size=order_request.quantity)
                else:
                    bt_order = self.sell(data=data, size=order_request.quantity)

            elif order_request.order_type == "LIMIT":
                if order_request.side == "BUY":
                    bt_order = self.buy(
                        data=data,
                        size=order_request.quantity,
                        price=order_request.limit_price,
                        exectype=bt.Order.Limit,
                    )
                else:
                    bt_order = self.sell(
                        data=data,
                        size=order_request.quantity,
                        price=order_request.limit_price,
                        exectype=bt.Order.Limit,
                    )
            else:
                logger.warning(f"Unsupported order type: {order_request.order_type}")
                return

            if bt_order:
                self._pending_orders[order_request.client_order_id or str(bt_order.ref)] = bt_order
                logger.debug(f"Order submitted to Backtrader: {order_request}")

        def _update_positions(self) -> None:
            """Update context positions from Backtrader broker."""
            for i, data in enumerate(self.datas):
                symbol = data._name or f"data{i}"
                position = self.getposition(data)

                if position.size != 0:
                    snapshot = PositionSnapshot(
                        symbol=symbol,
                        quantity=position.size,
                        avg_price=position.price,
                        timestamp=data.datetime.datetime(0),
                        source="backtrader",
                    )
                    self._context.positions[symbol] = snapshot
                elif symbol in self._context.positions:
                    del self._context.positions[symbol]

        def _get_timeframe(self, data: Any) -> str:
            """Get timeframe string from Backtrader data."""
            # Map Backtrader timeframe to string
            tf_map = {
                bt.TimeFrame.Minutes: "1m",
                bt.TimeFrame.Days: "1d",
                bt.TimeFrame.Weeks: "1w",
            }
            return tf_map.get(data._timeframe, "1d")

    def run_backtest_with_backtrader(
        apex_strategy_class: Type[Strategy],
        data_feeds: List[Any],
        initial_cash: float = 100000.0,
        commission: float = 0.001,
        strategy_params: Optional[Dict[str, Any]] = None,
        reality_pack: Optional[RealityModelPack] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest using Backtrader engine with Apex strategy.

        Args:
            apex_strategy_class: Apex strategy class to run.
            data_feeds: List of Backtrader data feeds.
            initial_cash: Starting capital.
            commission: Commission rate (legacy, used if reality_pack is None).
            strategy_params: Parameters for Apex strategy.
            reality_pack: Reality model pack for fees and slippage.

        Returns:
            Dictionary with results including analyzers.
        """
        cerebro = bt.Cerebro()

        # Add strategy
        cerebro.addstrategy(
            ApexStrategyWrapper,
            apex_strategy_class=apex_strategy_class,
            apex_params=strategy_params or {},
        )

        # Add data feeds
        for feed in data_feeds:
            cerebro.adddata(feed)

        # Set broker settings
        cerebro.broker.setcash(initial_cash)

        if reality_pack:
            # Wire fees (approximate using Backtrader's commission model)
            # For stocks, we use per-share if available in fee model
            # This is a simplification as Backtrader's model is less flexible
            # than Apex's internal reality models.
            comm = 0.0
            if hasattr(reality_pack.fee_model, "per_share"):
                comm = reality_pack.fee_model.per_share
            elif hasattr(reality_pack.fee_model, "stock_per_share"):
                comm = reality_pack.fee_model.stock_per_share
            else:
                # Fallback to commission parameter
                comm = commission

            cerebro.broker.setcommission(commission=comm, margin=None, mult=1.0)

            # Wire slippage
            if hasattr(reality_pack.slippage_model, "slippage_bps"):
                slippage_pct = reality_pack.slippage_model.slippage_bps / 10000.0
                cerebro.broker.set_slippage_perc(slippage_pct)
        else:
            cerebro.broker.setcommission(commission=commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe")
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        # Run backtest
        results = cerebro.run()
        strategy = results[0]

        # Extract analyzer results
        sharpe = strategy.analyzers.sharpe.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()

        return {
            "initial_cash": initial_cash,
            "final_value": cerebro.broker.getvalue(),
            "sharpe_ratio": sharpe.get("sharperatio"),
            "max_drawdown": drawdown.get("max", {}).get("drawdown"),
            "total_trades": trades.get("total", {}).get("total", 0),
            "returns": returns,
            "analyzers": {
                "sharpe": sharpe,
                "drawdown": drawdown,
                "trades": trades,
                "returns": returns,
            },
        }

else:
    # Stub classes when backtrader not available
    class ApexStrategyWrapper:  # type: ignore[no-redef]
        """Stub when backtrader not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("backtrader not installed. Run: pip install backtrader")

    def run_backtest_with_backtrader(  # type: ignore[misc]
        *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Stub when backtrader not installed."""
        raise ImportError("backtrader not installed. Run: pip install backtrader")
