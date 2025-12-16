"""
Backtrader integration adapter.

Wraps Apex strategies to run within Backtrader's execution engine.
This provides production-grade backtesting with Backtrader's robust
fill simulation, slippage models, and analyzers.

Usage:
    import backtrader as bt
    from src.infrastructure.backtest.backtrader_adapter import (
        ApexStrategyWrapper,
        run_backtest_with_backtrader,
    )
    from src.domain.strategy.examples import MovingAverageCrossStrategy

    # Create wrapper
    cerebro = bt.Cerebro()
    cerebro.addstrategy(
        ApexStrategyWrapper,
        apex_strategy_class=MovingAverageCrossStrategy,
        apex_params={'short_window': 10, 'long_window': 50},
    )

    # Add data and run
    cerebro.adddata(bt.feeds.YahooFinanceData(dataname='AAPL'))
    results = cerebro.run()

Note: Requires backtrader package: pip install backtrader
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Type, Dict, Any, Optional, List
import logging

# Check if backtrader is available
try:
    import backtrader as bt
    BACKTRADER_AVAILABLE = True
except ImportError:
    BACKTRADER_AVAILABLE = False
    bt = None

from ...domain.clock import BacktraderClock
from ...domain.strategy.base import Strategy, StrategyContext
from ...domain.strategy.scheduler import NullScheduler
from ...domain.events.domain_events import QuoteTick, BarData, PositionSnapshot
from ...domain.interfaces.execution_provider import OrderRequest

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BacktraderScheduler:
    """
    Scheduler implementation for Backtrader integration.

    Maps Apex scheduler calls to Backtrader's timer system.
    Scheduled actions are triggered via notify_timer and next() methods.
    """

    def __init__(self, bt_strategy: "bt.Strategy"):
        """
        Initialize Backtrader scheduler.

        Args:
            bt_strategy: Backtrader strategy instance.
        """
        self._bt = bt_strategy
        self._bar_close_callbacks: List[callable] = []
        self._timer_callbacks: Dict[str, callable] = {}

    def schedule_daily(
        self,
        action_id: str,
        callback: callable,
        time_of_day: "datetime.time",
    ) -> None:
        """Schedule daily action using Backtrader's timer."""
        if not BACKTRADER_AVAILABLE:
            return

        self._timer_callbacks[action_id] = callback

        # Note: Backtrader timer setup would go here
        # self._bt.add_timer(...)
        logger.debug(f"Scheduled daily action: {action_id} at {time_of_day}")

    def schedule_on_bar_close(
        self,
        action_id: str,
        callback: callable,
    ) -> None:
        """Schedule action for bar close."""
        self._bar_close_callbacks.append(callback)
        logger.debug(f"Scheduled on_bar_close: {action_id}")

    def schedule_before_close(
        self,
        action_id: str,
        callback: callable,
        minutes_before: int = 5,
    ) -> None:
        """Schedule action before market close."""
        # In Backtrader, this would use add_timer with session offset
        logger.debug(f"Scheduled before_close: {action_id}")

    def trigger_bar_close(self) -> None:
        """Trigger all bar close callbacks."""
        for callback in self._bar_close_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Bar close callback error: {e}")


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

        params = (
            ('apex_strategy_class', None),  # Required: Apex Strategy class
            ('apex_params', {}),            # Optional: Strategy parameters
            ('apex_strategy_id', 'bt-apex'), # Optional: Strategy ID
        )

        def __init__(self):
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

            logger.info(
                f"ApexStrategyWrapper initialized: {self._apex.__class__.__name__}"
            )

        def start(self):
            """Called when strategy starts."""
            self._apex.start()

        def stop(self):
            """Called when strategy stops."""
            self._apex.stop()

        def next(self):
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

            # Trigger scheduled bar close actions
            self._scheduler.trigger_bar_close()

        def notify_order(self, order):
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

        def _get_timeframe(self, data) -> str:
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
    ) -> Dict[str, Any]:
        """
        Run backtest using Backtrader engine with Apex strategy.

        Args:
            apex_strategy_class: Apex strategy class to run.
            data_feeds: List of Backtrader data feeds.
            initial_cash: Starting capital.
            commission: Commission rate.
            strategy_params: Parameters for Apex strategy.

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
        cerebro.broker.setcommission(commission=commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

        # Run backtest
        results = cerebro.run()
        strategy = results[0]

        # Extract analyzer results
        sharpe = strategy.analyzers.sharpe.get_analysis()
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trades = strategy.analyzers.trades.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()

        return {
            'initial_cash': initial_cash,
            'final_value': cerebro.broker.getvalue(),
            'sharpe_ratio': sharpe.get('sharperatio'),
            'max_drawdown': drawdown.get('max', {}).get('drawdown'),
            'total_trades': trades.get('total', {}).get('total', 0),
            'returns': returns,
            'analyzers': {
                'sharpe': sharpe,
                'drawdown': drawdown,
                'trades': trades,
                'returns': returns,
            },
        }

else:
    # Stub classes when backtrader not available
    class ApexStrategyWrapper:
        """Stub when backtrader not installed."""
        def __init__(self, *args, **kwargs):
            raise ImportError("backtrader not installed. Run: pip install backtrader")

    def run_backtest_with_backtrader(*args, **kwargs):
        """Stub when backtrader not installed."""
        raise ImportError("backtrader not installed. Run: pip install backtrader")
