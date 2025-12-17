"""
Backtest engine that orchestrates historical data replay.

The engine coordinates:
- SimulatedClock: Time progression
- DataFeed: Historical data replay
- SimulatedExecution: Order matching
- Strategy: Signal generation
- Metrics: Performance calculation

Usage:
    config = BacktestConfig(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 6, 30),
        symbols=["AAPL", "MSFT"],
        initial_capital=100000.0,
    )

    engine = BacktestEngine(config)
    engine.set_strategy(MyStrategy, params={"window": 20})
    engine.set_data_feed(CsvDataFeed(csv_dir="data"))

    result = await engine.run()
    result.print_summary()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Type, Any
import time
import logging

from ...domain.clock import SimulatedClock
from ...domain.strategy.base import Strategy, StrategyContext
from ...domain.strategy.scheduler import SimulatedScheduler
from ...domain.strategy.registry import get_strategy_class
from ...domain.backtest.backtest_result import (
    BacktestResult,
    PerformanceMetrics,
    RiskMetrics,
    TradeMetrics,
    CostMetrics,
    TradeRecord,
)
from ...domain.backtest.backtest_spec import BacktestSpec
from ...domain.events.domain_events import BarData, QuoteTick, PositionSnapshot

from .simulated_execution import SimulatedExecution, FillModel
from .data_feeds import DataFeed, CsvDataFeed
from .trade_tracker import TradeTracker

from ...domain.reality import RealityModelPack, get_preset_pack

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    start_date: date
    end_date: date
    symbols: List[str]
    initial_capital: float = 100000.0

    # Data settings
    data_type: str = "bars"  # "bars" or "ticks"
    bar_size: str = "1d"

    # Execution settings (legacy - used when reality_pack is None)
    fill_model: FillModel = FillModel.IMMEDIATE
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005

    # Reality modeling (preferred over legacy execution settings)
    # Can be a RealityModelPack instance or a preset name string
    reality_pack: Optional[RealityModelPack] = None
    reality_pack_name: Optional[str] = None  # e.g., "ib", "futu_us", "simple"

    # Strategy
    strategy_name: Optional[str] = None
    strategy_params: Dict[str, Any] = field(default_factory=dict)

    def get_reality_pack(self) -> Optional[RealityModelPack]:
        """Get the reality pack, resolving from name if needed."""
        if self.reality_pack is not None:
            return self.reality_pack
        if self.reality_pack_name:
            return get_preset_pack(self.reality_pack_name)
        return None

    @classmethod
    def from_spec(cls, spec: BacktestSpec) -> "BacktestConfig":
        """Create config from BacktestSpec."""
        # Check if spec has reality pack settings
        reality_pack_name = None
        if hasattr(spec.execution, 'reality_pack') and spec.execution.reality_pack:
            reality_pack_name = spec.execution.reality_pack

        return cls(
            start_date=spec.data.start_date or date(2024, 1, 1),
            end_date=spec.data.end_date or date(2024, 12, 31),
            symbols=spec.get_symbols(),
            initial_capital=spec.execution.initial_capital,
            bar_size=spec.data.bar_size,
            strategy_name=spec.strategy.name,
            strategy_params=spec.strategy.params,
            reality_pack_name=reality_pack_name,
        )


class BacktestEngine:
    """
    Backtest engine that replays historical data.

    Orchestrates the backtest process:
    1. Initialize simulated clock and execution
    2. Create strategy with simulated context
    3. Replay historical data through strategy
    4. Collect fills and update positions
    5. Calculate performance metrics
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration.
        """
        self._config = config

        # Initialize simulated clock
        start_time = datetime.combine(config.start_date, datetime.min.time())
        self._clock = SimulatedClock(start_time)

        # Initialize scheduler
        self._scheduler = SimulatedScheduler(self._clock)

        # Initialize simulated execution
        # Use reality pack if configured, otherwise use legacy parameters
        reality_pack = config.get_reality_pack()
        self._execution = SimulatedExecution(
            clock=self._clock,
            fill_model=config.fill_model,
            slippage_bps=config.slippage_bps,
            commission_per_share=config.commission_per_share,
            reality_pack=reality_pack,
        )

        if reality_pack:
            logger.info(f"Using reality pack: {reality_pack.name}")

        # Initialize context
        self._context = StrategyContext(
            clock=self._clock,
            scheduler=self._scheduler,
            positions={},
            account=None,
            execution=self._execution,
        )

        # Components to be set
        self._strategy: Optional[Strategy] = None
        self._data_feed: Optional[DataFeed] = None

        # Tracking
        self._cash = config.initial_capital
        self._equity_curve: List[Dict[str, Any]] = []
        self._trades: List[TradeRecord] = []
        self._running = False

        # Trade tracker for entry/exit matching
        self._trade_tracker = TradeTracker(matching_method="FIFO")

    @property
    def clock(self) -> SimulatedClock:
        """Get simulated clock."""
        return self._clock

    @property
    def execution(self) -> SimulatedExecution:
        """Get simulated execution."""
        return self._execution

    def set_strategy(
        self,
        strategy_class: Optional[Type[Strategy]] = None,
        strategy_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Set strategy to run.

        Args:
            strategy_class: Strategy class (if providing class directly).
            strategy_name: Strategy name in registry (alternative to class).
            params: Strategy parameters.
        """
        if strategy_class is None and strategy_name:
            strategy_class = get_strategy_class(strategy_name)
            if strategy_class is None:
                raise ValueError(f"Unknown strategy: {strategy_name}")

        if strategy_class is None:
            raise ValueError("Must provide either strategy_class or strategy_name")

        params = params or self._config.strategy_params

        self._strategy = strategy_class(
            strategy_id=f"backtest-{strategy_name or strategy_class.__name__}",
            symbols=self._config.symbols,
            context=self._context,
            **params,
        )

        # Wire up order callback
        self._strategy.on_order_callback(self._handle_order)

        logger.info(f"Strategy set: {self._strategy}")

    def set_data_feed(self, data_feed: DataFeed) -> None:
        """
        Set data feed for historical data.

        Args:
            data_feed: Data feed instance.
        """
        self._data_feed = data_feed
        logger.info(f"Data feed set: {data_feed.__class__.__name__}")

    async def run(self) -> BacktestResult:
        """
        Run the backtest.

        Returns:
            BacktestResult with all metrics.
        """
        if self._strategy is None:
            raise ValueError("Strategy not set. Call set_strategy() first.")

        if self._data_feed is None:
            raise ValueError("Data feed not set. Call set_data_feed() first.")

        start_time = time.time()
        logger.info(
            f"Starting backtest: {self._config.start_date} to {self._config.end_date}"
        )

        self._running = True
        self._cash = self._config.initial_capital
        self._equity_curve.clear()
        self._trades.clear()

        # Load data
        await self._data_feed.load()
        logger.info(f"Loaded {self._data_feed.bar_count} bars")

        # Start strategy
        self._strategy.start()

        # Process all bars
        bar_count = 0
        async for bar in self._data_feed.stream_bars():
            if not self._running:
                break

            # Advance clock to bar time
            self._clock.advance_to(bar.timestamp)

            # Process scheduled actions
            self._scheduler.advance_to(bar.timestamp)

            # Update execution with bar price
            tick = QuoteTick(
                symbol=bar.symbol,
                bid=bar.close,
                ask=bar.close,
                last=bar.close,
                volume=bar.volume,
                timestamp=bar.timestamp,
                source="backtest",
            )
            self._execution.update_price(tick)

            # Update context market data
            self._context.market_data[bar.symbol] = tick

            # Process any fills
            self._process_fills()

            # Feed bar to strategy
            self._strategy.on_bar(bar)

            # Trigger bar close scheduled actions
            self._scheduler.trigger_bar_close_actions()

            # Record equity
            equity = self._calculate_equity()
            self._equity_curve.append({
                "date": bar.timestamp.date().isoformat(),
                "timestamp": bar.timestamp.isoformat(),
                "equity": equity,
                "cash": self._cash,
                "position_value": self._execution.get_total_position_value(),
            })

            bar_count += 1

            # Progress logging
            if bar_count % 1000 == 0:
                logger.info(
                    f"Processed {bar_count} bars, "
                    f"time={self._clock.now().date()}, "
                    f"equity=${equity:,.2f}"
                )

        # Stop strategy
        self._strategy.stop()

        run_duration = time.time() - start_time
        logger.info(
            f"Backtest complete: {bar_count} bars in {run_duration:.2f}s"
        )

        return self._generate_result(run_duration)

    def _handle_order(self, order) -> None:
        """Handle order request from strategy."""
        # Use sync version for backtest context
        self._execution.submit_order_sync(order)

    def _process_fills(self) -> None:
        """Process pending fills and update positions."""
        fills = self._execution.get_pending_fills()

        for fill in fills:
            # Update cash (with multiplier for options/futures)
            multiplier = getattr(fill, 'multiplier', 1) or 1
            cost = fill.quantity * fill.price * multiplier
            if fill.side == "BUY":
                self._cash -= cost + fill.commission
            else:
                self._cash += cost - fill.commission

            # Update context positions
            pos = self._execution.get_position(fill.symbol)
            if pos and pos.quantity != 0:
                self._context.positions[fill.symbol] = PositionSnapshot(
                    symbol=fill.symbol,
                    underlying=fill.underlying,
                    asset_type=fill.asset_type,
                    quantity=pos.quantity,
                    avg_price=pos.avg_price,
                    timestamp=fill.timestamp,
                    source="backtest",
                )
            elif fill.symbol in self._context.positions:
                del self._context.positions[fill.symbol]

            # Route fill to strategy
            self._strategy.on_fill(fill)

            # Track fill for entry/exit matching
            trade = self._trade_tracker.record_fill(fill)
            if trade:
                self._trades.append(trade)

            logger.debug(f"Fill processed: {fill.side} {fill.quantity} {fill.symbol}")

    def _calculate_equity(self) -> float:
        """Calculate current portfolio equity."""
        return self._cash + self._execution.get_total_position_value()

    def _generate_result(self, run_duration: float) -> BacktestResult:
        """Generate backtest result with metrics."""
        from datetime import date

        final_equity = self._calculate_equity()
        initial = self._config.initial_capital

        total_return = (final_equity - initial) / initial if initial > 0 else 0

        # Calculate max drawdown with proper calendar day duration
        peak = initial
        max_drawdown = 0.0
        max_drawdown_duration_days = 0
        peak_date = None
        max_drawdown_peak_date = None
        max_drawdown_trough_date = None

        for point in self._equity_curve:
            equity = point["equity"]
            point_date = date.fromisoformat(point["date"])

            if equity > peak:
                peak = equity
                peak_date = point_date

            drawdown = (peak - equity) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_peak_date = peak_date
                max_drawdown_trough_date = point_date

        # Calculate actual calendar days in max drawdown
        if max_drawdown_peak_date and max_drawdown_trough_date:
            max_drawdown_duration_days = (max_drawdown_trough_date - max_drawdown_peak_date).days

        # Count trading days
        trading_days = len(set(p["date"] for p in self._equity_curve))

        # Performance metrics
        performance = PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return * 100,
            cagr=self._calculate_cagr(initial, final_equity, trading_days),
            annualized_return=total_return * 252 / max(trading_days, 1),
        )

        # Risk metrics
        risk = RiskMetrics(
            max_drawdown=max_drawdown * 100,
            max_drawdown_duration_days=max_drawdown_duration_days,
            sharpe_ratio=self._calculate_sharpe(),
        )

        # Trade metrics from tracker
        trade_metrics = self._trade_tracker.calculate_metrics()
        total_commission = sum(t.commission for t in self._trades)

        # Cost metrics
        costs = CostMetrics(
            total_commission=total_commission,
            cost_pct_of_capital=(total_commission / initial * 100) if initial > 0 else 0,
        )

        return BacktestResult(
            strategy_name=self._config.strategy_name or self._strategy.__class__.__name__,
            strategy_id=self._strategy.strategy_id,
            start_date=self._config.start_date,
            end_date=self._config.end_date,
            trading_days=trading_days,
            initial_capital=initial,
            final_capital=final_equity,
            symbols=self._config.symbols,
            performance=performance,
            risk=risk,
            trades=trade_metrics,
            costs=costs,
            equity_curve=self._equity_curve,
            trade_log=self._trades,
            run_duration_seconds=run_duration,
            engine="apex",
        )

    def _calculate_cagr(self, initial: float, final: float, days: int) -> float:
        """Calculate Compound Annual Growth Rate."""
        if initial <= 0 or days <= 0:
            return 0.0
        years = days / 252  # Trading days per year
        if years <= 0:
            return 0.0
        return ((final / initial) ** (1 / years) - 1) * 100

    def _calculate_sharpe(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio from equity curve.

        For intraday bars, aggregates to daily returns first to ensure
        correct annualization (252 trading days).
        """
        if len(self._equity_curve) < 2:
            return 0.0

        import statistics
        from collections import defaultdict

        # Aggregate equity by day (take last equity of each day)
        daily_equity: dict[str, float] = {}
        for point in self._equity_curve:
            day = point["date"]
            daily_equity[day] = point["equity"]  # Last value for each day wins

        # Sort by date and calculate daily returns
        sorted_days = sorted(daily_equity.keys())
        if len(sorted_days) < 2:
            return 0.0

        daily_returns = []
        for i in range(1, len(sorted_days)):
            prev_eq = daily_equity[sorted_days[i - 1]]
            curr_eq = daily_equity[sorted_days[i]]
            if prev_eq > 0:
                daily_returns.append((curr_eq - prev_eq) / prev_eq)

        if not daily_returns or len(daily_returns) < 2:
            return 0.0

        mean_return = statistics.mean(daily_returns)
        std_return = statistics.stdev(daily_returns)

        if std_return == 0:
            return 0.0

        # Annualize using 252 trading days
        daily_rf = risk_free_rate / 252
        sharpe = (mean_return - daily_rf) / std_return * (252 ** 0.5)
        return sharpe

    def stop(self) -> None:
        """Stop the backtest early."""
        self._running = False
        if self._strategy:
            self._strategy.stop()
