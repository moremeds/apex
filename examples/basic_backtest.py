"""Basic backtest example using VectorbtEngine."""

import asyncio
from datetime import datetime, timedelta

import polars as pl

from apex.core.types import MarketDataFrame
from apex.engine import EngineConfig, EngineHarness, EngineType, quick_backtest


class SimpleMovingAverageStrategy:
    """Simple moving average crossover strategy for demonstration."""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 20):
        """Initialize strategy with moving average periods."""
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signals(self, data: pl.DataFrame) -> pl.DataFrame:
        """Generate entry/exit signals based on moving average crossover."""
        # Calculate moving averages
        df = data.with_columns([
            pl.col("close").rolling_mean(window_size=self.fast_period).alias("fast_ma"),
            pl.col("close").rolling_mean(window_size=self.slow_period).alias("slow_ma"),
        ])
        
        # Generate signals
        df = df.with_columns([
            # Entry: fast MA crosses above slow MA
            (
                (pl.col("fast_ma") > pl.col("slow_ma")) &
                (pl.col("fast_ma").shift(1) <= pl.col("slow_ma").shift(1))
            ).alias("entry"),
            # Exit: fast MA crosses below slow MA  
            (
                (pl.col("fast_ma") < pl.col("slow_ma")) &
                (pl.col("fast_ma").shift(1) >= pl.col("slow_ma").shift(1))
            ).alias("exit")
        ])
        
        return df.select(["datetime", "entry", "exit"])
    
    def get_parameters(self) -> dict:
        """Return strategy parameters."""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period
        }
    
    def validate_data(self, data: pl.DataFrame) -> bool:
        """Validate that data has required columns."""
        required_columns = ["datetime", "open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)


def create_sample_data() -> MarketDataFrame:
    """Create sample market data for demonstration."""
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate sample price data with trend and noise
    dates = []
    prices = []
    
    base_price = 100.0
    trend = 0.0002  # Small upward trend per day
    volatility = 0.02
    
    import random
    random.seed(42)  # For reproducible results
    
    current_date = start_date
    current_price = base_price
    
    while current_date <= end_date:
        # Add trend and random walk
        price_change = trend + random.normalvariate(0, volatility)
        current_price *= (1 + price_change)
        
        # Generate OHLCV data
        high_price = current_price * (1 + abs(random.normalvariate(0, 0.01)))
        low_price = current_price * (1 - abs(random.normalvariate(0, 0.01)))
        
        # Ensure OHLC consistency
        open_price = current_price + random.normalvariate(0, 0.005) * current_price
        close_price = current_price
        
        if open_price > high_price:
            high_price = open_price * 1.001
        if open_price < low_price:
            low_price = open_price * 0.999
        if close_price > high_price:
            high_price = close_price * 1.001  
        if close_price < low_price:
            low_price = close_price * 0.999
            
        volume = int(1000000 + random.normalvariate(0, 200000))
        
        dates.append(current_date)
        prices.append({
            "datetime": current_date,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": max(volume, 100000)  # Ensure positive volume
        })
        
        current_date += timedelta(days=1)
    
    df = pl.DataFrame(prices)
    
    return MarketDataFrame(
        data=df,
        symbol="SAMPLE",
        start_date=start_date,
        end_date=end_date,
        source="synthetic",
        quality_scores={}
    )


async def main():
    """Run the backtest example."""
    print("🚀 Apex Backtesting Engine Example")
    print("==================================\n")
    
    # Create sample data
    print("📊 Creating sample market data...")
    market_data = create_sample_data()
    print(f"   • Symbol: {market_data.symbol}")
    print(f"   • Period: {market_data.start_date.date()} to {market_data.end_date.date()}")
    print(f"   • Data points: {len(market_data.data):,}")
    
    # Create strategy
    print("\n🎯 Setting up strategy...")
    strategy = SimpleMovingAverageStrategy(fast_period=10, slow_period=30)
    print(f"   • Strategy: {strategy.__class__.__name__}")
    print(f"   • Parameters: {strategy.get_parameters()}")
    
    # Configure engine
    print("\n⚙️  Configuring backtesting engine...")
    config = EngineConfig(
        initial_cash=100000.0,
        commission=0.001,  # 0.1% commission
        slippage=0.0001,   # 0.01% slippage
        max_position_size=0.95  # Use up to 95% of available cash
    )
    print(f"   • Initial cash: ${config.initial_cash:,.2f}")
    print(f"   • Commission: {config.commission:.3%}")
    print(f"   • Slippage: {config.slippage:.4%}")
    
    # Method 1: Using EngineHarness (recommended)
    print("\n🔧 Running backtest with EngineHarness...")
    harness = EngineHarness(config, EngineType.VECTORBT)
    result1 = await harness.run_backtest(strategy, market_data)
    
    print("\n📈 Results (EngineHarness):")
    print(f"   • Final value: ${result1.final_value:,.2f}")
    print(f"   • Total return: {result1.total_return:.2%}")
    print(f"   • Total trades: {result1.total_trades}")
    if result1.sharpe_ratio:
        print(f"   • Sharpe ratio: {result1.sharpe_ratio:.3f}")
    if result1.max_drawdown:
        print(f"   • Max drawdown: {result1.max_drawdown:.2%}")
    if result1.win_rate:
        print(f"   • Win rate: {result1.win_rate:.2%}")
    
    # Method 2: Using quick_backtest (convenient)
    print("\n🚀 Running backtest with quick_backtest...")
    result2 = await quick_backtest(
        strategy=strategy,
        data=market_data,
        initial_cash=100000.0,
        commission=0.001
    )
    
    print("\n📈 Results (quick_backtest):")
    print(f"   • Final value: ${result2.final_value:,.2f}")
    print(f"   • Total return: {result2.total_return:.2%}")
    print(f"   • Total trades: {result2.total_trades}")
    if result2.sharpe_ratio:
        print(f"   • Sharpe ratio: {result2.sharpe_ratio:.3f}")
    if result2.max_drawdown:
        print(f"   • Max drawdown: {result2.max_drawdown:.2%}")
    if result2.win_rate:
        print(f"   • Win rate: {result2.win_rate:.2%}")
    
    print("\n✅ Backtest completed successfully!")
    print("\n💡 Next steps:")
    print("   • Try different strategy parameters")
    print("   • Implement custom strategies")
    print("   • Add walk-forward validation")
    print("   • Generate detailed reports")


if __name__ == "__main__":
    asyncio.run(main())