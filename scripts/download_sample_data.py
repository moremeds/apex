#!/usr/bin/env python
"""
Download sample data for testing and development.
Uses yfinance to download historical market data.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import yfinance as yf
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Please run: uv pip install -e .[dev,ml]")
    sys.exit(1)


def download_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path
) -> Path:
    """Download stock data for a given symbol."""
    print(f"Downloading {symbol} data...")
    
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        print(f"⚠️  No data found for {symbol}")
        return None
    
    # Clean column names
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df = df.reset_index()
    
    # Save as CSV
    output_file = output_dir / f"{symbol.lower()}.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {symbol} data to {output_file}")
    
    return output_file


def download_crypto_data(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Path
) -> Path:
    """Download cryptocurrency data."""
    print(f"Downloading {symbol} crypto data...")
    
    ticker = yf.Ticker(f"{symbol}-USD")
    df = ticker.history(start=start_date, end=end_date)
    
    if df.empty:
        print(f"⚠️  No data found for {symbol}")
        return None
    
    # Clean column names
    df.columns = [col.lower().replace(" ", "_") for col in df.columns]
    df = df.reset_index()
    
    # Save as CSV
    output_file = output_dir / f"{symbol.lower()}_crypto.csv"
    df.to_csv(output_file, index=False)
    print(f"✓ Saved {symbol} crypto data to {output_file}")
    
    return output_file


def create_sample_config(config_dir: Path):
    """Create sample configuration files."""
    print("\n📝 Creating sample configurations...")
    
    # Default config
    default_config = """# Default Configuration for Apex Backtesting

data:
  cache_dir: "~/.apex/cache"
  quality_threshold: 0.7
  
engine:
  initial_cash: 100000
  commission: 0.001
  slippage: 0.001
  
logging:
  level: "INFO"
  file: "logs/apex.log"
"""
    
    config_file = config_dir / "default.yaml"
    config_file.write_text(default_config)
    print(f"✓ Created {config_file}")
    
    # Strategy config
    strategy_config = """# Moving Average Cross Strategy Configuration

strategy:
  name: "MovingAverageCross"
  parameters:
    fast_period: 20
    slow_period: 50
    
position_sizing:
  method: "fixed"
  size: 0.95  # Use 95% of available capital
  
risk_management:
  stop_loss: 0.02  # 2% stop loss
  take_profit: 0.05  # 5% take profit
"""
    
    strategy_file = config_dir / "strategies" / "ma_cross.yaml"
    strategy_file.parent.mkdir(exist_ok=True)
    strategy_file.write_text(strategy_config)
    print(f"✓ Created {strategy_file}")


def main():
    """Main function to download sample data."""
    print("📊 Downloading sample data for Apex")
    print("=" * 50)
    
    # Create data directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Set date range (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    print(f"Date range: {start_str} to {end_str}\n")
    
    # Download stock data
    stocks = ["SPY", "AAPL", "MSFT", "GOOGL", "TSLA"]
    print("📈 Downloading stock data...")
    for symbol in stocks:
        download_stock_data(symbol, start_str, end_str, raw_dir)
    
    # Download crypto data
    cryptos = ["BTC", "ETH"]
    print("\n🪙 Downloading cryptocurrency data...")
    for symbol in cryptos:
        download_crypto_data(symbol, start_str, end_str, raw_dir)
    
    # Create sample configs
    config_dir = Path("configs")
    config_dir.mkdir(exist_ok=True)
    create_sample_config(config_dir)
    
    print("\n" + "=" * 50)
    print("✅ Sample data download complete!")
    print(f"\nData saved to: {raw_dir.absolute()}")
    print(f"Configs saved to: {config_dir.absolute()}")
    print("\nYou can now run backtests with:")
    print("  uv run python scripts/run_backtest.py --strategy ma_cross --symbol SPY")


if __name__ == "__main__":
    main()