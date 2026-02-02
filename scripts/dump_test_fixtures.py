#!/usr/bin/env python3
"""
Dump test fixture data for TrendPulse unit tests.

Fetches AAPL and SPY 2024 daily bars via yfinance and saves as parquet
in tests/fixtures/. Includes real-world gaps and missing values.

Usage:
    python scripts/dump_test_fixtures.py
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


def main() -> None:
    output_dir = Path("tests/fixtures")
    output_dir.mkdir(parents=True, exist_ok=True)

    symbols = ["AAPL", "SPY"]
    start = "2024-01-01"
    end = "2024-12-31"

    for symbol in symbols:
        print(f"Fetching {symbol} {start} to {end}...")
        df = yf.download(symbol, start=start, end=end, auto_adjust=True)

        if df.empty:
            print(f"  WARNING: No data for {symbol}")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        df.index.name = "date"

        # Add symbol column
        df.insert(0, "symbol", symbol)

        # Keep: symbol, date, open, high, low, close, volume
        keep = ["symbol", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]

        out_path = output_dir / f"{symbol.lower()}_2024_daily.parquet"
        df.to_parquet(out_path)
        print(f"  Saved {len(df)} rows to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
