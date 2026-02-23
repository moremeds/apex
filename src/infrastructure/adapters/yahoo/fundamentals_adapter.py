"""Yahoo Finance Fundamentals Adapter — batch fundamental data fetcher.

Fetches fundamental/statistical data for multiple symbols in parallel
using yfinance's Ticker.info endpoint with ThreadPoolExecutor.

Usage:
    adapter = YahooFundamentalsAdapter(max_workers=20)
    data = adapter.fetch_fundamentals(
        symbols=["AAPL", "MSFT", "GOOGL"],
        fields=["floatShares", "sharesOutstanding", "beta"],
    )
    # → {"AAPL": {"floatShares": 14.6B, ...}, "MSFT": {...}, ...}
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

try:
    import yfinance as yf
except ImportError:
    yf = None

from ....utils.logging_setup import get_logger

logger = get_logger(__name__)

# All available info fields from yfinance Ticker.info that are useful
# for screening and fundamental analysis.
AVAILABLE_FIELDS = {
    # Share structure
    "floatShares",
    "sharesOutstanding",
    "sharesShort",
    "sharesPercentSharesOut",
    "shortRatio",
    "shortPercentOfFloat",
    "impliedSharesOutstanding",
    # Valuation
    "marketCap",
    "enterpriseValue",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "priceToSalesTrailing12Months",
    "enterpriseToRevenue",
    "enterpriseToEbitda",
    # Profitability
    "profitMargins",
    "operatingMargins",
    "grossMargins",
    "returnOnAssets",
    "returnOnEquity",
    # Dividends
    "dividendYield",
    "dividendRate",
    "payoutRatio",
    "fiveYearAvgDividendYield",
    # Growth
    "revenueGrowth",
    "earningsGrowth",
    "earningsQuarterlyGrowth",
    "revenueQuarterlyGrowth",
    # Risk
    "beta",
    "auditRisk",
    "overallRisk",
    # Trading
    "averageVolume",
    "averageVolume10days",
    "averageDailyVolume10Day",
    "fiftyDayAverage",
    "twoHundredDayAverage",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    # Financials
    "totalRevenue",
    "totalDebt",
    "totalCash",
    "totalCashPerShare",
    "debtToEquity",
    "currentRatio",
    "bookValue",
    "freeCashflow",
    "operatingCashflow",
    "earningsPerShare",
}


class YahooFundamentalsAdapter:
    """Batch fundamentals fetcher using yfinance.

    Fetches selected fields from Yahoo Finance's quoteSummary endpoint
    for many symbols in parallel. Designed for one-time batch operations
    (universe screening, cache building) rather than live trading.
    """

    def __init__(self, max_workers: int = 20) -> None:
        """Initialize adapter.

        Args:
            max_workers: Number of parallel threads for fetching.
                20 is a good balance between speed and Yahoo rate limits.
        """
        self._max_workers = max_workers
        if yf is None:
            logger.warning(
                "yfinance not installed. YahooFundamentalsAdapter will return empty data."
            )

    def fetch_fundamentals(
        self,
        symbols: list[str],
        fields: list[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Fetch selected fundamental fields for multiple symbols.

        Args:
            symbols: List of ticker symbols to fetch.
            fields: List of field names to extract (default: all available).
                See AVAILABLE_FIELDS for valid names.

        Returns:
            {symbol: {field: value, ...}} for symbols where at least one
            field was successfully fetched. Symbols that fail are omitted.
        """
        if yf is None:
            logger.error("yfinance not installed")
            return {}

        if not symbols:
            return {}

        requested = set(fields) if fields else AVAILABLE_FIELDS
        result: dict[str, dict[str, Any]] = {}
        success = 0
        failures = 0

        t0 = time.monotonic()
        logger.info(
            f"Fetching {len(requested)} fields for {len(symbols)} symbols "
            f"({self._max_workers} workers)..."
        )

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(self._fetch_one, sym, requested): sym for sym in symbols}
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    data = future.result()
                    if data:
                        result[sym] = data
                        success += 1
                    else:
                        failures += 1
                except Exception as e:
                    failures += 1
                    logger.debug(f"Failed to fetch {sym}: {e}")

                # Progress logging every 500 symbols
                done = success + failures
                if done % 500 == 0 and done > 0:
                    elapsed = time.monotonic() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  Progress: {done}/{len(symbols)} "
                        f"({success} ok, {failures} failed, {rate:.0f}/s)"
                    )

        elapsed = time.monotonic() - t0
        logger.info(
            f"Fetched fundamentals: {success}/{len(symbols)} ok, "
            f"{failures} failed in {elapsed:.1f}s "
            f"({len(symbols) / elapsed:.0f}/s)"
        )
        return result

    def fetch_float_shares(self, symbols: list[str]) -> dict[str, float]:
        """Convenience method: fetch floatShares for multiple symbols.

        Returns:
            {symbol: floatShares} for symbols with valid float data.
        """
        raw = self.fetch_fundamentals(symbols, fields=["floatShares"])
        result: dict[str, float] = {}
        for sym, data in raw.items():
            fs = data.get("floatShares")
            if fs is not None and isinstance(fs, (int, float)) and fs > 0:
                result[sym] = float(fs)
        logger.info(f"Float shares: {len(result)}/{len(symbols)} symbols have valid data")
        return result

    @staticmethod
    def _fetch_one(symbol: str, fields: set[str]) -> dict[str, Any] | None:
        """Fetch info for a single symbol, extract requested fields."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info:
                return None

            extracted: dict[str, Any] = {}
            for field in fields:
                val = info.get(field)
                if val is not None:
                    extracted[field] = val

            return extracted if extracted else None
        except Exception:
            return None
