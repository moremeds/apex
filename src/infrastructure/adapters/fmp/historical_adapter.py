"""FMP historical price data adapter.

Fetches daily and intraday OHLCV bars from Financial Modeling Prep.

Endpoints:
- Daily:    GET /stable/historical-price-eod/full?symbol=X&from=Y&to=Z
- Intraday: GET /stable/historical-chart/{interval}?symbol=X&from=Y&to=Z

Daily endpoint returns an ``adjClose`` column (split + dividend adjusted).
When present, the adapter uses it as ``close`` so that downstream consumers
(momentum ranking, backtests) operate on adjusted prices by default.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import pandas as pd
import requests

from src.utils.logging_setup import get_logger

from .index_constituents import _FMP_BASE, _FMP_DELAY, _load_fmp_key

logger = get_logger(__name__)

# APEX timeframe -> FMP interval string
_TIMEFRAME_MAP: dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "daily",
}


class FMPHistoricalAdapter:
    """FMP historical price data adapter.

    Provides daily and intraday OHLCV bar data with rate limiting and
    graceful error handling (402/429).
    """

    def __init__(
        self,
        api_key: str | None = None,
        request_delay: float | None = None,
    ) -> None:
        self._api_key = api_key or _load_fmp_key()
        if not self._api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY env var or add to config/secrets.yaml"
            )
        # Configurable delay between requests (seconds).
        # Falls back to the shared _FMP_DELAY constant from index_constituents.
        self._request_delay = request_delay if request_delay is not None else _FMP_DELAY

    def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars for a single symbol.

        Args:
            symbol: Ticker symbol (e.g. "AAPL").
            timeframe: One of 1m, 5m, 15m, 30m, 1h, 4h, 1d.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            DataFrame with columns: open, high, low, close, volume.
            ``close`` is adjusted (split + dividend) when the endpoint
            provides ``adjClose``.
            DatetimeIndex in ascending order. Empty DataFrame on failure.
        """
        fmp_interval = _TIMEFRAME_MAP.get(timeframe)
        if fmp_interval is None:
            logger.warning(f"Unsupported FMP timeframe: {timeframe}")
            return pd.DataFrame()

        if fmp_interval == "daily":
            return self._fetch_daily(symbol, start_date, end_date)
        return self._fetch_intraday(symbol, fmp_interval, start_date, end_date)

    def fetch_bars_batch(
        self,
        symbols: list[str],
        timeframe: str,
        start_date: date,
        end_date: date,
    ) -> dict[str, pd.DataFrame]:
        """Batch fetch with rate limiting.

        Args:
            symbols: List of ticker symbols.
            timeframe: Bar timeframe.
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            Dict mapping symbol -> OHLCV DataFrame.
        """
        result: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.fetch_bars(symbol, timeframe, start_date, end_date)
            if not df.empty:
                result[symbol] = df
        return result

    # ── Internal helpers ───────────────────────────────────────────

    def _fetch_daily(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch daily bars via /stable/historical-price-eod/full."""
        url = f"{_FMP_BASE}/stable/historical-price-eod/full"
        data = self._fmp_get(
            url,
            params={
                "symbol": symbol,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )
        return self._parse_response(data, symbol, use_adj_close=True)

    def _fetch_intraday(
        self,
        symbol: str,
        interval: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch intraday bars via /stable/historical-chart/{interval}."""
        url = f"{_FMP_BASE}/stable/historical-chart/{interval}"
        data = self._fmp_get(
            url,
            params={
                "symbol": symbol,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )
        return self._parse_response(data, symbol, use_adj_close=False)

    def _parse_response(
        self, data: Any, symbol: str, *, use_adj_close: bool = False
    ) -> pd.DataFrame:
        """Parse FMP JSON response into standardized DataFrame.

        Args:
            data: Raw JSON list from FMP.
            symbol: Ticker for logging.
            use_adj_close: If True and ``adjClose`` column is present,
                overwrite ``close`` with it so downstream gets adjusted prices.
        """
        if not isinstance(data, list) or not data:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(data)

            # FMP returns 'date' column for both daily and intraday
            if "date" not in df.columns:
                logger.warning(f"FMP response for {symbol} missing 'date' column")
                return pd.DataFrame()

            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            # Standardize column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Prefer adjusted close when available (daily endpoint)
            if use_adj_close and "adjclose" in df.columns:
                df["close"] = df["adjclose"]

            ohlcv_cols = ["open", "high", "low", "close", "volume"]
            available = [c for c in ohlcv_cols if c in df.columns]
            if "close" not in available:
                logger.warning(f"FMP response for {symbol} missing 'close' column")
                return pd.DataFrame()

            df = df[available]
            df = df.sort_index()  # Ensure ascending order
            df = df.dropna(subset=["close"])

            logger.debug(f"FMP: {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"FMP parse error for {symbol}: {e}")
            return pd.DataFrame()

    def _fmp_get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Make GET request to FMP API with rate limiting."""
        request_params: dict[str, Any] = {"apikey": self._api_key}
        if params:
            request_params.update(params)

        time.sleep(self._request_delay)
        try:
            resp = requests.get(url, params=request_params, timeout=30)
            if resp.status_code == 402:
                logger.warning(
                    f"FMP 402 (Payment Required) for {url}. "
                    "Check your FMP plan includes this endpoint."
                )
                return []
            if resp.status_code == 429:
                logger.warning("FMP rate limit hit, backing off 5s")
                time.sleep(5)
                resp = requests.get(url, params=request_params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            logger.error(f"FMP request failed: {e}")
            return []
