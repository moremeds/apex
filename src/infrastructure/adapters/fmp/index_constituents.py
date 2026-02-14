"""FMP index membership adapter.

Fetches current and historical index constituents from Financial Modeling Prep.

Live endpoints:
- /stable/sp500-constituent         → current S&P 500 members
- /stable/nasdaq-constituent        → current NASDAQ members

Historical (for backtest, anti-survivorship):
- /stable/historical/sp500-constituent → additions/removals with dates

Russell 2000 proxy:
- /stable/stock-screener with marketCapMoreThan/LowerThan + country=US
"""

from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

import requests
import yaml

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_FMP_BASE = "https://financialmodelingprep.com"
_FMP_DELAY = 0.3  # Rate limit delay between requests


def _load_fmp_key() -> str:
    """Load FMP API key from env var or config/secrets.yaml."""
    key = os.environ.get("FMP_API_KEY", "")
    if key:
        return key

    secrets_path = os.path.join(os.path.dirname(__file__), "../../../../config/secrets.yaml")
    secrets_path = os.path.normpath(secrets_path)
    try:
        with open(secrets_path) as f:
            data = yaml.safe_load(f) or {}
        key = data.get("fmp", {}).get("api_key", "")
    except FileNotFoundError:
        pass
    return key


class FMPIndexConstituentsAdapter:
    """FMP index membership fetcher.

    Provides current and historical index constituency data for building
    a momentum screener universe.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or _load_fmp_key()
        if not self._api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY env var or add to config/secrets.yaml"
            )

    def fetch_sp500(self) -> list[str]:
        """Fetch current S&P 500 constituent symbols."""
        data = self._fmp_get(f"{_FMP_BASE}/stable/sp500-constituent")
        return self._extract_symbols(data)

    def fetch_nasdaq(self) -> list[str]:
        """Fetch current NASDAQ constituent symbols."""
        data = self._fmp_get(f"{_FMP_BASE}/stable/nasdaq-constituent")
        return self._extract_symbols(data)

    def fetch_russell_proxy(
        self, cap_min: float = 300_000_000, cap_max: float = 10_000_000_000
    ) -> list[str]:
        """Fetch Russell 2000 proxy via stock screener.

        Uses FMP stock screener with market cap range $300M-$10B as proxy
        for Russell 2000 membership. Documented limitation: not exact R2K.

        Args:
            cap_min: Minimum market cap in USD.
            cap_max: Maximum market cap in USD.

        Returns:
            List of symbols matching the criteria.
        """
        data = self._fmp_get(
            f"{_FMP_BASE}/stable/stock-screener",
            params={
                "marketCapMoreThan": int(cap_min),
                "marketCapLowerThan": int(cap_max),
                "country": "US",
                "isActivelyTrading": True,
                "limit": 5000,
            },
        )
        return self._extract_symbols(data)

    def fetch_historical_sp500_changes(self) -> list[dict[str, Any]]:
        """Fetch historical S&P 500 additions and removals.

        Returns list of dicts with keys: date, symbol, addedSecurity,
        removedTicker, removedSecurity, etc.
        """
        data = self._fmp_get(f"{_FMP_BASE}/stable/historical/sp500-constituent")
        if not isinstance(data, list):
            return []
        return data

    def get_point_in_time_sp500(self, as_of: date) -> list[str]:
        """Reconstruct S&P 500 membership as of a specific date.

        Starts from current membership and reverses historical changes
        to build a point-in-time constituent list.

        Args:
            as_of: Date for which to reconstruct membership.

        Returns:
            List of symbols that were in S&P 500 on as_of date.
        """
        current = set(self.fetch_sp500())
        changes = self.fetch_historical_sp500_changes()

        # Sort changes newest first
        changes.sort(key=lambda c: c.get("date", ""), reverse=True)

        for change in changes:
            change_date_str = change.get("date", "")
            if not change_date_str:
                continue

            try:
                change_date = date.fromisoformat(change_date_str)
            except ValueError:
                continue

            # Only undo changes that happened AFTER as_of
            if change_date <= as_of:
                break

            added = change.get("symbol", "")
            removed = change.get("removedTicker", "")

            # Reverse: if it was added after as_of, remove it
            if added:
                current.discard(added)
            # Reverse: if it was removed after as_of, add it back
            if removed:
                current.add(removed)

        return sorted(current)

    def get_combined_universe(
        self,
        indices: list[str] | None = None,
        russell_proxy: bool = True,
        cap_min: float = 300_000_000,
        cap_max: float = 10_000_000_000,
    ) -> list[str]:
        """Fetch combined universe from multiple indices.

        Args:
            indices: List of index names to include ("sp500", "nasdaq").
            russell_proxy: Whether to include Russell 2000 proxy.
            cap_min: Min market cap for Russell proxy.
            cap_max: Max market cap for Russell proxy.

        Returns:
            Deduplicated sorted list of symbols.
        """
        all_symbols: set[str] = set()
        indices = indices or ["sp500", "nasdaq"]

        for idx in indices:
            if idx == "sp500":
                symbols = self.fetch_sp500()
                logger.info(f"S&P 500: {len(symbols)} symbols")
                all_symbols.update(symbols)
            elif idx == "nasdaq":
                symbols = self.fetch_nasdaq()
                logger.info(f"NASDAQ: {len(symbols)} symbols")
                all_symbols.update(symbols)
            else:
                logger.warning(f"Unknown index: {idx}")

        if russell_proxy:
            symbols = self.fetch_russell_proxy(cap_min, cap_max)
            logger.info(f"Russell proxy: {len(symbols)} symbols")
            all_symbols.update(symbols)

        logger.info(f"Combined universe: {len(all_symbols)} unique symbols")
        return sorted(all_symbols)

    def _fmp_get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Make GET request to FMP API with rate limiting."""
        request_params: dict[str, Any] = {"apikey": self._api_key}
        if params:
            request_params.update(params)

        time.sleep(_FMP_DELAY)
        try:
            resp = requests.get(url, params=request_params, timeout=30)
            if resp.status_code == 402:
                logger.warning(f"FMP free tier limit reached for {url}")
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

    @staticmethod
    def _extract_symbols(data: Any) -> list[str]:
        """Extract symbol list from FMP API response."""
        if not isinstance(data, list):
            return []
        symbols = []
        for item in data:
            sym = item.get("symbol", "")
            if sym and isinstance(sym, str):
                symbols.append(sym)
        return sorted(symbols)
