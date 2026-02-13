"""FMP (Financial Modeling Prep) earnings data adapter.

Fetches earnings calendar, per-symbol earnings history, and analyst grades
from FMP's stable API. Uses yfinance for OHLCV price data around earnings.

Free tier covers ~87 major stocks; 402 responses are skipped gracefully.

BMO/AMC handling:
    - BMO (Before Market Open): reaction date = report_date
    - AMC (After Market Close): reaction date = next trading day
    The FMP earnings calendar returns a `time` field with these values.
"""

from __future__ import annotations

import os
import time
from datetime import date, timedelta
from typing import Any

import pandas_market_calendars as mcal
import requests
import yaml

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_FMP_BASE = "https://financialmodelingprep.com"

# Rate limits (seconds between requests)
_FMP_DELAY = 0.3
_YF_DELAY = 0.5

# NYSE calendar for BMO/AMC trading-day arithmetic
_NYSE = mcal.get_calendar("NYSE")


def _next_trading_day(ref_date: date) -> date:
    """Return the next NYSE trading day after ref_date."""
    end = ref_date + timedelta(days=10)
    schedule = _NYSE.schedule(
        start_date=ref_date + timedelta(days=1),
        end_date=end,
    )
    if len(schedule) == 0:
        return ref_date + timedelta(days=1)
    return date.fromisoformat(schedule.index[0].strftime("%Y-%m-%d"))


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
    except Exception as e:
        logger.warning(f"Failed to read secrets.yaml: {e}")

    return key


class FMPEarningsAdapter:
    """Fetches earnings data from FMP + yfinance price supplement.

    API call budget per daily run:
        1 call:    earnings calendar (date range)
        ~5-20:     per-symbol earnings history
        ~5-20:     per-symbol analyst grades
        Total:     ~11-41 calls/day (well within 250/day free tier)
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or _load_fmp_key()
        if not self._api_key:
            raise ValueError(
                "FMP API key required. Set FMP_API_KEY env var or add to config/secrets.yaml"
            )

    # ── FMP Endpoints ─────────────────────────────────────────────────

    def fetch_earnings_calendar(
        self,
        from_date: date,
        to_date: date,
    ) -> list[dict[str, Any]]:
        """Fetch earnings calendar for date range (1 API call).

        Returns list of dicts with: symbol, date, epsActual, epsEstimated,
        revenueActual, revenueEstimated.
        """
        url = f"{_FMP_BASE}/stable/earnings-calendar"
        params = {
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "apikey": self._api_key,
        }
        data = self._fmp_get(url, params)
        if not isinstance(data, list):
            return []
        return data

    def fetch_earnings_history(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch historical earnings for SUE computation (1 API call per symbol).

        Returns list of dicts with: date, epsActual, epsEstimated,
        revenueActual, revenueEstimated.
        """
        url = f"{_FMP_BASE}/stable/earnings"
        params = {"symbol": symbol, "apikey": self._api_key}
        data = self._fmp_get(url, params)
        if not isinstance(data, list):
            return []
        return data

    def fetch_analyst_grades(self, symbol: str) -> list[dict[str, Any]]:
        """Fetch analyst grade changes (1 API call per symbol).

        Returns list of dicts with: date, gradingCompany, previousGrade,
        newGrade, action (upgrade/downgrade/maintain/reiterated).
        """
        url = f"{_FMP_BASE}/stable/grades"
        params = {"symbol": symbol, "apikey": self._api_key}
        data = self._fmp_get(url, params)
        if not isinstance(data, list):
            return []
        return data

    # ── yfinance Price Helper ─────────────────────────────────────────

    @staticmethod
    def fetch_price_data(symbol: str, reaction_date: date) -> dict[str, Any]:
        """Fetch OHLCV around the earnings reaction date via yfinance.

        Args:
            symbol: Ticker symbol.
            reaction_date: The trading day when the market reacts to earnings.
                For BMO reporters: same as report_date.
                For AMC reporters: next trading day after report_date.

        Returns dict with: prior_close, open, close, volume, avg_20d_volume,
        high_52w, current_price, forward_pe.
        """
        import yfinance as yf

        result: dict[str, Any] = {
            "prior_close": 0.0,
            "open": 0.0,
            "close": 0.0,
            "volume": 0.0,
            "avg_20d_volume": 0.0,
            "high_52w": 0.0,
            "current_price": 0.0,
            "forward_pe": None,
        }

        try:
            ticker = yf.Ticker(symbol)

            # Get 52w high and forward PE from info
            info = ticker.info or {}
            result["high_52w"] = info.get("fiftyTwoWeekHigh", 0.0) or 0.0
            result["forward_pe"] = info.get("forwardPE")
            result["current_price"] = (
                info.get("currentPrice", 0.0) or info.get("regularMarketPrice", 0.0) or 0.0
            )

            # Historical data around earnings reaction date
            start = reaction_date - timedelta(days=30)
            end = reaction_date + timedelta(days=5)
            hist = ticker.history(start=start.isoformat(), end=end.isoformat())

            if hist.empty:
                return result

            # Find reaction day row (nearest to reaction_date)
            reaction_str = reaction_date.isoformat()
            if reaction_str in hist.index.strftime("%Y-%m-%d").tolist():
                idx = hist.index.strftime("%Y-%m-%d").tolist().index(reaction_str)
            else:
                # Find nearest trading day
                idx = len(hist) - 1
                for i, d in enumerate(hist.index):
                    if d.date() >= reaction_date:
                        idx = i
                        break

            if idx > 0:
                result["prior_close"] = float(hist.iloc[idx - 1]["Close"])
            result["open"] = float(hist.iloc[idx]["Open"])
            result["close"] = float(hist.iloc[idx]["Close"])
            result["volume"] = float(hist.iloc[idx]["Volume"])

            # 20-day average volume (before earnings)
            vol_window = hist.iloc[max(0, idx - 20) : idx]
            if not vol_window.empty:
                result["avg_20d_volume"] = float(vol_window["Volume"].mean())

            # Use most recent close as current price if not from info
            if not result["current_price"]:
                result["current_price"] = float(hist.iloc[-1]["Close"])

        except Exception as e:
            logger.warning(f"yfinance error for {symbol}: {e}")

        return result

    # ── Orchestration ─────────────────────────────────────────────────

    def fetch_recent_earnings(
        self,
        symbols: list[str],
        lookback_days: int = 10,
    ) -> tuple[list[dict[str, Any]], int]:
        """Fetch and merge earnings data for recent reporters.

        Flow:
            1. Earnings calendar → who reported in window
            2. Intersect with requested symbols
            3. Per reporter: earnings history + grades + price data
            4. Return merged list + skipped count

        Args:
            symbols: Universe symbols to screen.
            lookback_days: How many calendar days back to check.

        Returns:
            (list of merged earning dicts, count of symbols skipped due to FMP tier)
        """
        symbol_set = set(symbols)
        today = date.today()
        from_date = today - timedelta(days=lookback_days)

        # Step 1: Calendar
        calendar = self.fetch_earnings_calendar(from_date, today)
        reporters = [
            e for e in calendar if e.get("symbol") in symbol_set and e.get("epsActual") is not None
        ]
        logger.info(f"PEAD: {len(reporters)} symbols reported in last {lookback_days} days")

        results: list[dict[str, Any]] = []
        skipped = 0

        for entry in reporters:
            symbol = entry["symbol"]

            # Earnings history for SUE
            time.sleep(_FMP_DELAY)
            history = self.fetch_earnings_history(symbol)
            if not history:
                if not entry.get("epsEstimated"):
                    skipped += 1
                    logger.info(f"Skipping {symbol}: no history and no estimate (FMP tier limit)")
                    continue
                # Calendar has estimate but no history — will use fallback SUE
                logger.info(f"{symbol}: no FMP history, using fallback SUE from calendar estimate")

            # Analyst grades
            time.sleep(_FMP_DELAY)
            grades = self.fetch_analyst_grades(symbol)

            # Price data via yfinance
            report_date_str = entry.get("date", "")
            try:
                # Truncate to date portion — FMP may return datetime strings
                report_date = date.fromisoformat(report_date_str[:10])
            except (ValueError, TypeError):
                logger.warning(f"Invalid date for {symbol}: {report_date_str}")
                continue

            # BMO/AMC: determine reaction date
            report_time = (entry.get("time") or "bmo").lower().strip()
            if report_time == "amc":
                reaction_date = _next_trading_day(report_date)
            else:
                reaction_date = report_date

            time.sleep(_YF_DELAY)
            price = self.fetch_price_data(symbol, reaction_date)

            # Build merged dict
            merged = self._merge_earning_data(
                entry, history, grades, price, report_date, report_time=report_time
            )
            results.append(merged)

        logger.info(
            f"Fetched earnings for {len(results)} symbols " f"({skipped} skipped: FMP tier limit)"
        )
        return results, skipped

    # ── Private Helpers ───────────────────────────────────────────────

    def _fmp_get(self, url: str, params: dict[str, Any]) -> Any:
        """Make a GET request to FMP with error handling."""
        try:
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code == 403:
                raise ValueError("FMP API key invalid or expired")

            if resp.status_code == 402:
                # Free tier limit — not an error, just skip
                return []

            if resp.status_code >= 500:
                logger.warning(f"FMP server error {resp.status_code}, retrying once")
                time.sleep(1)
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code >= 400:
                    return []

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            logger.warning(f"FMP timeout: {url}")
            return []
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"FMP request error: {e}")
            return []

    @staticmethod
    def _merge_earning_data(
        calendar_entry: dict[str, Any],
        history: list[dict[str, Any]],
        grades: list[dict[str, Any]],
        price: dict[str, Any],
        report_date: date,
        downgrade_window_days: int = 3,
        report_time: str = "bmo",
    ) -> dict[str, Any]:
        """Merge FMP calendar + history + grades + yfinance into screener input."""
        symbol = calendar_entry["symbol"]
        actual = calendar_entry.get("epsActual", 0.0) or 0.0
        estimated = calendar_entry.get("epsEstimated", 0.0) or 0.0

        # Sort history by date descending to ensure correct ordering
        # (FMP order is not contractually guaranteed)
        sorted_history = sorted(
            history,
            key=lambda h: h.get("date", "")[:10],
            reverse=True,
        )

        # Exclude current quarter from history to prevent SUE leakage
        # (current quarter's surprise must not be in its own SUE denominator)
        report_date_str = report_date.isoformat()
        past_history = [h for h in sorted_history if h.get("date", "")[:10] != report_date_str]

        # Historical surprises for SUE (last 12 quarters for multi-Q SUE)
        historical_surprises: list[float] = []
        for h in past_history[:12]:
            h_actual = h.get("epsActual")
            h_est = h.get("epsEstimated")
            if h_actual is not None and h_est is not None:
                historical_surprises.append(h_actual - h_est)

        # Revenue beat from calendar entry
        rev_actual = calendar_entry.get("revenueActual")
        rev_est = calendar_entry.get("revenueEstimated")
        revenue_beat = rev_actual is not None and rev_est is not None and rev_actual > rev_est

        # Analyst downgrade within configured window
        analyst_downgrade = False
        for g in grades:
            g_date_str = g.get("date", "")
            try:
                g_date = date.fromisoformat(g_date_str[:10])
            except (ValueError, TypeError):
                continue
            days_diff = (g_date - report_date).days
            if (
                0 <= days_diff <= downgrade_window_days
                and g.get("action", "").lower() == "downgrade"
            ):
                analyst_downgrade = True
                break

        # Earnings reaction from price data
        prior_close = price.get("prior_close", 0.0)
        open_price = price.get("open", 0.0)
        close_price = price.get("close", 0.0)
        volume = price.get("volume", 0.0)
        avg_vol = price.get("avg_20d_volume", 0.0)
        high_52w = price.get("high_52w", 0.0)

        gap_return = (open_price - prior_close) / prior_close if prior_close else 0.0
        day_return = (close_price - prior_close) / prior_close if prior_close else 0.0
        volume_ratio = volume / avg_vol if avg_vol else 0.0
        at_52w_high = prior_close >= (high_52w * 0.95) if high_52w else False

        return {
            "symbol": symbol,
            "report_date": report_date.isoformat(),
            "report_time": report_time,
            "actual_eps": actual,
            "consensus_eps": estimated,
            "historical_surprises": historical_surprises,
            "earnings_day_gap": gap_return,
            "earnings_day_return": day_return,
            "earnings_day_volume_ratio": volume_ratio,
            "revenue_beat": revenue_beat,
            "at_52w_high": at_52w_high,
            "analyst_downgrade": analyst_downgrade,
            "forward_pe": price.get("forward_pe"),
            "current_price": price.get("current_price", 0.0),
            "earnings_open": open_price,
        }
