"""Google Trends attention adapter for PEAD screener.

Measures retail attention for earnings symbols using Google Trends data.
Low-attention stocks show stronger post-earnings drift (under-reaction).
High-attention stocks revert faster (efficient market pricing).

Requires optional ``pytrends`` dependency.  When unavailable or rate-limited,
returns ``None`` for attention — screening continues unblocked.

Company name lookup prevents ticker-symbol ambiguity (CAT, AI, META).
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path
from typing import Any

import yfinance as yf

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "attention.json"

# Classification thresholds (relative Google Trends interest 0-100)
_LOW_THRESHOLD = 25
_HIGH_THRESHOLD = 65


class AttentionAdapter:
    """Fetches and classifies Google Trends attention for symbols.

    Caches results to ``data/cache/attention.json`` keyed by
    ``{symbol}_{report_date}`` to avoid redundant API calls.

    If ``pytrends`` is not installed, all methods return None gracefully.
    """

    def __init__(self, cache_path: Path | None = None) -> None:
        self._cache_path = cache_path or DEFAULT_CACHE_PATH
        self._cache: dict[str, Any] | None = None
        self._pytrends_available = self._check_pytrends()

    @staticmethod
    def _check_pytrends() -> bool:
        try:
            import pytrends  # type: ignore[import-not-found]  # noqa: F401

            return True
        except ImportError:
            logger.info("pytrends not installed — attention filter disabled")
            return False

    # ── Cache ────────────────────────────────────────────────────────────

    def _load_cache(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache

        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Attention cache load error: {e}")
                self._cache = {}
        else:
            self._cache = {}

        return self._cache

    def _save_cache(self) -> None:
        cache = self._load_cache()
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(cache, indent=2))

    # ── Public API ───────────────────────────────────────────────────────

    def get_attention_level(self, symbol: str, report_date: date) -> str | None:
        """Get attention level for a symbol around its earnings report.

        Returns:
            "low" / "medium" / "high" / None (if data unavailable).
        """
        cache_key = f"{symbol}_{report_date.isoformat()}"
        cache = self._load_cache()

        if cache_key in cache:
            level: str | None = cache[cache_key].get("level")
            return level

        if not self._pytrends_available:
            return None

        score = self._fetch_attention_score(symbol)
        if score is None:
            return None

        level = self._classify(score)
        cache[cache_key] = {"score": score, "level": level}
        self._save_cache()

        return level

    def update_attention_batch(
        self,
        symbols_dates: list[tuple[str, date]],
        delay_seconds: float = 2.0,
    ) -> int:
        """Pre-fetch attention data for a batch of (symbol, report_date) pairs.

        Rate-limits requests to avoid 429 errors. Skips cached entries.

        Returns:
            Count of newly fetched attention scores.
        """
        if not self._pytrends_available:
            logger.info("pytrends not installed — skipping attention batch")
            return 0

        cache = self._load_cache()
        fetched = 0

        for symbol, rdate in symbols_dates:
            cache_key = f"{symbol}_{rdate.isoformat()}"
            if cache_key in cache:
                continue

            score = self._fetch_attention_score(symbol)
            if score is not None:
                level = self._classify(score)
                cache[cache_key] = {"score": score, "level": level}
                fetched += 1
                logger.info(f"Attention: {symbol} → {level} (score={score})")
            else:
                logger.warning(f"Attention: {symbol} → unavailable")

            # Rate limit
            if delay_seconds > 0:
                time.sleep(delay_seconds)

        if fetched > 0:
            self._save_cache()

        return fetched

    # ── Internal ─────────────────────────────────────────────────────────

    def _fetch_attention_score(self, symbol: str) -> int | None:
        """Fetch Google Trends interest score for a symbol.

        Uses ``"{company_name} stock"`` as search query to avoid
        ticker-symbol ambiguity (CAT, AI, META).

        Returns:
            Peak interest score (0-100) in the last 7 days, or None.
        """
        query = self._build_search_query(symbol)
        if query is None:
            return None

        try:
            from pytrends.request import TrendReq  # type: ignore[import-not-found]

            pytrends = TrendReq(hl="en-US", tz=300)
            pytrends.build_payload([query], timeframe="now 7-d")
            interest = pytrends.interest_over_time()

            if interest.empty or query not in interest.columns:
                return None

            return int(interest[query].max())

        except Exception as e:
            logger.warning(f"Google Trends error for {symbol}: {e}")
            return None

    def _build_search_query(self, symbol: str) -> str | None:
        """Build unambiguous Google Trends query.

        Strategy: use "{company_name} stock" as primary query.
        Fallback: "{symbol} stock earnings" if company name unavailable.

        Examples:
            AAPL -> "Apple stock"
            CAT  -> "Caterpillar stock"
            META -> "Meta Platforms stock"
            AI   -> "C3.ai stock"
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            name = info.get("shortName") or info.get("longName")
            if name:
                # Strip common suffixes like "Inc.", "Corp.", "Ltd."
                for suffix in [", Inc.", " Inc.", " Corp.", " Ltd.", " LLC"]:
                    name = name.replace(suffix, "")
                return f"{name.strip()} stock"
        except Exception as e:
            logger.debug(f"yfinance company name lookup failed for {symbol}: {e}")

        # Fallback — less precise but still disambiguated
        return f"{symbol} stock earnings"

    @staticmethod
    def _classify(score: int) -> str:
        """Classify attention score into low/medium/high."""
        if score <= _LOW_THRESHOLD:
            return "low"
        if score >= _HIGH_THRESHOLD:
            return "high"
        return "medium"
