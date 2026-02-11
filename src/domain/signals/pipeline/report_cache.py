"""
Report frame cache for HTML/package generation.

Caches enriched per-symbol/timeframe DataFrames (price + indicators) to avoid
recomputing expensive indicator stacks between nearby report runs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pandas as pd

from src.utils.logging_setup import get_logger

logger = get_logger(__name__)


class ReportFrameCache:
    """
    Filesystem cache for enriched report DataFrames.

    Cache key is content-derived (symbol/timeframe + bar fingerprints +
    indicator signature), and freshness is additionally guarded by TTL.
    """

    def __init__(
        self,
        cache_dir: Path,
        max_age_minutes: int = 10,
        cleanup_max_files: int = 4000,
        enabled: bool = True,
        code_signature: str = "report-frame-cache-v1",
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._max_age_seconds = max(0, int(max_age_minutes)) * 60
        self._cleanup_max_files = max(1, int(cleanup_max_files))
        self._enabled = enabled
        self._code_signature = code_signature
        self._hits = 0
        self._misses = 0

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return (self._hits / total) if total > 0 else 0.0

    def _normalize_value(self, value: Any) -> Any:
        """Normalize values for deterministic JSON hashing."""
        if isinstance(value, dict):
            return {k: self._normalize_value(v) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [self._normalize_value(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def indicator_signature(self, indicators: List[Any]) -> str:
        """Build a stable hash of indicator names + params + warmup requirements."""
        records: List[Dict[str, Any]] = []
        for indicator in indicators:
            records.append(
                {
                    "name": str(getattr(indicator, "name", type(indicator).__name__)),
                    "warmup": int(getattr(indicator, "warmup_periods", 0)),
                    "params": self._normalize_value(getattr(indicator, "default_params", {}) or {}),
                }
            )
        payload = json.dumps(sorted(records, key=lambda x: x["name"]), separators=(",", ":"))
        return hashlib.sha256(f"{self._code_signature}|{payload}".encode("utf-8")).hexdigest()[:16]

    def make_cache_key(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        indicator_signature: str,
    ) -> str:
        """Build a content-derived cache key for a DataFrame input."""
        if df.empty:
            raw = f"{symbol}|{timeframe}|0|empty|{indicator_signature}|{self._code_signature}"
            return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

        first_ts = pd.Timestamp(df.index[0]).isoformat()
        last_ts = pd.Timestamp(df.index[-1]).isoformat()
        bar_count = len(df)

        close_tail = ""
        if "close" in df.columns:
            tail_vals = [float(v) for v in df["close"].tail(5)]
            close_tail = ",".join(f"{value:.6f}" for value in tail_vals)

        raw = (
            f"{symbol}|{timeframe}|{bar_count}|{first_ts}|{last_ts}|{close_tail}|"
            f"{indicator_signature}|{self._code_signature}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _path_for(self, symbol: str, timeframe: str, cache_key: str) -> Path:
        safe_symbol = symbol.replace("/", "_").upper()
        safe_tf = timeframe.replace("/", "_")
        return self._cache_dir / f"{safe_symbol}_{safe_tf}_{cache_key}.parquet"

    def _is_stale(self, path: Path) -> bool:
        """Check TTL on cache file mtime."""
        if self._max_age_seconds <= 0:
            return False
        try:
            age_seconds = max(0.0, float(pd.Timestamp.utcnow().timestamp()) - path.stat().st_mtime)
            return age_seconds > self._max_age_seconds
        except FileNotFoundError:
            return True

    def load(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        indicator_signature: str,
    ) -> Optional[pd.DataFrame]:
        """Load cached enriched DataFrame if present and fresh."""
        if not self._enabled:
            self._misses += 1
            return None

        cache_key = self.make_cache_key(symbol, timeframe, df, indicator_signature)
        cache_path = self._path_for(symbol, timeframe, cache_key)
        if not cache_path.exists():
            self._misses += 1
            return None

        if self._is_stale(cache_path):
            self._misses += 1
            try:
                cache_path.unlink()
            except OSError:
                pass
            return None

        try:
            cached_df = pd.read_parquet(cache_path)
            if not isinstance(cached_df.index, pd.DatetimeIndex):
                cached_df.index = pd.to_datetime(cached_df.index, errors="coerce", utc=True)
            self._hits += 1
            return cached_df
        except Exception as exc:
            self._misses += 1
            logger.warning(f"Failed to read report cache {cache_path}: {exc}")
            return None

    def save(
        self,
        symbol: str,
        timeframe: str,
        df_input: pd.DataFrame,
        indicator_signature: str,
        enriched_df: pd.DataFrame,
    ) -> None:
        """Save enriched DataFrame using atomic rename."""
        if not self._enabled:
            return

        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = self.make_cache_key(symbol, timeframe, df_input, indicator_signature)
            cache_path = self._path_for(symbol, timeframe, cache_key)
            temp_path = cache_path.with_suffix(f".{uuid4().hex}.tmp")
            enriched_df.to_parquet(temp_path)
            os.replace(temp_path, cache_path)
        except Exception as exc:
            logger.warning(f"Failed to write report cache for {symbol}/{timeframe}: {exc}")

    def cleanup(self) -> int:
        """
        Remove stale cache files and enforce a max file count.

        Returns:
            Number of files removed.
        """
        if not self._enabled or not self._cache_dir.exists():
            return 0

        removed = 0
        try:
            files = sorted(self._cache_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return removed

        # Remove stale files first.
        stale_cutoff = float(pd.Timestamp.utcnow().timestamp()) - float(self._max_age_seconds)
        for path in files:
            try:
                if self._max_age_seconds > 0 and path.stat().st_mtime < stale_cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue

        # Refresh file list and cap count.
        try:
            files = sorted(self._cache_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
        except OSError:
            return removed

        overflow = len(files) - self._cleanup_max_files
        if overflow > 0:
            for path in files[:overflow]:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    continue

        return removed
