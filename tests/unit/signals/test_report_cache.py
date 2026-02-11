from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from src.domain.signals.pipeline.report_cache import ReportFrameCache


class _DummyIndicator:
    name = "RSI"
    warmup_periods = 14
    default_params = {"period": 14}


def _make_df(start: str) -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=6, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [1, 2, 3, 4, 5, 6],
            "high": [2, 3, 4, 5, 6, 7],
            "low": [0, 1, 2, 3, 4, 5],
            "close": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
            "volume": [10, 20, 30, 40, 50, 60],
        },
        index=idx,
    )


class TestReportFrameCache:
    def test_cache_key_changes_when_last_bar_changes(self, tmp_path: Path) -> None:
        cache = ReportFrameCache(tmp_path, max_age_minutes=10, enabled=True)
        indicators = [_DummyIndicator()]
        signature = cache.indicator_signature(indicators)

        df_a = _make_df("2026-01-01")
        df_b = _make_df("2026-01-02")

        key_a = cache.make_cache_key("AAPL", "1h", df_a, signature)
        key_b = cache.make_cache_key("AAPL", "1h", df_b, signature)

        assert key_a != key_b

    def test_ttl_expiry_returns_miss_and_removes_file(self, tmp_path: Path) -> None:
        cache = ReportFrameCache(tmp_path, max_age_minutes=1, enabled=True)
        indicators = [_DummyIndicator()]
        signature = cache.indicator_signature(indicators)

        base_df = _make_df("2026-01-01")
        enriched_df = base_df.copy()
        enriched_df["RSI_value"] = [50.0] * len(enriched_df)
        cache.save("AAPL", "1h", base_df, signature, enriched_df)

        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1
        cache_file = parquet_files[0]

        old_time = time.time() - 7200
        os.utime(cache_file, (old_time, old_time))

        loaded = cache.load("AAPL", "1h", base_df, signature)
        assert loaded is None
        assert not cache_file.exists()
        assert cache.misses == 1

    def test_cleanup_enforces_max_file_count(self, tmp_path: Path) -> None:
        cache = ReportFrameCache(
            tmp_path,
            max_age_minutes=120,
            cleanup_max_files=2,
            enabled=True,
        )
        indicators = [_DummyIndicator()]
        signature = cache.indicator_signature(indicators)
        base_df = _make_df("2026-01-01")

        for idx in range(5):
            symbol = f"SYM{idx}"
            cache.save(symbol, "1h", base_df, signature, base_df)
            file_path = sorted(tmp_path.glob(f"{symbol}_1h_*.parquet"))[0]
            old_time = time.time() - (1000 - idx)
            os.utime(file_path, (old_time, old_time))

        removed = cache.cleanup()
        remaining = list(tmp_path.glob("*.parquet"))

        assert removed >= 3
        assert len(remaining) <= 2
