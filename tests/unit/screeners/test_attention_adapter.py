"""Unit tests for PEAD attention adapter (Google Trends)."""

from __future__ import annotations

import json
import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.infrastructure.adapters.earnings.attention_adapter import AttentionAdapter


class TestQueryBuilding:
    @patch("src.infrastructure.adapters.earnings.attention_adapter.yf")
    def test_build_query_with_company_name(self, mock_yf: MagicMock) -> None:
        """Uses '{company_name} stock' when yfinance returns a name."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"shortName": "Apple Inc."}
        mock_yf.Ticker.return_value = mock_ticker

        adapter = AttentionAdapter()
        query = adapter._build_search_query("AAPL")
        assert query == "Apple stock"

    @patch("src.infrastructure.adapters.earnings.attention_adapter.yf")
    def test_build_query_strips_corp_suffix(self, mock_yf: MagicMock) -> None:
        """Strips Inc., Corp., Ltd. suffixes."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"shortName": "Caterpillar Inc."}
        mock_yf.Ticker.return_value = mock_ticker

        adapter = AttentionAdapter()
        query = adapter._build_search_query("CAT")
        assert query == "Caterpillar stock"

    @patch("src.infrastructure.adapters.earnings.attention_adapter.yf")
    def test_build_query_fallback(self, mock_yf: MagicMock) -> None:
        """Falls back to '{symbol} stock earnings' when name unavailable."""
        mock_ticker = MagicMock()
        mock_ticker.info = {}
        mock_yf.Ticker.return_value = mock_ticker

        adapter = AttentionAdapter()
        query = adapter._build_search_query("XYZ")
        assert query == "XYZ stock earnings"

    @patch("src.infrastructure.adapters.earnings.attention_adapter.yf")
    def test_build_query_exception_fallback(self, mock_yf: MagicMock) -> None:
        """Falls back gracefully when yfinance raises."""
        mock_yf.Ticker.side_effect = Exception("network error")

        adapter = AttentionAdapter()
        query = adapter._build_search_query("FAIL")
        assert query == "FAIL stock earnings"


class TestClassification:
    def test_low_attention(self) -> None:
        assert AttentionAdapter._classify(10) == "low"
        assert AttentionAdapter._classify(25) == "low"

    def test_medium_attention(self) -> None:
        assert AttentionAdapter._classify(26) == "medium"
        assert AttentionAdapter._classify(50) == "medium"
        assert AttentionAdapter._classify(64) == "medium"

    def test_high_attention(self) -> None:
        assert AttentionAdapter._classify(65) == "high"
        assert AttentionAdapter._classify(100) == "high"


class TestCaching:
    def test_cache_hit_returns_level(self) -> None:
        """Cached entries are returned without API calls."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        cache = {"AAPL_2025-01-27": {"score": 80, "level": "high"}}
        path.write_text(json.dumps(cache))

        adapter = AttentionAdapter(cache_path=path)
        level = adapter.get_attention_level("AAPL", date(2025, 1, 27))
        assert level == "high"

        path.unlink(missing_ok=True)

    def test_cache_miss_without_pytrends_returns_none(self) -> None:
        """When pytrends is unavailable, uncached entries return None."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        adapter = AttentionAdapter(cache_path=path)
        adapter._pytrends_available = False

        level = adapter.get_attention_level("AAPL", date(2025, 1, 27))
        assert level is None

        path.unlink(missing_ok=True)


class TestBatchUpdate:
    def test_batch_skips_when_no_pytrends(self) -> None:
        """Batch update returns 0 when pytrends unavailable."""
        adapter = AttentionAdapter()
        adapter._pytrends_available = False

        count = adapter.update_attention_batch([("AAPL", date(2025, 1, 27))])
        assert count == 0

    def test_batch_skips_cached(self) -> None:
        """Already-cached entries are not re-fetched."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        cache = {"AAPL_2025-01-27": {"score": 80, "level": "high"}}
        path.write_text(json.dumps(cache))

        adapter = AttentionAdapter(cache_path=path)
        adapter._pytrends_available = True

        # Mock _fetch_attention_score to track calls
        adapter._fetch_attention_score = MagicMock(return_value=None)  # type: ignore[assignment]

        adapter.update_attention_batch([("AAPL", date(2025, 1, 27))], delay_seconds=0)
        adapter._fetch_attention_score.assert_not_called()  # type: ignore[attr-defined]

        path.unlink(missing_ok=True)


class TestAttentionModifierIntegration:
    def test_apply_attention_modifier_low_bonus(self) -> None:
        from src.domain.screeners.pead.scorer import apply_attention_modifier

        result = apply_attention_modifier(50.0, "low", low_bonus=5.0)
        assert result == 55.0

    def test_apply_attention_modifier_high_penalty(self) -> None:
        from src.domain.screeners.pead.scorer import apply_attention_modifier

        result = apply_attention_modifier(50.0, "high", high_penalty=-5.0)
        assert result == 45.0

    def test_apply_attention_modifier_medium_no_change(self) -> None:
        from src.domain.screeners.pead.scorer import apply_attention_modifier

        result = apply_attention_modifier(50.0, "medium")
        assert result == 50.0

    def test_apply_attention_modifier_none_no_change(self) -> None:
        from src.domain.screeners.pead.scorer import apply_attention_modifier

        result = apply_attention_modifier(50.0, None)
        assert result == 50.0
