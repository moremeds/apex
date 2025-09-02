"""Tests for data quality scorer."""

import pytest
from datetime import datetime, timedelta
import polars as pl

from apex.core.types import DataQualityDimension, MarketDataFrame
from apex.data.quality.scorer import DataQualityScorer


@pytest.fixture
def quality_scorer():
    """Create a data quality scorer for testing."""
    return DataQualityScorer(quality_threshold=0.7)


@pytest.fixture
def perfect_data():
    """Create perfect quality test data."""
    dates = [datetime(2023, 1, i+1) for i in range(10)]  # 10 consecutive days
    return pl.DataFrame({
        "datetime": dates,
        "open": [100.0 + i for i in range(10)],
        "high": [101.0 + i for i in range(10)],
        "low": [99.0 + i for i in range(10)],
        "close": [100.5 + i for i in range(10)],
        "volume": [1000000] * 10,
    })


@pytest.fixture
def problematic_data():
    """Create data with quality issues."""
    dates = [datetime(2023, 1, i+1) if i != 5 else None for i in range(10)]  # Missing date
    return pl.DataFrame({
        "datetime": dates,
        "open": [100.0 + i if i != 3 else None for i in range(10)],  # Missing value
        "high": [101.0 + i if i != 7 else 50.0 for i in range(10)],  # Inconsistent OHLC
        "low": [99.0 + i if i != 2 else -10.0 for i in range(10)],   # Negative price
        "close": [100.5 + i if i != 1 else 0.0 for i in range(10)],  # Zero price
        "volume": [1000000] * 10,
    })


class TestDataQualityScorer:
    """Test cases for data quality scorer."""

    def test_initialization(self):
        """Test scorer initialization."""
        scorer = DataQualityScorer(quality_threshold=0.8)
        assert scorer.quality_threshold == 0.8

    @pytest.mark.asyncio
    async def test_score_perfect_data(self, quality_scorer, perfect_data):
        """Test scoring of perfect quality data."""
        market_data = MarketDataFrame(
            data=perfect_data,
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            source="test",
        )
        
        result = await quality_scorer.score_data_quality(market_data)
        
        # All dimension scores should be perfect or near-perfect
        assert result.quality_scores[DataQualityDimension.COMPLETENESS].score >= 0.95
        assert result.quality_scores[DataQualityDimension.CONSISTENCY].score == 1.0
        assert result.quality_scores[DataQualityDimension.VALIDITY].score == 1.0
        assert result.quality_scores[DataQualityDimension.TIMELINESS].score >= 0.95
        assert result.quality_scores[DataQualityDimension.UNIQUENESS].score == 1.0
        
        # Overall score should be high
        assert result.overall_quality_score >= 0.95

    @pytest.mark.asyncio
    async def test_score_problematic_data(self, quality_scorer, problematic_data):
        """Test scoring of problematic data."""
        market_data = MarketDataFrame(
            data=problematic_data,
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            source="test",
        )
        
        result = await quality_scorer.score_data_quality(market_data)
        
        # Should detect quality issues
        assert result.quality_scores[DataQualityDimension.COMPLETENESS].score < 1.0
        assert result.quality_scores[DataQualityDimension.CONSISTENCY].score < 1.0
        assert result.quality_scores[DataQualityDimension.VALIDITY].score < 1.0
        
        # Overall score should be lower (but data has only a few issues)
        assert result.overall_quality_score < 1.0

    def test_score_completeness_perfect(self, quality_scorer, perfect_data):
        """Test completeness scoring with perfect data."""
        score = quality_scorer._score_completeness(perfect_data)
        
        assert score.dimension == DataQualityDimension.COMPLETENESS
        assert score.score >= 0.95  # Should be near perfect
        assert score.details["missing_cells"] == 0
        assert len(score.recommendations) == 0

    def test_score_completeness_missing_data(self, quality_scorer):
        """Test completeness scoring with missing data."""
        data_with_nulls = pl.DataFrame({
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "open": [100.0, None],  # Missing value
            "high": [101.0, 102.0],
            "low": [99.0, 101.0],
            "close": [100.5, 101.5],
            "volume": [1000000, 1000000],
        })
        
        score = quality_scorer._score_completeness(data_with_nulls)
        
        assert score.score < 1.0
        assert score.details["missing_cells"] == 1
        assert "imputation" in " ".join(score.recommendations).lower()

    def test_score_consistency_valid_ohlc(self, quality_scorer, perfect_data):
        """Test consistency scoring with valid OHLC relationships."""
        score = quality_scorer._score_consistency(perfect_data)
        
        assert score.dimension == DataQualityDimension.CONSISTENCY
        assert score.score == 1.0
        assert score.details["ohlc_violations"] == 0

    def test_score_consistency_invalid_ohlc(self, quality_scorer):
        """Test consistency scoring with invalid OHLC relationships."""
        invalid_data = pl.DataFrame({
            "datetime": [datetime(2023, 1, 1)],
            "open": [100.0],
            "high": [99.0],  # High < Open (invalid)
            "low": [101.0],  # Low > Open (invalid)
            "close": [100.5],
            "volume": [1000000],
        })
        
        score = quality_scorer._score_consistency(invalid_data)
        
        assert score.score < 1.0
        assert score.details["ohlc_violations"] > 0

    def test_score_validity_valid_prices(self, quality_scorer, perfect_data):
        """Test validity scoring with valid price data."""
        score = quality_scorer._score_validity(perfect_data)
        
        assert score.dimension == DataQualityDimension.VALIDITY
        assert score.score == 1.0

    def test_score_validity_invalid_prices(self, quality_scorer):
        """Test validity scoring with invalid prices."""
        invalid_data = pl.DataFrame({
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
            "open": [100.0, -50.0],  # Negative price
            "high": [101.0, 0.0],    # Zero price
            "low": [99.0, -51.0],
            "close": [100.5, 0.0],
            "volume": [1000000, 1000000],
        })
        
        score = quality_scorer._score_validity(invalid_data)
        
        assert score.score < 1.0
        assert "negative" in " ".join(score.recommendations).lower()

    def test_score_timeliness_full_coverage(self, quality_scorer):
        """Test timeliness scoring with full date coverage."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        data = pl.DataFrame({
            "datetime": [start_date + timedelta(days=i) for i in range(10)],
            "open": [100.0] * 10,
            "high": [101.0] * 10,
            "low": [99.0] * 10,
            "close": [100.5] * 10,
            "volume": [1000000] * 10,
        })
        
        score = quality_scorer._score_timeliness(data, start_date, end_date)
        
        assert score.dimension == DataQualityDimension.TIMELINESS
        assert score.score >= 0.9

    def test_score_timeliness_partial_coverage(self, quality_scorer):
        """Test timeliness scoring with partial date coverage."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 1, 10)
        
        # Data only covers half the requested range
        data = pl.DataFrame({
            "datetime": [start_date + timedelta(days=i) for i in range(5)],
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.5] * 5,
            "volume": [1000000] * 5,
        })
        
        score = quality_scorer._score_timeliness(data, start_date, end_date)
        
        assert score.score < 0.9
        assert score.details["end_gap_days"] > 0

    def test_score_uniqueness_unique_data(self, quality_scorer, perfect_data):
        """Test uniqueness scoring with unique data."""
        score = quality_scorer._score_uniqueness(perfect_data)
        
        assert score.dimension == DataQualityDimension.UNIQUENESS
        assert score.score == 1.0
        assert score.details["duplicates"] == 0

    def test_score_uniqueness_duplicate_data(self, quality_scorer):
        """Test uniqueness scoring with duplicate timestamps."""
        duplicate_data = pl.DataFrame({
            "datetime": [datetime(2023, 1, 1), datetime(2023, 1, 1)],  # Duplicate
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.5, 100.5],
            "volume": [1000000, 1000000],
        })
        
        score = quality_scorer._score_uniqueness(duplicate_data)
        
        assert score.score < 1.0
        assert score.details["duplicates"] == 1
        assert "duplicate" in " ".join(score.recommendations).lower()

    def test_detect_date_gaps(self, quality_scorer):
        """Test date gap detection."""
        # Data with a significant gap
        dates = [datetime(2023, 1, 1), datetime(2023, 1, 15)]  # 14-day gap
        data = pl.DataFrame({
            "datetime": dates,
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000000, 1000000],
        })
        
        gaps = quality_scorer._detect_date_gaps(data)
        
        assert len(gaps) > 0
        assert "2023-01-01 to 2023-01-15" in gaps[0]

    @pytest.mark.asyncio
    async def test_empty_dataset_handling(self, quality_scorer):
        """Test handling of empty datasets."""
        empty_data = pl.DataFrame()
        market_data = MarketDataFrame(
            data=empty_data,
            symbol="TEST",
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            source="test",
        )
        
        result = await quality_scorer.score_data_quality(market_data)
        
        # All scores should be 0 for empty data
        for dimension in DataQualityDimension:
            assert result.quality_scores[dimension].score == 0.0
            assert "empty" in result.quality_scores[dimension].details.get("error", "").lower()