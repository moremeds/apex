"""Data quality scoring implementation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

import polars as pl
import structlog

from apex.core.types import (
    DataQualityDimension,
    DataQualityScore,
    MarketDataFrame,
)

logger = structlog.get_logger(__name__)


class DataQualityScorer:
    """Data quality scorer implementing 5-dimensional quality assessment."""

    def __init__(self, quality_threshold: float = 0.7) -> None:
        """Initialize the quality scorer."""
        self.quality_threshold = quality_threshold

    async def score_data_quality(self, data: MarketDataFrame) -> MarketDataFrame:
        """Score data quality across all dimensions."""
        df = data.data
        scores = {}

        # Score each dimension
        scores[DataQualityDimension.COMPLETENESS] = self._score_completeness(df)
        scores[DataQualityDimension.CONSISTENCY] = self._score_consistency(df)
        scores[DataQualityDimension.VALIDITY] = self._score_validity(df)
        scores[DataQualityDimension.TIMELINESS] = self._score_timeliness(
            df, data.start_date, data.end_date
        )
        scores[DataQualityDimension.UNIQUENESS] = self._score_uniqueness(df)

        # Update the market data frame with quality scores
        data.quality_scores = scores
        
        overall_score = data.overall_quality_score
        logger.info(
            "Data quality scored",
            symbol=data.symbol,
            overall_score=overall_score,
            completeness=scores[DataQualityDimension.COMPLETENESS].score,
            consistency=scores[DataQualityDimension.CONSISTENCY].score,
            validity=scores[DataQualityDimension.VALIDITY].score,
            timeliness=scores[DataQualityDimension.TIMELINESS].score,
            uniqueness=scores[DataQualityDimension.UNIQUENESS].score,
        )

        return data

    def _score_completeness(self, df: pl.DataFrame) -> DataQualityScore:
        """Score data completeness - missing values and gaps."""
        if len(df) == 0:
            return DataQualityScore(
                dimension=DataQualityDimension.COMPLETENESS,
                score=0.0,
                details={"error": "Empty dataset"},
                recommendations=["Dataset is empty"],
            )
        
        total_cells = len(df) * len(df.columns)
        missing_cells = df.null_count().sum_horizontal().item() if len(df.columns) > 0 else 0
        
        completeness_ratio = 1.0 - (missing_cells / total_cells) if total_cells > 0 else 0.0
        
        # Check for date gaps (business days)
        date_gaps = self._detect_date_gaps(df)
        gap_penalty = min(len(date_gaps) * 0.1, 0.3)  # Max 30% penalty
        
        score = max(0.0, completeness_ratio - gap_penalty)
        
        details = {
            "missing_cells": missing_cells,
            "total_cells": total_cells,
            "missing_ratio": missing_cells / total_cells if total_cells > 0 else 0,
            "date_gaps": date_gaps,
        }
        
        recommendations = []
        if missing_cells > 0:
            recommendations.append("Consider imputation or data cleaning for missing values")
        if date_gaps:
            recommendations.append("Investigate and fill date gaps in the data")
            
        return DataQualityScore(
            dimension=DataQualityDimension.COMPLETENESS,
            score=score,
            details=details,
            recommendations=recommendations,
        )

    def _score_consistency(self, df: pl.DataFrame) -> DataQualityScore:
        """Score data consistency - OHLC relationships."""
        inconsistencies = 0
        total_rows = len(df)
        
        if total_rows == 0:
            return DataQualityScore(
                dimension=DataQualityDimension.CONSISTENCY,
                score=0.0,
                details={"error": "Empty dataset"},
                recommendations=["Dataset is empty"],
            )

        # Check OHLC relationships: High >= Open, Low <= Open, High >= Close, Low <= Close
        violations = df.filter(
            (pl.col("high") < pl.col("open")) |
            (pl.col("high") < pl.col("close")) |
            (pl.col("low") > pl.col("open")) |
            (pl.col("low") > pl.col("close"))
        )
        
        inconsistencies = len(violations)
        consistency_ratio = 1.0 - (inconsistencies / total_rows)
        
        details = {
            "ohlc_violations": inconsistencies,
            "total_rows": total_rows,
            "violation_ratio": inconsistencies / total_rows,
        }
        
        recommendations = []
        if inconsistencies > 0:
            recommendations.append("Review and correct OHLC relationship violations")
            recommendations.append("Validate data source quality controls")
            
        return DataQualityScore(
            dimension=DataQualityDimension.CONSISTENCY,
            score=consistency_ratio,
            details=details,
            recommendations=recommendations,
        )

    def _score_validity(self, df: pl.DataFrame) -> DataQualityScore:
        """Score data validity - outliers and invalid values."""
        issues = 0
        total_rows = len(df)
        
        if total_rows == 0:
            return DataQualityScore(
                dimension=DataQualityDimension.VALIDITY,
                score=0.0,
                details={"error": "Empty dataset"},
                recommendations=["Dataset is empty"],
            )

        # Check for negative prices
        negative_prices = df.filter(
            (pl.col("open") < 0) |
            (pl.col("high") < 0) |
            (pl.col("low") < 0) |
            (pl.col("close") < 0)
        )
        issues += len(negative_prices)
        
        # Check for zero prices (suspicious)
        zero_prices = df.filter(
            (pl.col("open") == 0) |
            (pl.col("high") == 0) |
            (pl.col("low") == 0) |
            (pl.col("close") == 0)
        )
        issues += len(zero_prices)
        
        # Check for extreme outliers (> 10x median)
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                median_val = df.select(pl.col(col).median()).item()
                if median_val and median_val > 0:
                    outliers = df.filter(pl.col(col) > median_val * 10)
                    issues += len(outliers)
        
        validity_ratio = 1.0 - min(issues / total_rows, 1.0) if total_rows > 0 else 0.0
        
        details = {
            "negative_prices": len(negative_prices),
            "zero_prices": len(zero_prices),
            "total_issues": issues,
            "total_rows": total_rows,
        }
        
        recommendations = []
        if issues > 0:
            recommendations.append("Investigate and correct invalid price values including negative and zero prices")
            recommendations.append("Implement data validation at source")
            
        return DataQualityScore(
            dimension=DataQualityDimension.VALIDITY,
            score=validity_ratio,
            details=details,
            recommendations=recommendations,
        )

    def _score_timeliness(
        self, df: pl.DataFrame, start_date: datetime, end_date: datetime
    ) -> DataQualityScore:
        """Score data timeliness - coverage of requested date range."""
        if len(df) == 0:
            return DataQualityScore(
                dimension=DataQualityDimension.TIMELINESS,
                score=0.0,
                details={"error": "Empty dataset"},
                recommendations=["Dataset is empty"],
            )

        # Get actual date range
        actual_start = df.select(pl.col("datetime").min()).item()
        actual_end = df.select(pl.col("datetime").max()).item()
        
        # Calculate coverage
        requested_days = (end_date - start_date).days
        if requested_days <= 0:
            coverage_ratio = 1.0  # Same day request
        else:
            actual_days = (actual_end - actual_start).days
            coverage_ratio = min(actual_days / requested_days, 1.0)
        
        # Penalty for significant gaps from requested range
        start_gap = max(0, (actual_start - start_date).days)
        end_gap = max(0, (end_date - actual_end).days)
        gap_penalty = min((start_gap + end_gap) / requested_days * 0.5, 0.3)
        
        score = max(0.0, coverage_ratio - gap_penalty)
        
        details = {
            "requested_start": start_date.isoformat(),
            "requested_end": end_date.isoformat(),
            "actual_start": actual_start.isoformat() if actual_start else None,
            "actual_end": actual_end.isoformat() if actual_end else None,
            "coverage_ratio": coverage_ratio,
            "start_gap_days": start_gap,
            "end_gap_days": end_gap,
        }
        
        recommendations = []
        if score < 0.8:
            recommendations.append("Consider alternative data sources for better coverage")
        if start_gap > 7 or end_gap > 7:
            recommendations.append("Significant gaps in requested date range")
            
        return DataQualityScore(
            dimension=DataQualityDimension.TIMELINESS,
            score=score,
            details=details,
            recommendations=recommendations,
        )

    def _score_uniqueness(self, df: pl.DataFrame) -> DataQualityScore:
        """Score data uniqueness - duplicate records."""
        total_rows = len(df)
        
        if total_rows == 0:
            return DataQualityScore(
                dimension=DataQualityDimension.UNIQUENESS,
                score=0.0,
                details={"error": "Empty dataset"},
                recommendations=["Dataset is empty"],
            )

        # Check for duplicate datetime entries
        unique_dates = df.select(pl.col("datetime")).n_unique()
        duplicates = total_rows - unique_dates
        
        uniqueness_ratio = unique_dates / total_rows if total_rows > 0 else 0.0
        
        details = {
            "total_rows": total_rows,
            "unique_dates": unique_dates,
            "duplicates": duplicates,
            "uniqueness_ratio": uniqueness_ratio,
        }
        
        recommendations = []
        if duplicates > 0:
            recommendations.append("Remove or consolidate duplicate datetime entries")
            recommendations.append("Investigate data collection process for duplicates")
            
        return DataQualityScore(
            dimension=DataQualityDimension.UNIQUENESS,
            score=uniqueness_ratio,
            details=details,
            recommendations=recommendations,
        )

    def _detect_date_gaps(self, df: pl.DataFrame) -> List[str]:
        """Detect significant gaps in date sequence."""
        if len(df) < 2 or "datetime" not in df.columns:
            return []
            
        # Filter out null dates and sort
        df_valid = df.filter(pl.col("datetime").is_not_null()).sort("datetime")
        
        if len(df_valid) < 2:
            return []
        
        # Calculate gaps between consecutive dates
        gaps = []
        dates = df_valid.select("datetime").to_series()
        
        for i in range(1, len(dates)):
            if dates[i] is not None and dates[i-1] is not None:
                gap = (dates[i] - dates[i-1]).total_seconds() / (24 * 3600)  # days
                if gap > 7:  # More than a week
                    gaps.append(f"{dates[i-1].date()} to {dates[i].date()}")
                
        return gaps