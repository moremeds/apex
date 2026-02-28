"""Tests for VRP calculation and BSM strike estimation."""

import math

import numpy as np
import pandas as pd
import pytest

from src.domain.services.advisor.vrp import (
    compute_term_structure,
    compute_vrp,
    estimate_strike_bsm,
)


class TestComputeVRP:
    def test_positive_vrp(self):
        """When IV > RV, VRP should be positive."""
        np.random.seed(42)
        n = 300
        vix = pd.Series(np.random.normal(20, 2, n).clip(10, 40))
        returns = pd.Series(np.random.normal(0.0003, 0.009, n))
        underlying = (1 + returns).cumprod() * 500

        result = compute_vrp(vix, underlying)
        assert result.iv30 == pytest.approx(vix.iloc[-1], abs=0.01)
        assert result.vrp > 0

    def test_negative_vrp(self):
        """When RV > IV (stress), VRP should be negative."""
        np.random.seed(42)
        n = 300
        vix = pd.Series(np.full(n, 12.0))
        returns = pd.Series(np.random.normal(-0.001, 0.016, n))
        underlying = (1 + returns).cumprod() * 500

        result = compute_vrp(vix, underlying)
        assert result.vrp < 0

    def test_vrp_zscore_range(self):
        """Z-score should be finite and reasonable."""
        np.random.seed(42)
        n = 300
        vix = pd.Series(np.random.normal(18, 3, n).clip(10, 40))
        returns = pd.Series(np.random.normal(0.0003, 0.01, n))
        underlying = (1 + returns).cumprod() * 500

        result = compute_vrp(vix, underlying)
        assert math.isfinite(result.vrp_zscore)
        assert -5 < result.vrp_zscore < 5

    def test_iv_percentile_range(self):
        np.random.seed(42)
        n = 300
        vix = pd.Series(np.random.normal(18, 3, n).clip(10, 40))
        returns = pd.Series(np.random.normal(0.0003, 0.01, n))
        underlying = (1 + returns).cumprod() * 500

        result = compute_vrp(vix, underlying)
        assert 0 <= result.iv_percentile <= 100

    def test_insufficient_data_fallback(self):
        """Short series should return safe defaults, not crash."""
        vix = pd.Series([18.0, 19.0, 17.5])
        underlying = pd.Series([500.0, 501.0, 499.0])

        result = compute_vrp(vix, underlying)
        assert result.vrp_zscore == 0.0
        assert result.iv_percentile == 50.0

    def test_empty_series_returns_fallback(self):
        """Empty series should return safe defaults."""
        vix = pd.Series(dtype=float)
        underlying = pd.Series(dtype=float)

        result = compute_vrp(vix, underlying)
        assert result.vrp_zscore == 0.0


class TestComputeTermStructure:
    def test_contango(self):
        ratio, state = compute_term_structure(vix=18.0, vix3m=20.0)
        assert ratio < 1.0
        assert state == "contango"

    def test_inverted(self):
        ratio, state = compute_term_structure(vix=25.0, vix3m=18.0)
        assert ratio > 1.2
        assert state == "inverted"

    def test_flat(self):
        ratio, state = compute_term_structure(vix=19.5, vix3m=19.0)
        assert state == "flat"

    def test_zero_vix3m(self):
        ratio, state = compute_term_structure(vix=18.0, vix3m=0.0)
        assert ratio == 1.0
        assert state == "flat"


class TestEstimateStrikeBSM:
    def test_put_below_spot(self):
        strike = estimate_strike_bsm(
            spot=500, iv=0.18, dte=35, target_delta=0.25, option_type="put"
        )
        assert strike < 500
        assert strike > 400

    def test_call_above_spot(self):
        strike = estimate_strike_bsm(
            spot=500, iv=0.18, dte=35, target_delta=0.25, option_type="call"
        )
        assert strike > 500
        assert strike < 600

    def test_higher_iv_wider_strike(self):
        strike_low = estimate_strike_bsm(
            spot=500, iv=0.15, dte=35, target_delta=0.25, option_type="put"
        )
        strike_high = estimate_strike_bsm(
            spot=500, iv=0.30, dte=35, target_delta=0.25, option_type="put"
        )
        assert strike_high < strike_low

    def test_longer_dte_wider_strike(self):
        strike_short = estimate_strike_bsm(
            spot=500, iv=0.18, dte=14, target_delta=0.25, option_type="put"
        )
        strike_long = estimate_strike_bsm(
            spot=500, iv=0.18, dte=60, target_delta=0.25, option_type="put"
        )
        assert strike_long < strike_short

    def test_50_delta_near_atm(self):
        strike = estimate_strike_bsm(
            spot=500, iv=0.18, dte=35, target_delta=0.50, option_type="put"
        )
        assert abs(strike - 500) < 10

    def test_zero_dte_clamped(self):
        """DTE=0 should not crash (clamped to 1)."""
        strike = estimate_strike_bsm(spot=500, iv=0.18, dte=0, target_delta=0.25, option_type="put")
        assert 400 < strike < 500
