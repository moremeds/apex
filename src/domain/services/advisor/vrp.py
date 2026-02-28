"""VRP (Volatility Risk Premium) calculation and BSM strike estimation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.domain.services.advisor.models import VRPResult

# Fallback when data is insufficient
_FALLBACK = VRPResult(iv30=0.0, rv30=0.0, vrp=0.0, vrp_zscore=0.0, iv_percentile=50.0)


def compute_vrp(
    vix_close: pd.Series,
    underlying_close: pd.Series,
    rv_window: int = 30,
    zscore_window: int = 252,
) -> VRPResult:
    """Compute Volatility Risk Premium.

    VRP = IV30 (VIX) - RV30 (realized vol from underlying), z-scored.
    """
    if vix_close is None or underlying_close is None:
        return _FALLBACK
    if len(vix_close) < rv_window + 2 or len(underlying_close) < rv_window + 2:
        return VRPResult(
            iv30=float(vix_close.iloc[-1]) if len(vix_close) > 0 else 0.0,
            rv30=0.0,
            vrp=0.0,
            vrp_zscore=0.0,
            iv_percentile=50.0,
        )

    # Realized vol: annualized std of log returns over rv_window
    log_returns = np.log(underlying_close / underlying_close.shift(1)).dropna()
    rv30 = log_returns.rolling(rv_window).std() * math.sqrt(252) * 100

    # Align by date — strip timezone info and normalize to date-only
    def _to_date_index(idx: pd.Index) -> pd.Index:
        if isinstance(idx, pd.DatetimeIndex):
            bare = idx.tz_localize(None) if idx.tz is not None else idx
            return bare.normalize()
        return idx

    if isinstance(vix_close.index, pd.DatetimeIndex) and isinstance(
        rv30.index, pd.DatetimeIndex
    ):
        vix_by_date = vix_close.copy()
        vix_by_date.index = _to_date_index(vix_close.index)
        rv30_by_date = rv30.copy()
        rv30_by_date.index = _to_date_index(rv30.index)
        common = vix_by_date.index.intersection(rv30_by_date.index)
        vrp = vix_by_date.loc[common] - rv30_by_date.loc[common]
    else:
        vrp = vix_close - rv30
    vrp = vrp.dropna()

    if len(vrp) == 0:
        return _FALLBACK

    if len(vrp) < zscore_window:
        current_vrp = vrp.iloc[-1]
        # Use whatever window we have (min 30) for z-score instead of returning 0
        usable = len(vrp)
        if usable >= 30:
            vrp_mean_val = vrp.mean()
            vrp_std_val = vrp.std()
            zscore_partial = (
                float((current_vrp - vrp_mean_val) / vrp_std_val) if vrp_std_val > 0 else 0.0
            )
            vix_tail = vix_close.iloc[-usable:]
            iv_pctile_partial = float(
                (vix_tail < vix_close.iloc[-1]).sum() / len(vix_tail) * 100
            )
            return VRPResult(
                iv30=float(vix_close.iloc[-1]),
                rv30=float(rv30.dropna().iloc[-1]) if len(rv30.dropna()) > 0 else 0.0,
                vrp=float(current_vrp),
                vrp_zscore=zscore_partial,
                iv_percentile=iv_pctile_partial,
            )
        return VRPResult(
            iv30=float(vix_close.iloc[-1]),
            rv30=float(rv30.dropna().iloc[-1]) if len(rv30.dropna()) > 0 else 0.0,
            vrp=float(current_vrp),
            vrp_zscore=0.0,
            iv_percentile=50.0,
        )

    vrp_mean = vrp.rolling(zscore_window).mean()
    vrp_std = vrp.rolling(zscore_window).std()

    current_vrp = vrp.iloc[-1]
    std_val = vrp_std.iloc[-1]
    zscore = (current_vrp - vrp_mean.iloc[-1]) / std_val if std_val > 0 else 0.0

    # IV percentile: rank of current VIX in trailing window
    vix_tail = vix_close.iloc[-zscore_window:]
    iv_pctile = float((vix_tail < vix_close.iloc[-1]).sum() / len(vix_tail) * 100)

    return VRPResult(
        iv30=float(vix_close.iloc[-1]),
        rv30=float(rv30.iloc[-1]),
        vrp=float(current_vrp),
        vrp_zscore=float(zscore),
        iv_percentile=iv_pctile,
    )


def compute_term_structure(vix: float, vix3m: float) -> tuple[float, str]:
    """Compute VIX/VIX3M term structure ratio and state."""
    if vix3m <= 0:
        return 1.0, "flat"

    ratio = vix / vix3m

    if ratio > 1.2:
        state = "inverted"
    elif ratio > 1.0:
        state = "flat"
    else:
        state = "contango"

    return round(ratio, 3), state


def estimate_strike_bsm(
    spot: float,
    iv: float,
    dte: int,
    target_delta: float,
    option_type: str,
) -> float:
    """Estimate strike price for a target delta using BSM approximation.

    DTE is clamped to min 1 to avoid division by zero.
    """
    t = max(dte, 1) / 365.0
    vol_sqrt_t = iv * math.sqrt(t)

    if option_type == "put":
        d1 = norm.ppf(1 - target_delta)
        strike = spot * math.exp(-vol_sqrt_t * d1 + 0.5 * iv * iv * t)
    else:
        d1 = norm.ppf(target_delta)
        strike = spot * math.exp(-vol_sqrt_t * d1 + 0.5 * iv * iv * t)

    return round(strike, 2)
