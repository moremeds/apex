"""The asset-class registry: one row per partition livewire publishes.

This module is the seam that makes the read surface extendible. Route dispatch,
path construction, timeframe validation, payload selection and adjustment policy all
resolve through ``get_asset_class`` -- adding a seventh class is a row here, not a
new route.

Verified against the production lake (macmini:/Volumes/DATA_LAKE/livewire/data-lake)
on 2026-08-23. Three distinct parquet schemas exist:

  * OHLCV        equity, volatility, fx, cmdty
  * OHLCV+       futures, which adds settlement / open_interest / contract identity
  * rates        trade_date + tenor_years + yield_pct -- a yield, not a price

``timeframes`` is the ladder the *class* publishes, measured by enumerating every
symbol directory in each non-equity partition. It is a capability ceiling, not a
per-symbol guarantee: 30 of 44 volatility symbols carry only ``1d``, and a request
for a timeframe a given symbol lacks reads an absent file and yields no bars.

Silver exists ONLY under asset_class=equity, so no other class may serve adjusted
prices. ``supports_adjusted`` encodes that; it is not a preference.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_ASSET_CLASS = "equity"


class UnknownAssetClass(ValueError):
    """Raised when a caller names an asset class the lake does not publish."""


@dataclass(frozen=True)
class AssetClassSpec:
    """Everything the read path needs to know about one lake partition."""

    name: str
    partition: str
    timeframes: tuple[str, ...]
    payload: str
    supports_adjusted: bool
    extra_bar_fields: tuple[str, ...] = ()


# Ordered finest-to-coarsest; error messages and consumers both rely on the order.
_FULL_LADDER = ("1m", "5m", "30m", "1h", "1d")
# Measured union across all 44 volatility symbols -- no 1m file exists anywhere.
_VOLATILITY_LADDER = ("5m", "30m", "1h", "1d")
_DAILY_ONLY = ("1d",)


def _daily_ohlcv(name: str) -> AssetClassSpec:
    return AssetClassSpec(
        name=name,
        partition=f"asset_class={name}",
        timeframes=_DAILY_ONLY,
        payload="bars",
        supports_adjusted=False,
    )


ASSET_CLASSES: dict[str, AssetClassSpec] = {
    "equity": AssetClassSpec(
        name="equity",
        partition="asset_class=equity",
        timeframes=_FULL_LADDER,
        payload="bars",
        supports_adjusted=True,
    ),
    "volatility": AssetClassSpec(
        name="volatility",
        partition="asset_class=volatility",
        timeframes=_VOLATILITY_LADDER,
        payload="bars",
        supports_adjusted=False,
    ),
    "fx": AssetClassSpec(
        name="fx",
        partition="asset_class=fx",
        timeframes=_FULL_LADDER,
        payload="bars",
        supports_adjusted=False,
    ),
    "cmdty": _daily_ohlcv("cmdty"),
    "futures": AssetClassSpec(
        name="futures",
        partition="asset_class=futures",
        timeframes=_DAILY_ONLY,
        payload="bars",
        supports_adjusted=False,
        extra_bar_fields=("settlement", "open_interest"),
    ),
    "rates": AssetClassSpec(
        name="rates",
        partition="asset_class=rates",
        timeframes=_DAILY_ONLY,
        payload="rates_series",
        supports_adjusted=False,
    ),
}


def get_asset_class(name: str) -> AssetClassSpec:
    """Return the spec for ``name``, or raise ``UnknownAssetClass``."""
    try:
        return ASSET_CLASSES[name]
    except KeyError:
        raise UnknownAssetClass(
            f"unknown asset class: {name!r} (have {sorted(ASSET_CLASSES)})"
        ) from None
