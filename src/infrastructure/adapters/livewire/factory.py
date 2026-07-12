"""Construct a LivewireOhlcProvider from config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ohlc_provider import LivewireOhlcProvider


def build_livewire_provider(config: Any) -> LivewireOhlcProvider:
    root = getattr(config, "livewire_bronze_root", None)
    if not root:
        raise ValueError("livewire_bronze_root is not configured")
    silver_root = getattr(config, "livewire_silver_root", None)
    return LivewireOhlcProvider(
        bronze_root=Path(root),
        silver_root=Path(silver_root) if silver_root else None,
        price_mode=getattr(config, "livewire_price_mode", "raw"),
    )
