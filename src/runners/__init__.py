"""
Application runners for different modes.

Provides entry points for:
- TradingRunner: Live/dry-run trading with validation gate
- SignalRunner: Standalone TA signal pipeline for validation/testing
"""

from .signal_runner import (
    SignalRunner,
    SignalRunnerConfig,
)
from .trading_runner import (
    ManifestLoadError,
    StrategyNotValidatedError,
    TradingConfig,
    TradingRunner,
    load_strategy_manifest,
)

__all__ = [
    "TradingRunner",
    "TradingConfig",
    "StrategyNotValidatedError",
    "ManifestLoadError",
    "load_strategy_manifest",
    # Signal pipeline runner
    "SignalRunner",
    "SignalRunnerConfig",
]
