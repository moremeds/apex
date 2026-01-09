"""
Application runners for different modes.

Provides entry points for:
- TradingRunner: Live/dry-run trading with validation gate
- SignalRunner: Standalone TA signal pipeline for validation/testing
"""

from .trading_runner import (
    TradingRunner,
    TradingConfig,
    StrategyNotValidatedError,
    ManifestLoadError,
    load_strategy_manifest,
)
from .signal_runner import (
    SignalRunner,
    SignalRunnerConfig,
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
