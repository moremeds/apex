"""
Parity testing module for comparing backtest engines.

Detects drift between different execution paths (VectorBT vs Apex,
live vs backtest) to prevent subtle production bugs.

Exports:
- StrategyParityHarness: Main harness for engine comparison
- compare_signal_parity: Signal-level parity comparison
- compare_directional_signal_parity: Directional signal comparison
- SignalCapture: Capture signals from event-driven strategies
- DirectionalSignalCapture: Capture directional signals
- DriftType, DriftDetail, ParityResult, ParityConfig: Result models
- SignalParityResult: Signal comparison results
"""

# Signal capture classes
from .capture import DirectionalSignalCapture, SignalCapture

# Harness
from .harness import StrategyParityHarness

# Models
from .models import (
    DriftDetail,
    DriftType,
    ParityConfig,
    ParityResult,
    SignalParityResult,
)

# Signal comparison functions
from .signals import compare_directional_signal_parity, compare_signal_parity

__all__ = [
    # Models
    "DriftType",
    "DriftDetail",
    "ParityResult",
    "ParityConfig",
    "SignalParityResult",
    # Harness
    "StrategyParityHarness",
    # Signal functions
    "compare_signal_parity",
    "compare_directional_signal_parity",
    # Capture classes
    "SignalCapture",
    "DirectionalSignalCapture",
]
