"""
Signal Pipeline Package - Modular components for signal runner.

This package contains the refactored signal runner components:
- config: Configuration dataclass and argument parser
- processor: Live and backfill signal processing
- validator: Bar validation and reporting
- trainer: Turning point model training
"""

from .config import SignalPipelineConfig, create_argument_parser
from .processor import SignalPipelineProcessor
from .trainer import TurningPointTrainer
from .validator import BarValidator

__all__ = [
    "SignalPipelineConfig",
    "create_argument_parser",
    "SignalPipelineProcessor",
    "BarValidator",
    "TurningPointTrainer",
]
