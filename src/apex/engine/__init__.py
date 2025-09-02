"""Backtesting engines for Apex."""

from apex.engine.base import BacktestResult, BaseEngine, EngineConfig, StrategyProtocol
from apex.engine.data_extraction import VectorbtDataExtractor
from apex.engine.harness import EngineHarness, EngineType, quick_backtest
from apex.engine.portfolio import PortfolioMetrics
from apex.engine.signal_processor import SignalProcessor
from apex.engine.vectorbt import VectorbtEngine

__all__ = [
    # Base classes and protocols
    "BaseEngine",
    "StrategyProtocol", 
    "EngineConfig",
    "BacktestResult",
    
    # Engine implementations
    "VectorbtEngine",
    
    # Harness and utilities
    "EngineHarness",
    "EngineType",
    "quick_backtest",
    
    # Engine components
    "VectorbtDataExtractor",
    "PortfolioMetrics", 
    "SignalProcessor",
]