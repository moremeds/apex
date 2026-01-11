#!/usr/bin/env python3
"""
Signal Engine Test Runner.

Tests the trading signal engine components:
- Individual indicator calculations
- Rule evaluation
- Signal generation pipeline

Usage:
    # Test all indicators
    python scripts/test_signal_engine.py --all

    # Test specific indicator
    python scripts/test_signal_engine.py --indicator rsi

    # Test all rules
    python scripts/test_signal_engine.py --rules

    # List available indicators
    python scripts/test_signal_engine.py --list

    # Verbose output
    python scripts/test_signal_engine.py --all -v
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_sample_bars(
    symbol: str = "AAPL",
    num_bars: int = 100,
    timeframe: str = "1h",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate sample OHLCV bar data for testing.

    Creates realistic-looking price data with trends and volatility.
    """
    np.random.seed(seed)

    bars = []
    base_price = 150.0
    base_volume = 1_000_000

    # Generate price series with trend + noise
    returns = np.random.normal(0.0002, 0.015, num_bars)  # Small upward drift
    prices = base_price * np.exp(np.cumsum(returns))

    start_time = datetime.now() - timedelta(hours=num_bars)

    for i in range(num_bars):
        price = prices[i]
        # Add intrabar variation
        high_var = abs(np.random.normal(0, 0.005))
        low_var = abs(np.random.normal(0, 0.005))

        open_price = price * (1 + np.random.normal(0, 0.002))
        close_price = price
        high_price = max(open_price, close_price) * (1 + high_var)
        low_price = min(open_price, close_price) * (1 - low_var)
        volume = int(base_volume * (1 + np.random.normal(0, 0.3)))

        bar = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": start_time + timedelta(hours=i),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": max(0, volume),
        }
        bars.append(bar)

    return bars


def test_indicator(indicator_name: str, verbose: bool = False) -> bool:
    """Test a specific indicator with sample data."""
    from src.domain.signals.indicators.registry import IndicatorRegistry

    registry = IndicatorRegistry()
    registry.discover()

    # Get indicator instance (registry stores instances, not classes)
    indicator = registry.get(indicator_name)
    if not indicator:
        print(f"  [ERROR] Indicator '{indicator_name}' not found")
        return False

    print(f"  Testing {indicator_name}...")

    try:
        import pandas as pd

        # Generate sample data
        bars = generate_sample_bars(num_bars=100)

        # Convert to pandas DataFrame (indicator protocol expects DataFrame)
        bar_data = pd.DataFrame(bars)
        bar_data = bar_data.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        })

        # Calculate indicator with default params
        result = indicator.calculate(bar_data, {})

        if result is None:
            print(f"    [WARN] Indicator returned None (may need more data)")
            return True

        if verbose:
            print(f"    Result type: {type(result)}")
            if isinstance(result, pd.DataFrame):
                print(f"    Columns: {list(result.columns)}")
                # Show last row values
                last_row = result.iloc[-1]
                for col in result.columns:
                    val = last_row[col]
                    if pd.notna(val):
                        print(f"    {col}: {val:.4f}")
            elif isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        print(f"    {key}: {value:.4f}")
                    elif isinstance(value, np.ndarray):
                        print(f"    {key}: array shape {value.shape}, last={value[-1]:.4f}")
                    else:
                        print(f"    {key}: {value}")
            elif isinstance(result, (int, float)):
                print(f"    Value: {result:.4f}")

        print(f"    [OK] {indicator_name} calculated successfully")
        return True

    except Exception as e:
        print(f"    [ERROR] {indicator_name} failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_all_indicators(verbose: bool = False) -> Dict[str, bool]:
    """Test all registered indicators."""
    from src.domain.signals.indicators.registry import IndicatorRegistry

    registry = IndicatorRegistry()
    registry.discover()

    results = {}
    indicators = registry.get_names()

    print(f"\nTesting {len(indicators)} indicators...")
    print("-" * 50)

    for name in sorted(indicators):
        results[name] = test_indicator(name, verbose)

    return results


def test_rules(verbose: bool = False) -> Dict[str, bool]:
    """Test all registered rules."""
    from src.domain.signals.rules import (
        MOMENTUM_RULES,
        TREND_RULES,
        VOLATILITY_RULES,
        VOLUME_RULES,
    )

    results = {}
    rule_sets = [
        ("momentum", MOMENTUM_RULES),
        ("trend", TREND_RULES),
        ("volatility", VOLATILITY_RULES),
        ("volume", VOLUME_RULES),
    ]

    print("\nTesting signal rules...")
    print("-" * 50)

    for category, rules in rule_sets:
        print(f"\n{category.upper()} Rules ({len(rules)}):")

        if not rules:
            print(f"  [WARN] No rules defined in {category}")
            continue

        for rule in rules:
            rule_name = getattr(rule, "name", str(rule))
            try:
                # Test rule condition evaluation with mock states
                prev_state = {"value": 25.0, "zone": "oversold", "fast": 10.0, "slow": 20.0}
                curr_state = {"value": 35.0, "zone": "neutral", "fast": 25.0, "slow": 20.0}

                result = rule.check_condition(prev_state, curr_state)

                if verbose:
                    print(f"  {rule_name}: condition={result}")
                else:
                    print(f"  {rule_name}: [OK]")

                results[f"{category}:{rule_name}"] = True

            except Exception as e:
                print(f"  {rule_name}: [ERROR] {e}")
                results[f"{category}:{rule_name}"] = False

    return results


def list_indicators():
    """List all available indicators."""
    from src.domain.signals.indicators.registry import IndicatorRegistry

    registry = IndicatorRegistry()
    registry.discover()

    indicator_names = registry.get_names()
    categories = {}

    for name in indicator_names:
        indicator = registry.get(name)
        if indicator:
            category = getattr(indicator, "category", "unknown")
            if hasattr(category, "value"):
                category = category.value
            categories.setdefault(category, []).append(name)

    print("\nAvailable Indicators:")
    print("-" * 50)

    for category in sorted(categories.keys()):
        print(f"\n{category.upper()}:")
        for name in sorted(categories[category]):
            print(f"  - {name}")

    print(f"\nTotal: {len(indicator_names)} indicators")


def run_full_pipeline(symbol: str = "AAPL", verbose: bool = False):
    """Run the full signal generation pipeline."""
    import pandas as pd
    from src.domain.signals.indicators.registry import IndicatorRegistry

    print(f"\nRunning full pipeline for {symbol}...")
    print("-" * 50)

    # Generate sample data
    bars = generate_sample_bars(symbol=symbol, num_bars=200)
    print(f"Generated {len(bars)} sample bars")

    # Convert to DataFrame for indicator protocol
    bar_data = pd.DataFrame(bars)
    bar_data = bar_data.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    })

    # Initialize registry
    indicator_registry = IndicatorRegistry()
    indicator_registry.discover()
    print(f"Loaded {len(indicator_registry)} indicators")

    # Test calculating key indicators
    print("\nCalculating indicators...")

    # Calculate a few key indicators (registry returns instances, not classes)
    for ind_name in ["rsi", "macd", "bollinger", "supertrend"]:
        indicator = indicator_registry.get(ind_name)
        if indicator:
            try:
                # Indicator protocol: calculate(DataFrame, params_dict) -> DataFrame
                result = indicator.calculate(bar_data, {})
                if result is not None and verbose:
                    if isinstance(result, pd.DataFrame):
                        last_row = result.iloc[-1].to_dict()
                        non_nan = {k: f"{v:.4f}" for k, v in last_row.items()
                                   if pd.notna(v) and isinstance(v, (int, float))}
                        print(f"  {ind_name}: {non_nan}")
                    else:
                        print(f"  {ind_name}: {result}")
                else:
                    print(f"  {ind_name}: calculated")
            except Exception as e:
                print(f"  {ind_name}: [ERROR] {e}")

    print("\n[OK] Pipeline test completed")


def main():
    parser = argparse.ArgumentParser(
        description="Test Signal Engine components",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true", help="Test all indicators")
    parser.add_argument("--indicator", "-i", type=str, help="Test specific indicator")
    parser.add_argument("--rules", action="store_true", help="Test all rules")
    parser.add_argument("--list", action="store_true", help="List available indicators")
    parser.add_argument("--pipeline", action="store_true", help="Run full pipeline test")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Default to --list if no args
    if not any([args.all, args.indicator, args.rules, args.list, args.pipeline]):
        args.list = True

    if args.list:
        list_indicators()

    if args.indicator:
        success = test_indicator(args.indicator, args.verbose)
        sys.exit(0 if success else 1)

    if args.all:
        results = test_all_indicators(args.verbose)
        passed = sum(1 for v in results.values() if v)
        failed = len(results) - passed
        print(f"\n{'='*50}")
        print(f"Results: {passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)

    if args.rules:
        results = test_rules(args.verbose)
        passed = sum(1 for v in results.values() if v)
        failed = len(results) - passed
        print(f"\n{'='*50}")
        print(f"Results: {passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)

    if args.pipeline:
        run_full_pipeline(verbose=args.verbose)


if __name__ == "__main__":
    main()
