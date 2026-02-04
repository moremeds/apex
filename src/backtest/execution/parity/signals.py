"""
Signal parity comparison functions.

Compare signals from VectorBT SignalGenerator vs event-driven Strategy
to detect parity drift at the signal level.
"""

from typing import List

import pandas as pd

from .models import SignalParityResult


def compare_signal_parity(
    vectorbt_entries: pd.Series,
    vectorbt_exits: pd.Series,
    captured_entries: pd.Series,
    captured_exits: pd.Series,
    warmup_bars: int,
) -> SignalParityResult:
    """
    Compare signals from VectorBT SignalGenerator vs event-driven Strategy.

    Args:
        vectorbt_entries: Boolean series from SignalGenerator.generate()
        vectorbt_exits: Boolean series from SignalGenerator.generate()
        captured_entries: Boolean series captured from event-driven strategy
        captured_exits: Boolean series captured from event-driven strategy
        warmup_bars: Number of initial bars to skip (indicator warmup)

    Returns:
        SignalParityResult with match/mismatch statistics

    Note:
        - All series must have the same index
        - Comparison is exact match (True == True, False == False)
        - NaN values are treated as False
    """
    # Ensure all 4 series have the same index
    indices_match = (
        vectorbt_entries.index.equals(captured_entries.index)
        and vectorbt_exits.index.equals(captured_exits.index)
        and vectorbt_entries.index.equals(vectorbt_exits.index)
    )
    if not indices_match:
        return SignalParityResult(
            passed=False,
            warmup_bars=warmup_bars,
            total_bars=len(vectorbt_entries),
            compared_bars=0,
            mismatches=["Index mismatch between signal series - all must have same index"],
        )

    total_bars = len(vectorbt_entries)
    compared_bars = total_bars - warmup_bars

    if compared_bars <= 0:
        return SignalParityResult(
            passed=True,
            warmup_bars=warmup_bars,
            total_bars=total_bars,
            compared_bars=0,
            mismatches=["No bars to compare after warmup"],
        )

    # Skip warmup bars
    vbt_entries = vectorbt_entries.iloc[warmup_bars:].fillna(False).astype(bool)
    vbt_exits = vectorbt_exits.iloc[warmup_bars:].fillna(False).astype(bool)
    cap_entries = captured_entries.iloc[warmup_bars:].fillna(False).astype(bool)
    cap_exits = captured_exits.iloc[warmup_bars:].fillna(False).astype(bool)

    # Compare entries
    entry_match = vbt_entries == cap_entries
    entry_matches = entry_match.sum()
    entry_mismatches = (~entry_match).sum()

    # Compare exits
    exit_match = vbt_exits == cap_exits
    exit_matches = exit_match.sum()
    exit_mismatches = (~exit_match).sum()

    # Find first mismatches
    first_entry_mismatch = None
    first_exit_mismatch = None
    mismatches: List[str] = []

    if entry_mismatches > 0:
        mismatch_idx = entry_match[~entry_match].index
        first_entry_mismatch = mismatch_idx[0] if len(mismatch_idx) > 0 else None
        # Report first 5 mismatches
        for idx in list(mismatch_idx)[:5]:
            mismatches.append(
                f"Entry mismatch at {idx}: "
                f"vectorbt={vbt_entries.loc[idx]}, captured={cap_entries.loc[idx]}"
            )

    if exit_mismatches > 0:
        mismatch_idx = exit_match[~exit_match].index
        first_exit_mismatch = mismatch_idx[0] if len(mismatch_idx) > 0 else None
        for idx in list(mismatch_idx)[:5]:
            mismatches.append(
                f"Exit mismatch at {idx}: "
                f"vectorbt={vbt_exits.loc[idx]}, captured={cap_exits.loc[idx]}"
            )

    passed = bool(entry_mismatches == 0 and exit_mismatches == 0)

    return SignalParityResult(
        passed=passed,
        warmup_bars=warmup_bars,
        total_bars=total_bars,
        compared_bars=compared_bars,
        entry_matches=int(entry_matches),
        entry_mismatches=int(entry_mismatches),
        exit_matches=int(exit_matches),
        exit_mismatches=int(exit_mismatches),
        first_entry_mismatch_idx=first_entry_mismatch,
        first_exit_mismatch_idx=first_exit_mismatch,
        mismatches=mismatches,
    )


def compare_directional_signal_parity(
    vectorbt_long_entries: pd.Series,
    vectorbt_long_exits: pd.Series,
    vectorbt_short_entries: pd.Series,
    vectorbt_short_exits: pd.Series,
    captured_long_entries: pd.Series,
    captured_long_exits: pd.Series,
    captured_short_entries: pd.Series,
    captured_short_exits: pd.Series,
    warmup_bars: int,
) -> SignalParityResult:
    """
    Compare directional signals from VectorBT SignalGenerator vs event-driven Strategy.

    Used for strategies that generate both long AND short signals (e.g., MTF RSI Trend).

    Args:
        vectorbt_long_entries: Long entry signals from SignalGenerator.generate_directional()
        vectorbt_long_exits: Long exit signals
        vectorbt_short_entries: Short entry signals
        vectorbt_short_exits: Short exit signals
        captured_long_entries: Long entries captured from event-driven strategy
        captured_long_exits: Long exits captured
        captured_short_entries: Short entries captured
        captured_short_exits: Short exits captured
        warmup_bars: Number of initial bars to skip

    Returns:
        SignalParityResult aggregating all 4 signal comparisons
    """
    # Compare long signals
    long_result = compare_signal_parity(
        vectorbt_long_entries,
        vectorbt_long_exits,
        captured_long_entries,
        captured_long_exits,
        warmup_bars,
    )

    # Compare short signals
    short_result = compare_signal_parity(
        vectorbt_short_entries,
        vectorbt_short_exits,
        captured_short_entries,
        captured_short_exits,
        warmup_bars,
    )

    # Aggregate results
    passed = long_result.passed and short_result.passed

    # Combine mismatches with side labels
    mismatches = []
    for m in long_result.mismatches:
        mismatches.append(f"[LONG] {m}")
    for m in short_result.mismatches:
        mismatches.append(f"[SHORT] {m}")

    return SignalParityResult(
        passed=passed,
        warmup_bars=warmup_bars,
        total_bars=long_result.total_bars,
        compared_bars=long_result.compared_bars,
        entry_matches=long_result.entry_matches + short_result.entry_matches,
        entry_mismatches=long_result.entry_mismatches + short_result.entry_mismatches,
        exit_matches=long_result.exit_matches + short_result.exit_matches,
        exit_mismatches=long_result.exit_mismatches + short_result.exit_mismatches,
        first_entry_mismatch_idx=(
            long_result.first_entry_mismatch_idx or short_result.first_entry_mismatch_idx
        ),
        first_exit_mismatch_idx=(
            long_result.first_exit_mismatch_idx or short_result.first_exit_mismatch_idx
        ),
        mismatches=mismatches,
    )
