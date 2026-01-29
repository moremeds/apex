"""
Regime Block Splitter — concatenates named date blocks into train/verify windows.

Used for DualMACD behavioral gate walk-forward validation where training
and verification happen on distinct market regime periods.

Walk-forward discipline (Review #7):
- Round 1: optimize parameters
- Rounds 2-3: verify only (no re-tuning). If verify fails, discard params.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterator, List, Tuple

from ...core import TimeWindow


@dataclass
class RegimeBlock:
    """A named date range representing a market regime period."""

    name: str
    start: date
    end: date
    role: str = "train"  # train / verify


@dataclass
class RegimeBlockConfig:
    """Configuration for regime-block walk-forward splitting."""

    blocks: List[RegimeBlock] = field(default_factory=list)
    purge_days: int = 5

    def add_block(self, name: str, start: date, end: date, role: str = "train") -> None:
        self.blocks.append(RegimeBlock(name=name, start=start, end=end, role=role))


class RegimeBlockSplitter:
    """
    Splits data into rounds based on named regime blocks.

    Each round consists of:
    - Training blocks (role="train") concatenated for parameter optimization
    - Verification blocks (role="verify") for out-of-sample validation

    Walk-forward discipline:
    - Round 1 optimizes → Rounds 2-3 only verify.
    - If verify fails, entire parameter set is discarded.
    """

    def __init__(self, config: RegimeBlockConfig) -> None:
        self.config = config

    def split(self) -> Iterator[Tuple[List[RegimeBlock], List[RegimeBlock]]]:
        """
        Yield (train_blocks, verify_blocks) tuples.

        First round: train + verify
        Subsequent rounds: verify only (using params from Round 1)
        """
        train_blocks = [b for b in self.config.blocks if b.role == "train"]
        verify_blocks = [b for b in self.config.blocks if b.role == "verify"]

        if not train_blocks:
            return

        # Round 1: train + first verify block
        if verify_blocks:
            yield train_blocks, [verify_blocks[0]]

        # Rounds 2+: verify only with remaining blocks
        for vb in verify_blocks[1:]:
            yield train_blocks, [vb]

    def to_time_windows(self) -> Iterator[Tuple[TimeWindow, TimeWindow]]:
        """
        Convert regime blocks into TimeWindow pairs for the backtest runner.

        Each (train_blocks, verify_blocks) pair produces one TimeWindow tuple.
        """
        for fold_idx, (train_blocks, verify_blocks) in enumerate(self.split()):
            train_start = min(b.start for b in train_blocks)
            train_end = max(b.end for b in train_blocks)

            verify_start = min(b.start for b in verify_blocks)
            verify_end = max(b.end for b in verify_blocks)

            train_window = TimeWindow(
                window_id=f"regime_fold_{fold_idx}",
                fold_index=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=verify_start,
                test_end=verify_end,
                purge_days=self.config.purge_days,
                is_train=True,
                is_oos=False,
            )
            test_window = TimeWindow(
                window_id=f"regime_fold_{fold_idx}",
                fold_index=fold_idx,
                train_start=train_start,
                train_end=train_end,
                test_start=verify_start,
                test_end=verify_end,
                purge_days=self.config.purge_days,
                is_train=False,
                is_oos=True,
            )

            yield train_window, test_window
