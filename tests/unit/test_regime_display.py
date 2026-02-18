"""Tests for src.utils.regime_display."""

from __future__ import annotations

import pytest

from src.utils.regime_display import regime_label


class TestRegimeLabel:
    """Test regime_label() utility."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("R0", "Healthy Uptrend"),
            ("R1", "Choppy/Extended"),
            ("R2", "Risk-Off"),
            ("R3", "Rebound Window"),
        ],
    )
    def test_all_regime_codes(self, code: str, expected: str) -> None:
        assert regime_label(code) == expected

    @pytest.mark.parametrize("bad_code", ["R9", "", "invalid", "r0"])
    def test_unknown_code(self, bad_code: str) -> None:
        assert regime_label(bad_code) == "Unknown"
