"""Tests for _map_regime_to_flex() — regime_detector state → RegimeFlexRow mapping."""

from src.server.pipeline import _map_regime_to_flex


class TestMapRegimeToFlex:
    def test_short_codes(self):
        """Short regime codes (R0-R3) map to correct exposure."""
        result = _map_regime_to_flex({"regime": "R0", "regime_changed": False})
        assert result["regime"] == "R0"
        assert result["target_exposure"] == 1.0
        assert result["signal"] == "NONE"

        result = _map_regime_to_flex({"regime": "R1", "regime_changed": False})
        assert result["target_exposure"] == 0.5

        result = _map_regime_to_flex({"regime": "R2", "regime_changed": False})
        assert result["target_exposure"] == 0.0

        result = _map_regime_to_flex({"regime": "R3", "regime_changed": False})
        assert result["target_exposure"] == 0.25

    def test_regime_transition_signal(self):
        """When regime_changed=True, signal shows transition."""
        result = _map_regime_to_flex({
            "regime": "R2",
            "regime_changed": True,
            "previous_regime": "R0",
        })
        assert result["regime"] == "R2"
        assert result["target_exposure"] == 0.0
        assert result["signal"] == "R0→R2"

    def test_unknown_regime_defaults_to_half_exposure(self):
        """Unknown regime code falls back to 0.5 exposure."""
        result = _map_regime_to_flex({"regime": "X9"})
        assert result["regime"] == "X9"
        assert result["target_exposure"] == 0.5

    def test_full_enum_name_handled(self):
        """Full enum names like R1_CHOPPY_EXTENDED are split to short code."""
        result = _map_regime_to_flex({
            "regime": "R1_CHOPPY_EXTENDED",
            "regime_changed": False,
        })
        assert result["regime"] == "R1"
        assert result["target_exposure"] == 0.5
        assert result["signal"] == "NONE"

    def test_transition_with_full_enum_names(self):
        """Transition signal with full enum names extracts short codes."""
        result = _map_regime_to_flex({
            "regime": "R0_HEALTHY_UPTREND",
            "regime_changed": True,
            "previous_regime": "R2_RISK_OFF",
        })
        assert result["regime"] == "R0"
        assert result["signal"] == "R2→R0"

    def test_date_passthrough(self):
        """Date field from state is preserved."""
        result = _map_regime_to_flex({
            "regime": "R0",
            "regime_changed": False,
            "date": "2026-03-02T16:00:00+00:00",
        })
        assert result["date"] == "2026-03-02T16:00:00+00:00"

    def test_missing_date_defaults_empty(self):
        """Missing date field defaults to empty string."""
        result = _map_regime_to_flex({"regime": "R0"})
        assert result["date"] == ""

    def test_missing_previous_regime_on_change(self):
        """Missing previous_regime with regime_changed=True produces arrow from empty."""
        result = _map_regime_to_flex({
            "regime": "R0",
            "regime_changed": True,
            "previous_regime": None,
        })
        assert result["signal"] == "→R0"
