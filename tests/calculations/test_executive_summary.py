"""Tests for executive_summary module."""
import pytest
import numpy as np

from modules.calculations.executive_summary import (
    SignalStatus,
    ConfidenceBreakdown,
    TrainingCard,
    LIMITER_TYPES,
)


class TestSignalStatus:
    """Tests for SignalStatus dataclass."""

    def test_create_signal_status(self):
        """Test creating SignalStatus."""
        status = SignalStatus(
            name="Heart Rate",
            status="valid",
            icon="❤️",
            note="Good signal quality",
        )

        assert status.name == "Heart Rate"
        assert status.status == "valid"
        assert status.icon == "❤️"
        assert status.note == "Good signal quality"

    def test_signal_status_with_warning(self):
        """Test SignalStatus with warning."""
        status = SignalStatus(
            name="SmO2",
            status="warning",
            icon="📊",
            note="Signal drop detected",
        )

        assert status.status == "warning"


class TestConfidenceBreakdown:
    """Tests for ConfidenceBreakdown dataclass."""

    def test_create_confidence_breakdown(self):
        """Test creating ConfidenceBreakdown."""
        breakdown = ConfidenceBreakdown(
            ve_stability=0.85,
            hr_lag=0.9,
            smo2_noise=0.7,
            protocol_quality=0.8,
            limiting_factor="central",
        )

        assert breakdown.ve_stability == 0.85
        assert breakdown.hr_lag == 0.9
        assert breakdown.limiting_factor == "central"

    def test_confidence_values_in_range(self):
        """Test that confidence values are in valid range."""
        breakdown = ConfidenceBreakdown(
            ve_stability=0.5,
            hr_lag=0.6,
            smo2_noise=0.4,
            protocol_quality=0.7,
            limiting_factor="peripheral",
        )

        assert 0.0 <= breakdown.ve_stability <= 1.0
        assert 0.0 <= breakdown.hr_lag <= 1.0

    def test_limiting_factor_options(self):
        """Test that limiting factor uses valid options."""
        # Test central limiter
        central = ConfidenceBreakdown(
            ve_stability=0.8,
            hr_lag=0.9,
            smo2_noise=0.7,
            protocol_quality=0.8,
            limiting_factor="central",
        )
        assert central.limiting_factor == "central"

        # Test peripheral limiter
        peripheral = ConfidenceBreakdown(
            ve_stability=0.8,
            hr_lag=0.9,
            smo2_noise=0.7,
            protocol_quality=0.8,
            limiting_factor="peripheral",
        )
        assert peripheral.limiting_factor == "peripheral"


class TestTrainingCard:
    """Tests for TrainingCard dataclass."""

    def test_create_training_card(self):
        """Test creating TrainingCard."""
        card = TrainingCard(
            strategy_name="Sweet Spot",
            power_range="88-94% FTP",
            volume="2x20min",
            adaptation_goal="Increase threshold power",
            expected_response="Improved lactate clearance",
            risk_level="low",
        )

        assert card.strategy_name == "Sweet Spot"
        assert card.power_range == "88-94% FTP"
        assert card.risk_level == "low"

    def test_training_card_risk_levels(self):
        """Test different risk levels."""
        low_risk = TrainingCard(
            strategy_name="Endurance",
            power_range="55-75% FTP",
            volume="60-90min",
            adaptation_goal="Aerobic base",
            expected_response="Fatigue resistance",
            risk_level="low",
        )
        assert low_risk.risk_level == "low"

        high_risk = TrainingCard(
            strategy_name="VO2max Intervals",
            power_range="105-120% FTP",
            volume="5x5min",
            adaptation_goal="Maximal aerobic power",
            expected_response="Increased VO2max",
            risk_level="high",
        )
        assert high_risk.risk_level == "high"


class TestLimiterTypes:
    """Tests for LIMITER_TYPES constant."""

    def test_limiter_types_structure(self):
        """Test LIMITER_TYPES has expected structure."""
        assert "central" in LIMITER_TYPES
        assert "peripheral" in LIMITER_TYPES

    def test_central_limiter_content(self):
        """Test central limiter content."""
        central = LIMITER_TYPES["central"]
        assert "name" in central
        assert "subtitle" in central
        assert "icon" in central

    def test_peripheral_limiter_content(self):
        """Test peripheral limiter content."""
        peripheral = LIMITER_TYPES["peripheral"]
        assert "name" in peripheral
        assert "subtitle" in peripheral

    def test_limiter_types_have_required_keys(self):
        """Test that all limiters have required keys."""
        required_keys = ["name", "subtitle", "icon", "color"]

        for limiter_type, config in LIMITER_TYPES.items():
            for key in required_keys:
                assert key in config, f"Missing {key} in {limiter_type}"


class TestConfidenceIntegration:
    """Integration tests for confidence calculations."""

    def test_overall_confidence_from_breakdown(self):
        """Test calculating overall confidence from breakdown."""
        breakdown = ConfidenceBreakdown(
            ve_stability=0.9,
            hr_lag=0.85,
            smo2_noise=0.8,
            protocol_quality=0.9,
            limiting_factor="central",
        )

        # Calculate weighted average (hypothetical)
        weights = {
            "ve_stability": 0.25,
            "hr_lag": 0.25,
            "smo2_noise": 0.2,
            "protocol_quality": 0.3,
        }
        overall = (
            breakdown.ve_stability * weights["ve_stability"]
            + breakdown.hr_lag * weights["hr_lag"]
            + breakdown.smo2_noise * weights["smo2_noise"]
            + breakdown.protocol_quality * weights["protocol_quality"]
        )

        assert 0.0 <= overall <= 1.0
        assert overall > 0.8  # Should be high with these values

    def test_low_confidence_signals(self):
        """Test identifying low confidence signals."""
        breakdown = ConfidenceBreakdown(
            ve_stability=0.3,
            hr_lag=0.4,
            smo2_noise=0.2,
            protocol_quality=0.5,
            limiting_factor="central",
        )

        # Check for low confidence signals
        low_confidence = [
            attr for attr, val in [
                ("ve_stability", breakdown.ve_stability),
                ("hr_lag", breakdown.hr_lag),
                ("smo2_noise", breakdown.smo2_noise),
            ]
            if val < 0.5
        ]

        assert len(low_confidence) >= 2
