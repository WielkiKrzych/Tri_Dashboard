"""Tests for sleep recovery calculations."""

import numpy as np
import pytest

from modules.calculations.sleep_recovery import (
    SleepData,
    SleepScore,
    CompositeRecovery,
    calculate_sleep_score,
    calculate_composite_recovery,
)


@pytest.fixture
def optimal_sleep():
    return SleepData(
        date="2026-01-15",
        total_sleep_hours=8.0,
        deep_sleep_hours=1.8,
        rem_sleep_hours=2.0,
        sleep_efficiency_pct=92.0,
        hr_during_sleep=55.0,
        hrv_during_sleep=65.0,
        wake_count=0,
    )


@pytest.fixture
def poor_sleep():
    return SleepData(
        date="2026-01-15",
        total_sleep_hours=4.5,
        deep_sleep_hours=0.3,
        rem_sleep_hours=0.5,
        sleep_efficiency_pct=60.0,
        hr_during_sleep=75.0,
        hrv_during_sleep=25.0,
        wake_count=5,
    )


class TestCalculateSleepScore:
    def test_optimal_sleep_high_score(self, optimal_sleep):
        result = calculate_sleep_score(optimal_sleep)
        assert result.score >= 75
        assert result.level

    def test_poor_sleep_low_score(self, poor_sleep):
        result = calculate_sleep_score(poor_sleep)
        assert result.score < 50

    def test_score_range_0_100(self, optimal_sleep):
        result = calculate_sleep_score(optimal_sleep)
        assert 0 <= result.score <= 100

    def test_has_date(self, optimal_sleep):
        result = calculate_sleep_score(optimal_sleep)
        assert result.date == "2026-01-15"

    def test_has_interpretation(self, optimal_sleep):
        result = calculate_sleep_score(optimal_sleep)
        assert result.interpretation

    def test_zero_sleep(self):
        data = SleepData(
            date="2026-01-15",
            total_sleep_hours=0,
            deep_sleep_hours=0,
            rem_sleep_hours=0,
            sleep_efficiency_pct=0,
            hr_during_sleep=None,
            hrv_during_sleep=None,
            wake_count=10,
        )
        result = calculate_sleep_score(data)
        assert result.score < 30

    def test_excessive_sleep(self):
        data = SleepData(
            date="2026-01-15",
            total_sleep_hours=12.0,
            deep_sleep_hours=2.0,
            rem_sleep_hours=3.0,
            sleep_efficiency_pct=85.0,
            hr_during_sleep=55.0,
            hrv_during_sleep=None,
            wake_count=0,
        )
        result = calculate_sleep_score(data)
        assert result.score < 85


class TestCalculateCompositeRecovery:
    def test_well_rested(self):
        result = calculate_composite_recovery(
            sleep_score=90.0,
            hrv_readiness_score=85.0,
            current_tsb=20.0,
        )
        assert result.composite_score >= 60
        assert result.recommendation

    def test_fatigued(self):
        result = calculate_composite_recovery(
            sleep_score=30.0,
            hrv_readiness_score=25.0,
            current_tsb=-25.0,
        )
        assert result.composite_score < 40

    def test_composite_fields(self):
        result = calculate_composite_recovery(
            sleep_score=75.0,
            hrv_readiness_score=70.0,
            current_tsb=5.0,
        )
        assert result.date
        assert result.sleep_score == 75.0
        assert result.hrv_readiness == 70.0
        assert result.training_load_status
        assert result.recommendation

    def test_load_status_fresh(self):
        result = calculate_composite_recovery(80, 80, 30)
        assert "Świeży" in result.training_load_status

    def test_load_status_tired(self):
        result = calculate_composite_recovery(80, 80, -20)
        assert "Zmęczony" in result.training_load_status

    def test_load_status_optimal(self):
        result = calculate_composite_recovery(80, 80, -5)
        assert "Optymalne" in result.training_load_status
