"""Tests for HRV readiness score calculations."""

import numpy as np
import pytest

from modules.calculations.hrv_readiness import (
    ReadinessScore,
    calculate_readiness_score,
    get_readiness_level,
    get_readiness_recommendation,
)


@pytest.fixture
def stable_hrv_history():
    return [
        {"date": f"2026-01-{d:02d}", "rmssd": 55.0 + np.random.normal(0, 2)} for d in range(1, 8)
    ]


@pytest.fixture
def unstable_hrv_history():
    return [
        {"date": f"2026-01-{d:02d}", "rmssd": 30.0 + np.random.uniform(-25, 25)}
        for d in range(1, 8)
    ]


class TestGetReadinessLevel:
    def test_green(self):
        name, color = get_readiness_level(85)
        assert "Wysoka" in name

    def test_moderate(self):
        name, color = get_readiness_level(65)
        assert "umiarkowana" in name

    def test_lowered(self):
        name, color = get_readiness_level(45)
        assert "Obniżona" in name

    def test_low(self):
        name, color = get_readiness_level(25)
        assert "Niska" in name

    def test_critical(self):
        name, color = get_readiness_level(10)
        assert "Krytycznie" in name


class TestGetReadinessRecommendation:
    def test_high_readiness_rec(self):
        rec = get_readiness_recommendation("Wysoka gotowość", 85)
        assert "gotowy" in rec.lower() or "intensywny" in rec.lower()

    def test_critical_readiness_rec(self):
        rec = get_readiness_recommendation("Krytycznie niska", 10)
        assert "odpoczynek" in rec.lower() or "całkowity" in rec.lower()


class TestCalculateReadinessScore:
    def test_stable_hrv_high_score(self, stable_hrv_history):
        result = calculate_readiness_score(stable_hrv_history)
        assert result is not None
        assert result.score >= 70

    def test_unstable_hrv_low_score(self, unstable_hrv_history):
        result = calculate_readiness_score(unstable_hrv_history)
        assert result is not None
        assert result.score < 70

    def test_too_few_entries_returns_none(self):
        history = [{"date": "2026-01-01", "rmssd": 50.0}]
        assert calculate_readiness_score(history) is None

    def test_empty_history_returns_none(self):
        assert calculate_readiness_score([]) is None

    def test_none_history_returns_none(self):
        assert calculate_readiness_score(None) is None

    def test_result_has_all_fields(self, stable_hrv_history):
        result = calculate_readiness_score(stable_hrv_history)
        assert result.date
        assert 0 <= result.score <= 100
        assert result.level
        assert result.color
        assert result.rmssd_7day_avg > 0
        assert result.rmssd_cv >= 0
        assert result.recommendation
