"""Tests for SmO2 longitudinal trend analysis."""

import numpy as np
import pytest

from modules.calculations.smo2_longitudinal import (
    SmO2SessionThreshold,
    SmO2LongitudinalTrend,
    calculate_longitudinal_trend,
    interpret_trend,
)


def _make_threshold(date: str, t1_power: float, grade: str = "A") -> SmO2SessionThreshold:
    return SmO2SessionThreshold(
        date=date,
        t1_power=t1_power,
        t1_smo2=60.0,
        t2_power=t1_power + 40,
        t2_smo2=40.0,
        quality_grade=grade,
    )


@pytest.fixture
def improving_thresholds():
    return [_make_threshold(f"2026-01-{d:02d}", 180 + i * 5) for i, d in enumerate(range(1, 9))]


@pytest.fixture
def declining_thresholds():
    return [_make_threshold(f"2026-01-{d:02d}", 220 - i * 5) for i, d in enumerate(range(1, 9))]


class TestCalculateLongitudinalTrend:
    def test_improving_trend(self, improving_thresholds):
        trend = calculate_longitudinal_trend(improving_thresholds)
        assert trend.direction == "improving"
        assert trend.power_trend_slope > 0

    def test_declining_trend(self, declining_thresholds):
        trend = calculate_longitudinal_trend(declining_thresholds)
        assert trend.direction == "declining"
        assert trend.power_trend_slope < 0

    def test_insufficient_data(self):
        single = [_make_threshold("2026-01-01", 200)]
        trend = calculate_longitudinal_trend(single)
        assert trend.direction == "stable"
        assert trend.power_trend_slope == 0.0

    def test_empty_thresholds(self):
        trend = calculate_longitudinal_trend([])
        assert trend.direction == "stable"
        assert len(trend.sessions) == 0

    def test_grade_filtering(self):
        thresholds = [
            _make_threshold("2026-01-01", 200, "A"),
            _make_threshold("2026-01-02", 210, "D"),
            _make_threshold("2026-01-03", 220, "A"),
        ]
        trend = calculate_longitudinal_trend(thresholds, min_grade="A")
        assert trend.power_trend_slope > 0

    def test_trend_includes_sessions(self, improving_thresholds):
        trend = calculate_longitudinal_trend(improving_thresholds)
        assert len(trend.sessions) == len(improving_thresholds)

    def test_r_squared_range(self, improving_thresholds):
        trend = calculate_longitudinal_trend(improving_thresholds)
        assert 0.0 <= trend.power_trend_r <= 1.0

    def test_confidence_range(self, improving_thresholds):
        trend = calculate_longitudinal_trend(improving_thresholds)
        assert 0.0 <= trend.confidence <= 1.0


class TestInterpretTrend:
    def test_improving_interpretation(self, improving_thresholds):
        trend = calculate_longitudinal_trend(improving_thresholds)
        text = interpret_trend(trend)
        assert "Poprawa" in text

    def test_declining_interpretation(self, declining_thresholds):
        trend = calculate_longitudinal_trend(declining_thresholds)
        text = interpret_trend(trend)
        assert "Spadek" in text

    def test_empty_sessions_interpretation(self):
        trend = SmO2LongitudinalTrend()
        text = interpret_trend(trend)
        assert "Brak danych" in text
