"""Tests for DFA alpha1 longitudinal threshold analysis."""

import numpy as np
import pytest

from modules.calculations.dfa_longitudinal import (
    DFAThresholdSession,
    DFALongitudinalTrend,
    extract_dfa_threshold,
    calculate_dfa_longitudinal_trend,
)


def _make_ramp_data(n: int = 200, cross_at: int = 100):
    time_arr = np.arange(n, dtype=float)
    alpha1 = np.ones(n) * 1.2
    alpha1[cross_at:] = 0.5
    power = np.linspace(100, 350, n)
    hr = np.linspace(100, 170, n)
    return time_arr, alpha1, power, hr


@pytest.fixture
def ramp_data_with_crossing():
    return _make_ramp_data(200, 100)


@pytest.fixture
def ramp_data_no_crossing():
    n = 200
    time_arr = np.arange(n, dtype=float)
    alpha1 = np.ones(n) * 1.2
    power = np.linspace(100, 350, n)
    hr = np.linspace(100, 170, n)
    return time_arr, alpha1, power, hr


def _make_session(date: str, power: float) -> DFAThresholdSession:
    return DFAThresholdSession(
        date=date,
        threshold_power=power,
        threshold_hr=150.0,
        alpha1_at_threshold=0.75,
        quality_grade="A",
    )


@pytest.fixture
def improving_sessions():
    return [_make_session(f"2026-01-{d:02d}", 200 + i * 3) for i, d in enumerate(range(1, 9))]


@pytest.fixture
def declining_sessions():
    return [_make_session(f"2026-01-{d:02d}", 230 - i * 3) for i, d in enumerate(range(1, 9))]


class TestExtractDFAThreshold:
    def test_clear_crossing(self, ramp_data_with_crossing):
        time_arr, alpha1, power, hr = ramp_data_with_crossing
        result = extract_dfa_threshold(time_arr, alpha1, power, hr, target_alpha1=0.75)
        assert result is not None
        threshold_power, threshold_hr = result
        assert threshold_power > 0
        assert threshold_hr > 0

    def test_no_crossing_returns_none(self, ramp_data_no_crossing):
        time_arr, alpha1, power, hr = ramp_data_no_crossing
        result = extract_dfa_threshold(time_arr, alpha1, power, hr, target_alpha1=0.75)
        assert result is None

    def test_empty_arrays(self):
        result = extract_dfa_threshold(np.array([]), np.array([]), np.array([]), np.array([]))
        assert result is None

    def test_short_arrays(self):
        result = extract_dfa_threshold(
            np.array([0, 1]), np.array([1.0, 0.5]), np.array([100, 200]), np.array([100, 150])
        )
        assert result is None


class TestCalculateDFALongitudinalTrend:
    def test_improving_trend(self, improving_sessions):
        trend = calculate_dfa_longitudinal_trend(improving_sessions)
        assert trend.direction == "improving"
        assert trend.power_trend_slope > 0

    def test_declining_trend(self, declining_sessions):
        trend = calculate_dfa_longitudinal_trend(declining_sessions)
        assert trend.direction == "declining"
        assert trend.power_trend_slope < 0

    def test_insufficient_data(self):
        single = [_make_session("2026-01-01", 200)]
        trend = calculate_dfa_longitudinal_trend(single)
        assert trend.direction == "insufficient_data"

    def test_confidence_range(self, improving_sessions):
        trend = calculate_dfa_longitudinal_trend(improving_sessions)
        assert 0.0 <= trend.confidence <= 1.0

    def test_sessions_preserved(self, improving_sessions):
        trend = calculate_dfa_longitudinal_trend(improving_sessions)
        assert len(trend.sessions) == len(improving_sessions)
