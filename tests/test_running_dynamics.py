"""Tests for running dynamics calculations."""

import numpy as np
import pytest

from modules.calculations.running_dynamics import (
    RunningDynamicsMetrics,
    calculate_running_dynamics,
    classify_running_economics,
    get_ideal_ranges,
)


def _make_accel_data(n: int = 500, cadence: float = 175.0):
    np.random.seed(42)
    t = np.arange(n, dtype=float) / 100.0
    freq = cadence / 60.0
    ay = np.sin(2 * np.pi * freq * t * 2) + np.random.normal(0, 0.1, n)
    ax = np.random.normal(0, 0.3, n)
    az = np.random.normal(0, 0.2, n)
    return ax, ay, az, t


@pytest.fixture
def good_running_data():
    return _make_accel_data(500, 180.0)


@pytest.fixture
def short_data():
    return _make_accel_data(20, 175.0)


class TestCalculateRunningDynamics:
    def test_synthetic_data(self, good_running_data):
        ax, ay, az, t = good_running_data
        result = calculate_running_dynamics(
            ax, ay, az, t, cadence=180.0, body_mass_kg=75.0, pace_min_km=5.0
        )
        assert result is not None
        assert result.ground_contact_time_ms > 0
        assert result.vertical_oscillation_mm > 0
        assert result.cadence_spm > 0

    def test_insufficient_data_returns_none(self, short_data):
        ax, ay, az, t = short_data
        result = calculate_running_dynamics(ax, ay, az, t)
        assert result is None

    def test_none_cadence_estimated(self, good_running_data):
        ax, ay, az, t = good_running_data
        result = calculate_running_dynamics(ax, ay, az, t, cadence=None)
        assert result is not None
        assert result.cadence_spm > 0

    def test_stride_length_with_pace(self, good_running_data):
        ax, ay, az, t = good_running_data
        result = calculate_running_dynamics(ax, ay, az, t, cadence=180.0, pace_min_km=5.0)
        if result is not None:
            assert result.stride_length_m > 0

    def test_vertical_ratio(self, good_running_data):
        ax, ay, az, t = good_running_data
        result = calculate_running_dynamics(ax, ay, az, t, cadence=180.0, pace_min_km=5.0)
        if result is not None and result.stride_length_m > 0:
            assert result.vertical_ratio_pct >= 0


class TestClassifyRunningEconomics:
    def test_good_economics(self):
        metrics = RunningDynamicsMetrics(
            ground_contact_time_ms=220,
            vertical_oscillation_mm=7.0,
            leg_spring_stiffness_kn_m=12.0,
            vertical_ratio_pct=6.0,
            cadence_spm=185,
            stride_length_m=1.2,
        )
        result = classify_running_economics(metrics)
        assert "Dobra" in result

    def test_poor_economics(self):
        metrics = RunningDynamicsMetrics(
            ground_contact_time_ms=320,
            vertical_oscillation_mm=12.0,
            leg_spring_stiffness_kn_m=4.0,
            vertical_ratio_pct=10.0,
            cadence_spm=160,
            stride_length_m=1.0,
        )
        result = classify_running_economics(metrics)
        assert "Słaba" in result


class TestGetIdealRanges:
    def test_fast_pace(self):
        ranges = get_ideal_ranges(3.5)
        assert "ground_contact_time_ms" in ranges
        assert isinstance(ranges["ground_contact_time_ms"], tuple)

    def test_moderate_pace(self):
        ranges = get_ideal_ranges(5.0)
        assert "vertical_oscillation_mm" in ranges

    def test_slow_pace(self):
        ranges = get_ideal_ranges(6.5)
        assert "cadence_spm" in ranges

    def test_very_fast_pace(self):
        ranges = get_ideal_ranges(3.0)
        assert ranges["ground_contact_time_ms"][0] < 250
