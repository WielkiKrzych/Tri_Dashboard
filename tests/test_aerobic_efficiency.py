"""Tests for aerobic efficiency calculations."""

import numpy as np
import pytest

from modules.calculations.aerobic_efficiency import (
    SessionEfficiency,
    EfficiencyTrend,
    calculate_session_efficiency,
    calculate_efficiency_trend,
    interpret_trend,
)


CP = 280.0


def _make_session_data(n: int = 600, power: float = 200.0, hr: float = 140.0):
    np.random.seed(42)
    power_arr = np.random.normal(power, 15, n).clip(50, 400)
    hr_arr = np.random.normal(hr, 8, n).clip(60, 200)
    time_arr = np.arange(n, dtype=float)
    return power_arr, hr_arr, time_arr


def _make_session_eff(ef: float, zone_ef: float = 0.0) -> SessionEfficiency:
    return SessionEfficiency(
        date="2026-01-01",
        overall_ef=ef,
        zone1_ef=zone_ef,
        zone2_ef=zone_ef,
        zone3_ef=zone_ef,
        zone4_ef=zone_ef,
        zone5_ef=zone_ef,
        ef_start=ef,
        ef_end=ef,
        ef_delta_pct=0.0,
    )


class TestCalculateSessionEfficiency:
    def test_known_values(self):
        power, hr, time = _make_session_data(power=200, hr=140)
        result = calculate_session_efficiency(power, hr, time, CP)
        expected_ef = 200.0 / 140.0
        assert abs(result.overall_ef - expected_ef) < 0.3

    def test_zero_hr_no_crash(self):
        n = 600
        power = np.full(n, 200.0)
        hr = np.full(n, 40.0)
        time = np.arange(n, dtype=float)
        result = calculate_session_efficiency(power, hr, time, CP)
        assert result is not None

    def test_all_zones_populated(self):
        power, hr, time = _make_session_data()
        result = calculate_session_efficiency(power, hr, time, CP)
        assert isinstance(result.zone1_ef, float)
        assert isinstance(result.zone2_ef, float)

    def test_ef_delta_pct(self):
        n = 600
        power = np.full(n, 200.0)
        hr = np.concatenate([np.full(150, 130.0), np.full(450, 150.0)])
        time = np.arange(n, dtype=float)
        result = calculate_session_efficiency(power, hr, time, CP)
        assert result.ef_delta_pct < 0

    def test_too_few_valid_points(self):
        n = 5
        power = np.array([200, 210, 190, 0, 0], dtype=float)
        hr = np.array([140, 142, 138, 40, 30], dtype=float)
        time = np.arange(n, dtype=float)
        result = calculate_session_efficiency(power, hr, time, CP)
        assert result.overall_ef == 0.0


class TestCalculateEfficiencyTrend:
    def test_improving_trend(self):
        sessions = [_make_session_eff(1.2 + i * 0.1) for i in range(6)]
        trend = calculate_efficiency_trend(sessions)
        assert trend.direction == "improving"
        assert trend.overall_trend_slope > 0

    def test_declining_trend(self):
        sessions = [_make_session_eff(2.0 - i * 0.1) for i in range(6)]
        trend = calculate_efficiency_trend(sessions)
        assert trend.direction == "declining"

    def test_insufficient_data(self):
        sessions = [_make_session_eff(1.5)]
        trend = calculate_efficiency_trend(sessions)
        assert trend.direction == "insufficient_data"

    def test_zone_trends_populated(self):
        sessions = [_make_session_eff(1.5, zone_ef=0.8 + i * 0.05) for i in range(4)]
        trend = calculate_efficiency_trend(sessions)
        assert isinstance(trend.zone_trends, dict)


class TestInterpretTrend:
    def test_improving(self):
        assert "Poprawa" in interpret_trend("improving")

    def test_declining(self):
        assert "Spadek" in interpret_trend("declining")

    def test_stable(self):
        assert "Stabilny" in interpret_trend("stable")

    def test_unknown(self):
        assert "Nieznany" in interpret_trend("something_else")
