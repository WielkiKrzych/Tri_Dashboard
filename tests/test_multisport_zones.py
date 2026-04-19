"""Tests for multisport zone calculations."""

import numpy as np
import pytest

from modules.calculations.multisport_zones import (
    SportZones,
    calculate_cycling_zones,
    calculate_running_zones,
    calculate_swim_zones,
    estimate_critical_pace,
    pace_to_str,
)


class TestCalculateCyclingZones:
    def test_seven_zones(self):
        result = calculate_cycling_zones(250)
        assert len(result.zones) == 7

    def test_zone_boundaries_ftp_250(self):
        result = calculate_cycling_zones(250)
        z4_lo = result.zones[3][1]
        z4_hi = result.zones[3][2]
        assert abs(z4_lo - 225.0) < 1
        assert abs(z4_hi - 262.5) < 1

    def test_sport_name(self):
        result = calculate_cycling_zones(250)
        assert result.sport == "Kolarstwo"

    def test_threshold_value(self):
        result = calculate_cycling_zones(250)
        assert result.threshold_value == 250


class TestCalculateRunningZones:
    def test_six_zones(self):
        result = calculate_running_zones(270)
        assert len(result.zones) == 6

    def test_sport_name(self):
        result = calculate_running_zones(270)
        assert result.sport == "Bieg"

    def test_zone_boundaries_reasonable(self):
        result = calculate_running_zones(270)
        for name, lo, hi in result.zones:
            assert lo >= 0
            assert hi >= 0


class TestCalculateSwimZones:
    def test_six_zones(self):
        result = calculate_swim_zones(100)
        assert len(result.zones) == 6

    def test_sport_name(self):
        result = calculate_swim_zones(100)
        assert result.sport == "Pływanie"


class TestEstimateCriticalPace:
    def test_known_distances(self):
        distances = [1000.0, 2000.0, 5000.0]
        times = [210.0, 470.0, 1300.0]
        cp, dp = estimate_critical_pace(distances, times)
        assert cp > 0
        assert dp > 0

    def test_insufficient_data(self):
        cp, dp = estimate_critical_pace([1000.0], [200.0])
        assert cp == 0.0
        assert dp == 0.0

    def test_negative_distance(self):
        cp, dp = estimate_critical_pace([-1000.0, 3000.0], [200.0, 660.0])
        assert cp == 0.0


class TestPaceToStr:
    def test_normal_pace(self):
        result = pace_to_str(270)
        assert "4:" in result
        assert "30" in result

    def test_zero_pace(self):
        assert pace_to_str(0) == "—"

    def test_negative_pace(self):
        assert pace_to_str(-1) == "—"

    def test_fast_pace(self):
        result = pace_to_str(210)
        assert "3:" in result
