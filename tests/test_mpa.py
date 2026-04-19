"""Tests for MPA (Maximum Power Available) calculations."""

import numpy as np
import pytest

from modules.calculations.mpa import (
    MPAProfile,
    calculate_mpa,
    calculate_time_to_exhaustion,
)


@pytest.fixture
def rest_data():
    return np.full(300, 100.0), np.arange(300, dtype=float)


@pytest.fixture
def depleting_data():
    return np.full(300, 350.0), np.arange(300, dtype=float)


@pytest.fixture
def recovering_data():
    watts = np.concatenate([np.full(100, 350.0), np.full(200, 150.0)])
    return watts, np.arange(300, dtype=float)


CP = 280.0
W_PRIME = 20000.0


class TestCalculateTimeToExhaustion:
    def test_above_cp_finite(self):
        tte = calculate_time_to_exhaustion(W_PRIME, 350.0, CP)
        assert tte > 0
        assert np.isfinite(tte)
        expected = W_PRIME / (350.0 - CP)
        assert abs(tte - expected) < 1.0

    def test_below_cp_infinite(self):
        tte = calculate_time_to_exhaustion(W_PRIME, 200.0, CP)
        assert tte == float("inf")

    def test_at_cp_infinite(self):
        tte = calculate_time_to_exhaustion(W_PRIME, CP, CP)
        assert tte == float("inf")

    def test_zero_wbal(self):
        tte = calculate_time_to_exhaustion(0.0, 350.0, CP)
        assert tte == 0.0


class TestCalculateMPA:
    def test_at_rest_high_mpa(self, rest_data):
        watts, time = rest_data
        result = calculate_mpa(watts, time, CP, W_PRIME)
        assert isinstance(result, MPAProfile)
        assert result.mpa_array[0] > CP

    def test_depleting_mpa_drops(self, depleting_data):
        watts, time = depleting_data
        result = calculate_mpa(watts, time, CP, W_PRIME)
        assert result.mpa_array[-1] < result.mpa_array[0]

    def test_recovering_mpa_rises(self, recovering_data):
        watts, time = recovering_data
        result = calculate_mpa(watts, time, CP, W_PRIME)
        end_mpa = result.mpa_array[-1]
        mid_mpa = result.mpa_array[150]
        assert end_mpa > mid_mpa

    def test_wbal_depletes_above_cp(self, depleting_data):
        watts, time = depleting_data
        result = calculate_mpa(watts, time, CP, W_PRIME)
        assert result.wbal_array[-1] < W_PRIME

    def test_profile_metadata(self, depleting_data):
        watts, time = depleting_data
        result = calculate_mpa(watts, time, CP, W_PRIME)
        assert result.cp == CP
        assert result.w_prime == W_PRIME
        assert len(result.mpa_array) == len(watts)

    def test_tte_at_peak(self, depleting_data):
        watts, time = depleting_data
        result = calculate_mpa(watts, time, CP, W_PRIME)
        assert result.time_to_exhaustion_at_peak is not None
        assert result.time_to_exhaustion_at_peak > 0

    def test_different_sports(self, rest_data):
        watts, time = rest_data
        cycling = calculate_mpa(watts, time, CP, W_PRIME, sport=0)
        running = calculate_mpa(watts, time, CP, W_PRIME, sport=1)
        assert cycling is not None
        assert running is not None
