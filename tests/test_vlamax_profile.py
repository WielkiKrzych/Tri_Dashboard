"""Tests for VLamax profile calculations."""

import numpy as np
import pytest

from modules.calculations.vlamax_profile import (
    VLamaxProfile,
    build_vlamax_profile,
    get_metabolic_profile_chart_data,
    compare_vlamax_longitudinal,
)


def _sprinter_pdc(weight: float = 75.0) -> dict:
    return {
        5: 16.0 * weight,
        15: 12.0 * weight,
        30: 10.0 * weight,
        60: 8.0 * weight,
        120: 6.0 * weight,
        240: 4.7 * weight,
        300: 4.2 * weight,
        480: 3.7 * weight,
        600: 3.3 * weight,
        960: 3.1 * weight,
        1920: 2.5 * weight,
    }


def _climber_pdc(weight: float = 65.0) -> dict:
    return {
        5: 8.0 * weight,
        15: 7.7 * weight,
        30: 6.6 * weight,
        60: 5.7 * weight,
        120: 4.9 * weight,
        240: 4.3 * weight,
        300: 4.0 * weight,
        480: 3.8 * weight,
        600: 3.6 * weight,
        960: 3.3 * weight,
        1920: 2.9 * weight,
    }


@pytest.fixture
def sprinter_data():
    return _sprinter_pdc(), 75.0


@pytest.fixture
def climber_data():
    return _climber_pdc(), 65.0


class TestBuildVLamaxProfile:
    def test_sprinter_high_vlamax(self, sprinter_data):
        pdc, weight = sprinter_data
        profile = build_vlamax_profile(pdc, weight)
        assert profile is not None
        assert profile.vlamax > 0.3

    def test_climber_low_vlamax(self, climber_data):
        pdc, weight = climber_data
        profile = build_vlamax_profile(pdc, weight)
        assert profile is not None
        assert profile.vlamax < 0.8

    def test_empty_pdc_returns_none(self):
        assert build_vlamax_profile({}, 75.0) is None

    def test_zero_weight_returns_none(self):
        pdc = _sprinter_pdc()
        assert build_vlamax_profile(pdc, 0.0) is None

    def test_negative_weight_returns_none(self):
        pdc = _sprinter_pdc()
        assert build_vlamax_profile(pdc, -10.0) is None

    def test_profile_has_rider_type(self, sprinter_data):
        pdc, weight = sprinter_data
        profile = build_vlamax_profile(pdc, weight)
        assert profile.rider_type in {"Sprinter", "Puncheur", "All-rounder", "Climber"}

    def test_confidence_range(self, sprinter_data):
        pdc, weight = sprinter_data
        profile = build_vlamax_profile(pdc, weight)
        assert 0.0 <= profile.confidence <= 1.0

    def test_with_vo2max(self, sprinter_data):
        pdc, weight = sprinter_data
        profile = build_vlamax_profile(pdc, weight)
        assert profile is not None


class TestGetMetabolicProfileChartData:
    def test_with_aerobic_data(self, sprinter_data):
        pdc, weight = sprinter_data
        profile = build_vlamax_profile(pdc, weight)
        data = get_metabolic_profile_chart_data(profile)
        assert "durations" in data
        assert "aerobic_pct" in data
        assert "anaerobic_pct" in data

    def test_empty_profile(self):
        profile = VLamaxProfile(vlamax=0.5, confidence=0.8, rider_type="All-rounder")
        data = get_metabolic_profile_chart_data(profile)
        assert data["durations"] == []


class TestCompareVLamaxLongitudinal:
    def test_single_profile(self, sprinter_data):
        pdc, weight = sprinter_data
        p = build_vlamax_profile(pdc, weight)
        result = compare_vlamax_longitudinal([p])
        assert "vlamax_values" in result

    def test_multiple_profiles(self, sprinter_data):
        pdc, weight = sprinter_data
        profiles = [build_vlamax_profile(pdc, weight) for _ in range(3)]
        result = compare_vlamax_longitudinal(profiles)
        assert len(result["vlamax_values"]) == 3

    def test_empty_list(self):
        result = compare_vlamax_longitudinal([])
        assert result["vlamax_values"] == []
