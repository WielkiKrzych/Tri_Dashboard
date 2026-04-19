"""Tests for training impact decomposition."""

import numpy as np
import pandas as pd
import pytest

from modules.calculations.training_impact import (
    TrainingImpact,
    calculate_session_impact,
    classify_session_intensity,
    calculate_rolling_impact,
)


@pytest.fixture
def below_cp_df():
    np.random.seed(42)
    n = 600
    return pd.DataFrame(
        {
            "watts": np.random.normal(150, 20, n).clip(50, 250),
            "time": np.arange(n, dtype=float),
        }
    )


@pytest.fixture
def above_cp_df():
    np.random.seed(42)
    n = 600
    return pd.DataFrame(
        {
            "watts": np.random.normal(350, 30, n).clip(200, 450),
            "time": np.arange(n, dtype=float),
        }
    )


@pytest.fixture
def mixed_df():
    np.random.seed(42)
    n = 600
    watts = np.concatenate(
        [
            np.random.normal(150, 10, 300),
            np.random.normal(350, 20, 300),
        ]
    )
    return pd.DataFrame(
        {
            "watts": watts.clip(50, 450),
            "time": np.arange(n, dtype=float),
        }
    )


class TestClassifySessionIntensity:
    def test_recovery(self):
        assert classify_session_intensity(0.95) == "recovery"

    def test_endurance(self):
        assert classify_session_intensity(0.82) == "endurance"

    def test_tempo(self):
        assert classify_session_intensity(0.68) == "tempo"

    def test_threshold(self):
        assert classify_session_intensity(0.50) == "threshold"

    def test_vo2max(self):
        assert classify_session_intensity(0.35) == "vo2max"

    def test_anaerobic(self):
        assert classify_session_intensity(0.10) == "anaerobic"


class TestCalculateSessionImpact:
    def test_below_cp_mostly_aerobic(self, below_cp_df):
        result = calculate_session_impact(below_cp_df, cp=280, w_prime=20000)
        assert result is not None
        assert result.aerobic_fraction > 0.8

    def test_above_cp_significant_anaerobic(self, above_cp_df):
        result = calculate_session_impact(above_cp_df, cp=280, w_prime=20000)
        assert result is not None
        assert result.aerobic_fraction < 0.7

    def test_total_tss_positive(self, mixed_df):
        result = calculate_session_impact(mixed_df, cp=280, w_prime=20000)
        assert result is not None
        assert result.total_tss > 0

    def test_empty_df_returns_none(self):
        result = calculate_session_impact(pd.DataFrame(), cp=280, w_prime=20000)
        assert result is None

    def test_none_df_returns_none(self):
        result = calculate_session_impact(None, cp=280, w_prime=20000)
        assert result is None

    def test_missing_power_col_returns_none(self):
        df = pd.DataFrame({"time": range(100)})
        result = calculate_session_impact(df, cp=280, w_prime=20000)
        assert result is None

    def test_zero_cp_returns_none(self, below_cp_df):
        result = calculate_session_impact(below_cp_df, cp=0, w_prime=20000)
        assert result is None

    def test_too_few_samples_returns_none(self):
        df = pd.DataFrame({"watts": [200, 210], "time": [0, 1]})
        result = calculate_session_impact(df, cp=280, w_prime=20000)
        assert result is None


class TestCalculateRollingImpact:
    def test_empty_impacts(self):
        result = calculate_rolling_impact([])
        assert result["total_tss"] == 0.0

    def test_single_impact(self):
        impacts = [TrainingImpact(50, 10, 60, 0.833, "endurance")]
        result = calculate_rolling_impact(impacts)
        assert result["total_tss"] == 60

    def test_multiple_impacts(self):
        impacts = [
            TrainingImpact(50, 10, 60, 0.833, "endurance"),
            TrainingImpact(30, 30, 60, 0.5, "threshold"),
        ]
        result = calculate_rolling_impact(impacts)
        assert result["total_tss"] == 120
        assert 0 < result["aerobic_fraction"] < 1
