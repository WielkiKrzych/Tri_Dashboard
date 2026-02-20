"""Tests for modules/intervals.py — interval detection and classification."""

import pytest
import pandas as pd
import numpy as np
from modules.intervals import detect_intervals


@pytest.fixture
def interval_df():
    """Create DataFrame with one clear work interval above CP."""
    n = 300  # 5 minutes
    time = np.arange(n, dtype=float)
    watts = np.full(n, 150.0)
    # Interval: 60s-180s at 300W (above CP=250 * 0.9 = 225W threshold)
    watts[60:180] = 300.0
    hr = 120 + (watts - 150) * 0.2
    cadence = np.full(n, 90.0)
    return pd.DataFrame({
        "time": time,
        "watts": watts,
        "heartrate": hr,
        "cadence": cadence,
    })


class TestDetectIntervals:
    def test_no_watts_column(self):
        df = pd.DataFrame({"hr": [120, 130, 140], "time": [0, 1, 2]})
        result = detect_intervals(df, cp=250)
        assert len(result) == 0

    def test_all_below_threshold(self):
        df = pd.DataFrame({
            "watts": [100.0] * 100,
            "time": np.arange(100, dtype=float),
        })
        result = detect_intervals(df, cp=250, min_power_pct=0.9)
        assert len(result) == 0

    def test_single_interval_detected(self, interval_df):
        result = detect_intervals(interval_df, cp=250, min_duration=30, min_power_pct=0.9)
        assert len(result) == 1
        assert result["Avg Power"].iloc[0] >= 280
        assert result["Duration (s)"].iloc[0] >= 100

    def test_interval_has_correct_stats(self, interval_df):
        result = detect_intervals(interval_df, cp=250, min_duration=30, min_power_pct=0.9)
        assert "Avg HR" in result.columns
        assert "Avg Cadence" in result.columns

    def test_min_duration_filter(self):
        """Short burst below min_duration should be filtered out."""
        n = 200
        time = np.arange(n, dtype=float)
        watts = np.full(n, 100.0)
        watts[50:65] = 300.0  # Only 15s above threshold
        df = pd.DataFrame({"watts": watts, "time": time})
        result = detect_intervals(df, cp=250, min_duration=30, min_power_pct=0.9)
        assert len(result) == 0

    def test_merge_close_intervals(self):
        """Two intervals separated by short gap should merge."""
        n = 300
        time = np.arange(n, dtype=float)
        watts = np.full(n, 100.0)
        watts[50:100] = 300.0   # Block 1: 50s
        watts[110:160] = 300.0  # Block 2: 50s, gap = 10s
        df = pd.DataFrame({
            "watts": watts,
            "time": time,
            "heartrate": np.full(n, 140.0),
        })
        result = detect_intervals(df, cp=250, min_duration=30, min_power_pct=0.9, recovery_time_limit=20)
        # Should merge into 1 interval because gap (10s) < recovery_time_limit (20s)
        assert len(result) == 1

    def test_smo2_included_if_present(self):
        n = 200
        time = np.arange(n, dtype=float)
        watts = np.full(n, 100.0)
        watts[30:100] = 300.0
        df = pd.DataFrame({
            "watts": watts,
            "time": time,
            "smo2": np.full(n, 65.0),
        })
        result = detect_intervals(df, cp=250, min_duration=30, min_power_pct=0.9)
        if len(result) > 0:
            assert "Avg SmO2" in result.columns
