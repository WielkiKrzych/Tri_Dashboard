"""Tests for the ramp test pipeline orchestration (run_ramp_test_pipeline)."""

import pandas as pd
import numpy as np
import pytest

from modules.calculations.pipeline import run_ramp_test_pipeline


class TestRunRampTestPipeline:
    """Tests for the full ramp test pipeline orchestration."""

    @pytest.fixture
    def valid_ramp_df(self):
        """Create a synthetic ramp test DataFrame with realistic data."""
        n = 300
        time = np.arange(n, dtype=float)
        watts = np.linspace(50, 350, n)
        hr = np.linspace(80, 185, n) + np.random.normal(0, 2, n)
        ve = np.linspace(20, 120, n) + np.random.normal(0, 3, n)
        return pd.DataFrame(
            {
                "time": time,
                "watts": watts,
                "hr": hr,
                "tymeventilation": ve,
            }
        )

    def test_returns_ramp_test_result(self, valid_ramp_df):
        """Pipeline should return a RampTestResult object."""
        result = run_ramp_test_pipeline(valid_ramp_df)
        assert result is not None
        assert hasattr(result, "validity")

    def test_invalid_empty_df(self):
        """Pipeline should handle empty DataFrame gracefully."""
        df = pd.DataFrame()
        result = run_ramp_test_pipeline(df)
        assert result is not None

    def test_with_smo2_column(self, valid_ramp_df):
        """Pipeline should handle optional SmO2 column."""
        valid_ramp_df["smo2"] = np.linspace(80, 30, len(valid_ramp_df))
        result = run_ramp_test_pipeline(valid_ramp_df)
        assert result is not None

    def test_with_custom_columns(self, valid_ramp_df):
        """Pipeline should accept custom column names."""
        renamed = valid_ramp_df.rename(columns={"watts": "power", "hr": "heart_rate"})
        result = run_ramp_test_pipeline(
            renamed,
            power_column="power",
            hr_column="heart_rate",
        )
        assert result is not None

    def test_with_cp_watts(self, valid_ramp_df):
        """Pipeline should accept optional CP watts."""
        result = run_ramp_test_pipeline(valid_ramp_df, cp_watts=280.0)
        assert result is not None
