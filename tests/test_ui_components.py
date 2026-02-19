"""
Tests for pure (non-Streamlit) functions in UI modules.

Only functions that do NOT call Streamlit directly are tested here.
Streamlit rendering functions (render_*, show_*) require the Streamlit
test runner and are out of scope for standard pytest.
"""
import pandas as pd
import numpy as np
import pytest

from modules.ui.header import extract_header_data


class TestExtractHeaderData:
    """Tests for header.extract_header_data() — a pure data-extraction function."""

    def test_returns_all_expected_keys(self, sample_power_df):
        """Result must contain all 6 keys expected by render_sticky_header."""
        metrics = {
            "avg_watts": 200.0,
            "avg_hr": 140.0,
            "avg_cadence": 90.0,
            "avg_vent": 55.0,
        }
        result = extract_header_data(sample_power_df, metrics)

        expected_keys = {"avg_power", "avg_hr", "avg_smo2", "avg_cadence", "avg_ve", "duration_min"}
        assert set(result.keys()) == expected_keys

    def test_with_smo2_column_returns_nonzero(self, sample_power_df):
        """When df has 'smo2' column, avg_smo2 should reflect actual data mean."""
        metrics = {}
        result = extract_header_data(sample_power_df, metrics)

        assert result["avg_smo2"] > 0
        assert abs(result["avg_smo2"] - sample_power_df["smo2"].mean()) < 0.01

    def test_without_smo2_column_returns_zero(self, sample_power_df):
        """When df lacks 'smo2', avg_smo2 must be 0.0 (no KeyError)."""
        df_no_smo2 = sample_power_df.drop(columns=["smo2"])
        metrics = {}
        result = extract_header_data(df_no_smo2, metrics)

        assert result["avg_smo2"] == 0

    def test_duration_is_rows_divided_by_60(self, sample_power_df):
        """600 rows of 1-Hz data → 10.0 minutes."""
        metrics = {}
        result = extract_header_data(sample_power_df, metrics)

        assert len(sample_power_df) == 600
        assert result["duration_min"] == pytest.approx(10.0)

    def test_metrics_values_passed_through(self, sample_power_df):
        """avg_power, avg_hr, avg_cadence, avg_ve come directly from the metrics dict."""
        metrics = {
            "avg_watts": 250.0,
            "avg_hr": 155.0,
            "avg_cadence": 95.0,
            "avg_vent": 62.5,
        }
        result = extract_header_data(sample_power_df, metrics)

        assert result["avg_power"] == 250.0
        assert result["avg_hr"] == 155.0
        assert result["avg_cadence"] == 95.0
        assert result["avg_ve"] == 62.5

    def test_empty_metrics_dict_returns_zeros(self, sample_power_df):
        """Missing keys in metrics dict must default to 0 (no KeyError)."""
        result = extract_header_data(sample_power_df, {})

        assert result["avg_power"] == 0
        assert result["avg_hr"] == 0
        assert result["avg_cadence"] == 0
        assert result["avg_ve"] == 0

    def test_zero_row_dataframe_returns_zero_duration(self, empty_df):
        """Empty DataFrame → duration_min == 0 (no ZeroDivisionError)."""
        result = extract_header_data(empty_df, {})

        assert result["duration_min"] == 0
        assert result["avg_smo2"] == 0
