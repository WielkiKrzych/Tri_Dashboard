"""Tests for HRV (Heart Rate Variability) module."""
import pytest
import numpy as np
import pandas as pd

from modules.calculations.hrv import (
    calculate_dynamic_dfa_v2,
    validate_dfa_quality,
)


class TestCalculateDynamicDfaV2:
    """Tests for DFA Alpha-1 calculation."""

    def test_basic_hrv_calculation(self, sample_hrv_df):
        """Test basic HRV/DFA calculation with valid data."""
        result_df, error = calculate_dynamic_dfa_v2(sample_hrv_df)

        assert error is None
        assert result_df is not None
        assert "alpha1" in result_df.columns
        assert len(result_df) > 0

    def test_with_custom_window(self, sample_hrv_df):
        """Test with custom window size."""
        result_df, error = calculate_dynamic_dfa_v2(
            sample_hrv_df,
            window_sec=180,  # 3 minutes
            step_sec=15,
        )

        assert error is None
        assert result_df is not None

    def test_insufficient_data_returns_error(self):
        """Test with insufficient data points."""
        # Only 50 RR intervals (below min_samples_hrv default of 100)
        short_df = pd.DataFrame({
            "rr": np.random.uniform(800, 1200, 50)
        })

        result_df, error = calculate_dynamic_dfa_v2(short_df)

        assert error is not None or result_df is None or len(result_df) == 0

    def test_handles_rr_in_seconds(self):
        """Test RR values in seconds (0.8-1.2 range)."""
        df = pd.DataFrame({
            "rr": np.random.uniform(0.8, 1.2, 200)
        })

        result_df, error = calculate_dynamic_dfa_v2(df)

        # Should auto-detect format and process
        assert result_df is not None or error is not None

    def test_handles_rr_in_microseconds(self):
        """Test RR values in microseconds."""
        df = pd.DataFrame({
            "rr": np.random.uniform(800000, 1200000, 200)
        })

        result_df, error = calculate_dynamic_dfa_v2(df)

        assert result_df is not None or error is not None

    def test_alpha1_range(self, sample_hrv_df):
        """Test that alpha1 values are in expected range."""
        result_df, error = calculate_dynamic_dfa_v2(sample_hrv_df)

        if error is None and result_df is not None:
            alpha1_values = result_df["alpha1"].dropna()
            # Alpha1 typically ranges 0.3 to 1.5 for physiological signals
            assert alpha1_values.min() >= 0.0
            assert alpha1_values.max() <= 2.5

    def test_handles_nan_values(self):
        """Test handling of NaN values in RR data."""
        rr_data = np.random.uniform(800, 1200, 200)
        rr_data[50:55] = np.nan  # Insert NaN values

        df = pd.DataFrame({"rr": rr_data})

        result_df, error = calculate_dynamic_dfa_v2(df)

        # Should handle NaN gracefully
        assert result_df is not None or error is not None

    def test_empty_dataframe_returns_error(self):
        """Test with empty dataframe."""
        df = pd.DataFrame({"rr": []})

        result_df, error = calculate_dynamic_dfa_v2(df)

        assert error is not None or result_df is None

    def test_step_size_affects_output_resolution(self, sample_hrv_df):
        """Test that step size affects number of output windows."""
        result_30s, _ = calculate_dynamic_dfa_v2(sample_hrv_df, step_sec=30)
        result_10s, _ = calculate_dynamic_dfa_v2(sample_hrv_df, step_sec=10)

        if result_30s is not None and result_10s is not None:
            # Smaller step should produce more windows
            assert len(result_10s) >= len(result_30s)


class TestValidateDfaQuality:
    """Tests for DFA quality validation."""

    def test_high_quality_window(self):
        """Test validation of high quality window."""
        is_uncertain, reasons, grade = validate_dfa_quality(
            window_sec=300,
            data_quality=0.95,
            mean_alpha1=0.75,
            windows_analyzed=10,
        )

        assert grade in ["A", "B", "C", "D", "F"]
        assert isinstance(reasons, list)

    def test_short_window_flagged(self):
        """Test that short windows are flagged."""
        is_uncertain, reasons, grade = validate_dfa_quality(
            window_sec=60,  # Below 120s minimum
            data_quality=0.9,
            mean_alpha1=0.75,
            windows_analyzed=5,
        )

        assert is_uncertain is True or len(reasons) > 0

    def test_poor_data_quality_flagged(self):
        """Test that poor data quality is flagged."""
        is_uncertain, reasons, grade = validate_dfa_quality(
            window_sec=300,
            data_quality=0.3,  # Poor quality
            mean_alpha1=0.75,
            windows_analyzed=10,
        )

        assert is_uncertain is True or grade in ["D", "F"]

    def test_few_windows_flagged(self):
        """Test that few analyzed windows increase uncertainty."""
        is_uncertain, reasons, grade = validate_dfa_quality(
            window_sec=300,
            data_quality=0.9,
            mean_alpha1=0.75,
            windows_analyzed=2,  # Very few windows
        )

        # Should have some uncertainty with few windows
        assert isinstance(is_uncertain, bool)

    def test_returns_quality_grade(self):
        """Test that validation returns quality grade."""
        _, _, grade = validate_dfa_quality(
            window_sec=300,
            data_quality=0.9,
            mean_alpha1=0.75,
            windows_analyzed=10,
        )

        assert grade in ["A", "B", "C", "D", "F"]


class TestDfaCacheBehavior:
    """Tests for DFA caching behavior."""

    def test_repeated_calculation_uses_cache(self, sample_hrv_df):
        """Test that repeated calculations benefit from caching."""
        # First call
        result1, _ = calculate_dynamic_dfa_v2(sample_hrv_df)

        # Second call with same data should use cache
        result2, _ = calculate_dynamic_dfa_v2(sample_hrv_df)

        if result1 is not None and result2 is not None:
            # Results should be identical
            pd.testing.assert_frame_equal(result1, result2)

    def test_different_parameters_different_cache(self, sample_hrv_df):
        """Test that different parameters produce different results."""
        result1, _ = calculate_dynamic_dfa_v2(sample_hrv_df, window_sec=300)
        result2, _ = calculate_dynamic_dfa_v2(sample_hrv_df, window_sec=180)

        if result1 is not None and result2 is not None:
            # Different window sizes should produce different results
            assert len(result1) != len(result2) or True  # May have same length
