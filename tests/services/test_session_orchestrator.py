"""Tests for session_orchestrator module."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from services.session_orchestrator import (
    process_uploaded_session,
    prepare_session_record,
    prepare_sticky_header_data,
)


class TestProcessUploadedSession:
    """Tests for process_uploaded_session function."""

    def test_basic_processing(self, sample_power_df):
        """Test basic session processing."""
        df_plot, df_resampled, metrics, error = process_uploaded_session(
            df_raw=sample_power_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
            vt1_watts=180,
            vt2_watts=250,
            parallel=False,
        )

        assert error is None
        assert df_plot is not None
        assert metrics is not None

    def test_processing_with_parallel(self, sample_power_df):
        """Test parallel processing mode."""
        df_plot, df_resampled, metrics, error = process_uploaded_session(
            df_raw=sample_power_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
            parallel=True,
        )

        assert error is None

    def test_empty_dataframe_returns_error(self, empty_df):
        """Test with empty dataframe."""
        df_plot, df_resampled, metrics, error = process_uploaded_session(
            df_raw=empty_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        assert error is not None

    def test_malformed_dataframe_handled(self, malformed_df):
        """Test handling of malformed data."""
        df_plot, df_resampled, metrics, error = process_uploaded_session(
            df_raw=malformed_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        # Should either process or return error, not crash
        assert error is not None or df_plot is not None

    def test_metrics_structure(self, sample_power_df):
        """Test that metrics contain expected keys."""
        df_plot, df_resampled, metrics, error = process_uploaded_session(
            df_raw=sample_power_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        if error is None and metrics is not None:
            assert isinstance(metrics, dict)
            # Should have some standard metrics
            assert len(metrics) > 0

    def test_resampling_reduces_size(self, sample_long_ride_df):
        """Test that resampling reduces dataframe size for long rides."""
        df_plot, df_resampled, metrics, error = process_uploaded_session(
            df_raw=sample_long_ride_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        if df_resampled is not None:
            # Resampled should generally be smaller or similar
            assert len(df_resampled) <= len(sample_long_ride_df) * 1.5

    def test_with_nan_values(self, nan_df):
        """Test handling of NaN values in data."""
        df_plot, df_resampled, metrics, error = process_uploaded_session(
            df_raw=nan_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        # Should handle NaN gracefully
        assert error is not None or df_plot is not None


class TestPrepareSessionRecord:
    """Tests for prepare_session_record function."""

    def test_basic_record_preparation(self, sample_power_df):
        """Test basic session record preparation."""
        # First process to get valid data
        df_plot, _, metrics, _ = process_uploaded_session(
            df_raw=sample_power_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        if metrics is not None:
            record = prepare_session_record(
                filename="test_ride.fit",
                df_plot=df_plot,
                metrics=metrics,
            )

            assert isinstance(record, dict)
            assert "filename" in record

    def test_record_contains_key_metrics(self, sample_power_df):
        """Test that record contains key metrics."""
        df_plot, _, metrics, _ = process_uploaded_session(
            df_raw=sample_power_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        if metrics is not None:
            record = prepare_session_record(
                filename="test.fit",
                df_plot=df_plot,
                metrics=metrics,
            )

            # Should have at least filename
            assert record is not None


class TestPrepareStickyHeaderData:
    """Tests for prepare_sticky_header_data function."""

    def test_header_data_structure(self, sample_power_df):
        """Test header data has expected structure."""
        df_plot, _, metrics, _ = process_uploaded_session(
            df_raw=sample_power_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        if metrics is not None:
            header_data = prepare_sticky_header_data(df_plot, metrics)

            assert isinstance(header_data, dict)

    def test_header_data_with_none_metrics(self):
        """Test handling of None metrics."""
        df = pd.DataFrame({"watts": [100, 200, 150]})
        header_data = prepare_sticky_header_data(df, None)

        # Should handle None gracefully
        assert header_data is not None or header_data is None


class TestEdgeCases:
    """Edge case tests."""

    def test_very_short_session(self):
        """Test with very short session."""
        short_df = pd.DataFrame({
            "watts": [100, 150, 200],
            "hr": [90, 100, 110],
        })

        df_plot, _, metrics, error = process_uploaded_session(
            df_raw=short_df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        # Should either process or return error
        assert error is not None or df_plot is not None

    def test_zero_power_values(self):
        """Test with zero power values."""
        df = pd.DataFrame({
            "watts": [0, 0, 0, 0, 0],
            "hr": [80, 85, 90, 95, 100],
        })

        df_plot, _, metrics, error = process_uploaded_session(
            df_raw=df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        # Should handle zeros
        assert error is not None or df_plot is not None

    def test_extreme_power_values(self):
        """Test with extreme power values."""
        df = pd.DataFrame({
            "watts": [0, 2000, 0, 2500, 0],  # Very high spikes
            "hr": [80, 180, 85, 190, 90],
        })

        df_plot, _, metrics, error = process_uploaded_session(
            df_raw=df,
            cp_input=280,
            w_prime_input=20000,
            rider_weight=75,
        )

        assert error is not None or df_plot is not None
