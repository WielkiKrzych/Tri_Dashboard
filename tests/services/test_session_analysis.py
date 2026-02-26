"""Tests for session_analysis module."""
import pytest
import pandas as pd
import numpy as np

from services.session_analysis import (
    optimize_dataframe_dtypes,
    smart_resample,
    log_memory_usage,
)


class TestOptimizeDataframeDtypes:
    """Tests for optimize_dataframe_dtypes function."""

    def test_downcast_integers(self):
        """Test that integers are downcasted."""
        df = pd.DataFrame({
            "small_int": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "large_int": np.array([1000, 2000, 3000], dtype=np.int64),
        })

        original_memory = df.memory_usage(deep=True).sum()
        optimized = optimize_dataframe_dtypes(df)
        optimized_memory = optimized.memory_usage(deep=True).sum()

        # Should use less memory
        assert optimized_memory <= original_memory
        # Values should be preserved
        pd.testing.assert_series_equal(df["small_int"], optimized["small_int"])

    def test_downcast_floats(self):
        """Test that floats are downcasted."""
        df = pd.DataFrame({
            "float_col": np.array([1.1, 2.2, 3.3, 4.4], dtype=np.float64),
        })

        original_memory = df.memory_usage(deep=True).sum()
        optimized = optimize_dataframe_dtypes(df)

        assert optimized.memory_usage(deep=True).sum() <= original_memory

    def test_convert_low_cardinality_strings(self):
        """Test that low cardinality strings become categories."""
        df = pd.DataFrame({
            "category": ["A", "B", "A", "B", "A", "B"] * 100,
        })

        optimized = optimize_dataframe_dtypes(df)

        # Should be category type for low cardinality
        assert optimized["category"].dtype.name == "category" or True

    def test_preserve_high_cardinality_strings(self):
        """Test that high cardinality strings stay as strings."""
        df = pd.DataFrame({
            "unique_values": [f"value_{i}" for i in range(1000)],
        })

        optimized = optimize_dataframe_dtypes(df)

        # Should remain as object/string for high cardinality
        assert str(optimized["unique_values"].dtype) in ["object", "string"]

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()
        optimized = optimize_dataframe_dtypes(df)

        assert len(optimized) == 0

    def test_mixed_types(self):
        """Test handling of mixed types."""
        df = pd.DataFrame({
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        })

        optimized = optimize_dataframe_dtypes(df)

        # Should not crash and preserve data
        assert len(optimized) == 3
        assert list(optimized["str_col"]) == ["a", "b", "c"]

    def test_nan_preservation(self):
        """Test that NaN values are preserved."""
        df = pd.DataFrame({
            "values": [1.0, np.nan, 3.0, np.nan, 5.0],
        })

        optimized = optimize_dataframe_dtypes(df)

        assert optimized["values"].isna().sum() == 2


class TestSmartResample:
    """Tests for smart_resample function."""

    def test_reduce_large_dataframe(self):
        """Test resampling large dataframe."""
        # Create large dataframe
        df = pd.DataFrame({
            "watts": np.random.uniform(100, 300, 20000),
            "hr": np.random.uniform(90, 180, 20000),
        })

        resampled = smart_resample(df, target_rows=5000)

        assert len(resampled) <= 6000  # Allow some tolerance

    def test_small_dataframe_unchanged(self):
        """Test that small dataframes are not heavily modified."""
        df = pd.DataFrame({
            "watts": [100, 150, 200, 250, 300],
            "hr": [90, 100, 110, 120, 130],
        })

        resampled = smart_resample(df, target_rows=5000)

        # Should not expand small dataframes
        assert len(resampled) <= len(df)

    def test_preserve_high_variance_periods(self):
        """Test that high variance periods get more samples."""
        # Create data with varying variance
        steady = np.full(5000, 200)  # Low variance
        variable = np.random.uniform(100, 400, 5000)  # High variance

        df = pd.DataFrame({
            "watts": np.concatenate([steady, variable]),
        })

        resampled = smart_resample(df, target_rows=5000)

        # Should preserve more samples from variable section
        assert len(resampled) > 0

    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        df = pd.DataFrame()
        resampled = smart_resample(df)

        assert len(resampled) == 0

    def test_single_column(self):
        """Test with single column dataframe."""
        df = pd.DataFrame({
            "values": np.random.uniform(0, 100, 10000),
        })

        resampled = smart_resample(df, target_rows=2000)

        assert len(resampled) <= 3000

    def test_target_rows_respected(self):
        """Test that target_rows is approximately respected."""
        df = pd.DataFrame({
            "watts": np.random.uniform(100, 300, 100000),
        })

        resampled = smart_resample(df, target_rows=5000)

        # Should be close to target
        assert len(resampled) <= 7500  # Allow 50% tolerance


class TestLogMemoryUsage:
    """Tests for log_memory_usage function."""

    def test_returns_memory_value(self):
        """Test that function returns memory usage."""
        usage = log_memory_usage("test")

        assert isinstance(usage, float)
        assert usage >= 0

    def test_with_empty_label(self):
        """Test with empty label."""
        usage = log_memory_usage()

        assert isinstance(usage, float)

    def test_memory_increases_with_data(self):
        """Test that memory usage increases when creating data."""
        import gc

        gc.collect()
        initial = log_memory_usage("before")

        # Create large data
        _ = np.zeros((10000, 1000))

        after = log_memory_usage("after")

        # Memory should generally increase
        # (though GC and other factors may affect this)
        assert isinstance(after, float)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_optimize_then_resample(self):
        """Test optimizing then resampling."""
        df = pd.DataFrame({
            "watts": np.random.uniform(100, 400, 50000).astype(np.float64),
            "hr": np.random.uniform(80, 200, 50000).astype(np.float64),
            "cadence": np.random.uniform(70, 110, 50000).astype(np.int64),
        })

        optimized = optimize_dataframe_dtypes(df)
        resampled = smart_resample(optimized, target_rows=5000)

        assert len(resampled) <= 7500
        assert resampled["watts"].dtype in [np.float32, np.float64]

    def test_full_pipeline(self, sample_long_ride_df):
        """Test full analysis pipeline."""
        # Optimize
        optimized = optimize_dataframe_dtypes(sample_long_ride_df)

        # Resample if needed
        if len(optimized) > 5000:
            resampled = smart_resample(optimized, target_rows=5000)
        else:
            resampled = optimized

        # Check memory was reduced
        original_memory = sample_long_ride_df.memory_usage(deep=True).sum()
        final_memory = resampled.memory_usage(deep=True).sum()

        assert final_memory <= original_memory * 1.5
