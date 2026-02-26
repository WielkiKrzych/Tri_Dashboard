"""Tests for pipeline module."""
import pytest
import pandas as pd
import numpy as np

from modules.calculations.pipeline import (
    validate_test,
)
from models.results import ValidityLevel


class TestValidateTest:
    """Tests for validate_test function."""

    def test_valid_ramp_test(self):
        """Test validation of a valid ramp test."""
        # Create a proper ramp test: 10 minutes, 200W range
        time = np.arange(0, 600, 1)  # 10 minutes in seconds
        power = 100 + (time / 600) * 200  # Ramp from 100 to 300W
        hr = 90 + (time / 600) * 80  # HR from 90 to 170

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": hr,
        })

        validity = validate_test(df)

        assert validity.validity == ValidityLevel.VALID
        assert validity.ramp_duration_sec >= 480
        assert validity.power_range_watts >= 150

    def test_conditional_short_ramp(self):
        """Test validation of a short but acceptable ramp."""
        # 7 minutes ramp (between 6-8 min threshold)
        time = np.arange(0, 420, 1)
        power = 100 + (time / 420) * 200
        hr = 90 + (time / 420) * 80

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": hr,
        })

        validity = validate_test(df)

        assert validity.validity in [ValidityLevel.VALID, ValidityLevel.CONDITIONAL]

    def test_invalid_too_short(self):
        """Test validation of too short ramp test."""
        # 4 minutes ramp (below 6 min threshold)
        time = np.arange(0, 240, 1)
        power = 100 + (time / 240) * 150
        hr = 90 + (time / 240) * 60

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": hr,
        })

        validity = validate_test(df)

        assert validity.validity == ValidityLevel.INVALID
        assert "krótki" in validity.get_issues_summary().lower() or len(validity.issues) > 0

    def test_invalid_small_power_range(self):
        """Test validation with insufficient power range."""
        # 10 minutes but only 100W range
        time = np.arange(0, 600, 1)
        power = 100 + (time / 600) * 100  # Only 100W range
        hr = 90 + (time / 600) * 40

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": hr,
        })

        validity = validate_test(df)

        # Should be conditional or invalid due to small range
        assert validity.validity in [ValidityLevel.CONDITIONAL, ValidityLevel.INVALID]

    def test_missing_power_column(self):
        """Test validation with missing power column."""
        df = pd.DataFrame({
            "time": np.arange(0, 600, 1),
            "hr": 100 + np.random.uniform(-5, 5, 600),
        })

        validity = validate_test(df, power_column="watts")

        assert validity.validity == ValidityLevel.INVALID
        assert len(validity.issues) > 0

    def test_missing_hr_column(self):
        """Test validation with missing HR column."""
        time = np.arange(0, 600, 1)
        df = pd.DataFrame({
            "time": time,
            "watts": 100 + (time / 600) * 200,
        })

        validity = validate_test(df, hr_column="heartrate")

        assert validity.validity == ValidityLevel.INVALID

    def test_custom_column_names(self):
        """Test validation with custom column names."""
        time = np.arange(0, 600, 1)
        df = pd.DataFrame({
            "seconds": time,
            "power": 100 + (time / 600) * 200,
            "heart_rate": 90 + (time / 600) * 80,
        })

        validity = validate_test(
            df,
            power_column="power",
            hr_column="heart_rate",
            time_column="seconds",
        )

        assert validity.validity == ValidityLevel.VALID

    def test_empty_dataframe(self):
        """Test validation with empty dataframe."""
        df = pd.DataFrame()

        validity = validate_test(df)

        assert validity.validity == ValidityLevel.INVALID

    def test_nan_values_in_power(self):
        """Test handling of NaN values in power."""
        time = np.arange(0, 600, 1)
        power = 100 + (time / 600) * 200
        power[100:150] = np.nan  # Insert NaN values

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": 100 + np.random.uniform(-5, 5, 600),
        })

        validity = validate_test(df)

        # Should handle NaN gracefully
        assert validity is not None

    def test_very_long_ramp(self):
        """Test validation of very long ramp test."""
        # 20 minutes ramp
        time = np.arange(0, 1200, 1)
        power = 100 + (time / 1200) * 300
        hr = 90 + (time / 1200) * 90

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": hr,
        })

        validity = validate_test(df)

        assert validity.validity == ValidityLevel.VALID
        assert validity.ramp_duration_sec >= 480

    def test_custom_duration_threshold(self):
        """Test with custom minimum duration threshold."""
        # 5 minutes ramp
        time = np.arange(0, 300, 1)
        power = 100 + (time / 300) * 200
        hr = 90 + (time / 300) * 80

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": hr,
        })

        # With 4 min threshold, should be valid
        validity = validate_test(df, min_ramp_duration_sec=240)

        assert validity.validity in [ValidityLevel.VALID, ValidityLevel.CONDITIONAL]

    def test_custom_power_range_threshold(self):
        """Test with custom power range threshold."""
        time = np.arange(0, 600, 1)
        power = 100 + (time / 600) * 100  # 100W range

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": 100 + np.random.uniform(-5, 5, 600),
        })

        # With 80W threshold, should be valid
        validity = validate_test(df, min_power_range_watts=80.0)

        assert validity.power_range_watts >= 80.0


class TestValidityMethods:
    """Tests for TestValidity methods."""

    def test_get_issues_summary(self):
        """Test issues summary generation."""
        time = np.arange(0, 300, 1)  # Too short
        power = 100 + (time / 300) * 100  # Small range

        df = pd.DataFrame({
            "time": time,
            "watts": power,
            "hr": 100 + np.random.uniform(-5, 5, 300),
        })

        validity = validate_test(df)

        summary = validity.get_issues_summary()
        assert isinstance(summary, str)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        time = np.arange(0, 600, 1)
        df = pd.DataFrame({
            "time": time,
            "watts": 100 + (time / 600) * 200,
            "hr": 90 + (time / 600) * 80,
        })

        validity = validate_test(df)
        result = validity.to_dict()

        assert isinstance(result, dict)
        assert "validity" in result
