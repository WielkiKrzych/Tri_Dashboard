"""Tests for modules/utils.py — utility functions for data parsing and normalization."""

import pytest
import pandas as pd
import numpy as np
from modules.utils import (
    parse_time_input,
    _serialize_df_to_parquet_bytes,
    normalize_columns_pandas,
    _clean_hrv_value,
    _process_hrv_column,
    _convert_numeric_types,
    _process_large_dataframe,
)


# =========================================================================
# parse_time_input
# =========================================================================

class TestParseTimeInput:
    def test_hhmmss(self):
        assert parse_time_input("01:02:03") == 3723

    def test_mmss(self):
        assert parse_time_input("05:30") == 330

    def test_seconds_only(self):
        assert parse_time_input("90") == 90

    def test_zero(self):
        assert parse_time_input("0") == 0

    def test_invalid_string(self):
        assert parse_time_input("abc") is None

    def test_empty_string(self):
        assert parse_time_input("") is None

    def test_none_input(self):
        assert parse_time_input(None) is None


# =========================================================================
# _serialize_df_to_parquet_bytes
# =========================================================================

class TestSerializeDfToParquetBytes:
    def test_round_trip(self):
        df = pd.DataFrame({"watts": [100, 200, 300], "hr": [120, 130, 140]})
        result = _serialize_df_to_parquet_bytes(df)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_empty_df(self):
        df = pd.DataFrame()
        result = _serialize_df_to_parquet_bytes(df)
        assert isinstance(result, bytes)


# =========================================================================
# normalize_columns_pandas
# =========================================================================

class TestNormalizeColumnsPandas:
    def test_hr_alias_remapped(self):
        df = pd.DataFrame({"hr": [120], "watts": [200]})
        result = normalize_columns_pandas(df)
        assert "heartrate" in result.columns

    def test_ve_alias_remapped(self):
        df = pd.DataFrame({"ve": [30.0], "watts": [200]})
        result = normalize_columns_pandas(df)
        assert "tymeventilation" in result.columns

    def test_br_alias_remapped(self):
        df = pd.DataFrame({"br": [20.0]})
        result = normalize_columns_pandas(df)
        assert "tymebreathrate" in result.columns

    def test_canonical_name_unchanged(self):
        df = pd.DataFrame({"heartrate": [120], "watts": [200]})
        result = normalize_columns_pandas(df)
        assert "heartrate" in result.columns

    def test_uppercase_lowered(self):
        df = pd.DataFrame({"HR": [120], "WATTS": [200]})
        result = normalize_columns_pandas(df)
        assert "heartrate" in result.columns
        assert "watts" in result.columns

    def test_thb_alias(self):
        df = pd.DataFrame({"total_hemoglobin": [12.0]})
        result = normalize_columns_pandas(df)
        assert "thb" in result.columns


# =========================================================================
# _clean_hrv_value
# =========================================================================

class TestCleanHrvValue:
    def test_plain_number(self):
        assert _clean_hrv_value("55.3") == pytest.approx(55.3)

    def test_colon_separated(self):
        result = _clean_hrv_value("50:60:55")
        assert result == pytest.approx(55.0)

    def test_nan_string(self):
        assert np.isnan(_clean_hrv_value("nan"))

    def test_empty_string(self):
        assert np.isnan(_clean_hrv_value(""))

    def test_invalid_string(self):
        assert np.isnan(_clean_hrv_value("abc"))

    def test_integer_string(self):
        assert _clean_hrv_value("42") == pytest.approx(42.0)


# =========================================================================
# _process_hrv_column
# =========================================================================

class TestProcessHrvColumn:
    def test_hrv_column_cleaned(self):
        df = pd.DataFrame({"hrv": ["50", "60:70", "nan", "45.5"]})
        result = _process_hrv_column(df)
        assert "hrv" in result.columns
        assert result["hrv"].dtype == np.float64 or pd.api.types.is_float_dtype(result["hrv"])
        assert not result["hrv"].isna().all()

    def test_no_hrv_column_unchanged(self):
        df = pd.DataFrame({"watts": [100, 200]})
        result = _process_hrv_column(df)
        assert "hrv" not in result.columns


# =========================================================================
# _convert_numeric_types
# =========================================================================

class TestConvertNumericTypes:
    def test_string_watts_converted(self):
        df = pd.DataFrame({"watts": ["100", "200", "300"]})
        result = _convert_numeric_types(df)
        assert pd.api.types.is_numeric_dtype(result["watts"])

    def test_mixed_types_converted(self):
        df = pd.DataFrame({"watts": ["100", "abc", "300"]})
        result = _convert_numeric_types(df)
        assert pd.api.types.is_numeric_dtype(result["watts"])
        assert pd.isna(result["watts"].iloc[1])

    def test_unrelated_column_unchanged(self):
        df = pd.DataFrame({"foo": ["hello", "world"]})
        result = _convert_numeric_types(df)
        assert result["foo"].iloc[0] == "hello"


# =========================================================================
# _process_large_dataframe
# =========================================================================

class TestProcessLargeDataframe:
    def test_chunked_processing(self):
        n = 200
        df = pd.DataFrame({
            "hr": np.random.randint(60, 180, n),
            "watts": np.random.randint(50, 400, n),
        })
        result = _process_large_dataframe(df, chunk_size=50)
        assert len(result) == n
        assert "heartrate" in result.columns
        assert "time" in result.columns
