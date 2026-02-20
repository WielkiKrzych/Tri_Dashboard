"""Extended tests for modules/tte.py — TTE computation and export."""

import json
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from modules.tte import (
    compute_tte,
    compute_tte_result,
    rolling_tte,
    compute_trend_data,
    export_tte_json,
    format_tte,
    TTEResult,
)


class TestComputeTrend:
    def test_multiple_windows(self):
        now = datetime.now()
        history = [
            {"date": (now - timedelta(days=i)).isoformat(), "tte_seconds": 300 + i * 10}
            for i in range(10)
        ]
        result = compute_trend_data(history, windows=[30, 90])
        assert 30 in result
        assert 90 in result
        assert result[30]["count"] > 0

    def test_empty_history(self):
        result = compute_trend_data([], windows=[30])
        assert result[30]["count"] == 0


class TestExportTteJson:
    def test_valid_json_output(self):
        result = TTEResult(
            session_id="test_001",
            tte_seconds=300,
            target_pct=100.0,
            ftp=280.0,
            tolerance_pct=5.0,
            target_power_min=266.0,
            target_power_max=294.0,
            timestamp="2024-01-01T12:00:00",
        )
        json_str = export_tte_json(result)
        data = json.loads(json_str)
        assert data["tte_seconds"] == 300
        assert data["tte_formatted"] == "05:00"
        assert data["target_power_range"]["min"] == 266.0


class TestFormatTte:
    def test_zero(self):
        assert format_tte(0) == "00:00"

    def test_negative(self):
        assert format_tte(-5) == "00:00"

    def test_five_minutes(self):
        assert format_tte(300) == "05:00"

    def test_mixed(self):
        assert format_tte(125) == "02:05"


class TestRollingTteExtended:
    def test_filters_by_window(self):
        now = datetime.now()
        history = [
            {"date": (now - timedelta(days=5)).isoformat(), "tte_seconds": 300},
            {"date": (now - timedelta(days=60)).isoformat(), "tte_seconds": 200},
        ]
        result = rolling_tte(history, window_days=30)
        assert result["count"] == 1  # Only the 5-day-old entry
        assert result["mean"] == 300

    def test_skips_invalid_dates(self):
        history = [
            {"date": "not-a-date", "tte_seconds": 300},
        ]
        result = rolling_tte(history, window_days=30)
        assert result["count"] == 0

    def test_skips_zero_tte(self):
        now = datetime.now()
        history = [
            {"date": now.isoformat(), "tte_seconds": 0},
        ]
        result = rolling_tte(history, window_days=30)
        assert result["count"] == 0
