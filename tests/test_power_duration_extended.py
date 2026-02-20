"""Extended tests for modules/power_duration.py — PDC, CP model, exports."""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from modules.power_duration import (
    compute_historical_pdc,
    _format_duration,
    _cp_model,
    export_model_json,
    export_pr_csv,
    CPModelResult,
    PersonalRecord,
)


class TestComputeHistoricalPdc:
    def test_empty_sessions(self):
        assert compute_historical_pdc([]) == {}

    def test_max_method(self):
        sessions = [
            {5: 800, 60: 400, 300: 280},
            {5: 750, 60: 420, 300: 290},
        ]
        result = compute_historical_pdc(sessions, method="max")
        assert result[5] == 800
        assert result[60] == 420
        assert result[300] == 290

    def test_mean_method(self):
        sessions = [
            {60: 400},
            {60: 500},
        ]
        result = compute_historical_pdc(sessions, method="mean")
        assert result[60] == pytest.approx(450.0)

    def test_median_method(self):
        sessions = [
            {60: 400},
            {60: 500},
            {60: 600},
        ]
        result = compute_historical_pdc(sessions, method="median")
        assert result[60] == pytest.approx(500.0)

    def test_none_values_skipped(self):
        sessions = [
            {5: 800, 60: None},
            {5: 750, 60: 400},
        ]
        result = compute_historical_pdc(sessions, method="max")
        assert result[5] == 800
        assert result[60] == 400

    def test_all_none_for_duration(self):
        sessions = [
            {60: None},
            {60: None},
        ]
        result = compute_historical_pdc(sessions, method="max")
        assert result[60] is None


class TestFormatDuration:
    def test_seconds(self):
        assert _format_duration(45) == "45s"

    def test_minutes_no_seconds(self):
        assert _format_duration(300) == "5min"

    def test_minutes_with_seconds(self):
        assert _format_duration(330) == "5min30s"

    def test_hours_no_minutes(self):
        assert _format_duration(3600) == "1h"

    def test_hours_with_minutes(self):
        assert _format_duration(3660) == "1h1min"


class TestCpModel:
    def test_basic_calculation(self):
        t = np.array([120.0, 300.0, 600.0, 1200.0])
        result = _cp_model(t, 20000.0, 250.0)
        # At t=120: 20000/120 + 250 = 416.67
        assert result[0] == pytest.approx(20000 / 120 + 250, abs=0.1)

    def test_asymptote_is_cp(self):
        t = np.array([100000.0])  # Very long duration
        result = _cp_model(t, 20000.0, 250.0)
        assert result[0] == pytest.approx(250.0, abs=1.0)


class TestExportModelJson:
    def test_creates_valid_json(self, tmp_path):
        model = CPModelResult(
            cp=260.0, w_prime=18000.0, rmse=5.2, r_squared=0.98,
            durations_used=[120, 300, 600, 1200]
        )
        output = tmp_path / "model.json"
        result_path = export_model_json(model, str(output))
        assert Path(result_path).exists()

        with open(result_path) as f:
            data = json.load(f)
        assert data["CP"] == 260.0
        assert data["W_prime_kJ"] == pytest.approx(18.0)


class TestExportPrCsv:
    def test_creates_valid_csv(self, tmp_path):
        prs = [
            PersonalRecord(duration=5, power=900.0, timestamp="2024-01-01"),
            PersonalRecord(duration=300, power=350.0, timestamp="2024-01-01"),
        ]
        output = tmp_path / "prs.csv"
        result_path = export_pr_csv(prs, str(output))
        assert Path(result_path).exists()

        df = pd.read_csv(result_path)
        assert len(df) == 2
        assert "duration_formatted" in df.columns
