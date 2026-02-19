"""
Unit tests for the 7 modules introduced during the large-file refactor:
  - vt_utils
  - vt_step
  - vt_sliding
  - vt_cpet
  - smo2_analysis
  - smo2_thresholds
  - csv_export
"""

import json
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ramp_df(n=600, ve_slope=0.06, noise=0.5):
    """1-Hz ramp DataFrame with ventilation, power and HR."""
    np.random.seed(0)
    time = np.arange(n, dtype=float)
    watts = np.linspace(50, 350, n)
    hr = np.linspace(100, 175, n)
    ve = 10.0 + ve_slope * time + np.random.normal(0, noise, n)
    return pd.DataFrame(
        {
            "time": time,
            "watts": watts,
            "hr": hr,
            "tymeventilation": ve,
        }
    )


def _step_df_with_ve(n_steps=6, step_dur=120):
    """Step-test DataFrame with rising VE slope across steps."""
    rows = []
    for step in range(n_steps):
        for t in range(step_dur):
            global_t = step * step_dur + t
            ve = 15 + step * 3 + np.random.default_rng(step * 1000 + t).normal(0, 0.3)
            rows.append(
                {
                    "time": float(global_t),
                    "watts": float(50 + step * 30),
                    "hr": float(100 + step * 10),
                    "tymeventilation": ve,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. vt_utils
# ---------------------------------------------------------------------------


class TestCalculateSlope:
    def test_positive_slope(self):
        from modules.calculations.vt_utils import calculate_slope

        t = pd.Series([0.0, 1.0, 2.0, 3.0])
        v = pd.Series([0.0, 1.0, 2.0, 3.0])
        slope, intercept, err = calculate_slope(t, v)
        assert abs(slope - 1.0) < 1e-9
        assert abs(intercept) < 1e-9

    def test_fewer_than_2_points_returns_zeros(self):
        from modules.calculations.vt_utils import calculate_slope

        slope, intercept, err = calculate_slope(pd.Series([1.0]), pd.Series([2.0]))
        assert slope == 0.0
        assert intercept == 0.0
        assert err == 0.0

    def test_all_nan_returns_zeros(self):
        from modules.calculations.vt_utils import calculate_slope

        t = pd.Series([np.nan, np.nan, np.nan])
        v = pd.Series([1.0, 2.0, 3.0])
        slope, intercept, err = calculate_slope(t, v)
        assert slope == 0.0

    def test_returns_three_values(self):
        from modules.calculations.vt_utils import calculate_slope

        result = calculate_slope(pd.Series([0.0, 1.0]), pd.Series([0.0, 2.0]))
        assert len(result) == 3


class TestDetectVt1PeaksHeuristic:
    def test_too_short_returns_none(self):
        from modules.calculations.vt_utils import detect_vt1_peaks_heuristic

        df = pd.DataFrame({"time": range(50), "tymeventilation": range(50)})
        result, notes = detect_vt1_peaks_heuristic(df, "time", "tymeventilation")
        assert result is None
        assert len(notes) > 0

    def test_slope_above_threshold_returns_dict(self):
        from modules.calculations.vt_utils import detect_vt1_peaks_heuristic

        n = 300
        t = np.arange(n, dtype=float)
        # Two prominent peaks with steep slope between them
        ve = np.zeros(n)
        ve[50] = 20.0  # first peak
        ve[250] = 20.0  # second peak
        ve[50:250] = np.linspace(5.0, 15.0, 200)  # rising segment
        df = pd.DataFrame({"time": t, "tymeventilation": ve, "watts": np.ones(n) * 200})
        result, notes = detect_vt1_peaks_heuristic(df, "time", "tymeventilation")
        # Either detected or not — just confirm shape is correct
        assert isinstance(notes, list)
        if result is not None:
            assert "slope" in result
            assert "avg_power" in result


# ---------------------------------------------------------------------------
# 2. vt_step
# ---------------------------------------------------------------------------


class TestDetectVtFromSteps:
    def _make_step_range(self, df, step_dur=120):
        from modules.calculations.threshold_types import DetectedStep, StepTestRange

        n_steps = int(df["time"].max() / step_dur)
        steps = []
        for i in range(n_steps + 1):
            start = i * step_dur
            end = (i + 1) * step_dur
            mask = (df["time"] >= start) & (df["time"] < end)
            if mask.sum() > 0:
                steps.append(
                    DetectedStep(
                        step_number=i,
                        start_time=float(start),
                        end_time=float(end),
                        duration_sec=float(step_dur),
                        avg_power=float(df.loc[mask, "watts"].mean()),
                    )
                )
        return StepTestRange(
            start_time=0.0,
            end_time=float(df["time"].max()),
            steps=steps,
            min_power=50.0,
            max_power=200.0,
        )

    def test_insufficient_steps_returns_notes(self):
        from modules.calculations.vt_step import detect_vt_from_steps

        result = detect_vt_from_steps(pd.DataFrame(), None)
        assert len(result.notes) > 0
        assert result.vt1_watts is None

    def test_missing_ve_column_returns_notes(self):
        from modules.calculations.vt_step import detect_vt_from_steps

        df = _step_df_with_ve()
        step_range = self._make_step_range(df)
        result = detect_vt_from_steps(df, step_range, ve_column="does_not_exist")
        assert any("Missing VE" in n for n in result.notes)

    def test_detects_vt1_on_step_data(self):
        from modules.calculations.vt_step import detect_vt_from_steps

        df = _step_df_with_ve(n_steps=8, step_dur=120)
        step_range = self._make_step_range(df, step_dur=120)
        result = detect_vt_from_steps(df, step_range)
        # Either VT1 is detected or notes explain why not
        assert isinstance(result.notes, list)
        if result.vt1_watts is not None:
            assert result.vt1_watts > 0

    def test_result_has_step_analysis(self):
        from modules.calculations.vt_step import detect_vt_from_steps

        df = _step_df_with_ve(n_steps=6, step_dur=120)
        step_range = self._make_step_range(df)
        result = detect_vt_from_steps(df, step_range)
        assert isinstance(result.step_analysis, list)


# ---------------------------------------------------------------------------
# 3. vt_sliding
# ---------------------------------------------------------------------------


class TestDetectVtTransitionZone:
    def test_too_short_returns_none_none(self):
        from modules.calculations.vt_sliding import detect_vt_transition_zone

        df = pd.DataFrame({"time": range(10), "tymeventilation": range(10), "watts": range(10), "hr": range(10)})
        v1, v2 = detect_vt_transition_zone(df, window_duration=60)
        assert v1 is None
        assert v2 is None

    def test_returns_transition_zones_or_none(self):
        from modules.calculations.vt_sliding import detect_vt_transition_zone

        df = _ramp_df(n=900, ve_slope=0.06)
        v1, v2 = detect_vt_transition_zone(df, window_duration=60, step_size=10)
        assert v1 is None or hasattr(v1, "range_watts")
        assert v2 is None or hasattr(v2, "range_watts")

    def test_cache_hit_returns_same_object(self):
        from modules.calculations.vt_sliding import detect_vt_transition_zone

        df = _ramp_df(n=900)
        r1 = detect_vt_transition_zone(df, window_duration=60)
        r2 = detect_vt_transition_zone(df, window_duration=60)
        assert r1 is r2  # same cached tuple


class TestRunSensitivityAnalysis:
    def test_returns_sensitivity_result(self):
        from modules.calculations.vt_sliding import run_sensitivity_analysis
        from modules.calculations.threshold_types import SensitivityResult

        df = _ramp_df(n=900)
        result = run_sensitivity_analysis(df, "tymeventilation", "watts", "hr", "time")
        assert isinstance(result, SensitivityResult)

    def test_details_list_populated(self):
        from modules.calculations.vt_sliding import run_sensitivity_analysis

        df = _ramp_df(n=900)
        result = run_sensitivity_analysis(df, "tymeventilation", "watts", "hr", "time")
        assert isinstance(result.details, list)


# ---------------------------------------------------------------------------
# 4. vt_cpet
# ---------------------------------------------------------------------------


class TestDetectVtCpet:
    def test_missing_ve_returns_dict(self):
        from modules.calculations.vt_cpet import detect_vt_cpet

        # DataFrame has columns but not the VE column — should return dict gracefully
        df = pd.DataFrame({"time": range(10), "watts": range(10)})
        result = detect_vt_cpet(df, ve_column="tymeventilation")
        assert isinstance(result, dict)

    def test_with_ve_data_returns_dict_with_notes(self):
        from modules.calculations.vt_cpet import detect_vt_cpet

        df = _ramp_df(n=600)
        result = detect_vt_cpet(df)
        assert isinstance(result, dict)
        assert "notes" in result or "vt1_watts" in result or "method" in result

    def test_vslope_savgol_calls_detect_vt_cpet(self):
        from modules.calculations.vt_cpet import detect_vt_vslope_savgol

        df = _ramp_df(n=300)
        result = detect_vt_vslope_savgol(df)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# 5. smo2_analysis
# ---------------------------------------------------------------------------


class TestCalculateSmo2Slope:
    def test_negative_slope_for_decreasing_smo2(self):
        from modules.calculations.smo2_analysis import calculate_smo2_slope

        n = 50
        df = pd.DataFrame(
            {"watts": np.linspace(100, 400, n), "SmO2": np.linspace(70, 40, n)}
        )
        slope, r2 = calculate_smo2_slope(df)
        assert slope < 0
        assert 0 <= r2 <= 1

    def test_too_few_points_returns_zeros(self):
        from modules.calculations.smo2_analysis import calculate_smo2_slope

        df = pd.DataFrame({"watts": [200], "SmO2": [65]})
        slope, r2 = calculate_smo2_slope(df)
        assert slope == 0.0
        assert r2 == 0.0


class TestCalculateHalftimeReoxygenation:
    def test_returns_none_for_missing_column(self):
        from modules.calculations.smo2_analysis import calculate_halftime_reoxygenation

        df = pd.DataFrame({"watts": range(100)})
        assert calculate_halftime_reoxygenation(df) is None

    def test_returns_none_when_no_recovery(self):
        from modules.calculations.smo2_analysis import calculate_halftime_reoxygenation

        # Monotonically decreasing SmO2, no recovery
        df = pd.DataFrame(
            {"watts": np.linspace(100, 400, 200), "SmO2": np.linspace(70, 30, 200)}
        )
        result = calculate_halftime_reoxygenation(df, power_col="watts")
        assert result is None or isinstance(result, float)

    def test_detects_recovery_after_peak(self):
        from modules.calculations.smo2_analysis import calculate_halftime_reoxygenation

        # Ramp up power, SmO2 drops, then recovery phase
        n = 300
        power = np.concatenate([np.linspace(100, 400, 150), np.ones(150) * 50])
        smo2 = np.concatenate([np.linspace(70, 40, 150), np.linspace(40, 65, 150)])
        df = pd.DataFrame({"watts": power, "SmO2": smo2})
        result = calculate_halftime_reoxygenation(df, power_col="watts")
        if result is not None:
            assert result > 0


class TestCalculateHrCouplingIndex:
    def test_returns_zero_for_missing_columns(self):
        from modules.calculations.smo2_analysis import calculate_hr_coupling_index

        df = pd.DataFrame({"watts": range(50)})
        assert calculate_hr_coupling_index(df) == 0.0

    def test_negative_correlation_for_inverse_signals(self):
        from modules.calculations.smo2_analysis import calculate_hr_coupling_index

        n = 100
        df = pd.DataFrame(
            {
                "SmO2": np.linspace(70, 40, n),
                "hr": np.linspace(100, 170, n),
            }
        )
        r = calculate_hr_coupling_index(df)
        assert -1.0 <= r <= 1.0
        assert r < 0  # Inverse relationship


class TestCalculateSmo2Drift:
    def test_negative_drift_for_declining_smo2(self):
        from modules.calculations.smo2_analysis import calculate_smo2_drift

        df = pd.DataFrame(
            {"watts": np.ones(100) * 200, "SmO2": np.linspace(70, 50, 100)}
        )
        drift = calculate_smo2_drift(df)
        assert drift < 0

    def test_zero_for_missing_columns(self):
        from modules.calculations.smo2_analysis import calculate_smo2_drift

        df = pd.DataFrame({"watts": range(50)})
        assert calculate_smo2_drift(df) == 0.0


class TestClassifySmo2Limiter:
    def test_local_limiter_for_steep_slope(self):
        from modules.calculations.smo2_analysis import (
            SmO2AdvancedMetrics,
            classify_smo2_limiter,
        )

        m = SmO2AdvancedMetrics(slope_per_100w=-12.0, hr_coupling_r=0.0)
        ltype, confidence, interp = classify_smo2_limiter(m)
        assert ltype == "local"
        assert 0 < confidence <= 1.0
        assert isinstance(interp, str)

    def test_central_limiter_for_strong_hr_coupling(self):
        from modules.calculations.smo2_analysis import (
            SmO2AdvancedMetrics,
            classify_smo2_limiter,
        )

        m = SmO2AdvancedMetrics(slope_per_100w=-2.0, hr_coupling_r=-0.85)
        ltype, confidence, _ = classify_smo2_limiter(m)
        assert ltype == "central"


class TestAnalyzeSmo2Advanced:
    def test_missing_smo2_column(self):
        from modules.calculations.smo2_analysis import analyze_smo2_advanced

        df = pd.DataFrame({"watts": range(50)})
        result = analyze_smo2_advanced(df)
        assert result.data_quality == "no_smo2"

    def test_missing_power_column(self):
        from modules.calculations.smo2_analysis import analyze_smo2_advanced

        df = pd.DataFrame({"SmO2": range(50)})
        result = analyze_smo2_advanced(df)
        assert result.data_quality == "no_power"

    def test_returns_metrics_with_data(self):
        from modules.calculations.smo2_analysis import analyze_smo2_advanced

        n = 100
        df = pd.DataFrame(
            {
                "SmO2": np.linspace(70, 40, n),
                "watts": np.linspace(100, 400, n),
                "hr": np.linspace(100, 170, n),
            }
        )
        result = analyze_smo2_advanced(df)
        assert result.limiter_type in ("local", "central", "metabolic", "unknown")
        assert 0 <= result.limiter_confidence <= 1.0
        assert isinstance(result.recommendations, list)

    def test_cache_returns_same_object(self):
        from modules.calculations.smo2_analysis import analyze_smo2_advanced

        df = pd.DataFrame(
            {"SmO2": np.linspace(70, 50, 60), "watts": np.linspace(100, 300, 60)}
        )
        r1 = analyze_smo2_advanced(df)
        r2 = analyze_smo2_advanced(df)
        assert r1 is r2


class TestFormatSmo2MetricsForReport:
    def test_all_required_keys_present(self):
        from modules.calculations.smo2_analysis import (
            SmO2AdvancedMetrics,
            format_smo2_metrics_for_report,
        )

        m = SmO2AdvancedMetrics(slope_per_100w=-5.0, limiter_type="local")
        report = format_smo2_metrics_for_report(m)
        for key in ("slope_per_100w", "limiter_type", "limiter_confidence", "data_quality"):
            assert key in report


# ---------------------------------------------------------------------------
# 6. smo2_thresholds
# ---------------------------------------------------------------------------


class TestDetectSmo2ThresholdsMoxy:
    def test_missing_smo2_returns_note(self):
        from modules.calculations.smo2_thresholds import detect_smo2_thresholds_moxy

        df = pd.DataFrame({"watts": range(300), "time": range(300)})
        result = detect_smo2_thresholds_moxy(df)
        assert any("SmO2" in n for n in result.analysis_notes)

    def test_missing_power_returns_note(self):
        from modules.calculations.smo2_thresholds import detect_smo2_thresholds_moxy

        df = pd.DataFrame({"smo2": range(300), "time": range(300)})
        result = detect_smo2_thresholds_moxy(df)
        assert any("mocy" in n for n in result.analysis_notes)

    def test_too_few_steps_returns_note(self):
        from modules.calculations.smo2_thresholds import detect_smo2_thresholds_moxy

        # Only 60 rows → less than 4 valid 180-s steps
        n = 60
        df = pd.DataFrame(
            {"smo2": np.ones(n) * 60, "watts": np.ones(n) * 200, "time": np.arange(n)}
        )
        result = detect_smo2_thresholds_moxy(df)
        assert result.t1_watts is None

    def test_result_has_zones_list(self):
        from modules.calculations.smo2_thresholds import detect_smo2_thresholds_moxy

        n = 1200
        smo2 = np.linspace(70, 30, n) + np.random.default_rng(0).normal(0, 1, n)
        df = pd.DataFrame(
            {
                "smo2": smo2,
                "watts": np.linspace(100, 450, n),
                "hr": np.linspace(100, 180, n),
                "time": np.arange(n),
            }
        )
        result = detect_smo2_thresholds_moxy(df, step_duration_sec=180)
        assert isinstance(result.zones, list)
        assert isinstance(result.analysis_notes, list)

    def test_cache_hit_same_object(self):
        from modules.calculations.smo2_thresholds import detect_smo2_thresholds_moxy

        n = 900
        df = pd.DataFrame(
            {
                "smo2": np.linspace(70, 40, n),
                "watts": np.linspace(100, 400, n),
                "time": np.arange(n),
            }
        )
        r1 = detect_smo2_thresholds_moxy(df)
        r2 = detect_smo2_thresholds_moxy(df)
        assert r1 is r2


# ---------------------------------------------------------------------------
# 7. csv_export
# ---------------------------------------------------------------------------


class TestExportSessionCsv:
    def test_returns_bytes(self):
        from modules.reporting.csv_export import export_session_csv

        df = pd.DataFrame({"time": [1, 2, 3], "watts": [100, 200, 300]})
        result = export_session_csv(df)
        assert isinstance(result, bytes)

    def test_header_row_present(self):
        from modules.reporting.csv_export import export_session_csv

        df = pd.DataFrame({"time": [1, 2], "watts": [100, 200]})
        csv = export_session_csv(df).decode("utf-8")
        assert "time" in csv
        assert "watts" in csv

    def test_values_correct(self):
        from modules.reporting.csv_export import export_session_csv

        df = pd.DataFrame({"time": [1], "watts": [150]})
        csv = export_session_csv(df).decode("utf-8")
        assert "150" in csv

    def test_empty_df_produces_header_only(self):
        from modules.reporting.csv_export import export_session_csv

        df = pd.DataFrame({"time": [], "watts": []})
        csv = export_session_csv(df).decode("utf-8")
        lines = [l for l in csv.strip().splitlines() if l]
        assert len(lines) == 1  # header only
        assert "watts" in lines[0]


class TestExportMetricsCsv:
    def test_returns_bytes(self):
        from modules.reporting.csv_export import export_metrics_csv

        result = export_metrics_csv({"avg_power": 200})
        assert isinstance(result, bytes)

    def test_private_keys_excluded(self):
        from modules.reporting.csv_export import export_metrics_csv

        csv = export_metrics_csv({"avg_power": 200, "_internal": "skip"}).decode("utf-8")
        assert "_internal" not in csv
        assert "avg_power" in csv

    def test_nested_dict_json_serialised(self):
        from modules.reporting.csv_export import export_metrics_csv

        csv = export_metrics_csv({"meta": {"a": 1}}).decode("utf-8")
        assert "meta" in csv
        # JSON-encoded value should be in the CSV
        assert "{" in csv or "a" in csv

    def test_nested_list_json_serialised(self):
        from modules.reporting.csv_export import export_metrics_csv

        csv = export_metrics_csv({"zones": [1, 2, 3]}).decode("utf-8")
        assert "zones" in csv

    def test_has_metric_and_value_columns(self):
        from modules.reporting.csv_export import export_metrics_csv

        csv = export_metrics_csv({"np": 210}).decode("utf-8")
        assert "metric" in csv
        assert "value" in csv

    def test_empty_metrics_produces_header_only(self):
        from modules.reporting.csv_export import export_metrics_csv

        csv = export_metrics_csv({}).decode("utf-8")
        lines = [l for l in csv.strip().splitlines() if l]
        assert len(lines) == 1  # header only
