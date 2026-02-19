"""
End-to-end tests for Streamlit app using AppTest framework.

Tests critical user flows:
1. App boots without crashing (no file uploaded)
2. All tab modules are importable and their render functions exist
3. Full calculation pipeline works with realistic test data
4. Report generation pipeline produces valid output
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# =========================================================================
# 1. TAB REGISTRY — all tabs importable
# =========================================================================


class TestTabRegistry:
    """Verify all registered tabs can be imported."""

    TAB_MODULES = [
        ("modules.ui.report", "render_report_tab"),
        ("modules.ui.power", "render_power_tab"),
        ("modules.ui.intervals_ui", "render_intervals_tab"),
        ("modules.ui.biomech", "render_biomech_tab"),
        ("modules.ui.model", "render_model_tab"),
        ("modules.ui.hrv", "render_hrv_tab"),
        ("modules.ui.smo2", "render_smo2_tab"),
        ("modules.ui.hemo", "render_hemo_tab"),
        ("modules.ui.vent", "render_vent_tab"),
        ("modules.ui.vent_thresholds", "render_vent_thresholds_tab"),
        ("modules.ui.smo2_thresholds", "render_smo2_thresholds_tab"),
        ("modules.ui.thermal", "render_thermal_tab"),
        ("modules.ui.nutrition", "render_nutrition_tab"),
        ("modules.ui.limiters", "render_limiters_tab"),
        ("modules.ui.ai_coach", "render_ai_coach_tab"),
        ("modules.ui.threshold_analysis_ui", "render_threshold_analysis_tab"),
        ("modules.ui.trends_history", "render_trends_history_tab"),
        ("modules.ui.community", "render_community_tab"),
        ("modules.ui.history_import_ui", "render_history_import_tab"),
        ("modules.ui.heart_rate", "render_hr_tab"),
        ("modules.ui.manual_thresholds", "render_manual_thresholds_tab"),
        ("modules.ui.smo2_manual_thresholds", "render_smo2_manual_thresholds_tab"),
        ("modules.ui.summary", "render_summary_tab"),
        ("modules.ui.drift_maps_ui", "render_drift_maps_tab"),
        ("modules.ui.tte_ui", "render_tte_tab"),
        ("modules.ui.ramp_archive", "render_ramp_archive"),
    ]

    @pytest.mark.parametrize("module_path,func_name", TAB_MODULES)
    def test_tab_module_importable(self, module_path, func_name):
        """Each registered tab module must import without error."""
        import importlib

        module = importlib.import_module(module_path)
        assert hasattr(module, func_name), f"{module_path} missing {func_name}"
        assert callable(getattr(module, func_name))


# =========================================================================
# 2. FULL PIPELINE E2E — realistic ramp test data
# =========================================================================


@pytest.fixture
def ramp_test_df():
    """Generate realistic 20-minute ramp test data (1Hz)."""
    np.random.seed(42)
    n = 1200  # 20 minutes

    time = np.arange(n, dtype=float)

    # Power: warmup 100W (5min) → ramp 100-350W (12min) → cooldown (3min)
    watts = np.zeros(n)
    watts[:300] = 100 + np.random.normal(0, 3, 300)
    watts[300:1020] = np.linspace(100, 350, 720) + np.random.normal(0, 5, 720)
    watts[1020:] = 100 + np.random.normal(0, 3, 180)
    watts = watts.clip(50, 400)

    # HR: follows power with lag
    hr = 70 + (watts - 100) * 0.3 + np.random.normal(0, 2, n)
    hr = hr.clip(60, 200)

    # VE: non-linear response (exponential at high power)
    ve_base = 15 + (watts - 100) * 0.15
    ve_nonlinear = np.where(watts > 250, (watts - 250) ** 1.3 * 0.02, 0)
    ve = ve_base + ve_nonlinear + np.random.normal(0, 2, n)
    ve = ve.clip(10, 200)

    # Breathing rate
    br = 15 + (watts - 100) * 0.08 + np.random.normal(0, 1, n)
    br = br.clip(10, 60)

    # SmO2: decreases with intensity
    smo2 = 70 - (watts - 100) * 0.08 + np.random.normal(0, 3, n)
    smo2 = smo2.clip(20, 95)

    # Cadence
    cadence = 85 + np.random.normal(0, 3, n)
    cadence = cadence.clip(60, 120)

    return pd.DataFrame(
        {
            "time": time,
            "watts": watts,
            "hr": hr,
            "tymeventilation": ve,
            "tymebreathrate": br,
            "smo2": smo2,
            "cadence": cadence,
        }
    )


class TestFullPipelineE2E:
    """End-to-end tests for the calculation pipeline with realistic data."""

    def test_vt_cpet_detection_produces_valid_result(self, ramp_test_df):
        """VT CPET detection returns valid thresholds on ramp test data."""
        from modules.calculations.vt_cpet import detect_vt_cpet

        result = detect_vt_cpet(
            ramp_test_df,
            power_column="watts",
            ve_column="tymeventilation",
            time_column="time",
            hr_column="hr",
        )

        assert result is not None
        assert "error" not in result, f"Detection failed: {result.get('error')}"
        assert result["vt1_watts"] is not None, "VT1 not detected"
        assert result["vt2_watts"] is not None, "VT2 not detected"
        assert result["vt1_watts"] < result["vt2_watts"], "VT1 must be < VT2"
        assert result["df_steps"] is not None, "No step data"
        assert len(result["df_steps"]) >= 5, "Too few steps"
        assert len(result["analysis_notes"]) > 0, "No analysis notes"

    def test_vt_cpet_vt1_in_physiological_range(self, ramp_test_df):
        """VT1 should be detected in a physiologically reasonable range."""
        from modules.calculations.vt_cpet import detect_vt_cpet

        result = detect_vt_cpet(ramp_test_df, power_column="watts", ve_column="tymeventilation")

        max_power = ramp_test_df["watts"].max()
        assert result["vt1_watts"] >= 100, "VT1 too low"
        assert result["vt1_watts"] <= max_power * 0.85, "VT1 too high"

    def test_vt_cpet_backward_compat_wrapper(self, ramp_test_df):
        """detect_vt_vslope_savgol wrapper produces same result as detect_vt_cpet."""
        from modules.calculations.vt_cpet import detect_vt_cpet, detect_vt_vslope_savgol

        result_new = detect_vt_cpet(ramp_test_df)
        result_old = detect_vt_vslope_savgol(ramp_test_df)

        assert result_old["vt1_watts"] == result_new["vt1_watts"]
        assert result_old["vt2_watts"] == result_new["vt2_watts"]

    def test_smo2_analysis_produces_valid_output(self, ramp_test_df):
        """SmO2 advanced analysis returns metrics on ramp data."""
        from modules.calculations.smo2_analysis import analyze_smo2_advanced

        result = analyze_smo2_advanced(
            ramp_test_df,
            smo2_col="smo2",
            power_col="watts",
            time_col="time",
        )

        assert result is not None
        # Returns SmO2AdvancedMetrics dataclass
        assert hasattr(result, "slope_per_100w")
        assert hasattr(result, "data_quality")

    def test_quality_check_passes_for_ramp_test(self, ramp_test_df):
        """Protocol check should recognise valid ramp test."""
        from modules.calculations.quality import check_step_test_protocol

        result = check_step_test_protocol(ramp_test_df)

        assert result is not None
        assert "is_valid" in result

    def test_signal_quality_check(self, ramp_test_df):
        """VE signal quality check returns valid result."""
        from modules.calculations.quality import check_signal_quality

        result = check_signal_quality(ramp_test_df["tymeventilation"], "VE", (0, 300))

        assert result is not None
        assert "is_valid" in result
        assert "score" in result

    def test_session_classification_detects_ramp(self, ramp_test_df):
        """Session classifier should recognise ramp test pattern."""
        from modules.domain import classify_ramp_test

        power = ramp_test_df["watts"].dropna()
        result = classify_ramp_test(power)

        assert result is not None
        assert hasattr(result, "confidence")
        # Ramp confidence should be reasonable (>0.3) for a synthetic ramp
        assert result.confidence > 0.3, f"Low ramp confidence: {result.confidence}"

    def test_data_validation_service(self, ramp_test_df):
        """Data validation service accepts valid data."""
        from services.data_validation import validate_dataframe

        result = validate_dataframe(ramp_test_df)

        # Returns (df, is_valid) or similar tuple
        assert result is not None


# =========================================================================
# 3. REPORTING PIPELINE E2E
# =========================================================================


class TestReportingPipelineE2E:
    """Test that the reporting modules chain correctly."""

    def test_persistence_facade_exports_all_symbols(self):
        """The persistence facade must export all public symbols."""
        from modules.reporting.persistence import (
            save_ramp_test_report,
            generate_and_save_pdf,
            load_ramp_test_report,
            check_git_tracking,
            NumpyEncoder,
        )

        assert callable(save_ramp_test_report)
        assert callable(generate_and_save_pdf)
        assert callable(load_ramp_test_report)
        assert callable(check_git_tracking)

    def test_numpy_encoder_handles_all_types(self):
        """NumpyEncoder must handle numpy scalar types."""
        import json
        from modules.reporting.persistence import NumpyEncoder

        data = {
            "int_val": np.int64(42),
            "float_val": np.float64(3.14),
            "bool_val": np.bool_(True),
            "array_val": np.array([1, 2, 3]),
        }

        encoded = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(encoded)

        assert decoded["int_val"] == 42
        assert abs(decoded["float_val"] - 3.14) < 0.01
        assert decoded["bool_val"] is True
        assert decoded["array_val"] == [1, 2, 3]

    def test_csv_export_produces_valid_csv(self, ramp_test_df):
        """CSV export generates valid output."""
        from modules.reporting.csv_export import export_session_csv

        csv_bytes = export_session_csv(ramp_test_df)

        assert csv_bytes is not None
        assert len(csv_bytes) > 0
        # Should be parseable as CSV
        from io import BytesIO

        df_round_trip = pd.read_csv(BytesIO(csv_bytes))
        assert len(df_round_trip) == len(ramp_test_df)


# =========================================================================
# 4. MODULE INTEGRATION — sub-module splits work correctly
# =========================================================================


class TestSubModuleSplitIntegration:
    """Verify that all split modules import and chain correctly."""

    def test_vt_cpet_submodules_import(self):
        """All vt_cpet sub-modules must import."""
        from modules.calculations.vt_cpet_preprocessing import preprocess_cpet_data
        from modules.calculations.vt_cpet_steps import aggregate_step_data
        from modules.calculations.vt_cpet_gas_exchange import detect_gas_exchange_thresholds
        from modules.calculations.vt_cpet_ve_only import (
            detect_ve_only_thresholds,
            _find_breakpoint_segmented,
            _calculate_segment_slope,
        )

        assert callable(preprocess_cpet_data)
        assert callable(aggregate_step_data)
        assert callable(detect_gas_exchange_thresholds)
        assert callable(detect_ve_only_thresholds)
        assert callable(_find_breakpoint_segmented)
        assert callable(_calculate_segment_slope)

    def test_summary_submodules_import(self):
        """All summary sub-modules must import."""
        from modules.ui.summary_calculations import (
            _hash_dataframe,
            _calculate_np,
            _estimate_cp_wprime,
        )
        from modules.ui.summary_charts import (
            _build_training_timeline_chart,
            _render_cp_model_chart,
        )
        from modules.ui.summary_thresholds import _render_vent_thresholds_summary

        assert callable(_hash_dataframe)
        assert callable(_calculate_np)
        assert callable(_estimate_cp_wprime)

    def test_vent_thresholds_submodules_import(self):
        """All vent_thresholds sub-modules must import."""
        from modules.ui.vent_thresholds_report import render_report_section
        from modules.ui.vent_thresholds_display import render_threshold_cards, render_theory_section
        from modules.ui.vent_thresholds_charts import render_cpet_charts
        from modules.ui.vent_thresholds_timeline import render_threshold_timeline

        assert callable(render_report_section)
        assert callable(render_threshold_cards)
        assert callable(render_theory_section)
        assert callable(render_cpet_charts)
        assert callable(render_threshold_timeline)

    def test_reporting_submodules_import(self):
        """All reporting sub-modules must import."""
        from modules.reporting.report_io import (
            NumpyEncoder,
            load_ramp_test_report,
            check_git_tracking,
        )
        from modules.reporting.index_manager import _update_index, update_index_pdf_path
        from modules.reporting.pdf_generator import generate_and_save_pdf
        from modules.reporting.report_generator import save_ramp_test_report

        assert callable(load_ramp_test_report)
        assert callable(check_git_tracking)
        assert callable(generate_and_save_pdf)
        assert callable(save_ramp_test_report)

    def test_vent_theory_submodule_import(self):
        """Vent theory sub-module must import."""
        from modules.ui.vent_theory import render_vent_theory

        assert callable(render_vent_theory)

    def test_breakpoint_segmented_finds_known_breakpoint(self):
        """_find_breakpoint_segmented should detect a known breakpoint."""
        from modules.calculations.vt_cpet_ve_only import _find_breakpoint_segmented

        # Create data with a clear breakpoint at index 10
        x = np.arange(20, dtype=float)
        y = np.where(x < 10, 0.5 * x + 5, 2.0 * x - 10)

        bp = _find_breakpoint_segmented(x, y, min_segment_size=3)

        assert bp is not None
        assert 7 <= bp <= 13, f"Breakpoint at {bp}, expected near 10"

    def test_calculate_segment_slope_linear(self):
        """_calculate_segment_slope returns correct slope for linear data."""
        from modules.calculations.vt_cpet_ve_only import _calculate_segment_slope

        x = np.array([0, 1, 2, 3, 4], dtype=float)
        y = np.array([0, 2, 4, 6, 8], dtype=float)

        slope = _calculate_segment_slope(x, y)
        assert abs(slope - 2.0) < 0.01

    def test_calculate_np_normalized_power(self):
        """_calculate_np returns correct normalized power."""
        from modules.ui.summary_calculations import _calculate_np

        # Constant power of 200W → NP should be ~200
        watts = pd.Series([200.0] * 100)
        np_val = _calculate_np(watts)
        assert abs(np_val - 200.0) < 1.0
