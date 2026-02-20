"""Tests for SmO2 subpackage, vent_advanced, and smo2_breakpoints modules."""

import pytest
import numpy as np
import pandas as pd


# =========================================================================
# SmO2 subpackage (types, calculator, classifier, __init__)
# =========================================================================

class TestSmO2Types:
    def test_advanced_metrics_defaults(self):
        from modules.calculations.smo2.types import SmO2AdvancedMetrics
        m = SmO2AdvancedMetrics()
        assert m.slope_per_100w == 0.0
        assert m.data_quality == "unknown"

    def test_advanced_metrics_to_dict(self):
        from modules.calculations.smo2.types import SmO2AdvancedMetrics
        m = SmO2AdvancedMetrics(slope_per_100w=-5.5, limiter_type="local")
        d = m.to_dict()
        assert d["slope_per_100w"] == -5.5
        assert d["limiter_type"] == "local"

    def test_threshold_result_defaults(self):
        from modules.calculations.smo2.types import SmO2ThresholdResult
        r = SmO2ThresholdResult()
        assert r.t1_watts is None
        assert r.method == "moxy_3point"
        assert r.zones == []


class TestSmO2Calculator:
    @pytest.fixture
    def smo2_df(self):
        n = 200
        np.random.seed(42)
        watts = np.linspace(100, 400, n)
        smo2 = 70 - watts * 0.08 + np.random.normal(0, 2, n)
        hr = 100 + watts * 0.2 + np.random.normal(0, 2, n)
        return pd.DataFrame({
            "watts": watts,
            "SmO2": smo2.clip(20, 95),
            "hr": hr.clip(60, 200),
            "seconds": np.arange(n, dtype=float),
        })

    def test_calculate_smo2_slope(self, smo2_df):
        from modules.calculations.smo2.calculator import SmO2MetricsCalculator
        slope, r2 = SmO2MetricsCalculator.calculate_smo2_slope(smo2_df)
        assert slope < 0  # SmO2 should decrease with power
        assert 0 <= r2 <= 1

    def test_calculate_smo2_slope_no_power(self):
        from modules.calculations.smo2.calculator import SmO2MetricsCalculator
        df = pd.DataFrame({"SmO2": [70, 65, 60]})
        slope, r2 = SmO2MetricsCalculator.calculate_smo2_slope(df)
        assert slope == 0.0

    def test_calculate_hr_coupling_index(self, smo2_df):
        from modules.calculations.smo2.calculator import SmO2MetricsCalculator
        coupling = SmO2MetricsCalculator.calculate_hr_coupling_index(smo2_df)
        assert -1 <= coupling <= 1

    def test_calculate_smo2_drift(self, smo2_df):
        from modules.calculations.smo2.calculator import SmO2MetricsCalculator
        drift = SmO2MetricsCalculator.calculate_smo2_drift(smo2_df)
        assert isinstance(drift, float)

    def test_halftime_no_recovery(self):
        from modules.calculations.smo2.calculator import SmO2MetricsCalculator
        # Flat power — no peak/recovery pattern
        df = pd.DataFrame({
            "watts": [200.0] * 100,
            "SmO2": [65.0] * 100,
            "seconds": np.arange(100, dtype=float),
        })
        result = SmO2MetricsCalculator.calculate_halftime_reoxygenation(df)
        assert result is None


class TestSmO2Classifier:
    def test_classify_local_limiter(self):
        from modules.calculations.smo2.classifier import SmO2LimiterClassifier
        from modules.calculations.smo2.types import SmO2AdvancedMetrics
        m = SmO2AdvancedMetrics(
            slope_per_100w=-12.0, halftime_reoxy_sec=90.0,
            hr_coupling_r=-0.2, drift_pct=-10.0
        )
        limiter, confidence, interp = SmO2LimiterClassifier.classify(m)
        assert limiter == "local"
        assert confidence > 0

    def test_classify_central_limiter(self):
        from modules.calculations.smo2.classifier import SmO2LimiterClassifier
        from modules.calculations.smo2.types import SmO2AdvancedMetrics
        m = SmO2AdvancedMetrics(
            slope_per_100w=-2.0, halftime_reoxy_sec=30.0,
            hr_coupling_r=-0.9, drift_pct=-3.0
        )
        limiter, confidence, interp = SmO2LimiterClassifier.classify(m)
        assert limiter == "central"

    def test_get_recommendations(self):
        from modules.calculations.smo2.classifier import SmO2LimiterClassifier
        recs = SmO2LimiterClassifier.get_recommendations("local")
        assert isinstance(recs, tuple)
        assert len(recs) > 0

    def test_get_recommendations_unknown(self):
        from modules.calculations.smo2.classifier import SmO2LimiterClassifier
        recs = SmO2LimiterClassifier.get_recommendations("unknown")
        assert isinstance(recs, tuple)


class TestSmO2Init:
    def test_analyze_smo2_advanced_no_smo2(self):
        from modules.calculations.smo2 import analyze_smo2_advanced
        df = pd.DataFrame({"watts": [200, 250, 300]})
        result = analyze_smo2_advanced(df)
        assert result.data_quality == "no_smo2"

    def test_analyze_smo2_advanced_no_power(self):
        from modules.calculations.smo2 import analyze_smo2_advanced
        df = pd.DataFrame({"SmO2": [70, 65, 60]})
        result = analyze_smo2_advanced(df)
        assert result.data_quality == "no_power"

    def test_analyze_smo2_advanced_full(self):
        from modules.calculations.smo2 import analyze_smo2_advanced
        n = 200
        np.random.seed(42)
        df = pd.DataFrame({
            "watts": np.linspace(100, 400, n),
            "SmO2": 70 - np.linspace(0, 30, n) + np.random.normal(0, 1, n),
            "hr": 100 + np.linspace(0, 60, n) + np.random.normal(0, 1, n),
            "seconds": np.arange(n, dtype=float),
        })
        result = analyze_smo2_advanced(df)
        assert result.limiter_type in ("local", "central", "metabolic", "unknown")
        assert result.slope_per_100w != 0.0


# =========================================================================
# vent_advanced
# =========================================================================

class TestVentAdvanced:
    @pytest.fixture
    def vent_df(self):
        n = 300
        np.random.seed(42)
        watts = np.linspace(100, 350, n) + np.random.normal(0, 3, n)
        ve = 15 + (watts - 100) * 0.15 + np.random.normal(0, 1, n)
        rr = 15 + (watts - 100) * 0.08 + np.random.normal(0, 0.5, n)
        return pd.DataFrame({
            "watts": watts.clip(50, 400),
            "tymeventilation": ve.clip(10, 200),
            "tymebreathrate": rr.clip(10, 60),
            "time": np.arange(n, dtype=float),
        })

    def test_calculate_ve_metrics(self, vent_df):
        from modules.calculations.vent_advanced import calculate_ve_metrics
        ve_avg, ve_max, ve_slope = calculate_ve_metrics(vent_df)
        assert ve_avg > 0
        assert ve_max > ve_avg
        assert ve_slope > 0  # VE should increase with power

    def test_calculate_ve_metrics_no_ve(self):
        from modules.calculations.vent_advanced import calculate_ve_metrics
        df = pd.DataFrame({"watts": [200, 250, 300]})
        assert calculate_ve_metrics(df) == (0.0, 0.0, 0.0)

    def test_calculate_rr_metrics(self, vent_df):
        from modules.calculations.vent_advanced import calculate_rr_metrics
        rr_avg, rr_max = calculate_rr_metrics(vent_df)
        assert rr_avg > 0
        assert rr_max >= rr_avg

    def test_calculate_ve_rr_ratio(self):
        from modules.calculations.vent_advanced import calculate_ve_rr_ratio
        assert calculate_ve_rr_ratio(60.0, 20.0) == pytest.approx(3.0)
        assert calculate_ve_rr_ratio(60.0, 0.0) == 0.0

    def test_classify_breathing_pattern_efficient(self):
        from modules.calculations.vent_advanced import classify_breathing_pattern
        pattern, desc = classify_breathing_pattern(rr_avg=25, rr_max=40, ve_rr_ratio=3.0, ve_slope=0.2)
        assert pattern == "efficient"

    def test_classify_breathing_pattern_shallow(self):
        from modules.calculations.vent_advanced import classify_breathing_pattern
        pattern, desc = classify_breathing_pattern(rr_avg=45, rr_max=55, ve_rr_ratio=1.2, ve_slope=0.3)
        assert pattern == "shallow"

    def test_classify_breathing_pattern_hyperventilation(self):
        from modules.calculations.vent_advanced import classify_breathing_pattern
        pattern, desc = classify_breathing_pattern(rr_avg=30, rr_max=40, ve_rr_ratio=2.5, ve_slope=0.6)
        assert pattern == "hyperventilation"

    def test_classify_ventilatory_control(self):
        from modules.calculations.vent_advanced import classify_ventilatory_control, VentilationMetrics
        m = VentilationMetrics(
            ve_avg=50, ve_max=120, rr_avg=25, rr_max=40,
            ve_rr_ratio=3.0, ve_slope=0.2, breathing_pattern="efficient"
        )
        status, confidence, interp = classify_ventilatory_control(m)
        assert status in ("controlled", "compensatory", "unstable")
        assert 0 <= confidence <= 1

    def test_find_ve_breakpoint(self, vent_df):
        from modules.calculations.vent_advanced import find_ve_breakpoint
        bp = find_ve_breakpoint(vent_df)
        # May return None if breakpoint not clear enough
        assert bp is None or bp > 100


# =========================================================================
# smo2_breakpoints
# =========================================================================

class TestSmO2Breakpoints:
    def test_detect_breakpoints_valid_data(self):
        from modules.calculations.smo2_breakpoints import detect_smo2_breakpoints_segmented, SmO2Breakpoints
        n = 200
        np.random.seed(42)
        watts = np.linspace(100, 400, n)
        # Three phases: flat, moderate decline, steep decline
        smo2 = np.piecewise(
            watts,
            [watts < 250, (watts >= 250) & (watts < 350), watts >= 350],
            [lambda x: 70 - 0.02 * (x - 100),
             lambda x: 67 - 0.1 * (x - 250),
             lambda x: 57 - 0.3 * (x - 350)]
        ) + np.random.normal(0, 1, n)

        df = pd.DataFrame({"watts": watts, "smo2": smo2})
        result = detect_smo2_breakpoints_segmented(df)
        assert isinstance(result, SmO2Breakpoints)

    def test_detect_breakpoints_insufficient_data(self):
        from modules.calculations.smo2_breakpoints import detect_smo2_breakpoints_segmented
        df = pd.DataFrame({"watts": [200, 250], "smo2": [70, 65]})
        result = detect_smo2_breakpoints_segmented(df)
        assert not result.is_valid

    def test_detect_breakpoints_forced(self):
        from modules.calculations.smo2_breakpoints import detect_smo2_breakpoints_segmented
        n = 200
        watts = np.linspace(100, 400, n)
        smo2 = 70 - watts * 0.05 + np.random.normal(0, 1, n)
        df = pd.DataFrame({"watts": watts, "smo2": smo2})
        result = detect_smo2_breakpoints_segmented(df, force_bp1=250.0, force_bp2=350.0)
        assert isinstance(result.bp1_power, (float, type(None)))


# =========================================================================
# smo2/constants
# =========================================================================

class TestSmO2Constants:
    def test_constants_exist(self):
        from modules.calculations.smo2.constants import LIMITER_THRESHOLDS, RECOMMENDATIONS
        assert "slope_severe" in LIMITER_THRESHOLDS
        assert "local" in RECOMMENDATIONS
        assert "central" in RECOMMENDATIONS
        assert "unknown" in RECOMMENDATIONS
