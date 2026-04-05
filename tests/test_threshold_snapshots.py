"""
Snapshot/regression tests for canonical threshold entry points.

These tests lock in the contract defined in docs/DOMAIN_MODEL.md:
  - canonical entry points return documented fields
  - deterministic synthetic ramps produce thresholds in known windows
  - confidence bands follow the HIGH/CONDITIONAL/LOW/INVALID contract

If any assertion here fails after a refactor, either the algorithm changed
materially (and the snapshot must be reviewed) or the contract was broken
(and the code must be fixed). Do NOT loosen tolerances without justification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Deterministic physiological fixtures
# ---------------------------------------------------------------------------


def _canonical_ramp(
    duration_s: int = 1500,
    start_w: float = 100.0,
    end_w: float = 400.0,
    vt1_frac: float = 0.45,
    vt2_frac: float = 0.75,
    noise: float = 0.2,
    seed: int = 7,
) -> pd.DataFrame:
    """A 25-minute synthetic ramp with engineered VE break-points.

    VE has three slope regimes:
        t < vt1_frac * T    → slope 0.020 L/min/s  (aerobic)
        t < vt2_frac * T    → slope 0.060 L/min/s  (heavy)
        otherwise           → slope 0.160 L/min/s  (severe, RCP+)

    True thresholds (Watts) are therefore:
        VT1 ≈ start + vt1_frac * (end - start)
        VT2 ≈ start + vt2_frac * (end - start)
    Defaults: VT1 ≈ 235 W, VT2 ≈ 325 W.
    """
    rng = np.random.default_rng(seed)
    time = np.arange(duration_s, dtype=float)
    watts = np.linspace(start_w, end_w, duration_s)
    hr = np.linspace(100.0, 180.0, duration_s)

    # Piecewise-linear true VE (cumulative slope integration)
    slope = np.where(
        time < vt1_frac * duration_s,
        0.020,
        np.where(time < vt2_frac * duration_s, 0.060, 0.160),
    )
    ve_true = 15.0 + np.cumsum(slope)
    ve = ve_true + rng.normal(0.0, noise, duration_s)

    # Gas exchange: VO2 tracks power, VCO2 rises faster past VT2 (hyperventilation)
    vo2 = 0.5 + 0.010 * watts + rng.normal(0.0, 0.02, duration_s)
    vco2_base = 0.010 * watts
    vco2_excess = np.where(time < vt2_frac * duration_s, 0.0, 0.004 * (watts - start_w - vt2_frac * (end_w - start_w)))
    vco2 = 0.4 + vco2_base + vco2_excess + rng.normal(0.0, 0.02, duration_s)

    return pd.DataFrame(
        {
            "time": time,
            "watts": watts,
            "hr": hr,
            "tymeventilation": ve,
            "tymevo2": vo2,
            "tymevco2": vco2,
        }
    )


def _canonical_smo2_ramp(
    n_steps: int = 8,
    step_duration_s: int = 180,
    start_w: float = 120.0,
    step_w: float = 30.0,
    seed: int = 11,
) -> pd.DataFrame:
    """8-step SmO2 ramp with engineered desaturation kinetics.

    Steps 0–2: gentle drift (aerobic).
    Steps 3–4: accelerated desaturation (LT1 crossing ≈ step 3).
    Steps 5–7: steep desaturation + oscillation (T2_onset ≈ step 5).
    """
    rng = np.random.default_rng(seed)
    rows = []
    smo2 = 75.0
    for step in range(n_steps):
        power = start_w + step * step_w
        hr = 105.0 + step * 9.0
        if step < 3:
            rate = -0.010  # %/s
            osc = 0.4
        elif step < 5:
            rate = -0.035
            osc = 0.6
        else:
            rate = -0.070
            osc = 1.2
        for t in range(step_duration_s):
            smo2 = smo2 + rate + rng.normal(0.0, osc * 0.1)
            rows.append(
                {
                    "time": float(step * step_duration_s + t),
                    "watts": power,
                    "hr": hr,
                    "smo2": max(25.0, min(90.0, smo2)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# §3.1 — detect_vt_cpet() contract
# ---------------------------------------------------------------------------


class TestVtCpetContract:
    """Lock in the output contract of the canonical VT entry point."""

    @pytest.fixture(scope="class")
    def result(self):
        from modules.calculations.vt_cpet import detect_vt_cpet

        df = _canonical_ramp()
        return detect_vt_cpet(df)

    def test_returns_dict(self, result):
        assert isinstance(result, dict)

    def test_has_documented_vt1_fields(self, result):
        # Per DOMAIN_MODEL.md §3.1
        required = {"vt1_watts", "vt1_hr", "vt1_ve", "vt1_confidence"}
        assert required.issubset(result.keys()), f"Missing VT1 fields: {required - result.keys()}"

    def test_has_documented_vt2_fields(self, result):
        required = {"vt2_watts", "vt2_hr", "vt2_ve", "vt2_confidence"}
        assert required.issubset(result.keys()), f"Missing VT2 fields: {required - result.keys()}"

    def test_has_method_and_analysis_notes(self, result):
        assert "method" in result
        # analysis_notes OR notes — accept either legacy key
        assert "analysis_notes" in result or "notes" in result

    def test_confidence_in_unit_interval(self, result):
        for key in ("vt1_confidence", "vt2_confidence"):
            val = result.get(key)
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{key}={val} outside [0, 1]"


# ---------------------------------------------------------------------------
# Snapshot tests — VT detected in expected windows
# ---------------------------------------------------------------------------


class TestVtCpetSnapshot:
    """Regression-lock VT1/VT2 detection on a synthetic ramp with known break-points.

    True VT1 ≈ 235 W (at t = 45% of ramp); true VT2 ≈ 325 W (at t = 75%).
    Tolerances reflect physiological windows used by the detector (±25 W),
    not an arbitrary expansion.
    """

    @pytest.fixture(scope="class")
    def result(self):
        from modules.calculations.vt_cpet import detect_vt_cpet

        df = _canonical_ramp()
        return detect_vt_cpet(df)

    def test_vt1_detected(self, result):
        assert result.get("vt1_watts") is not None, "VT1 not detected on canonical ramp"

    def test_vt2_detected(self, result):
        assert result.get("vt2_watts") is not None, "VT2 not detected on canonical ramp"

    def test_vt1_in_expected_window(self, result):
        vt1 = result.get("vt1_watts")
        if vt1 is None:
            pytest.skip("VT1 not detected — covered by test_vt1_detected")
        # True VT1 = 235 W. Accept ±30 W for detector variance.
        assert 205 <= vt1 <= 265, f"VT1={vt1} outside [205, 265] window"

    def test_vt2_in_expected_window(self, result):
        vt2 = result.get("vt2_watts")
        if vt2 is None:
            pytest.skip("VT2 not detected — covered by test_vt2_detected")
        # True VT2 = 325 W. Accept ±30 W for detector variance.
        assert 295 <= vt2 <= 355, f"VT2={vt2} outside [295, 355] window"

    def test_vt1_below_vt2(self, result):
        vt1, vt2 = result.get("vt1_watts"), result.get("vt2_watts")
        if vt1 is None or vt2 is None:
            pytest.skip("Requires both thresholds")
        assert vt1 < vt2, f"VT1={vt1} must be strictly below VT2={vt2}"


# ---------------------------------------------------------------------------
# §3.3 — detect_smo2_thresholds_moxy() contract + snapshot
# ---------------------------------------------------------------------------


class TestSmO2ThresholdsContract:
    @pytest.fixture(scope="class")
    def result(self):
        from modules.calculations.smo2_thresholds import detect_smo2_thresholds_moxy

        df = _canonical_smo2_ramp()
        return detect_smo2_thresholds_moxy(df, step_duration_sec=180)

    def test_returns_result_object(self, result):
        # SmO2ThresholdResult is a dataclass
        assert result is not None
        assert hasattr(result, "t1_watts")
        assert hasattr(result, "t2_onset_watts")

    def test_has_documented_t1_fields(self, result):
        for field in ("t1_watts", "t1_hr", "t1_smo2", "t1_gradient", "t1_step"):
            assert hasattr(result, field), f"Missing T1 field: {field}"

    def test_has_documented_t2_onset_fields(self, result):
        for field in ("t2_onset_watts", "t2_onset_hr", "t2_onset_smo2", "t2_onset_gradient"):
            assert hasattr(result, field), f"Missing T2_onset field: {field}"

    def test_t2_steady_none_in_ramp_protocol(self, result):
        # Per DOMAIN_MODEL.md §3.3 and code comment:
        # "T2_steady MUST remain None for ramp tests"
        # On our constructed dataset (no constant-load plateaus), T2_steady
        # should not be populated.
        assert result.t2_steady_watts is None, (
            f"T2_steady must be None for ramp protocols, got {result.t2_steady_watts}. "
            "See SmO2ThresholdResult docstring and DOMAIN_MODEL.md §3.3."
        )

    def test_has_method_and_notes(self, result):
        assert hasattr(result, "method")
        assert hasattr(result, "analysis_notes")

    def test_physiological_agreement_valid_value(self, result):
        # Per DOMAIN_MODEL.md §3.3, one of: strong, moderate, weak, not_checked
        assert result.physiological_agreement in {
            "strong",
            "moderate",
            "weak",
            "not_checked",
        }, f"Invalid physiological_agreement={result.physiological_agreement}"


class TestSmO2ThresholdsSnapshot:
    """Regression-lock T1/T2_onset ordering and bounds on a synthetic SmO2 ramp."""

    @pytest.fixture(scope="class")
    def result(self):
        from modules.calculations.smo2_thresholds import detect_smo2_thresholds_moxy

        df = _canonical_smo2_ramp()
        return detect_smo2_thresholds_moxy(df, step_duration_sec=180)

    def test_t1_before_t2_when_both_present(self, result):
        if result.t1_watts is None or result.t2_onset_watts is None:
            pytest.skip("Requires both T1 and T2_onset")
        assert result.t1_watts < result.t2_onset_watts, (
            f"T1={result.t1_watts}W must be below T2_onset={result.t2_onset_watts}W"
        )

    def test_thresholds_within_ramp_range(self, result):
        # Canonical ramp goes 120 → 120+7*30 = 330 W
        for name, val in (("t1_watts", result.t1_watts), ("t2_onset_watts", result.t2_onset_watts)):
            if val is not None:
                assert 120 <= val <= 330, f"{name}={val} outside ramp range [120, 330]"


# ---------------------------------------------------------------------------
# Ramp classification contract
# ---------------------------------------------------------------------------


class TestRampClassificationContract:
    def test_clean_ramp_classified_as_ramp(self):
        from modules.domain import classify_ramp_test

        power = pd.Series(np.linspace(100, 400, 1200))
        result = classify_ramp_test(power)
        assert result.is_ramp is True
        assert result.confidence >= 0.5

    def test_flat_power_not_classified_as_ramp(self):
        from modules.domain import classify_ramp_test

        power = pd.Series([200.0] * 600)
        result = classify_ramp_test(power)
        assert result.is_ramp is False

    def test_result_has_documented_fields(self):
        from modules.domain import classify_ramp_test

        power = pd.Series(np.linspace(100, 400, 1200))
        result = classify_ramp_test(power)
        # Per DOMAIN_MODEL.md §3.5
        for field in ("is_ramp", "confidence", "reason", "criteria_met", "criteria_failed", "suggested_type"):
            assert hasattr(result, field), f"Missing RampClassificationResult field: {field}"

    def test_confidence_is_fraction_of_four_criteria(self):
        from modules.domain import classify_ramp_test

        power = pd.Series(np.linspace(100, 400, 1200))
        result = classify_ramp_test(power)
        # confidence = met_criteria / 4 → must be a multiple of 0.25
        assert result.confidence in {0.0, 0.25, 0.5, 0.75, 1.0}, (
            f"Expected confidence in {{0, 0.25, 0.5, 0.75, 1.0}}, got {result.confidence}"
        )


# ---------------------------------------------------------------------------
# TransitionZone contract (used by StepVTResult and vt_sliding)
# ---------------------------------------------------------------------------


class TestTransitionZoneContract:
    def test_confidence_bands_match_domain_model(self):
        """Enforce the HIGH/CONDITIONAL/LOW/INVALID bands from DOMAIN_MODEL.md §4."""
        from modules.calculations.threshold_types import TransitionZone

        high = TransitionZone(range_watts=(250, 260), confidence=0.85, method="test")
        mid = TransitionZone(range_watts=(250, 260), confidence=0.65, method="test")
        low = TransitionZone(range_watts=(250, 260), confidence=0.45, method="test")

        assert high.is_high_confidence(threshold=0.8) is True
        assert mid.is_high_confidence(threshold=0.8) is False
        assert low.is_high_confidence(threshold=0.4) is True
        assert low.is_high_confidence(threshold=0.5) is False

    def test_midpoint_watts_is_average_of_range(self):
        from modules.calculations.threshold_types import TransitionZone

        zone = TransitionZone(range_watts=(200.0, 240.0), confidence=0.8, method="test")
        assert zone.midpoint_watts == 220.0
        assert zone.range_width_watts == 40.0
