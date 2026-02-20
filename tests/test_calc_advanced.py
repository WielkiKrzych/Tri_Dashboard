"""Tests for advanced calculation modules at 0% coverage.

Covers:
- cardiac_drift.py
- thermoregulation.py
- biomech_occlusion.py
- cardio_advanced.py
- gas_exchange_estimation.py
"""

import pytest
import numpy as np
import pandas as pd


# =========================================================================
# cardiac_drift
# =========================================================================

class TestCardiacDrift:
    def test_calculate_efficiency_factor(self):
        from modules.calculations.cardiac_drift import calculate_efficiency_factor
        power = np.array([200.0, 250.0, 300.0])
        hr = np.array([120.0, 140.0, 160.0])
        ef = calculate_efficiency_factor(power, hr)
        assert ef[0] == pytest.approx(200 / 120)
        assert ef[1] == pytest.approx(250 / 140)

    def test_ef_low_hr_returns_nan(self):
        from modules.calculations.cardiac_drift import calculate_efficiency_factor
        power = np.array([200.0])
        hr = np.array([30.0])  # Below 50 threshold
        ef = calculate_efficiency_factor(power, hr)
        assert np.isnan(ef[0])

    def test_analyze_cardiac_drift_basic(self):
        from modules.calculations.cardiac_drift import analyze_cardiac_drift, CardiacDriftProfile
        n = 600
        time = np.arange(n, dtype=float)
        power = np.full(n, 200.0)
        # HR drifts upward over time
        hr = 120 + np.linspace(0, 20, n) + np.random.normal(0, 1, n)
        hr = hr.clip(60, 200)

        profile = analyze_cardiac_drift(power, hr, time)
        assert isinstance(profile, CardiacDriftProfile)
        assert profile.ef_start > 0
        assert profile.drift_classification in ("minimal", "moderate", "high")

    def test_analyze_cardiac_drift_with_temp(self):
        from modules.calculations.cardiac_drift import analyze_cardiac_drift
        n = 600
        time = np.arange(n, dtype=float)
        power = np.full(n, 200.0)
        hr = 120 + np.linspace(0, 15, n)
        core_temp = 37.0 + np.linspace(0, 1.5, n)

        profile = analyze_cardiac_drift(power, hr, time, core_temp=core_temp)
        assert profile.data_points > 0

    def test_analyze_cardiac_drift_short_data(self):
        from modules.calculations.cardiac_drift import analyze_cardiac_drift
        power = np.array([200.0, 200.0])
        hr = np.array([120.0, 125.0])
        time = np.array([0.0, 1.0])
        profile = analyze_cardiac_drift(power, hr, time)
        assert profile.confidence == 0.0


# =========================================================================
# thermoregulation
# =========================================================================

class TestThermoregulation:
    def test_analyze_thermoregulation_basic(self):
        from modules.calculations.thermoregulation import analyze_thermoregulation, ThermoProfile
        n = 600
        time = np.arange(n, dtype=float)
        core_temp = 37.0 + np.linspace(0, 1.5, n) + np.random.normal(0, 0.05, n)

        profile = analyze_thermoregulation(core_temp, time)
        assert isinstance(profile, ThermoProfile)
        assert profile.max_core_temp > 37.0
        assert profile.delta_core_temp > 0

    def test_analyze_thermoregulation_with_hr_power(self):
        from modules.calculations.thermoregulation import analyze_thermoregulation
        n = 600
        time = np.arange(n, dtype=float)
        core_temp = 37.0 + np.linspace(0, 2.0, n)
        hr = 120 + np.linspace(0, 30, n)
        power = np.full(n, 200.0)

        profile = analyze_thermoregulation(core_temp, time, hr=hr, power=power)
        assert profile.data_points > 0

    def test_analyze_thermoregulation_short_data(self):
        from modules.calculations.thermoregulation import analyze_thermoregulation
        core_temp = np.array([37.0, 37.1])
        time = np.array([0.0, 1.0])
        profile = analyze_thermoregulation(core_temp, time)
        assert profile.confidence == 0.0

    def test_thresholds_detected(self):
        from modules.calculations.thermoregulation import analyze_thermoregulation
        n = 1200
        time = np.arange(n, dtype=float)
        # Temp rising from 36.5 to 39.0 — should cross 38.0 and 38.5
        core_temp = 36.5 + np.linspace(0, 2.5, n)

        profile = analyze_thermoregulation(core_temp, time)
        assert profile.time_to_38_0 is not None or profile.max_core_temp >= 38.0


# =========================================================================
# biomech_occlusion
# =========================================================================

class TestBiomechOcclusion:
    def test_analyze_biomech_occlusion_basic(self):
        from modules.calculations.biomech_occlusion import analyze_biomech_occlusion, OcclusionProfile
        n = 100
        np.random.seed(42)
        torque = np.linspace(20, 80, n) + np.random.normal(0, 2, n)
        # SmO2 decreases as torque increases (occlusion)
        smo2 = 70 - torque * 0.3 + np.random.normal(0, 2, n)
        smo2 = smo2.clip(20, 95)

        profile = analyze_biomech_occlusion(torque, smo2)
        assert isinstance(profile, OcclusionProfile)
        assert profile.data_points > 0
        assert profile.regression_slope != 0

    def test_analyze_biomech_occlusion_short_data(self):
        from modules.calculations.biomech_occlusion import analyze_biomech_occlusion
        torque = np.array([20.0, 30.0])
        smo2 = np.array([70.0, 65.0])
        profile = analyze_biomech_occlusion(torque, smo2)
        assert "Niewystarczające" in profile.mechanism_description

    def test_analyze_with_cadence(self):
        from modules.calculations.biomech_occlusion import analyze_biomech_occlusion
        n = 100
        torque = np.linspace(20, 80, n)
        smo2 = 70 - torque * 0.3
        cadence = np.full(n, 90.0)
        profile = analyze_biomech_occlusion(torque, smo2, cadence=cadence)
        assert profile.data_points > 0


# =========================================================================
# cardio_advanced
# =========================================================================

class TestCardioAdvanced:
    @pytest.fixture
    def cardio_df(self):
        n = 600
        np.random.seed(42)
        watts = np.linspace(100, 350, n) + np.random.normal(0, 5, n)
        hr = 100 + (watts - 100) * 0.3 + np.random.normal(0, 2, n)
        return pd.DataFrame({
            "watts": watts.clip(50, 400),
            "hr": hr.clip(60, 200),
            "time": np.arange(n, dtype=float),
        })

    def test_calculate_pulse_power(self, cardio_df):
        from modules.calculations.cardio_advanced import calculate_pulse_power
        avg_pp, pp_series = calculate_pulse_power(cardio_df)
        assert avg_pp > 0
        assert len(pp_series) == len(cardio_df)

    def test_calculate_efficiency_factor(self, cardio_df):
        from modules.calculations.cardio_advanced import calculate_efficiency_factor
        ef = calculate_efficiency_factor(cardio_df)
        assert ef > 0

    def test_calculate_hr_drift(self, cardio_df):
        from modules.calculations.cardio_advanced import calculate_hr_drift
        drift = calculate_hr_drift(cardio_df)
        assert isinstance(drift, float)

    def test_calculate_hr_recovery(self, cardio_df):
        from modules.calculations.cardio_advanced import calculate_hr_recovery
        recovery = calculate_hr_recovery(cardio_df)
        # May be None if no clear recovery detected
        assert recovery is None or isinstance(recovery, float)

    def test_calculate_cci(self, cardio_df):
        from modules.calculations.cardio_advanced import calculate_cci
        avg_cci, bp_watts, bp_hr, cci_profile = calculate_cci(cardio_df)
        assert isinstance(avg_cci, float)
        assert isinstance(cci_profile, list)

    def test_analyze_cardiovascular(self, cardio_df):
        from modules.calculations.cardio_advanced import analyze_cardiovascular
        metrics = analyze_cardiovascular(cardio_df)
        assert hasattr(metrics, "pulse_power")
        assert hasattr(metrics, "hr_drift_pct")


# =========================================================================
# gas_exchange_estimation
# =========================================================================

class TestGasExchangeEstimation:
    def test_estimate_vo2_from_power(self):
        from modules.calculations.gas_exchange_estimation import estimate_vo2_from_power
        power = np.array([100.0, 200.0, 300.0])
        vo2 = estimate_vo2_from_power(power)
        assert len(vo2) == 3
        assert vo2[1] > vo2[0]  # VO2 increases with power

    def test_estimate_vo2_from_hr(self):
        from modules.calculations.gas_exchange_estimation import estimate_vo2_from_hr
        hr = np.array([100.0, 140.0, 180.0])
        vo2 = estimate_vo2_from_hr(hr)
        assert len(vo2) == 3
        assert vo2[1] > vo2[0]

    def test_estimate_vco2_from_vo2(self):
        from modules.calculations.gas_exchange_estimation import estimate_vco2_from_vo2
        vo2 = np.array([1000.0, 2000.0, 3000.0])
        vco2 = estimate_vco2_from_vo2(vo2)
        assert len(vco2) == 3
        assert np.all(vco2 > 0)

    def test_estimate_vco2_with_rer(self):
        from modules.calculations.gas_exchange_estimation import estimate_vco2_from_vo2
        vo2 = np.array([2000.0, 2000.0])
        rer = np.array([0.85, 1.1])
        vco2 = estimate_vco2_from_vo2(vo2, rer=rer)
        assert vco2[1] > vco2[0]  # Higher RER → higher VCO2
