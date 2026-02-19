"""
Ventilatory Threshold — CPET-grade detection (V2.0, laboratory standard).

Orchestrates the full CPET pipeline:
  vt_cpet_preprocessing  — smoothing, unit normalisation, artifact removal
  vt_cpet_steps          — per-step steady-state aggregation
  vt_cpet_gas_exchange   — VE/VO2 + VE/VCO2 breakpoint detection (when VO2/VCO2 available)
  vt_cpet_ve_only        — 4-point VE-only detection + 4-domain zone construction

Also contains detect_vt_vslope_savgol() — deprecated wrapper for backward compatibility.
"""

from typing import Optional, Any
import numpy as np
import pandas as pd

from .vt_cpet_preprocessing import preprocess_cpet_data
from .vt_cpet_steps import aggregate_step_data
from .vt_cpet_gas_exchange import detect_gas_exchange_thresholds
from .vt_cpet_ve_only import detect_ve_only_thresholds


def detect_vt_vslope_savgol(
    df: pd.DataFrame,
    step_range: Optional[Any] = None,
    power_column: str = "watts",
    ve_column: str = "tymeventilation",
    time_column: str = "time",
    min_power_watts: Optional[int] = None,
) -> dict:
    """
    DEPRECATED: Use detect_vt_cpet() for CPET-grade detection.
    This wrapper calls the new function for backward compatibility.
    """
    return detect_vt_cpet(
        df, step_range, power_column, ve_column, time_column, min_power_watts=min_power_watts
    )


def detect_vt_cpet(
    df: pd.DataFrame,
    step_range: Optional[Any] = None,
    power_column: str = "watts",
    ve_column: str = "tymeventilation",
    time_column: str = "time",
    vo2_column: str = "tymevo2",
    vco2_column: str = "tymevco2",
    hr_column: str = "hr",
    step_duration_sec: int = 180,
    smoothing_window_sec: int = 25,
    min_power_watts: Optional[int] = None,
) -> dict:
    """
    CPET-Grade VT1/VT2 Detection using Ventilatory Equivalents.

    See sub-modules for algorithm details:
      - vt_cpet_preprocessing: smoothing + unit normalisation
      - vt_cpet_steps: steady-state step aggregation
      - vt_cpet_gas_exchange: VE/VO2 + VE/VCO2 breakpoint method
      - vt_cpet_ve_only: 4-point VE-only + zone construction

    Returns:
        dict with vt1_watts, vt2_watts, metabolic_zones, df_steps, analysis_notes, etc.
    """
    result = {
        "vt1_watts": None,
        "vt2_watts": None,
        "vt1_hr": None,
        "vt2_hr": None,
        "vt1_ve": None,
        "vt2_ve": None,
        "vt1_br": None,
        "vt2_br": None,
        "vt1_vo2": None,
        "vt2_vo2": None,
        "vt1_step": None,
        "vt2_step": None,
        "vt1_pct_vo2max": None,
        "vt2_pct_vo2max": None,
        "df_steps": None,
        "method": "cpet_segmented_regression",
        "has_gas_exchange": False,
        "analysis_notes": [],
        "validation": {"vt1_lt_vt2": False, "ve_vo2_rises_first": False},
        "ramp_start_step": None,
    }

    cols = {
        "power": power_column.lower(),
        "ve": ve_column.lower(),
        "time": time_column.lower(),
        "vo2": vo2_column.lower(),
        "vco2": vco2_column.lower(),
        "hr": hr_column.lower(),
    }

    # 1. Preprocess (copy, validate, normalise units, smooth, remove artifacts)
    data, has_vo2, has_vco2, has_hr = preprocess_cpet_data(
        df, cols, smoothing_window_sec, result
    )
    if result.get("error"):
        return result

    # 2. Aggregate per-step steady-state values
    df_steps = aggregate_step_data(
        data, cols, step_range, min_power_watts, has_vo2, has_vco2, has_hr, result
    )
    if df_steps is None:
        return result
    result["df_steps"] = df_steps

    # 3. Detect thresholds
    if has_vo2 and has_vco2:
        detect_gas_exchange_thresholds(df_steps, result)
    else:
        detect_ve_only_thresholds(df_steps, data, cols, result)

    # 4. Global fallback defaults (both paths)
    if result["vt1_watts"] is None:
        vt1_power = int(np.percentile(df_steps["power"].values, 60))
        result["vt1_watts"] = vt1_power
        result["analysis_notes"].append(f"VT1 not detected - using 60th percentile ({vt1_power}W)")

    if result["vt2_watts"] is None:
        vt2_power = int(np.percentile(df_steps["power"].values, 80))
        result["vt2_watts"] = vt2_power
        result["analysis_notes"].append(f"VT2 not detected - using 80th percentile ({vt2_power}W)")

    # 5. Physiological validation
    if result["vt1_watts"] >= result["vt2_watts"]:
        result["analysis_notes"].append("⚠️ VT1 >= VT2 - adjusted VT2 to VT1 + 15%")
        result["vt2_watts"] = int(result["vt1_watts"] * 1.15)
    result["validation"]["vt1_lt_vt2"] = result["vt1_watts"] < result["vt2_watts"]

    if result["vt1_step"] and result["vt2_step"]:
        result["validation"]["ve_vo2_rises_first"] = result["vt1_step"] < result["vt2_step"]

    return result
