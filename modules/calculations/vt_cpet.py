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
import pandas as pd

from .vt_cpet_preprocessing import preprocess_cpet_data
from .vt_cpet_steps import aggregate_step_data
from .vt_cpet_gas_exchange import detect_gas_exchange_thresholds
from .vt_cpet_ve_only import detect_ve_only_thresholds
from .common import validate_threshold_vs_pmax


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
    import warnings

    warnings.warn(
        "detect_vt_vslope_savgol() is deprecated — use detect_vt_cpet() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
        "cross_validation": None,  # [Issue #7]
        "vt1_confidence": None,
        "vt1_range_low": None,
        "vt1_range_high": None,
        "vt1_confidence_penalty": 0.0,
        "vt2_confidence": None,
        "vt2_range_low": None,
        "vt2_range_high": None,
        "vt2_confidence_penalty": 0.0,
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
    data, has_vo2, has_vco2, has_hr = preprocess_cpet_data(df, cols, smoothing_window_sec, result)
    if result.get("error"):
        return result

    # 2. Aggregate per-step steady-state values
    df_steps = aggregate_step_data(
        data, cols, step_range, min_power_watts, has_vo2, has_vco2, has_hr, result
    )
    if df_steps is None:
        return result
    result["df_steps"] = df_steps

    # 3.5 [Issue #7] Cross-validation between detection methods
    cross_validation = {
        "enabled": False,
        "vt1_methods": [],
        "vt2_methods": [],
        "vt1_deviation_watts": None,
        "vt2_deviation_watts": None,
        "warning": None,
    }

    # 3. Detect thresholds
    if has_vo2 and has_vco2:
        detect_gas_exchange_thresholds(df_steps, result)

        # [Issue #7] Store gas exchange results for cross-validation
        cross_validation["enabled"] = True
        if result.get("vt1_watts"):
            cross_validation["vt1_methods"].append(
                {
                    "method": "gas_exchange",
                    "value": result["vt1_watts"],
                    "confidence": 0.7,  # Default confidence for gas exchange
                }
            )
        if result.get("vt2_watts"):
            cross_validation["vt2_methods"].append(
                {
                    "method": "gas_exchange",
                    "value": result["vt2_watts"],
                    "confidence": 0.7,
                }
            )

        # Also run VE-only method for comparison
        ve_result = {"vt1_watts": None, "vt2_watts": None, "analysis_notes": []}
        detect_ve_only_thresholds(df_steps, data, cols, ve_result)

        if ve_result.get("vt1_watts"):
            cross_validation["vt1_methods"].append(
                {
                    "method": "ve_only",
                    "value": ve_result["vt1_watts"],
                    "confidence": 0.6,  # Lower confidence for VE-only
                }
            )
        if ve_result.get("vt2_watts"):
            cross_validation["vt2_methods"].append(
                {
                    "method": "ve_only",
                    "value": ve_result["vt2_watts"],
                    "confidence": 0.6,
                }
            )
    else:
        detect_ve_only_thresholds(df_steps, data, cols, result)

    # [Issue #7] Calculate weighted average if multiple methods
    if len(cross_validation["vt1_methods"]) >= 2:
        methods = cross_validation["vt1_methods"]
        total_conf = sum(m["confidence"] for m in methods)
        if total_conf > 0:
            weighted_vt1 = sum(m["value"] * m["confidence"] for m in methods) / total_conf
            deviation = max(m["value"] for m in methods) - min(m["value"] for m in methods)
            cross_validation["vt1_deviation_watts"] = deviation

            if deviation > 30:
                cross_validation["warning"] = (
                    f"⚠️ VT1 methods deviate by {deviation:.0f}W - results uncertain"
                )
                result["analysis_notes"].append(cross_validation["warning"])
            else:
                # Use weighted average as final value
                result["vt1_watts"] = int(weighted_vt1)
                result["analysis_notes"].append(
                    f"VT1 cross-validated: weighted average {int(weighted_vt1)}W "
                    f"(deviation: {deviation:.0f}W)"
                )

    if len(cross_validation["vt2_methods"]) >= 2:
        methods = cross_validation["vt2_methods"]
        total_conf = sum(m["confidence"] for m in methods)
        if total_conf > 0:
            weighted_vt2 = sum(m["value"] * m["confidence"] for m in methods) / total_conf
            deviation = max(m["value"] for m in methods) - min(m["value"] for m in methods)
            cross_validation["vt2_deviation_watts"] = deviation

            if deviation > 30:
                if cross_validation["warning"]:
                    cross_validation["warning"] += f", VT2 deviates by {deviation:.0f}W"
                else:
                    cross_validation["warning"] = (
                        f"⚠️ VT2 methods deviate by {deviation:.0f}W - results uncertain"
                    )
                result["analysis_notes"].append(f"⚠️ VT2 methods deviate by {deviation:.0f}W")
            else:
                result["vt2_watts"] = int(weighted_vt2)
                result["analysis_notes"].append(
                    f"VT2 cross-validated: weighted average {int(weighted_vt2)}W "
                    f"(deviation: {deviation:.0f}W)"
                )

    result["cross_validation"] = cross_validation

    # 4. Global fallback defaults (both paths) [Issue #3]
    # Use Pmax-relative formula for physiological plausibility
    pmax = df_steps["power"].max() if len(df_steps) > 0 else 0

    VT1_PMAX_RATIO_MIN = 0.55  # 55% of Pmax
    VT1_PMAX_RATIO_MAX = 0.65  # 65% of Pmax
    VT2_PMAX_RATIO_MIN = 0.75  # 75% of Pmax
    VT2_PMAX_RATIO_MAX = 0.85  # 85% of Pmax

    if result["vt1_watts"] is None and pmax > 0:
        # Use midpoint of physiological range
        vt1_power = int(pmax * ((VT1_PMAX_RATIO_MIN + VT1_PMAX_RATIO_MAX) / 2))
        result["vt1_watts"] = vt1_power
        result["analysis_notes"].append(
            f"VT1 not detected - using Pmax-relative estimate ({vt1_power}W = 60% of {pmax}W Pmax)"
        )

    if result["vt2_watts"] is None and pmax > 0:
        # Use midpoint of physiological range
        vt2_power = int(pmax * ((VT2_PMAX_RATIO_MIN + VT2_PMAX_RATIO_MAX) / 2))
        result["vt2_watts"] = vt2_power
        result["analysis_notes"].append(
            f"VT2 not detected - using Pmax-relative estimate ({vt2_power}W = 80% of {pmax}W Pmax)"
        )

    # 5. Physiological validation
    if result["vt1_watts"] >= result["vt2_watts"]:
        result["analysis_notes"].append("⚠️ VT1 >= VT2 - adjusted VT2 to VT1 + 15%")
        result["vt2_watts"] = int(result["vt1_watts"] * 1.15)
    result["validation"]["vt1_lt_vt2"] = result["vt1_watts"] < result["vt2_watts"]

    # C3: Confidence intervals for threshold values
    # VT shifts ±10-20W day-to-day (hydration, glycogen, temperature)
    # Gronwald et al. 2024 meta-analysis confirms VT/LT correspondence
    # but inherent measurement uncertainty exists
    vt1_penalty = result.get("vt1_confidence_penalty", 0.0)
    vt2_penalty = result.get("vt2_confidence_penalty", 0.0)

    # Base confidence from detection method
    base_confidence = 0.80 if has_vo2 and has_vco2 else 0.65

    if result["vt1_watts"] is not None:
        vt1_conf = max(0.3, base_confidence - vt1_penalty)
        # Margin: high confidence = ±5W, low = ±20W
        vt1_margin = int(5 + (1 - vt1_conf) * 25)
        result["vt1_confidence"] = round(vt1_conf, 2)
        result["vt1_range_low"] = result["vt1_watts"] - vt1_margin
        result["vt1_range_high"] = result["vt1_watts"] + vt1_margin

    if result["vt2_watts"] is not None:
        vt2_conf = max(0.3, base_confidence - vt2_penalty)
        vt2_margin = int(5 + (1 - vt2_conf) * 25)
        result["vt2_confidence"] = round(vt2_conf, 2)
        result["vt2_range_low"] = result["vt2_watts"] - vt2_margin
        result["vt2_range_high"] = result["vt2_watts"] + vt2_margin

    if result["vt1_step"] and result["vt2_step"]:
        result["validation"]["ve_vo2_rises_first"] = result["vt1_step"] < result["vt2_step"]

    # 6. [Issue #2] VT2 vs Pmax sanity check
    if result["vt2_watts"] is not None and pmax > 0:
        validation = validate_threshold_vs_pmax(result["vt2_watts"], pmax, "VT2", max_ratio=0.95)
        if not validation["is_valid"]:
            result["analysis_notes"].append(validation["message"])
            result["vt2_confidence_penalty"] = validation["confidence_penalty"]
            result["vt2_unreliable"] = True

    return result
