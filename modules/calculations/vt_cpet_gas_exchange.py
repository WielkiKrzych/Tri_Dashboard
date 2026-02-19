"""
CPET Gas Exchange Detection — VE/VO2 + VE/VCO2 breakpoint analysis.
"""

import numpy as np
import pandas as pd

from .vt_cpet_ve_only import _find_breakpoint_segmented, _calculate_segment_slope


def detect_gas_exchange_thresholds(df_steps: pd.DataFrame, result: dict) -> None:
    """
    Detect VT1 and VT2 using ventilatory equivalents (CPET mode).

    Uses segmented piecewise regression on VE/VO2 (VT1) and VE/VCO2 (VT2).
    Modifies df_steps in place (adds ve_vo2, ve_vco2, rer columns).
    Modifies result dict in place with threshold values and analysis notes.

    Args:
        df_steps: Per-step steady-state DataFrame with ve, vo2, vco2, hr, br columns.
        result: Result dict to populate.
    """
    df_steps["ve_vo2"] = df_steps["ve"] / df_steps["vo2"].replace(0, np.nan)
    df_steps["ve_vco2"] = df_steps["ve"] / df_steps["vco2"].replace(0, np.nan)
    df_steps["rer"] = df_steps["vco2"] / df_steps["vo2"].replace(0, np.nan)

    result["analysis_notes"].append("Using CPET mode: VE/VO2 and VE/VCO2 analysis")

    vo2max = df_steps["vo2"].max()

    vt1_idx = _find_breakpoint_segmented(
        df_steps["power"].values, df_steps["ve_vo2"].values, min_segment_size=3
    )

    if vt1_idx is not None and 1 < vt1_idx < len(df_steps) - 1:
        vco2_slope_before = _calculate_segment_slope(
            df_steps["power"].values[:vt1_idx], df_steps["ve_vco2"].values[:vt1_idx]
        )
        vco2_slope_at = _calculate_segment_slope(
            df_steps["power"].values[max(0, vt1_idx - 2) : vt1_idx + 2],
            df_steps["ve_vco2"].values[max(0, vt1_idx - 2) : vt1_idx + 2],
        )

        if abs(vco2_slope_at) < 0.1 or vco2_slope_at < vco2_slope_before * 1.5:
            result["vt1_watts"] = int(df_steps.loc[vt1_idx, "power"])
            result["vt1_ve"] = round(df_steps.loc[vt1_idx, "ve"], 1)
            result["vt1_vo2"] = round(df_steps.loc[vt1_idx, "vo2"], 2)
            result["vt1_step"] = int(df_steps.loc[vt1_idx, "step"])
            result["vt1_pct_vo2max"] = (
                round(df_steps.loc[vt1_idx, "vo2"] / vo2max * 100, 1) if vo2max > 0 else None
            )
            if "hr" in df_steps.columns and pd.notna(df_steps.loc[vt1_idx, "hr"]):
                result["vt1_hr"] = int(df_steps.loc[vt1_idx, "hr"])
            if "br" in df_steps.columns and pd.notna(df_steps.loc[vt1_idx, "br"]):
                result["vt1_br"] = int(df_steps.loc[vt1_idx, "br"])
            result["analysis_notes"].append(
                f"VT1 detected at step {result['vt1_step']} via VE/VO2 breakpoint"
            )
        else:
            result["analysis_notes"].append("VT1 candidate rejected: VE/VCO2 already rising")

    search_start = vt1_idx + 1 if vt1_idx else 3

    if search_start >= len(df_steps) - 4:
        search_start = max(3, len(df_steps) // 2)

    remaining_points = len(df_steps) - search_start
    if remaining_points >= 4:
        vt2_idx = _find_breakpoint_segmented(
            df_steps["power"].values[search_start:],
            df_steps["ve_vco2"].values[search_start:],
            min_segment_size=2,
        )

        if vt2_idx is not None:
            vt2_idx += search_start

            if vt2_idx < len(df_steps):
                rer_at_vt2 = df_steps.loc[vt2_idx, "rer"]

                if pd.notna(rer_at_vt2) and 0.95 <= rer_at_vt2 <= 1.15:
                    result["vt2_watts"] = int(df_steps.loc[vt2_idx, "power"])
                    result["vt2_ve"] = round(df_steps.loc[vt2_idx, "ve"], 1)
                    result["vt2_vo2"] = (
                        round(df_steps.loc[vt2_idx, "vo2"], 2)
                        if "vo2" in df_steps.columns
                        else None
                    )
                    result["vt2_step"] = int(df_steps.loc[vt2_idx, "step"])
                    result["vt2_pct_vo2max"] = (
                        round(df_steps.loc[vt2_idx, "vo2"] / vo2max * 100, 1)
                        if vo2max > 0 and "vo2" in df_steps.columns
                        else None
                    )
                    if "hr" in df_steps.columns and pd.notna(df_steps.loc[vt2_idx, "hr"]):
                        result["vt2_hr"] = int(df_steps.loc[vt2_idx, "hr"])
                    if "br" in df_steps.columns and pd.notna(df_steps.loc[vt2_idx, "br"]):
                        result["vt2_br"] = int(df_steps.loc[vt2_idx, "br"])
                    result["analysis_notes"].append(
                        f"VT2 detected at step {result['vt2_step']} (RER={rer_at_vt2:.2f})"
                    )
                else:
                    result["vt2_watts"] = int(df_steps.loc[vt2_idx, "power"])
                    result["vt2_ve"] = round(df_steps.loc[vt2_idx, "ve"], 1)
                    result["vt2_step"] = int(df_steps.loc[vt2_idx, "step"])
                    if "hr" in df_steps.columns and pd.notna(df_steps.loc[vt2_idx, "hr"]):
                        result["vt2_hr"] = int(df_steps.loc[vt2_idx, "hr"])
                    if "br" in df_steps.columns and pd.notna(df_steps.loc[vt2_idx, "br"]):
                        result["vt2_br"] = int(df_steps.loc[vt2_idx, "br"])
                    rer_str = f"{rer_at_vt2:.2f}" if pd.notna(rer_at_vt2) else "N/A"
                    result["analysis_notes"].append(
                        f"VT2 detected but RER={rer_str} (expected ~1.0)"
                    )
