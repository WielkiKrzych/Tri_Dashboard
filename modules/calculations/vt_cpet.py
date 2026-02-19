"""
Ventilatory Threshold — CPET-grade detection (V2.0, laboratory standard).

Implements full CPET pipeline:
- Ventilatory Equivalents method (VE/VO2, VE/VCO2) when gas exchange is available
- VE-only 4-point CPET detection (VT1_onset, VT1_steady, RCP_onset, RCP_steady)
- Segmented piecewise regression for breakpoint detection
- 4-domain metabolic zone construction

Also contains the deprecated detect_vt_vslope_savgol() wrapper for backward
compatibility with older UI callers.
"""

from typing import Optional, Any
import numpy as np
import pandas as pd
from scipy import stats

from .vt_utils import calculate_slope


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

    Algorithm:
    1. Preprocessing: Smooth signals with rolling window, remove artifacts
    2. Calculate VE/VO2, VE/VCO2, RER for each step (steady-state)
    3. VT1: Segmented regression on VE/VO2 - find first breakpoint
       - VE/VO2 slope increases while VE/VCO2 remains flat
    4. VT2: Segmented regression on VE/VCO2 - find breakpoint
       - VE/VCO2 transitions from flat to rising, RER ≈ 1.0
    5. Physiological validation: VT1 < VT2 < max power

    Args:
        df: DataFrame with test data (1Hz or breath-by-breath)
        step_range: Optional detected step ranges
        power_column: Power column name [W]
        ve_column: Ventilation column name [L/min or L/s]
        time_column: Time column name [s]
        vo2_column: VO2 column name [L/min or ml/min]
        vco2_column: VCO2 column name [L/min or ml/min]
        hr_column: Heart rate column name [bpm]
        step_duration_sec: Expected step duration (default 180s = 3min)
        smoothing_window_sec: Smoothing window size (default 25s)
        min_power_watts: Manual override - minimum power to start VT search (skip warmup)

    Returns:
        dict with vt1_watts, vt2_watts, charts data, analysis notes
    """
    from scipy.signal import savgol_filter
    from scipy.optimize import minimize

    # Initialize result
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

    # =========================================================================
    # 1. DATA PREPARATION
    # =========================================================================
    data = df.copy()
    data.columns = data.columns.str.lower().str.strip()

    cols = {
        "power": power_column.lower(),
        "ve": ve_column.lower(),
        "time": time_column.lower(),
        "vo2": vo2_column.lower(),
        "vco2": vco2_column.lower(),
        "hr": hr_column.lower(),
    }

    if cols["power"] not in data.columns:
        result["error"] = f"Missing {power_column}"
        return result

    if cols["ve"] not in data.columns:
        result["error"] = f"Missing {ve_column}"
        return result

    has_vo2 = cols["vo2"] in data.columns and data[cols["vo2"]].notna().sum() > 10
    has_vco2 = cols["vco2"] in data.columns and data[cols["vco2"]].notna().sum() > 10
    has_hr = cols["hr"] in data.columns and data[cols["hr"]].notna().sum() > 10
    result["has_gas_exchange"] = has_vo2 and has_vco2

    # =========================================================================
    # 2. UNIT NORMALIZATION
    # =========================================================================
    if data[cols["ve"]].mean() < 10:
        data["ve_lmin"] = data[cols["ve"]] * 60
    else:
        data["ve_lmin"] = data[cols["ve"]]

    if has_vo2:
        if data[cols["vo2"]].mean() > 100:
            data["vo2_lmin"] = data[cols["vo2"]] / 1000
        else:
            data["vo2_lmin"] = data[cols["vo2"]]

    if has_vco2:
        if data[cols["vco2"]].mean() > 100:
            data["vco2_lmin"] = data[cols["vco2"]] / 1000
        else:
            data["vco2_lmin"] = data[cols["vco2"]]

    # =========================================================================
    # 3. SMOOTHING (Artifact Removal)
    # =========================================================================
    window = min(smoothing_window_sec, len(data) // 4)
    if window < 3:
        window = 3

    data["ve_smooth"] = data["ve_lmin"].rolling(window, center=True, min_periods=1).mean()

    if has_vo2:
        data["vo2_smooth"] = data["vo2_lmin"].rolling(window, center=True, min_periods=1).mean()
    if has_vco2:
        data["vco2_smooth"] = data["vco2_lmin"].rolling(window, center=True, min_periods=1).mean()

    if has_vo2 and has_vco2:
        ve_diff = data["ve_smooth"].diff().abs()
        vo2_diff = data["vo2_smooth"].diff().abs()
        vco2_diff = data["vco2_smooth"].diff().abs()

        ve_threshold = ve_diff.std() * 3
        gas_threshold = max(vo2_diff.std(), vco2_diff.std())

        artifact_mask = (ve_diff > ve_threshold) & (
            (vo2_diff < gas_threshold) & (vco2_diff < gas_threshold)
        )
        artifact_count = artifact_mask.sum()
        if artifact_count > 0:
            result["analysis_notes"].append(f"Removed {artifact_count} respiratory artifacts")
            data.loc[artifact_mask, "ve_smooth"] = np.nan
            data["ve_smooth"] = data["ve_smooth"].interpolate(method="linear")

    # =========================================================================
    # 4. STEP AGGREGATION (Steady-State from last 60-90s)
    # =========================================================================
    step_data = []

    br_col = None
    for col in ["tymebreathrate", "br", "resprate", "breathing_rate", "rf", "rr"]:
        if col in data.columns:
            br_col = col
            break
    has_br = br_col is not None

    if step_range and hasattr(step_range, "steps") and step_range.steps:
        for i, step in enumerate(step_range.steps):
            mask = (data[cols["time"]] >= step.start_time) & (data[cols["time"]] <= step.end_time)
            step_df = data[mask]

            if len(step_df) < 30:
                continue

            step_duration = step.end_time - step.start_time
            ss_start_ratio = max(0.5, 1 - (90 / step_duration)) if step_duration > 90 else 0.5
            cutoff = int(len(step_df) * ss_start_ratio)
            ss_df = step_df.iloc[cutoff:]

            row = {
                "step": i + 1,
                "power": step.avg_power,
                "ve": ss_df["ve_smooth"].mean(),
                "time": step.start_time,
                "duration": step_duration,
            }

            if has_vo2:
                row["vo2"] = ss_df["vo2_smooth"].mean()
            if has_vco2:
                row["vco2"] = ss_df["vco2_smooth"].mean()
            if has_hr and cols["hr"] in ss_df.columns:
                row["hr"] = ss_df[cols["hr"]].mean()
            if br_col and br_col in ss_df.columns:
                row["br"] = ss_df[br_col].mean()

            step_data.append(row)
    else:
        data["power_bin"] = (data[cols["power"]] // 20) * 20

        raw_steps = []
        for power_bin, group in data.groupby("power_bin"):
            if len(group) < 30:
                continue

            if cols["time"] in group.columns:
                duration = group[cols["time"]].max() - group[cols["time"]].min()
            else:
                duration = len(group)

            if cols["time"] in group.columns:
                step_start_time = group[cols["time"]].min()
                step_end_time = group[cols["time"]].max()

                kinetics_cutoff = step_start_time + 30
                steady_group = group[group[cols["time"]] >= kinetics_cutoff]

                if len(steady_group) > 0:
                    ss_start_time = step_end_time - 60
                    ss_df = steady_group[steady_group[cols["time"]] >= ss_start_time]
                else:
                    ss_df = group.iloc[-60:]
            else:
                if len(group) > 90:
                    ss_df = group.iloc[-60:]
                else:
                    ss_df = group.iloc[30:] if len(group) > 30 else group

            row = {
                "power": power_bin,
                "ve": ss_df["ve_smooth"].mean(),
                "time": group[cols["time"]].iloc[0] if cols["time"] in group.columns else 0,
                "duration": duration,
            }

            if has_vo2:
                row["vo2"] = ss_df["vo2_smooth"].mean()
            if has_vco2:
                row["vco2"] = ss_df["vco2_smooth"].mean()
            if has_hr and cols["hr"] in ss_df.columns:
                row["hr"] = ss_df[cols["hr"]].mean()
            if br_col and br_col in ss_df.columns:
                row["br"] = ss_df[br_col].mean()

            raw_steps.append(row)

        raw_steps = sorted(raw_steps, key=lambda x: x["power"])

        ramp_start_idx = 0

        if min_power_watts is not None and min_power_watts > 0:
            for i, step in enumerate(raw_steps):
                if step["power"] >= min_power_watts:
                    ramp_start_idx = i
                    result["ramp_start_step"] = i + 1
                    result["analysis_notes"].append(
                        f"Manual override: Starting analysis from {int(min_power_watts)}W (step {i + 1})"
                    )
                    break
        else:
            min_step_duration = 120
            power_increment_range = (15, 40)

            for i in range(len(raw_steps) - 2):
                step1 = raw_steps[i]
                step2 = raw_steps[i + 1]
                step3 = raw_steps[i + 2]

                dur_ok = all(s["duration"] >= min_step_duration for s in [step1, step2, step3])

                inc1 = step2["power"] - step1["power"]
                inc2 = step3["power"] - step2["power"]
                inc_ok = (
                    power_increment_range[0] <= inc1 <= power_increment_range[1]
                    and power_increment_range[0] <= inc2 <= power_increment_range[1]
                )

                if dur_ok and inc_ok:
                    ramp_start_idx = i
                    result["ramp_start_step"] = i + 1
                    result["analysis_notes"].append(
                        f"Ramp test detected starting at step {i + 1} ({int(step1['power'])}W)"
                    )
                    break

        if ramp_start_idx > 0:
            result["analysis_notes"].append(f"Skipping first {ramp_start_idx} warmup steps")

        for i, step in enumerate(raw_steps[ramp_start_idx:]):
            step["step"] = i + 1
            step_data.append(step)

    if len(step_data) < 5:
        result["error"] = f"Insufficient steps ({len(step_data)}). Need at least 5."
        result["analysis_notes"].append("Not enough steps for reliable analysis")
        return result

    df_steps = pd.DataFrame(step_data).sort_values("power").reset_index(drop=True)
    result["df_steps"] = df_steps

    # =========================================================================
    # 5. CALCULATE VENTILATORY EQUIVALENTS
    # =========================================================================
    if has_vo2 and has_vco2:
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

    else:
        # =========================================================================
        # VE-ONLY MODE: 4-POINT CPET DETECTION
        # =========================================================================
        result["analysis_notes"].append("VE-only mode: 4-point CPET detection")
        result["method"] = "ve_only_4point_cpet"

        try:
            window = min(5, len(df_steps) if len(df_steps) % 2 == 1 else len(df_steps) - 1)
            if window < 3:
                window = 3
            df_steps["ve_smooth"] = savgol_filter(
                df_steps["ve"].values, window_length=window, polyorder=2
            )
        except Exception:
            df_steps["ve_smooth"] = df_steps["ve"].rolling(3, center=True, min_periods=1).mean()

        has_hr = "hr" in df_steps.columns and df_steps["hr"].notna().sum() > 3
        has_br = "br" in df_steps.columns and df_steps["br"].notna().sum() > 3

        if has_br:
            df_steps["br_smooth"] = df_steps["br"].rolling(3, center=True, min_periods=1).mean()
            df_steps["vt_calc"] = df_steps["ve_smooth"] / df_steps["br_smooth"].replace(0, np.nan)
            df_steps["vt_smooth"] = (
                df_steps["vt_calc"].rolling(3, center=True, min_periods=1).mean()
            )

        ve_slope = np.gradient(df_steps["ve_smooth"].values, df_steps["power"].values)
        df_steps["ve_slope"] = ve_slope

        ve_accel = np.gradient(ve_slope, df_steps["power"].values)
        df_steps["ve_accel"] = ve_accel
        df_steps["ve_accel_smooth"] = (
            df_steps["ve_accel"].rolling(3, center=True, min_periods=1).mean()
        )

        if has_br:
            br_slope = np.gradient(df_steps["br_smooth"].values, df_steps["power"].values)
            df_steps["br_slope"] = br_slope
            df_steps["br_slope_smooth"] = (
                df_steps["br_slope"].rolling(3, center=True, min_periods=1).mean()
            )

            vt_slope = np.gradient(
                df_steps["vt_smooth"].fillna(method="ffill").values, df_steps["power"].values
            )
            df_steps["vt_slope"] = vt_slope
            df_steps["vt_slope_smooth"] = (
                df_steps["vt_slope"].rolling(3, center=True, min_periods=1).mean()
            )

        if has_hr:
            hr_slope = np.gradient(df_steps["hr"].values, df_steps["power"].values)
            df_steps["hr_slope"] = hr_slope
            baseline_hr_slope = np.mean(hr_slope[: min(4, len(hr_slope))])
            if baseline_hr_slope > 0:
                df_steps["hr_drift"] = hr_slope / baseline_hr_slope
            else:
                df_steps["hr_drift"] = np.ones(len(hr_slope))

        baseline_ve_slope = np.mean(df_steps["ve_slope"].iloc[: min(4, len(df_steps))])
        baseline_ve_accel = np.mean(df_steps["ve_accel_smooth"].iloc[: min(4, len(df_steps))])
        baseline_br_slope = 0
        if has_br:
            baseline_br_slope = np.mean(df_steps["br_slope_smooth"].iloc[: min(4, len(df_steps))])

        vt1_onset_idx = None
        vt1_steady_idx = None
        rcp_onset_idx = None
        rcp_steady_idx = None

        for i in range(3, len(df_steps) - 4):
            accel_prev = df_steps["ve_accel_smooth"].iloc[i - 1]
            accel_curr = df_steps["ve_accel_smooth"].iloc[i]
            sign_change = (accel_prev <= 0 and accel_curr > 0) or (
                accel_curr > baseline_ve_accel * 2
            )

            slope_rising = (
                df_steps["ve_slope"].iloc[i] > baseline_ve_slope * 1.15
                and df_steps["ve_slope"].iloc[i + 1] > baseline_ve_slope * 1.10
            )

            br_not_spiking = True
            if has_br and baseline_br_slope > 0:
                br_not_spiking = df_steps["br_slope_smooth"].iloc[i] < baseline_br_slope * 2.0

            if (sign_change or slope_rising) and br_not_spiking:
                if df_steps["ve_slope"].iloc[i + 2] > baseline_ve_slope * 1.05:
                    vt1_onset_idx = i
                    break

        vt1_steady_is_real = False

        if vt1_onset_idx is not None:
            vt1_onset_slope = df_steps["ve_slope"].iloc[vt1_onset_idx]

            for i in range(vt1_onset_idx + 2, len(df_steps) - 3):
                curr_slope = df_steps["ve_slope"].iloc[i]
                next_slope = df_steps["ve_slope"].iloc[i + 1]
                next2_slope = df_steps["ve_slope"].iloc[i + 2]

                elevated = curr_slope > baseline_ve_slope * 1.15

                delta_1 = abs(next_slope - curr_slope) / max(abs(curr_slope), 0.01)
                delta_2 = abs(next2_slope - curr_slope) / max(abs(curr_slope), 0.01)
                slope_stable = delta_1 < 0.12 and delta_2 < 0.15

                br_ok = True
                if has_br and baseline_br_slope > 0:
                    br_at_i = df_steps["br_slope_smooth"].iloc[i]
                    br_ok = br_at_i < baseline_br_slope * 1.5

                ve_accel_at_i = df_steps["ve_accel_smooth"].iloc[i]
                no_exponential = ve_accel_at_i < baseline_ve_accel * 2.5

                if elevated and slope_stable and br_ok and no_exponential:
                    vt1_steady_idx = i
                    vt1_steady_is_real = True
                    break

        result["vt1_steady_is_real"] = vt1_steady_is_real

        search_start = (
            vt1_steady_idx if vt1_steady_idx else (vt1_onset_idx + 2 if vt1_onset_idx else 4)
        )

        if search_start and search_start < len(df_steps) - 3:
            search_df = df_steps.iloc[search_start:]

            if len(search_df) >= 3:
                max_accel_idx = search_df["ve_accel_smooth"].idxmax()
                max_accel_val = df_steps.loc[max_accel_idx, "ve_accel_smooth"]

                rcp_candidates = []

                for idx in search_df.index:
                    ve_accel_high = df_steps.loc[idx, "ve_accel_smooth"] > max_accel_val * 0.6

                    br_spike = True
                    vt_plateau = True

                    if has_br:
                        br_slope_val = df_steps.loc[idx, "br_slope_smooth"]
                        br_spike = (
                            br_slope_val > baseline_br_slope * 2.0
                            if baseline_br_slope > 0
                            else br_slope_val > 0.1
                        )

                        vt_slope_val = abs(df_steps.loc[idx, "vt_slope_smooth"])
                        baseline_vt_slope = abs(
                            np.mean(df_steps["vt_slope_smooth"].iloc[: min(4, len(df_steps))])
                        )
                        vt_plateau = (
                            vt_slope_val < baseline_vt_slope * 0.7
                            if baseline_vt_slope > 0
                            else True
                        )

                    if ve_accel_high and br_spike and vt_plateau:
                        rcp_candidates.append((idx, df_steps.loc[idx, "ve_accel_smooth"]))

                if rcp_candidates:
                    rcp_onset_idx = max(rcp_candidates, key=lambda x: x[1])[0]
                else:
                    rcp_onset_idx = max_accel_idx

        if rcp_onset_idx is not None:
            rcp_onset_loc = df_steps.index.get_loc(rcp_onset_idx)

            for i in range(rcp_onset_loc + 1, len(df_steps) - 1):
                idx = df_steps.index[i]

                ve_accel_high = df_steps.loc[idx, "ve_accel_smooth"] > baseline_ve_accel * 3

                br_dominates = True
                if has_br:
                    br_slope_val = df_steps.loc[idx, "br_slope_smooth"]
                    vt_slope_val = df_steps.loc[idx, "vt_slope_smooth"]
                    br_dominates = (
                        abs(br_slope_val) > abs(vt_slope_val) * 1.5 if vt_slope_val != 0 else True
                    )

                hr_drift_strong = True
                if has_hr:
                    hr_drift_strong = df_steps.loc[idx, "hr_drift"] > 1.5

                if ve_accel_high and br_dominates and hr_drift_strong:
                    rcp_steady_idx = idx
                    break

            if rcp_steady_idx is None and rcp_onset_loc + 1 < len(df_steps):
                rcp_steady_idx = df_steps.index[min(rcp_onset_loc + 1, len(df_steps) - 1)]

        def get_point_data(idx, point_name):
            if idx is None:
                return None
            row = (
                df_steps.loc[idx]
                if isinstance(idx, (int, np.integer)) and idx in df_steps.index
                else df_steps.iloc[idx]
            )
            data_pt = {
                "watts": int(row["power"]),
                "ve": round(row["ve"], 1),
                "step": int(row["step"]),
                "time": row.get("time", 0),
            }
            if "hr" in row and pd.notna(row["hr"]):
                data_pt["hr"] = int(row["hr"])
            if "br" in row and pd.notna(row["br"]):
                data_pt["br"] = int(row["br"])
            if "vt_smooth" in row and pd.notna(row["vt_smooth"]):
                data_pt["vt"] = round(row["vt_smooth"], 2)
            return data_pt

        if vt1_onset_idx is not None:
            pt = get_point_data(vt1_onset_idx, "vt1_onset")
            result["vt1_onset_watts"] = pt["watts"]
            result["vt1_watts"] = pt["watts"]
            result["vt1_ve"] = pt["ve"]
            result["vt1_step"] = pt["step"]
            result["vt1_hr"] = pt.get("hr")
            result["vt1_br"] = pt.get("br")
            result["vt1_vt"] = pt.get("vt")
            result["vt1_onset_time"] = pt.get("time")
            result["analysis_notes"].append(
                f"VT1_onset (GET/LT1): {pt['watts']}W @ step {pt['step']}"
            )

        if vt1_steady_idx is not None and result.get("vt1_steady_is_real", False):
            pt = get_point_data(vt1_steady_idx, "vt1_steady")
            result["vt1_steady_watts"] = pt["watts"]
            result["vt1_steady_ve"] = pt["ve"]
            result["vt1_steady_hr"] = pt.get("hr")
            result["vt1_steady_br"] = pt.get("br")
            result["vt1_steady_vt"] = pt.get("vt")
            result["vt1_steady_time"] = pt.get("time")
            result["vt1_steady_is_interpolated"] = False
            result["analysis_notes"].append(
                f"VT1_steady (LT1 steady): {pt['watts']}W @ step {pt['step']} ✓plateau"
            )
        elif result.get("vt1_onset_watts") and result.get("rcp_onset_watts"):
            vt1_onset_w = result["vt1_onset_watts"]
            rcp_onset_w = result["rcp_onset_watts"]

            vt1_steady_virtual_w = int((vt1_onset_w + rcp_onset_w) / 2)

            virtual_mask = (df_steps["power"] >= vt1_steady_virtual_w - 15) & (
                df_steps["power"] <= vt1_steady_virtual_w + 15
            )
            if virtual_mask.any():
                closest_idx = (
                    df_steps.loc[virtual_mask, "power"].sub(vt1_steady_virtual_w).abs().idxmin()
                )
                pt = get_point_data(closest_idx, "vt1_steady_virtual")
                result["vt1_steady_watts"] = pt["watts"]
                result["vt1_steady_ve"] = pt["ve"]
                result["vt1_steady_hr"] = pt.get("hr")
                result["vt1_steady_br"] = pt.get("br")
                result["vt1_steady_time"] = pt.get("time")
            else:
                result["vt1_steady_watts"] = vt1_steady_virtual_w

            result["vt1_steady_is_interpolated"] = True
            result["analysis_notes"].append(
                f"VT1_steady (interpolated): {result['vt1_steady_watts']}W - no physiological plateau detected"
            )

            result["no_steady_state_interpretation"] = (
                "Pomiędzy VT1_onset a RCP_onset nie występuje stabilny stan ustalony wentylacji. "
                "Krzywa VE wykazuje ciągłe przyspieszanie, co wskazuje na: "
                "wąską strefę przejściową, szybkie narastanie buforowania H⁺, "
                "wczesne wejście w domenę heavy. "
                "Profil typowy dla sportowca o wysokiej pojemności tlenowej i stromym przejściu do kompensacji oddechowej."
            )

        if rcp_onset_idx is not None:
            pt = get_point_data(rcp_onset_idx, "rcp_onset")
            result["rcp_onset_watts"] = pt["watts"]
            result["vt2_watts"] = pt["watts"]
            result["vt2_ve"] = pt["ve"]
            result["vt2_step"] = pt["step"]
            result["vt2_hr"] = pt.get("hr")
            result["vt2_br"] = pt.get("br")
            result["rcp_onset_vt"] = pt.get("vt")
            result["rcp_onset_time"] = pt.get("time")
            result["analysis_notes"].append(
                f"RCP_onset (VT2/LT2): {pt['watts']}W @ step {pt['step']}"
            )

        if rcp_steady_idx is not None:
            pt = get_point_data(rcp_steady_idx, "rcp_steady")
            result["rcp_steady_watts"] = pt["watts"]
            result["rcp_steady_ve"] = pt["ve"]
            result["rcp_steady_hr"] = pt.get("hr")
            result["rcp_steady_br"] = pt.get("br")
            result["rcp_steady_vt"] = pt.get("vt")
            result["rcp_steady_time"] = pt.get("time")
            result["analysis_notes"].append(f"RCP_steady (Full RCP): {pt['watts']}W")

        max_power = int(df_steps["power"].max())

        vt1_onset_w = result.get("vt1_onset_watts")
        vt1_steady_w = result.get("vt1_steady_watts")
        rcp_onset_w = result.get("rcp_onset_watts") or result.get("vt2_watts")
        rcp_steady_w = result.get("rcp_steady_watts")
        is_vt1_steady_interpolated = result.get("vt1_steady_is_interpolated", False)

        if vt1_onset_w is None:
            vt1_onset_w = int(np.percentile(df_steps["power"].values, 60))
            result["vt1_onset_watts"] = vt1_onset_w
            result["vt1_watts"] = vt1_onset_w
            result["analysis_notes"].append(
                f"⚠️ VT1_onset nie wykryty - szacunek 60 percentyl ({vt1_onset_w}W)"
            )

        if rcp_onset_w is None:
            rcp_onset_w = int(np.percentile(df_steps["power"].values, 80))
            result["rcp_onset_watts"] = rcp_onset_w
            result["vt2_watts"] = rcp_onset_w
            result["analysis_notes"].append(
                f"⚠️ RCP_onset nie wykryty - szacunek 80 percentyl ({rcp_onset_w}W)"
            )

        if vt1_steady_w is None:
            vt1_steady_w = int((vt1_onset_w + rcp_onset_w) / 2)
            result["vt1_steady_watts"] = vt1_steady_w
            result["vt1_steady_is_interpolated"] = True
            is_vt1_steady_interpolated = True
            result["analysis_notes"].append(
                f"⚠️ VT1_steady interpolowany: {vt1_steady_w}W = (VT1_onset + RCP_onset) / 2"
            )
            result["no_steady_state_interpretation"] = (
                "Nie wykryto stabilnego LT1 steady-state pomiędzy VT1_onset a RCP_onset. "
                "Wentylacja wykazuje ciągłe przyspieszanie, co wskazuje na wąską strefę przejściową "
                "i szybkie wejście w domenę heavy. "
                "Zastosowano interpolowany VT1_steady jako punkt referencyjny dla stref treningowych."
            )

        if rcp_steady_w is None:
            rcp_steady_w = int(rcp_onset_w + (max_power - rcp_onset_w) * 0.4)
            result["rcp_steady_watts"] = rcp_steady_w

        boundaries_valid = vt1_onset_w < vt1_steady_w < rcp_onset_w < rcp_steady_w <= max_power

        if not boundaries_valid:
            result["analysis_notes"].append("⚠️ Korekta granic: wymuszenie monotoniczności")

            if vt1_steady_w <= vt1_onset_w:
                vt1_steady_w = vt1_onset_w + int((rcp_onset_w - vt1_onset_w) * 0.4)
                result["vt1_steady_watts"] = vt1_steady_w

            if rcp_onset_w <= vt1_steady_w:
                rcp_onset_w = vt1_steady_w + int((max_power - vt1_steady_w) * 0.5)
                result["rcp_onset_watts"] = rcp_onset_w
                result["vt2_watts"] = rcp_onset_w

            if rcp_steady_w <= rcp_onset_w:
                rcp_steady_w = rcp_onset_w + 15
                result["rcp_steady_watts"] = rcp_steady_w

        result["boundaries_valid"] = vt1_onset_w < vt1_steady_w < rcp_onset_w < rcp_steady_w

        raw_power = data[data[cols["power"]] >= 100][cols["power"]].values
        power_60th_raw = int(np.percentile(raw_power, 60))
        power_80th_raw = int(np.percentile(raw_power, 80))

        if vt1_onset_w < power_60th_raw:
            result["analysis_notes"].append(
                f"⚠️ VT1_onset ({vt1_onset_w}W) poniżej 60 percentyla ({power_60th_raw}W) - korekta"
            )
            vt1_onset_w = power_60th_raw
            result["vt1_onset_watts"] = vt1_onset_w
            result["vt1_watts"] = vt1_onset_w

        if rcp_onset_w < power_80th_raw:
            result["analysis_notes"].append(
                f"⚠️ RCP_onset ({rcp_onset_w}W) poniżej 80 percentyla ({power_80th_raw}W) - korekta"
            )
            rcp_onset_w = power_80th_raw
            result["rcp_onset_watts"] = rcp_onset_w
            result["vt2_watts"] = rcp_onset_w

        if vt1_steady_w <= vt1_onset_w:
            vt1_steady_w = int((vt1_onset_w + rcp_onset_w) / 2)
            result["vt1_steady_watts"] = vt1_steady_w
            result["vt1_steady_is_interpolated"] = True

        zones = []

        zones.append(
            {
                "zone": 1,
                "name": "Pure Aerobic",
                "power_min": 0,
                "power_max": vt1_onset_w,
                "hr_min": None,
                "hr_max": result.get("vt1_hr"),
                "description": "Full homeostasis, linear VE, no H⁺ buffering",
                "training": "Recovery / Endurance Base",
                "domain": "Moderate",
            }
        )

        zone2_name = "Upper Aerobic (Unstable)" if is_vt1_steady_interpolated else "Upper Aerobic"
        zone2_desc = (
            "Strefa niestabilna - brak plateau VE, ciągłe przyspieszanie"
            if is_vt1_steady_interpolated
            else "Buffering onset, rising VE, stable metabolism"
        )
        zones.append(
            {
                "zone": 2,
                "name": zone2_name,
                "power_min": vt1_onset_w,
                "power_max": vt1_steady_w,
                "hr_min": result.get("vt1_hr"),
                "hr_max": result.get("vt1_steady_hr"),
                "description": zone2_desc,
                "training": "Tempo / Sweet Spot",
                "domain": "Moderate-Heavy Transition",
                "is_interpolated": is_vt1_steady_interpolated,
            }
        )

        zones.append(
            {
                "zone": 3,
                "name": "Heavy Domain",
                "power_min": vt1_steady_w,
                "power_max": rcp_onset_w,
                "hr_min": result.get("vt1_steady_hr"),
                "hr_max": result.get("vt2_hr"),
                "description": "Building acidosis, VE > VO₂, pre-compensation",
                "training": "Threshold / FTP Work",
                "domain": "Heavy",
            }
        )

        zones.append(
            {
                "zone": 4,
                "name": "Severe Domain",
                "power_min": rcp_onset_w,
                "power_max": max_power,
                "hr_min": result.get("vt2_hr"),
                "hr_max": None,
                "description": "Compensatory hyperventilation, no steady-state possible",
                "training": "VO₂max / Anaerobic",
                "domain": "Severe",
                "subzones": [
                    {
                        "name": "4a - Severe Onset",
                        "power_min": rcp_onset_w,
                        "power_max": rcp_steady_w,
                        "description": "RCP onset → full RCP transition",
                    },
                    {
                        "name": "4b - Full Severe",
                        "power_min": rcp_steady_w,
                        "power_max": max_power,
                        "description": "Full respiratory compensation, exhaustion imminent",
                    },
                ],
            }
        )

        if len(zones) != 4:
            result["analysis_notes"].append(f"❌ BŁĄD: Wygenerowano {len(zones)} stref zamiast 4!")
        else:
            result["analysis_notes"].append("✓ 4 strefy wygenerowane poprawnie")

        result["metabolic_zones"] = zones

    # =========================================================================
    # 6. FALLBACK DEFAULTS
    # =========================================================================
    max_power = df_steps["power"].max()

    if result["vt1_watts"] is None:
        vt1_power = int(np.percentile(df_steps["power"].values, 60))
        result["vt1_watts"] = vt1_power
        result["analysis_notes"].append(f"VT1 not detected - using 60th percentile ({vt1_power}W)")

    if result["vt2_watts"] is None:
        vt2_power = int(np.percentile(df_steps["power"].values, 80))
        result["vt2_watts"] = vt2_power
        result["analysis_notes"].append(f"VT2 not detected - using 80th percentile ({vt2_power}W)")

    # =========================================================================
    # 7. PHYSIOLOGICAL VALIDATION
    # =========================================================================
    if result["vt1_watts"] >= result["vt2_watts"]:
        result["analysis_notes"].append("⚠️ VT1 >= VT2 - adjusted VT2 to VT1 + 15%")
        result["vt2_watts"] = int(result["vt1_watts"] * 1.15)
    result["validation"]["vt1_lt_vt2"] = result["vt1_watts"] < result["vt2_watts"]

    if result["vt1_step"] and result["vt2_step"]:
        result["validation"]["ve_vo2_rises_first"] = result["vt1_step"] < result["vt2_step"]

    return result


def _find_breakpoint_segmented(
    x: np.ndarray, y: np.ndarray, min_segment_size: int = 3
) -> Optional[int]:
    """
    Find optimal breakpoint using piecewise linear regression.

    Tests each potential breakpoint and returns the one minimizing total SSE.

    Args:
        x: Independent variable (power)
        y: Dependent variable (VE/VO2 or VE/VCO2)
        min_segment_size: Minimum points in each segment

    Returns:
        Index of optimal breakpoint, or None if not found
    """
    if len(x) < 2 * min_segment_size:
        return None

    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 2 * min_segment_size:
        return None

    best_idx = None
    best_sse = np.inf
    best_slope_ratio = 0

    for i in range(min_segment_size, len(x) - min_segment_size):
        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]

        try:
            slope1, intercept1, _, _, _ = stats.linregress(x1, y1)
            pred1 = slope1 * x1 + intercept1
            sse1 = np.sum((y1 - pred1) ** 2)

            slope2, intercept2, _, _, _ = stats.linregress(x2, y2)
            pred2 = slope2 * x2 + intercept2
            sse2 = np.sum((y2 - pred2) ** 2)

            total_sse = sse1 + sse2

            slope_ratio = slope2 / slope1 if slope1 != 0 else slope2

            if total_sse < best_sse and slope_ratio > 1.1:
                best_sse = total_sse
                best_idx = i
                best_slope_ratio = slope_ratio

        except Exception:
            continue

    return best_idx


def _calculate_segment_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate slope of a segment using linear regression."""
    if len(x) < 2:
        return 0.0

    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return 0.0

    try:
        slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
        return slope
    except Exception:
        return 0.0
