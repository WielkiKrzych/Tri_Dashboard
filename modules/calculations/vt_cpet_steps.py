"""
CPET Step Aggregation — builds steady-state step data from raw time-series.
"""

from typing import Optional
import numpy as np
import pandas as pd


def aggregate_step_data(
    data: pd.DataFrame,
    cols: dict,
    step_range,
    min_power_watts: Optional[int],
    has_vo2: bool,
    has_vco2: bool,
    has_hr: bool,
    result: dict,
) -> Optional[pd.DataFrame]:
    """
    Aggregate raw CPET data into per-step steady-state values.

    Args:
        data: Preprocessed DataFrame (from preprocess_cpet_data)
        cols: Column mapping dict
        step_range: Optional detected step ranges (with .steps attribute)
        min_power_watts: Manual override — minimum power to start VT search
        has_vo2, has_vco2, has_hr: Data availability flags
        result: Result dict — modified in place for analysis_notes

    Returns:
        Sorted df_steps DataFrame, or None if fewer than 5 steps found.
    """
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
        step_data = _aggregate_from_power_bins(
            data, cols, min_power_watts, has_vo2, has_vco2, has_hr, br_col, result
        )

    if len(step_data) < 5:
        result["error"] = f"Insufficient steps ({len(step_data)}). Need at least 5."
        result["analysis_notes"].append("Not enough steps for reliable analysis")
        return None

    return pd.DataFrame(step_data).sort_values("power").reset_index(drop=True)


def _aggregate_from_power_bins(
    data: pd.DataFrame,
    cols: dict,
    min_power_watts: Optional[int],
    has_vo2: bool,
    has_vco2: bool,
    has_hr: bool,
    br_col: Optional[str],
    result: dict,
) -> list:
    """Auto-detect steps via power bins when no step_range is provided."""
    data["power_bin"] = (data[cols["power"]] // 20) * 20

    raw_steps = []
    for power_bin, group in data.groupby("power_bin"):
        if len(group) < 30:
            continue

        if cols["time"] in group.columns:
            duration = group[cols["time"]].max() - group[cols["time"]].min()
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
            duration = len(group)
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

    ramp_start_idx = _find_ramp_start(raw_steps, min_power_watts, result)

    if ramp_start_idx > 0:
        result["analysis_notes"].append(f"Skipping first {ramp_start_idx} warmup steps")

    step_data = []
    for i, step in enumerate(raw_steps[ramp_start_idx:]):
        step["step"] = i + 1
        step_data.append(step)

    return step_data


def _find_ramp_start(raw_steps: list, min_power_watts: Optional[int], result: dict) -> int:
    """Find the index where the ramp test starts (skip warmup)."""
    if min_power_watts is not None and min_power_watts > 0:
        for i, step in enumerate(raw_steps):
            if step["power"] >= min_power_watts:
                result["ramp_start_step"] = i + 1
                result["analysis_notes"].append(
                    f"Manual override: Starting analysis from {int(min_power_watts)}W (step {i + 1})"
                )
                return i
        return 0

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
            result["ramp_start_step"] = i + 1
            result["analysis_notes"].append(
                f"Ramp test detected starting at step {i + 1} ({int(step1['power'])}W)"
            )
            return i

    return 0
