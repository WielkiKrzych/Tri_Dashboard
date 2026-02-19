"""
Ventilatory Threshold — step-test VT detection.

Detects VT1 and VT2 from staircase protocol data using recursive window scan
over step stages. Handles range-based detection with confidence scoring.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from .threshold_types import TransitionZone, StepVTResult, StepTestRange
from .common import (
    VT1_SLOPE_THRESHOLD,
    VT1_SLOPE_SPIKE_SKIP,
    SLOPE_CONFIDENCE_MAX,
    STABILITY_CONFIDENCE_MAX,
    BASE_CONFIDENCE,
    MAX_CONFIDENCE,
    LOWER_STEP_WEIGHT,
    UPPER_STEP_WEIGHT,
)
from .vt_utils import calculate_slope


def detect_vt_from_steps(
    df: pd.DataFrame,
    step_range: StepTestRange,
    ve_column: str = "tymeventilation",
    power_column: str = "watts",
    hr_column: str = "hr",
    time_column: str = "time",
    vt1_slope_threshold: float = 0.05,
    vt2_slope_threshold: float = 0.05,
) -> StepVTResult:
    """Detect VT1 and VT2 using recursive window scan."""
    result = StepVTResult()

    if not step_range or not step_range.is_valid or len(step_range.steps) < 2:
        result.notes.append("Insufficient steps for VT detection")
        return result

    if ve_column not in df.columns:
        result.notes.append(f"Missing VE column: {ve_column}")
        return result

    has_hr = hr_column in df.columns
    br_column = next(
        (c for c in ["tymebreathrate", "br", "rr", "breath_rate"] if c in df.columns), None
    )

    stages = []
    for step in step_range.steps[1:]:
        step_mask = (df[time_column] >= step.start_time) & (df[time_column] < step.end_time)
        full_step_data = df[step_mask]

        if len(full_step_data) < 10:
            continue

        slope, _, _ = calculate_slope(full_step_data[time_column], full_step_data[ve_column])

        stages.append(
            {
                "data": full_step_data,
                "avg_power": full_step_data[power_column].mean(),
                "avg_hr": full_step_data[hr_column].mean() if has_hr else None,
                "avg_ve": full_step_data[ve_column].mean(),
                "avg_br": full_step_data[br_column].mean() if br_column else None,
                "step_number": step.step_number,
                "start_time": full_step_data[time_column].min(),
                "end_time": full_step_data[time_column].max(),
                "ve_slope": slope,
            }
        )

    if not stages:
        result.notes.append("No valid stages generated for analysis")
        return result

    def search(start_idx, threshold):
        n = len(stages)
        for i in range(start_idx, n):
            max_w = min(6, n - i)
            for w in range(1, max_w + 1):
                combined = pd.concat([s["data"] for s in stages[i : i + w]])
                if len(combined) < 5:
                    continue
                slope, _, _ = calculate_slope(combined[time_column], combined[ve_column])
                if slope > threshold:
                    return i + w, slope, stages[i + w - 1], i  # Return start index too
        return None, None, None, None

    s_idx, s_slope, s_stage, _ = search(0, VT1_SLOPE_SPIKE_SKIP)
    vt1_start = s_idx if s_stage else 0
    if s_stage:
        result.notes.append(
            f"Skipped spike > {VT1_SLOPE_SPIKE_SKIP} at Step {s_stage['step_number']} (Slope: {s_slope:.4f})"
        )

    # VT1 Detection with RANGE logic
    v1_idx, v1_slope, v1_stage, v1_start_idx = search(vt1_start, VT1_SLOPE_THRESHOLD)
    if v1_stage and v1_start_idx is not None:
        lower_step_idx = max(0, v1_start_idx - 1)
        lower_power = (
            stages[lower_step_idx]["avg_power"]
            if lower_step_idx < len(stages)
            else v1_stage["avg_power"]
        )
        upper_power = v1_stage["avg_power"]

        central_power = lower_power * LOWER_STEP_WEIGHT + upper_power * UPPER_STEP_WEIGHT

        slope_confidence = min(SLOPE_CONFIDENCE_MAX, v1_slope * 4)
        range_width = upper_power - lower_power
        stability_confidence = max(0.0, STABILITY_CONFIDENCE_MAX - range_width / 100)

        step_ve_values = [s["avg_ve"] for s in stages[v1_start_idx:v1_idx]]
        if len(step_ve_values) > 1:
            ve_variance = np.var(step_ve_values)
            variance_penalty = min(0.5, ve_variance / 20)
        else:
            variance_penalty = 0.0

        total_confidence = min(
            MAX_CONFIDENCE,
            BASE_CONFIDENCE + slope_confidence + stability_confidence - variance_penalty,
        )

        lower_hr = stages[lower_step_idx]["avg_hr"] if stages[lower_step_idx]["avg_hr"] else None
        upper_hr = v1_stage["avg_hr"]
        central_hr = (lower_hr * 0.3 + upper_hr * 0.7) if lower_hr and upper_hr else upper_hr

        lower_ve = stages[lower_step_idx]["avg_ve"]
        upper_ve = v1_stage["avg_ve"]
        central_ve = lower_ve * 0.3 + upper_ve * 0.7

        result.vt1_zone = TransitionZone(
            range_watts=(lower_power, upper_power),
            range_hr=(lower_hr, upper_hr) if lower_hr and upper_hr else None,
            midpoint_ve=central_ve,
            range_ve=[lower_ve, upper_ve],
            confidence=total_confidence,
            stability_score=stability_confidence / 0.4 if stability_confidence > 0 else 0.5,
            method="step_ve_slope_range",
            description=f"VT1 zone spanning Steps {stages[lower_step_idx]['step_number']}-{v1_stage['step_number']}",
            detection_sources=["VE"],
            variability_watts=range_width,
        )

        result.vt1_watts = round(central_power, 0)
        result.vt1_hr = round(central_hr, 0) if central_hr else None
        result.vt1_ve = round(v1_stage["avg_ve"], 1)
        result.vt1_br = round(v1_stage["avg_br"], 0) if v1_stage["avg_br"] else None
        result.vt1_ve_slope = round(v1_slope, 4)
        result.vt1_step_number = v1_stage["step_number"]

        result.notes.append(
            f"VT1 zone: {lower_power:.0f}–{upper_power:.0f} W "
            f"(central: {central_power:.0f} W, confidence: {total_confidence:.2f})"
        )

    # VT2 Detection with RANGE logic
    v2_start = v1_idx if v1_stage else vt1_start
    v2_idx, v2_slope, v2_stage, v2_start_idx = search(v2_start, vt2_slope_threshold)
    if v2_stage and v2_start_idx is not None:
        if not v1_stage or (v2_stage["avg_power"] > result.vt1_watts):
            lower_step_idx = max(0, v2_start_idx - 1)
            lower_power = (
                stages[lower_step_idx]["avg_power"]
                if lower_step_idx < len(stages)
                else v2_stage["avg_power"]
            )
            upper_power = v2_stage["avg_power"]
            central_power = lower_power * 0.3 + upper_power * 0.7

            slope_confidence = min(0.4, v2_slope * 4)
            range_width = upper_power - lower_power
            stability_confidence = max(0.0, 0.4 - range_width / 100)
            base_confidence = 0.2
            total_confidence = min(0.95, base_confidence + slope_confidence + stability_confidence)

            lower_hr = (
                stages[lower_step_idx]["avg_hr"] if stages[lower_step_idx]["avg_hr"] else None
            )
            upper_hr = v2_stage["avg_hr"]
            central_hr = (lower_hr * 0.3 + upper_hr * 0.7) if lower_hr and upper_hr else upper_hr

            lower_ve = stages[lower_step_idx]["avg_ve"]
            upper_ve = v2_stage["avg_ve"]
            central_ve = lower_ve * 0.3 + upper_ve * 0.7

            result.vt2_zone = TransitionZone(
                range_watts=(lower_power, upper_power),
                range_hr=(lower_hr, upper_hr) if lower_hr and upper_hr else None,
                midpoint_ve=central_ve,
                range_ve=[lower_ve, upper_ve],
                confidence=total_confidence,
                stability_score=stability_confidence / 0.4 if stability_confidence > 0 else 0.5,
                method="step_ve_slope_range",
                description=f"VT2 zone spanning Steps {stages[lower_step_idx]['step_number']}-{v2_stage['step_number']}",
                detection_sources=["VE"],
                variability_watts=range_width,
            )

            result.vt2_watts = round(central_power, 0)
            result.vt2_hr = round(central_hr, 0) if central_hr else None
            result.vt2_ve = round(v2_stage["avg_ve"], 1)
            result.vt2_br = round(v2_stage["avg_br"], 0) if v2_stage["avg_br"] else None
            result.vt2_ve_slope = round(v2_slope, 4)
            result.vt2_step_number = v2_stage["step_number"]

            result.notes.append(
                f"VT2 zone: {lower_power:.0f}–{upper_power:.0f} W "
                f"(central: {central_power:.0f} W, confidence: {total_confidence:.2f})"
            )

    result.step_analysis = [
        {
            "step_number": s["step_number"],
            "avg_power": s["avg_power"],
            "avg_hr": s["avg_hr"],
            "avg_ve": s["avg_ve"],
            "avg_br": s["avg_br"],
            "ve_slope": s["ve_slope"],
            "start_time": s["start_time"],
            "end_time": s["end_time"],
            "is_skipped": s_stage and s["step_number"] == s_stage["step_number"],
            "is_vt1": v1_stage and s["step_number"] == result.vt1_step_number,
            "is_vt2": v2_stage and s["step_number"] == result.vt2_step_number,
        }
        for s in stages
    ]
    return result
