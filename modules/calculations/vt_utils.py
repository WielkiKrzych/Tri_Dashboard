"""
Ventilatory Threshold — utility helpers.

Contains low-level slope calculation and peak-based heuristic detection.
Used internally by vt_step, vt_sliding, and vt_cpet modules.
"""

import numpy as np
import pandas as pd
from scipy import stats, signal
from typing import Optional, List, Tuple


def calculate_slope(time_series: pd.Series, value_series: pd.Series) -> Tuple[float, float, float]:
    """Calculate linear regression slope and standard error."""
    if len(time_series) < 2:
        return 0.0, 0.0, 0.0

    mask = ~(time_series.isna() | value_series.isna())
    if mask.sum() < 2:
        return 0.0, 0.0, 0.0

    slope, intercept, _, _, std_err = stats.linregress(time_series[mask], value_series[mask])
    return slope, intercept, std_err


def detect_vt1_peaks_heuristic(
    df: pd.DataFrame, time_column: str, ve_column: str
) -> Tuple[Optional[dict], List[str]]:
    """Heuristic VT1 Detection based on peaks."""
    if len(df) < 120:
        return None, ["Data too short for peak analysis"]

    ve_smooth = df[ve_column].rolling(window=20, center=True).mean().fillna(df[ve_column])
    peaks, _ = signal.find_peaks(ve_smooth, distance=60, prominence=1.5)

    if len(peaks) < 2:
        return None, [f"Found only {len(peaks)} peaks (need 2+)"]

    p1_idx = peaks[0]
    p2_idx = peaks[1]
    p1_time = df.iloc[p1_idx][time_column]
    p2_time = df.iloc[p2_idx][time_column]

    mask = (df[time_column] >= p1_time) & (df[time_column] <= p2_time)
    segment = df[mask]

    if len(segment) < 10:
        return None, ["Segment between peaks too short"]

    slope, _, _ = calculate_slope(segment[time_column], segment[ve_column])

    if slope > 0.05:
        return {
            "slope": slope,
            "start_time": p1_time,
            "end_time": p2_time,
            "avg_power": segment["watts"].mean() if "watts" in segment else 0,
            "avg_hr": segment["hr"].mean() if "hr" in segment else 0,
            "avg_ve": segment[ve_column].mean(),
            "idx_end": p2_idx,
        }, [f"Peak-to-Peak: Found Slope {slope:.4f} between {p1_time:.0f}s and {p2_time:.0f}s"]
    else:
        return None, [f"Peak-to-Peak: Slope {slope:.4f} too low (<= 0.05) between peaks"]
