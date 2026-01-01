"""
Ventilatory Threshold Detection (VT1/VT2).
"""
import numpy as np
import pandas as pd
from scipy import stats, signal
from typing import Optional, List, Tuple

from .threshold_types import (
    TransitionZone, 
    SensitivityResult, 
    StepVTResult,
    StepTestRange
)

def calculate_slope(time_series: pd.Series, value_series: pd.Series) -> Tuple[float, float, float]:
    """Calculate linear regression slope and standard error."""
    if len(time_series) < 2:
        return 0.0, 0.0, 0.0
    
    mask = ~(time_series.isna() | value_series.isna())
    if mask.sum() < 2:
        return 0.0, 0.0, 0.0
    
    slope, intercept, _, _, std_err = stats.linregress(
        time_series[mask], 
        value_series[mask]
    )
    return slope, intercept, std_err

def detect_vt1_peaks_heuristic(
    df: pd.DataFrame,
    time_column: str,
    ve_column: str
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
            'slope': slope,
            'start_time': p1_time,
            'end_time': p2_time,
            'avg_power': segment['watts'].mean() if 'watts' in segment else 0,
            'avg_hr': segment['hr'].mean() if 'hr' in segment else 0,
            'avg_ve': segment[ve_column].mean(),
            'idx_end': p2_idx
        }, [f"Peak-to-Peak: Found Slope {slope:.4f} between {p1_time:.0f}s and {p2_time:.0f}s"]
    else:
        return None, [f"Peak-to-Peak: Slope {slope:.4f} too low (<= 0.05) between peaks"]

def detect_vt_from_steps(
    df: pd.DataFrame,
    step_range: StepTestRange,
    ve_column: str = 'tymeventilation',
    power_column: str = 'watts',
    hr_column: str = 'hr',
    time_column: str = 'time',
    vt1_slope_threshold: float = 0.05,
    vt2_slope_threshold: float = 0.05
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
    br_column = next((c for c in ['tymebreathrate', 'br', 'rr', 'breath_rate'] if c in df.columns), None)

    stages = []
    for step in step_range.steps[1:]:
        step_mask = (df[time_column] >= step.start_time) & (df[time_column] < step.end_time)
        full_step_data = df[step_mask]
        
        if len(full_step_data) < 10: continue 
            
        slope, _, _ = calculate_slope(full_step_data[time_column], full_step_data[ve_column])
        
        stages.append({
            'data': full_step_data,
            'avg_power': full_step_data[power_column].mean(),
            'avg_hr': full_step_data[hr_column].mean() if has_hr else None,
            'avg_ve': full_step_data[ve_column].mean(),
            'avg_br': full_step_data[br_column].mean() if br_column else None,
            'step_number': step.step_number,
            'start_time': full_step_data[time_column].min(),
            'end_time': full_step_data[time_column].max(),
            've_slope': slope
        })

    if not stages:
        result.notes.append("No valid stages generated for analysis")
        return result

    def search(start_idx, threshold):
        n = len(stages)
        for i in range(start_idx, n):
            max_w = min(6, n - i) 
            for w in range(1, max_w + 1):
                combined = pd.concat([s['data'] for s in stages[i : i+w]])
                if len(combined) < 5: continue
                slope, _, _ = calculate_slope(combined[time_column], combined[ve_column])
                if slope > threshold:
                    return i + w, slope, stages[i + w - 1]
        return None, None, None

    s_idx, s_slope, s_stage = search(0, 0.10)
    vt1_start = s_idx if s_stage else 0
    if s_stage:
        result.notes.append(f"Skipped spike > 0.1 at Step {s_stage['step_number']} (Slope: {s_slope:.4f})")

    v1_idx, v1_slope, v1_stage = search(vt1_start, vt1_slope_threshold)
    if v1_stage:
        result.vt1_watts = round(v1_stage['avg_power'], 0)
        result.vt1_hr = round(v1_stage['avg_hr'], 0) if v1_stage['avg_hr'] else None
        result.vt1_ve = round(v1_stage['avg_ve'], 1)
        result.vt1_br = round(v1_stage['avg_br'], 0) if v1_stage['avg_br'] else None
        result.vt1_ve_slope = round(v1_slope, 4)
        result.vt1_step_number = v1_stage['step_number']
        result.notes.append(f"VT1 found at Step {v1_stage['step_number']} (Slope: {v1_slope:.4f})")

    v2_start = v1_idx if v1_stage else vt1_start
    _, v2_slope, v2_stage = search(v2_start, vt2_slope_threshold)
    if v2_stage:
        if not v1_stage or (v2_stage['avg_power'] > result.vt1_watts):
            result.vt2_watts = round(v2_stage['avg_power'], 0)
            result.vt2_hr = round(v2_stage['avg_hr'], 0) if v2_stage['avg_hr'] else None
            result.vt2_ve = round(v2_stage['avg_ve'], 1)
            result.vt2_br = round(v2_stage['avg_br'], 0) if v2_stage['avg_br'] else None
            result.vt2_ve_slope = round(v2_slope, 4)
            result.vt2_step_number = v2_stage['step_number']
            result.notes.append(f"VT2 found at Step {v2_stage['step_number']} (Slope: {v2_slope:.4f})")

    result.step_analysis = [
        {
            'step_number': s['step_number'], 'avg_power': s['avg_power'],
            'avg_hr': s['avg_hr'], 'avg_ve': s['avg_ve'], 'avg_br': s['avg_br'],
            've_slope': s['ve_slope'], 'start_time': s['start_time'], 'end_time': s['end_time'],
            'is_skipped': s_stage and s['step_number'] == s_stage['step_number'],
            'is_vt1': v1_stage and s['step_number'] == result.vt1_step_number,
            'is_vt2': v2_stage and s['step_number'] == result.vt2_step_number
        }
        for s in stages
    ]
    return result

def detect_vt_transition_zone(
    df: pd.DataFrame,
    window_duration: int = 60,
    step_size: int = 10,
    ve_column: str = 'tymeventilation',
    power_column: str = 'watts',
    hr_column: str = 'hr',
    time_column: str = 'time'
) -> Tuple[Optional[TransitionZone], Optional[TransitionZone]]:
    """Detect VT transition zones using sliding window."""
    if len(df) < window_duration: return None, None
    min_time, max_time = df[time_column].min(), df[time_column].max()
    vt1_c, vt2_c = [], []
    Z = 1.96
    
    for t in range(int(min_time), int(max_time) - window_duration, step_size):
        mask = (df[time_column] >= t) & (df[time_column] < t + window_duration)
        w = df[mask]
        if len(w) < 10: continue
        slope, _, err = calculate_slope(w[time_column], w[ve_column])
        l, u = slope - Z*err, slope + Z*err
        if l <= 0.05 <= u and 0.02 <= slope <= 0.08:
            vt1_c.append({'avg_watts': w[power_column].mean(), 'avg_hr': w[hr_column].mean() if hr_column in w else None, 'std_err': err})
        if l <= 0.15 <= u and 0.10 <= slope <= 0.20:
            vt2_c.append({'avg_watts': w[power_column].mean(), 'avg_hr': w[hr_column].mean() if hr_column in w else None, 'std_err': err})

    def process_c(c, threshold, err_scale):
        if not c: return None
        df_c = pd.DataFrame(c)
        avg_err = df_c['std_err'].mean()
        return TransitionZone(
            range_watts=(df_c['avg_watts'].min(), df_c['avg_watts'].max()),
            range_hr=(df_c['avg_hr'].min(), df_c['avg_hr'].max()) if 'avg_hr' in df_c and df_c['avg_hr'].min() else None,
            confidence=max(0.1, min(1.0, 1.0 - (avg_err * err_scale))),
            method=f"Sliding Window (threshold {threshold})",
            description=f"Region where slope CI overlaps {threshold}."
        )

    return process_c(vt1_c, 0.05, 100), process_c(vt2_c, 0.15, 50)

def run_sensitivity_analysis(
    df: pd.DataFrame,
    ve_column: str,
    power_column: str,
    hr_column: str,
    time_column: str
) -> SensitivityResult:
    """Check stability by varying window size."""
    wins = [30, 45, 60, 90]
    r1, r2 = [], []
    for w in wins:
        v1, v2 = detect_vt_transition_zone(df, w, 5, ve_column, power_column, hr_column, time_column)
        if v1: r1.append(sum(v1.range_watts)/2)
        if v2: r2.append(sum(v2.range_watts)/2)
    
    res = SensitivityResult()
    def analyze(r, name):
        if len(r) >= 2:
            std = np.std(r)
            score = max(0.0, min(1.0, 1.0 - (std / 50.0)))
            res.details.append(f"{name} variability: {std:.1f}W")
            return std, score, std > 30.0
        res.details.append(f"{name} not enough data points.")
        return 0.0, 0.5, False

    res.vt1_variability_watts, res.vt1_stability_score, res.is_vt1_unreliable = analyze(r1, "VT1")
    res.vt2_variability_watts, res.vt2_stability_score, res.is_vt2_unreliable = analyze(r2, "VT2")
    return res
