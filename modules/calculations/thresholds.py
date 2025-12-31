"""
Threshold Detection Functions for Step Tests.

Pure functions for detecting VT1/VT2 (ventilatory) and LT1/LT2 (metabolic) thresholds.
Extracted from UI modules for reuse in MCP Server.
Refactored to support Sliding Window and Transition Zones.
"""
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TransitionZone:
    """Represents a transition zone instead of a single point."""
    range_watts: Tuple[float, float]
    range_hr: Optional[Tuple[float, float]]
    confidence: float  # 0.0 to 1.0
    method: str
    description: str = ""


@dataclass
class ThresholdResult:
    """Result of threshold detection (Legacy/Simple)."""
    zone_name: str
    zone_type: str  # info, success, warning, error
    description: str
    slope_value: float
    power_at_threshold: Optional[float] = None
    hr_at_threshold: Optional[float] = None


@dataclass
class StepTestResult:
    """Complete result of step test analysis."""
    vt1_watts: Optional[float] = None
    vt2_watts: Optional[float] = None
    lt1_watts: Optional[float] = None
    lt2_watts: Optional[float] = None
    
    # New Zone Fields
    vt1_zone: Optional[TransitionZone] = None
    vt2_zone: Optional[TransitionZone] = None
    
    vt1_hr: Optional[float] = None
    vt2_hr: Optional[float] = None
    steps_analyzed: int = 0
    analysis_notes: List[str] = field(default_factory=list)


def calculate_slope(time_series: pd.Series, value_series: pd.Series) -> Tuple[float, float, float]:
    """
    Calculate linear regression slope and standard error.
    
    Args:
        time_series: Time values
        value_series: Measurement values
    
    Returns:
        Tuple of (slope, intercept, std_err)
    """
    if len(time_series) < 2:
        return 0.0, 0.0, 0.0
    
    # Remove NaN values
    mask = ~(time_series.isna() | value_series.isna())
    if mask.sum() < 2:
        return 0.0, 0.0, 0.0
    
    slope, intercept, _, _, std_err = stats.linregress(
        time_series[mask], 
        value_series[mask]
    )
    return slope, intercept, std_err


def detect_vt_transition_zone(
    df: pd.DataFrame,
    window_duration: int = 60,
    step_size: int = 10,
    ve_column: str = 'tymeventilation',
    power_column: str = 'watts',
    hr_column: str = 'hr',
    time_column: str = 'time'
) -> Tuple[Optional[TransitionZone], Optional[TransitionZone]]:
    """
    Detect VT1 and VT2 transition zones using a sliding window approach.
    Definition: A Transition Zone is the range of time/power where the 
    slope's 95% Confidence Interval overlaps the threshold value (0.05 or 0.15).
    
    Args:
        df: DataFrame with training data (must be time-sorted)
        window_duration: Duration of sliding window in seconds
        step_size: Step size for sliding window in seconds
        
    Returns:
        Tuple of (VT1_Zone, VT2_Zone)
    """
    if len(df) < window_duration:
        return None, None

    min_time = df[time_column].min()
    max_time = df[time_column].max()
    
    # List to store results for each window
    vt1_candidates = []
    vt2_candidates = []
    
    # Z-score for 95% CI (approx 1.96)
    Z_SCORE = 1.96
    
    # Slide window
    for t_start in range(int(min_time), int(max_time) - window_duration, step_size):
        t_end = t_start + window_duration
        mask = (df[time_column] >= t_start) & (df[time_column] < t_end)
        window = df[mask]
        
        if len(window) < 10:  # Require min points
            continue
            
        slope_ve, _, std_err = calculate_slope(window[time_column], window[ve_column])
        avg_watts = window[power_column].mean()
        avg_hr = window[hr_column].mean() if hr_column in window.columns else None
        
        # Calculate 95% CI bounds
        ci_lower = slope_ve - (Z_SCORE * std_err)
        ci_upper = slope_ve + (Z_SCORE * std_err)
        
        # VT1 Overlap Check (Threshold 0.05)
        if (ci_lower <= 0.05 <= ci_upper):
             # Sanity check: slope shouldn't be wildly off
             if 0.02 <= slope_ve <= 0.08:
                vt1_candidates.append({
                    'avg_watts': avg_watts,
                    'avg_hr': avg_hr,
                    'std_err': std_err
                })

        # VT2 Overlap Check (Threshold 0.15)
        if (ci_lower <= 0.15 <= ci_upper):
             if 0.10 <= slope_ve <= 0.20:
                vt2_candidates.append({
                    'avg_watts': avg_watts,
                    'avg_hr': avg_hr,
                    'std_err': std_err
                })
        
    # Process VT1 Candidates
    vt1_zone = None
    if vt1_candidates:
        df_vt1 = pd.DataFrame(vt1_candidates)
        watts_min = df_vt1['avg_watts'].min()
        watts_max = df_vt1['avg_watts'].max()
        hr_min = df_vt1['avg_hr'].min() if 'avg_hr' in df_vt1.columns else 0
        hr_max = df_vt1['avg_hr'].max() if 'avg_hr' in df_vt1.columns else 0
        
        avg_err = df_vt1['std_err'].mean()
        confidence = max(0.1, min(1.0, 1.0 - (avg_err * 100))) 
        
        vt1_zone = TransitionZone(
            range_watts=(watts_min, watts_max),
            range_hr=(hr_min, hr_max) if hr_min else None,
            confidence=confidence,
            method="Sliding Window CI Overlap (0.05)",
            description=f"Region where slope CI overlaps 0.05. Avg Err: {avg_err:.4f}"
        )

    # Process VT2 Candidates
    vt2_zone = None
    if vt2_candidates:
        df_vt2 = pd.DataFrame(vt2_candidates)
        watts_min = df_vt2['avg_watts'].min()
        watts_max = df_vt2['avg_watts'].max()
        hr_min = df_vt2['avg_hr'].min() if 'avg_hr' in df_vt2.columns else 0
        hr_max = df_vt2['avg_hr'].max() if 'avg_hr' in df_vt2.columns else 0
        
        avg_err = df_vt2['std_err'].mean()
        confidence = max(0.1, min(1.0, 1.0 - (avg_err * 50)))
        
        vt2_zone = TransitionZone(
            range_watts=(watts_min, watts_max),
            range_hr=(hr_min, hr_max) if hr_min else None,
            confidence=confidence,
            method="Sliding Window CI Overlap (0.15)",
            description=f"Region where slope CI overlaps 0.15. Avg Err: {avg_err:.4f}"
        )
        
    return vt1_zone, vt2_zone


def analyze_step_test(
    df: pd.DataFrame,
    step_duration_sec: int = 180,
    power_column: str = 'watts',
    ve_column: str = 'tymeventilation',
    smo2_column: str = 'smo2',
    hr_column: str = 'hr',
    time_column: str = 'time'
) -> StepTestResult:
    """
    Analyze a step test (ramp test) for threshold detection.
    Updated to use Sliding Window for VT and classic step analysis for SmO2 (hybrid).
    """
    result = StepTestResult()
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    has_ve = ve_column in df.columns
    has_smo2 = smo2_column in df.columns
    has_power = power_column in df.columns
    has_time = time_column in df.columns
    
    if not has_time:
        result.analysis_notes.append("Brak kolumny czasu")
        return result
        
    # 1. Sliding Window Analysis for VT (New)
    if has_ve and has_power:
        vt1_zone, vt2_zone = detect_vt_transition_zone(
            df, 
            window_duration=60, # 60s windows
            step_size=5,        # 5s steps for high resolution
            ve_column=ve_column,
            power_column=power_column,
            hr_column=hr_column,
            time_column=time_column
        )
        
        result.vt1_zone = vt1_zone
        result.vt2_zone = vt2_zone
        
        if vt1_zone:
            # For backward compatibility, pick the midpoint of the zone
            result.vt1_watts = (vt1_zone.range_watts[0] + vt1_zone.range_watts[1]) / 2
            if vt1_zone.range_hr:
                result.vt1_hr = (vt1_zone.range_hr[0] + vt1_zone.range_hr[1]) / 2
            result.analysis_notes.append(f"VT1 Zone Detected: {vt1_zone.range_watts[0]:.0f}-{vt1_zone.range_watts[1]:.0f}W (Conf: {vt1_zone.confidence:.2f})")
            
        if vt2_zone:
            result.vt2_watts = (vt2_zone.range_watts[0] + vt2_zone.range_watts[1]) / 2
            if vt2_zone.range_hr:
                result.vt2_hr = (vt2_zone.range_hr[0] + vt2_zone.range_hr[1]) / 2
            result.analysis_notes.append(f"VT2 Zone Detected: {vt2_zone.range_watts[0]:.0f}-{vt2_zone.range_watts[1]:.0f}W (Conf: {vt2_zone.confidence:.2f})")

    # 2. Classic Step Analysis (Legacy fallback + SmO2)
    # This preserves the step-by-step functionality for SmO2 which we haven't refactored yet
    # ... (Keeping the logic for preparing step_data to pass to SmO2 detection) ...
    
    total_duration = df[time_column].max() - df[time_column].min()
    num_steps = int(total_duration / max(step_duration_sec, 1)) # avoid div by 0
    
    # We need step_data for SmO2
    step_data = []
    if num_steps >= 3:
        start_time = df[time_column].min()
        for step in range(num_steps):
            step_start = start_time + step * step_duration_sec
            step_end = step_start + step_duration_sec
            
            stable_start = step_end - 60
            mask = (df[time_column] >= stable_start) & (df[time_column] < step_end)
            step_df = df[mask]
            
            if len(step_df) < 10: continue
            
            step_info = {'step': step + 1, 'start': step_start, 'end': step_end}
            if has_power: step_info['avg_power'] = step_df[power_column].mean()
            if has_smo2:
                smo2_slope, _, _ = calculate_slope(step_df[time_column], step_df[smo2_column])
                step_info['smo2_slope'] = smo2_slope
                step_info['avg_smo2'] = step_df[smo2_column].mean()
            
            step_data.append(step_info)

    # Detect LT1/LT2 from SmO2 data (Legacy single-point)
    if has_smo2 and len(step_data) >= 3:
        smo2_thresholds = _detect_smo2_thresholds(step_data)
        if smo2_thresholds:
            result.lt1_watts = smo2_thresholds.get('lt1_power')
            result.lt2_watts = smo2_thresholds.get('lt2_power')
            result.analysis_notes.append(
                f"LT1/LT2 (SmO2) detected at steps {smo2_thresholds.get('lt1_step')}/{smo2_thresholds.get('lt2_step')}"
            )
            
    return result


def _detect_smo2_thresholds(step_data: List[dict]) -> Optional[dict]:
    """Detect LT1 and LT2 from step test SmO2 data (Legacy implementation)."""
    if len(step_data) < 3: return None
    
    smo2_steps = [s for s in step_data if 'smo2_slope' in s and 'avg_power' in s]
    if len(smo2_steps) < 3: return None
    
    slopes = [s['smo2_slope'] for s in smo2_steps]
    
    # LT1: transition to near-zero
    lt1_idx = None
    for i, slope in enumerate(slopes):
        if slope <= 0.005: 
            lt1_idx = max(0, i - 1)
            break
            
    # LT2: transition to neg
    lt2_idx = None
    for i, slope in enumerate(slopes):
        if slope < -0.01:
            lt2_idx = max(0, i - 1)
            break
            
    result = {}
    if lt1_idx is not None:
        result['lt1_power'] = smo2_steps[lt1_idx].get('avg_power')
        result['lt1_step'] = smo2_steps[lt1_idx].get('step')
    if lt2_idx is not None:
        result['lt2_power'] = smo2_steps[lt2_idx].get('avg_power')
        result['lt2_step'] = smo2_steps[lt2_idx].get('step')
        
    return result if result else None


def calculate_training_zones_from_thresholds(
    vt1_watts: int,
    vt2_watts: int,
    cp=None,
    max_hr: int = 185
) -> dict:
    """Calculate training zones based on detected thresholds."""
    if cp is None:
        cp = vt2_watts
    
    # Fallback to 0 if None
    vt1_watts = vt1_watts if vt1_watts else 0
    vt2_watts = vt2_watts if vt2_watts else 0
    
    return {
        "power_zones": {
            "Z1_Recovery": (0, int(vt1_watts * 0.75)),
            "Z2_Endurance": (int(vt1_watts * 0.75), int(vt1_watts)),
            "Z3_Tempo": (int(vt1_watts), int((vt1_watts + vt2_watts) / 2)),
            "Z4_Threshold": (int((vt1_watts + vt2_watts) / 2), int(vt2_watts)),
            "Z5_VO2max": (int(vt2_watts), int(cp * 1.2)),
            "Z6_Anaerobic": (int(cp * 1.2), int(cp * 1.5))
        },
        "hr_zones": {
            "Z1_Recovery": (0, int(max_hr * 0.6)),
            "Z2_Endurance": (int(max_hr * 0.6), int(max_hr * 0.7)),
            "Z3_Tempo": (int(max_hr * 0.7), int(max_hr * 0.8)),
            "Z4_Threshold": (int(max_hr * 0.8), int(max_hr * 0.9)),
            "Z5_VO2max": (int(max_hr * 0.9), max_hr)
        },
        "zone_descriptions": {
            "Z1_Recovery": "Regeneracja",
            "Z2_Endurance": "Baza tlenowa",
            "Z3_Tempo": "Sweet spot",
            "Z4_Threshold": "Próg FTP",
            "Z5_VO2max": "VO2max",
            "Z6_Anaerobic": "Beztlenowa"
        }
    }
