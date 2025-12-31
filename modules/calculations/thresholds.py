"""
Threshold Detection Functions for Step Tests.

Pure functions for detecting VT1/VT2 (ventilatory) and LT1/LT2 (metabolic) thresholds.
Extracted from UI modules for reuse in MCP Server.
Refactored to support Sliding Window, Transition Zones, Hysteresis, and Sensitivity Analysis.
"""
from typing import Optional, List, Tuple, Dict
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
class HysteresisResult:
    """Result of directional analysis (hysteresis)."""
    vt1_inc_zone: Optional[TransitionZone] = None
    vt1_dec_zone: Optional[TransitionZone] = None
    vt2_inc_zone: Optional[TransitionZone] = None
    vt2_dec_zone: Optional[TransitionZone] = None
    
    vt1_shift_watts: Optional[float] = None # Positive = Upward shift, Negative = Downward shift (Fatigue)
    vt2_shift_watts: Optional[float] = None
    
    warnings: List[str] = field(default_factory=list)


@dataclass
class SensitivityResult:
    """Result of sensitivity/stability analysis."""
    vt1_stability_score: float = 0.0 # 0.0 - 1.0
    vt2_stability_score: float = 0.0 # 0.0 - 1.0
    
    vt1_variability_watts: float = 0.0 # StdDev of detected watts
    vt2_variability_watts: float = 0.0
    
    is_vt1_unreliable: bool = False
    is_vt2_unreliable: bool = False
    
    details: List[str] = field(default_factory=list)


@dataclass
class StepTestResult:
    """Complete result of step test analysis."""
    vt1_watts: Optional[float] = None
    vt2_watts: Optional[float] = None
    lt1_watts: Optional[float] = None
    lt2_watts: Optional[float] = None
    
    # New Zone Fields (Primary - usually Increasing Load)
    vt1_zone: Optional[TransitionZone] = None
    vt2_zone: Optional[TransitionZone] = None
    
    # Advanced Diagnostics
    hysteresis: Optional[HysteresisResult] = None
    sensitivity: Optional[SensitivityResult] = None
    
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


def segment_load_phases(
    df: pd.DataFrame, 
    power_col: str = 'watts',
    time_col: str = 'time',
    min_phase_duration: int = 180
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into Increasing Load (Ramp Up) and Decreasing Load (Ramp Down) phases.
    The split point is the peak power.
    """
    if df.empty or power_col not in df.columns:
        return df, pd.DataFrame()
    
    # Find index of peak power (smoothed to avoid spikes)
    if 'watts_smooth_temp' not in df.columns:
         s_watts = df[power_col].rolling(window=30, center=True).mean().fillna(df[power_col])
    else:
         s_watts = df['watts_smooth_temp']
         
    peak_idx = s_watts.idxmax()
    peak_time = df.loc[peak_idx, time_col]
    
    # Ensure strict time ordering just in case
    df = df.sort_values(time_col)
    
    inc_mask = df[time_col] <= peak_time
    dec_mask = df[time_col] > peak_time
    
    df_inc = df[inc_mask].copy()
    df_dec = df[dec_mask].copy()
    
    dur_inc = (df_inc[time_col].max() - df_inc[time_col].min()) if not df_inc.empty else 0
    dur_dec = (df_dec[time_col].max() - df_dec[time_col].min()) if not df_dec.empty else 0
    
    if dur_inc < min_phase_duration:
        return df, pd.DataFrame()
        
    if dur_dec < min_phase_duration:
        return df_inc, pd.DataFrame()
        
    return df_inc, df_dec


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
        # 0.02 <= slope <= 0.08 constraint
        if (ci_lower <= 0.05 <= ci_upper):
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


def run_sensitivity_analysis(
    df: pd.DataFrame,
    ve_column: str,
    power_column: str,
    hr_column: str,
    time_column: str
) -> SensitivityResult:
    """
    Run detection with multiple window sizes to check stability.
    """
    windows = [30, 45, 60, 90]
    results_vt1 = []
    results_vt2 = []
    
    for w in windows:
        v1, v2 = detect_vt_transition_zone(
            df, 
            window_duration=w,
            step_size=5,
            ve_column=ve_column,
            power_column=power_column,
            hr_column=hr_column,
            time_column=time_column
        )
        if v1:
            # Use midpoint of the zone as the representative value
            mid = (v1.range_watts[0] + v1.range_watts[1]) / 2
            results_vt1.append(mid)
        if v2:
            mid = (v2.range_watts[0] + v2.range_watts[1]) / 2
            results_vt2.append(mid)
            
    # Calculate Statistics
    res = SensitivityResult()
    
    # VT1 Analysis
    if len(results_vt1) >= 2:
        std_dev = np.std(results_vt1)
        res.vt1_variability_watts = float(std_dev)
        # Relaxed heuristic: < 10W is very stable, > 50W is unstable
        score = max(0.0, min(1.0, 1.0 - (std_dev / 50.0)))
        res.vt1_stability_score = score
        res.is_vt1_unreliable = std_dev > 30.0
        res.details.append(f"VT1 Variability: {std_dev:.1f}W across windows {windows}")
    elif len(results_vt1) > 0:
        res.vt1_stability_score = 0.5 # Unknown
        res.details.append("VT1 detected in some but not all windows.")
    else:
        res.details.append("VT1 not detected in sensitivity sweep.")
        
    # VT2 Analysis
    if len(results_vt2) >= 2:
        std_dev = np.std(results_vt2)
        res.vt2_variability_watts = float(std_dev)
        score = max(0.0, min(1.0, 1.0 - (std_dev / 50.0)))
        res.vt2_stability_score = score
        res.is_vt2_unreliable = std_dev > 30.0
        res.details.append(f"VT2 Variability: {std_dev:.1f}W across windows {windows}")
    elif len(results_vt2) > 0:
        res.vt2_stability_score = 0.5
    else:
        res.details.append("VT2 not detected in sensitivity sweep.")
        
    return res


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
    Includes Sliding Window VT, Hysteresis, and Sensitivity Analysis.
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
        
    # 1. Sliding Window Analysis for VT with Hysteresis
    if has_ve and has_power:
        # Segment data
        df_inc, df_dec = segment_load_phases(df, power_col=power_column, time_col=time_column)
        
        # Analyze Increasing Phase (Primary)
        vt1_inc, vt2_inc = detect_vt_transition_zone(
            df_inc, 
            window_duration=60, 
            step_size=5,
            ve_column=ve_column, 
            power_column=power_column, 
            hr_column=hr_column, 
            time_column=time_column
        )
        
        # Set Primary Results
        result.vt1_zone = vt1_inc
        result.vt2_zone = vt2_inc
        
        if vt1_inc:
            result.vt1_watts = (vt1_inc.range_watts[0] + vt1_inc.range_watts[1]) / 2
            if vt1_inc.range_hr:
                result.vt1_hr = (vt1_inc.range_hr[0] + vt1_inc.range_hr[1]) / 2
            result.analysis_notes.append(f"VT1 (Inc): {vt1_inc.range_watts[0]:.0f}-{vt1_inc.range_watts[1]:.0f}W")
        
        if vt2_inc:
            result.vt2_watts = (vt2_inc.range_watts[0] + vt2_inc.range_watts[1]) / 2
            if vt2_inc.range_hr:
                result.vt2_hr = (vt2_inc.range_hr[0] + vt2_inc.range_hr[1]) / 2
            result.analysis_notes.append(f"VT2 (Inc): {vt2_inc.range_watts[0]:.0f}-{vt2_inc.range_watts[1]:.0f}W")

        # Analyze Decreasing Phase (Secondary/Hysteresis)
        vt1_dec = None
        vt2_dec = None
        hysteresis = HysteresisResult()
        
        if not df_dec.empty:
            vt1_dec, vt2_dec = detect_vt_transition_zone(
                df_dec, 
                window_duration=60, 
                step_size=5,
                ve_column=ve_column, 
                power_column=power_column, 
                hr_column=hr_column, 
                time_column=time_column
            )
            
            hysteresis.vt1_inc_zone = vt1_inc
            hysteresis.vt1_dec_zone = vt1_dec
            hysteresis.vt2_inc_zone = vt2_inc
            hysteresis.vt2_dec_zone = vt2_dec
            
            # Calculate Shifts
            if vt1_inc and vt1_dec:
                mid_inc = (vt1_inc.range_watts[0] + vt1_inc.range_watts[1]) / 2
                mid_dec = (vt1_dec.range_watts[0] + vt1_dec.range_watts[1]) / 2
                diff = mid_dec - mid_inc 
                hysteresis.vt1_shift_watts = diff
                if abs(diff) > 20: 
                    hysteresis.warnings.append(f"Significant VT1 hysteresis: {diff:.0f}W")
                    result.analysis_notes.append(f"Hysteresis VT1: {diff:+.0f}W")
            
            if vt2_inc and vt2_dec:
                mid_inc = (vt2_inc.range_watts[0] + vt2_inc.range_watts[1]) / 2
                mid_dec = (vt2_dec.range_watts[0] + vt2_dec.range_watts[1]) / 2
                diff = mid_dec - mid_inc
                hysteresis.vt2_shift_watts = diff
                if abs(diff) > 20: 
                    hysteresis.warnings.append(f"Significant VT2 hysteresis: {diff:.0f}W")
                    result.analysis_notes.append(f"Hysteresis VT2: {diff:+.0f}W")
            
            result.hysteresis = hysteresis
            
        # 3. Sensitivity Analysis (On Increasing Phase Only)
        sensitivity = run_sensitivity_analysis(
            df_inc,
            ve_column=ve_column,
            power_column=power_column,
            hr_column=hr_column,
            time_column=time_column
        )
        result.sensitivity = sensitivity
        if sensitivity.is_vt1_unreliable:
            result.analysis_notes.append("⚠️ VT1 Detection Unstable (High Variability)")
        if sensitivity.is_vt2_unreliable:
            result.analysis_notes.append("⚠️ VT2 Detection Unstable (High Variability)")

    # 4. Classic Step Analysis (Legacy fallback + SmO2) - using FULL dataframe
    
    total_duration = df[time_column].max() - df[time_column].min()
    num_steps = int(total_duration / max(step_duration_sec, 1))
    
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
