"""
Metabolic Threshold Detection (SmO2/LT).

IMPORTANT: SmO₂ is a LOCAL/REGIONAL signal - see limitations below.
"""
import pandas as pd
from typing import Optional, List
from .threshold_types import StepSmO2Result, StepTestRange
from .ventilatory import calculate_slope

def detect_smo2_from_steps(
    df: pd.DataFrame,
    step_range: StepTestRange, 
    smo2_column: str = 'smo2',
    power_column: str = 'watts',
    hr_column: str = 'hr',
    time_column: str = 'time',
    smo2_t1_slope_threshold: float = -0.01,
    smo2_t2_slope_threshold: float = -0.02
) -> StepSmO2Result:
    """Detect SmO2 thresholds from step data.
    
    ⚠️ CRITICAL LIMITATIONS - SmO₂ is a LOCAL signal:
    
    1. SmO₂ reflects oxygen saturation in ONE muscle group only.
       Example: Vastus lateralis SmO₂ ≠ whole-body VO₂.
       
    2. This function detects POTENTIAL thresholds that should:
       - SUPPORT ventilatory thresholds (VT1, VT2)
       - NOT be used as standalone decision points
       - Be interpreted WITH VE/HR data, not instead of it
       
    3. Sensor placement, subcutaneous fat, and movement artifacts
       significantly affect readings.
       
    4. Results are returned with is_supporting_only=True to indicate
       these should influence interpretation of VT, not replace it.
    
    Args:
        df: DataFrame with SmO2 data
        step_range: Detected step test range
        smo2_column: Column name for SmO2 (default: 'smo2')
        power_column: Column name for power (default: 'watts')
        hr_column: Column name for HR (default: 'hr')
        time_column: Column name for time (default: 'time')
        smo2_t1_slope_threshold: Slope threshold for LT1 detection
        smo2_t2_slope_threshold: Slope threshold for LT2 detection
    
    Returns:
        StepSmO2Result with is_supporting_only=True (local signal)
    """
    result = StepSmO2Result()
    
    # Add interpretation note to results
    result.notes.append("⚠️ SmO₂ = sygnał LOKALNY - potwierdzaj z VT z wentylacji")
    
    if not step_range or not step_range.is_valid or len(step_range.steps) < 3:
        result.notes.append("Insufficient steps for SmO2 detection (need > 2)")
        return result
    
    if smo2_column not in df.columns:
        result.notes.append(f"Missing SmO2 column: {smo2_column}")
        return result
    
    has_hr = hr_column in df.columns
    all_steps = []
    for step in step_range.steps[1:]: 
        mask = (df[time_column] >= step.start_time) & (df[time_column] < step.end_time)
        data = df[mask]
        if len(data) < 10: continue
        slope, _, _ = calculate_slope(data[time_column], data[smo2_column])
        all_steps.append({
            'step_number': step.step_number,
            'start_time': data[time_column].min(),
            'end_time': data[time_column].max(),
            'avg_power': round(data[power_column].mean(), 0),
            'avg_hr': round(data[hr_column].mean(), 0) if has_hr else None,
            'avg_smo2': round(data[smo2_column].mean(), 1),
            'slope': round(slope, 5),
            'is_t1': False, 'is_t2': False, 'is_skipped': False
        })
            
    # Detect LT1: Skip the first encounter if it meets the threshold (spike/transient)
    first_found_idx = -1
    lt1_idx = -1
    
    for i, item in enumerate(all_steps):
        if item['slope'] < smo2_t1_slope_threshold:
            if first_found_idx == -1:
                first_found_idx = i
                item['is_skipped'] = True  # Added for UI labeling
                result.notes.append(f"Skipped initial SmO2 drop at Step {item['step_number']} (Slope: {item['slope']:.4f})")
                continue
            
            # This is the second encounter
            item['is_t1'] = True
            result.smo2_1_watts = item['avg_power']
            result.smo2_1_hr = item['avg_hr']
            result.smo2_1_step_number = item['step_number']
            result.smo2_1_slope = item['slope']
            lt1_idx = i
            result.notes.append(f"LT1 (SmO2) found at Step {item['step_number']} (Slope: {item['slope']:.4f})")
            break
            
    if lt1_idx != -1:
        for i in range(lt1_idx + 1, len(all_steps)):
            if all_steps[i]['slope'] < smo2_t2_slope_threshold:
                all_steps[i]['is_t2'] = True
                result.smo2_2_watts = all_steps[i]['avg_power']
                result.smo2_2_hr = all_steps[i]['avg_hr']
                result.smo2_2_step_number = all_steps[i]['step_number']
                result.smo2_2_slope = all_steps[i]['slope']
                result.notes.append(f"LT2 (SmO2) found at Step {all_steps[i]['step_number']} (Slope: {all_steps[i]['slope']:.4f})")
                break
    result.step_analysis = all_steps
    return result

def _detect_smo2_thresholds_legacy(step_data: List[dict]) -> Optional[dict]:
    """Legacy LT1/LT2 detection logic."""
    if len(step_data) < 3: return None
    s_steps = [s for s in step_data if 'smo2_slope' in s and 'avg_power' in s]
    if len(s_steps) < 3: return None
    
    lt1_idx = next((i for i, s in enumerate(s_steps) if s['smo2_slope'] <= 0.005), None)
    if lt1_idx is not None: lt1_idx = max(0, lt1_idx - 1)
    
    lt2_idx = next((i for i, s in enumerate(s_steps) if s['smo2_slope'] < -0.01), None)
    if lt2_idx is not None: lt2_idx = max(0, lt2_idx - 1)
            
    res = {}
    if lt1_idx is not None:
        res['lt1_power'] = s_steps[lt1_idx]['avg_power']
        res['lt1_step'] = s_steps[lt1_idx]['step']
    if lt2_idx is not None:
        res['lt2_power'] = s_steps[lt2_idx]['avg_power']
        res['lt2_step'] = s_steps[lt2_idx]['step']
    return res if res else None
