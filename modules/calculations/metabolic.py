"""
Metabolic Threshold Detection (SmO2/LT).

IMPORTANT: SmO₂ is a LOCAL/REGIONAL signal - see limitations below.
"""
import pandas as pd
from typing import List  # Optional removed - was only used by deleted legacy function
from .threshold_types import StepSmO2Result, StepTestRange, TransitionZone
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
                item['is_skipped'] = True
                result.notes.append(f"Skipped initial SmO2 drop at Step {item['step_number']} (Slope: {item['slope']:.4f})")
                continue
            
            # LT1 Detection with RANGE logic
            item['is_t1'] = True
            lt1_idx = i
            
            # Calculate RANGE from adjacent steps
            lower_step_idx = max(0, i - 1)
            lower_power = all_steps[lower_step_idx]['avg_power'] if lower_step_idx < len(all_steps) else item['avg_power']
            upper_power = item['avg_power']
            
            # Central tendency (weighted toward detection step)
            central_power = lower_power * 0.3 + upper_power * 0.7
            
            # Confidence: SmO₂ is LOCAL signal → max 0.6
            # Components: slope sharpness + stability
            slope_confidence = min(0.3, abs(item['slope']) * 10)
            range_width = upper_power - lower_power
            stability_confidence = max(0.0, 0.2 - range_width / 100)
            base_confidence = 0.1  # Local signal gets lower base
            total_confidence = min(0.6, base_confidence + slope_confidence + stability_confidence)
            
            # HR range
            lower_hr = all_steps[lower_step_idx]['avg_hr']
            upper_hr = item['avg_hr']
            central_hr = (lower_hr * 0.3 + upper_hr * 0.7) if lower_hr and upper_hr else upper_hr
            
            # Create TransitionZone (PRIMARY OUTPUT)
            result.smo2_1_zone = TransitionZone(
                range_watts=(lower_power, upper_power),
                range_hr=(lower_hr, upper_hr) if lower_hr and upper_hr else None,
                confidence=total_confidence,
                stability_score=stability_confidence / 0.2 if stability_confidence > 0 else 0.5,
                method="smo2_slope_range",
                description=f"SmO₂ LT1 zone Steps {all_steps[lower_step_idx]['step_number']}-{item['step_number']} (LOCAL)",
                detection_sources=["SmO2"],
                variability_watts=range_width
            )
            
            # Legacy point values (DERIVED)
            result.smo2_1_watts = round(central_power, 0)
            result.smo2_1_hr = round(central_hr, 0) if central_hr else None
            result.smo2_1_step_number = item['step_number']
            result.smo2_1_slope = item['slope']
            
            result.notes.append(
                f"LT1 (SmO2) zone: {lower_power:.0f}–{upper_power:.0f} W "
                f"(central: {central_power:.0f} W, confidence: {total_confidence:.2f})"
            )
            break
            
    if lt1_idx != -1:
        for i in range(lt1_idx + 1, len(all_steps)):
            if all_steps[i]['slope'] < smo2_t2_slope_threshold:
                all_steps[i]['is_t2'] = True
                
                # LT2 Detection with RANGE logic
                lower_step_idx = max(0, i - 1)
                lower_power = all_steps[lower_step_idx]['avg_power']
                upper_power = all_steps[i]['avg_power']
                central_power = lower_power * 0.3 + upper_power * 0.7
                
                # Confidence for LT2
                slope_confidence = min(0.3, abs(all_steps[i]['slope']) * 10)
                range_width = upper_power - lower_power
                stability_confidence = max(0.0, 0.2 - range_width / 100)
                total_confidence = min(0.6, 0.1 + slope_confidence + stability_confidence)
                
                lower_hr = all_steps[lower_step_idx]['avg_hr']
                upper_hr = all_steps[i]['avg_hr']
                central_hr = (lower_hr * 0.3 + upper_hr * 0.7) if lower_hr and upper_hr else upper_hr
                
                # Create TransitionZone (PRIMARY OUTPUT)
                result.smo2_2_zone = TransitionZone(
                    range_watts=(lower_power, upper_power),
                    range_hr=(lower_hr, upper_hr) if lower_hr and upper_hr else None,
                    confidence=total_confidence,
                    stability_score=stability_confidence / 0.2 if stability_confidence > 0 else 0.5,
                    method="smo2_slope_range",
                    description=f"SmO₂ LT2 zone Steps {all_steps[lower_step_idx]['step_number']}-{all_steps[i]['step_number']} (LOCAL)",
                    detection_sources=["SmO2"],
                    variability_watts=range_width
                )
                
                # Legacy point values (DERIVED)
                result.smo2_2_watts = round(central_power, 0)
                result.smo2_2_hr = round(central_hr, 0) if central_hr else None
                result.smo2_2_step_number = all_steps[i]['step_number']
                result.smo2_2_slope = all_steps[i]['slope']
                
                result.notes.append(
                    f"LT2 (SmO2) zone: {lower_power:.0f}–{upper_power:.0f} W "
                    f"(central: {central_power:.0f} W, confidence: {total_confidence:.2f})"
                )
                break
    result.step_analysis = all_steps
    return result

# NOTE: _detect_smo2_thresholds_legacy was removed (2026-01-02)
# REASON: Function was never called - detect_smo2_from_steps is the active implementation

