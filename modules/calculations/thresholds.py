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
    step_ve_analysis: List[dict] = field(default_factory=list)  # Per-step VE slope data
    
    # New Fields for VT1/VT2 Metrics
    vt1_ve: Optional[float] = None
    vt1_br: Optional[float] = None
    vt2_ve: Optional[float] = None
    vt2_br: Optional[float] = None
    
    # SmO2 Thresholds (AeT/AnT equivalent)
    smo2_1_watts: Optional[float] = None
    smo2_2_watts: Optional[float] = None
    smo2_1_hr: Optional[float] = None
    smo2_2_hr: Optional[float] = None
    step_smo2_analysis: List[dict] = field(default_factory=list)


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


@dataclass
class DetectedStep:
    """Represents a detected step in the step test."""
    step_number: int
    start_time: float
    end_time: float
    duration_sec: float
    avg_power: float
    power_diff_from_prev: float = 0.0


@dataclass
class StepTestRange:
    """Detected range of a valid step test."""
    start_time: float
    end_time: float
    steps: List[DetectedStep]
    min_power: float
    max_power: float
    is_valid: bool = True
    notes: List[str] = field(default_factory=list)


def detect_step_test_range(
    df: pd.DataFrame,
    power_column: str = 'watts',
    time_column: str = 'time',
    min_step_duration: int = 120,  # 2 min
    max_step_duration: int = 240,  # 4 min
    min_power_increment: int = 15,  # W (lower tolerance for real data)
    max_power_increment: int = 40,  # W (higher tolerance for real data)
    min_steps: int = 4,
    power_variation_threshold: float = 0.15,  # 15% CoV allowed within step
    end_power_drop_threshold: float = 0.5  # 50% drop signals end
) -> Optional[StepTestRange]:
    """
    Detect step test boundaries by finding consecutive steps with consistent duration and power increments.
    
    Algorithm (simplified for robustness):
    1. Divide data into fixed-duration segments (30s)
    2. Calculate average power per segment
    3. Group consecutive segments with similar power into steps
    4. Find sequence of 4+ steps with 20-30W increments
    5. End detection: look for sudden power drop (>50% from max)
    
    Returns:
        StepTestRange with detected steps, or None if no valid step test found
    """
    if df.empty or power_column not in df.columns or time_column not in df.columns:
        return None
    
    df = df.sort_values(time_column).reset_index(drop=True)
    
    min_time = df[time_column].min()
    max_time = df[time_column].max()
    total_duration = max_time - min_time
    
    if total_duration < min_step_duration * min_steps:
        return StepTestRange(
            start_time=min_time, end_time=max_time,
            steps=[], min_power=0, max_power=0,
            is_valid=False,
            notes=[f"Data too short ({total_duration:.0f}s) for {min_steps} steps"]
        )
    
    # Step 1: Divide into 30-second segments and calculate averages
    segment_duration = 30  # seconds
    segments = []
    
    for t in range(int(min_time), int(max_time) - segment_duration + 1, segment_duration):
        mask = (df[time_column] >= t) & (df[time_column] < t + segment_duration)
        segment_data = df[mask]
        
        if len(segment_data) >= 5:
            avg_power = segment_data[power_column].mean()
            std_power = segment_data[power_column].std()
            segments.append({
                'start': t,
                'end': t + segment_duration,
                'avg_power': avg_power,
                'std_power': std_power
            })
    
    if len(segments) < min_steps * 4:  # Need at least 4 segments per step (2min / 30s)
        return StepTestRange(
            start_time=min_time, end_time=max_time,
            steps=[], min_power=0, max_power=0,
            is_valid=False,
            notes=[f"Only {len(segments)} segments found, need more data"]
        )
    
    # Step 2: Group consecutive segments with similar power into steps
    power_tolerance = 10  # W - segments within this range are same step
    
    steps_raw = []
    current_step_segments = [segments[0]]
    
    for i in range(1, len(segments)):
        prev_avg = np.mean([s['avg_power'] for s in current_step_segments])
        curr_avg = segments[i]['avg_power']
        
        if abs(curr_avg - prev_avg) <= power_tolerance:
            # Same step
            current_step_segments.append(segments[i])
        else:
            # New step - save previous
            step_duration = current_step_segments[-1]['end'] - current_step_segments[0]['start']
            step_power = np.mean([s['avg_power'] for s in current_step_segments])
            
            steps_raw.append({
                'start': current_step_segments[0]['start'],
                'end': current_step_segments[-1]['end'],
                'duration': step_duration,
                'avg_power': step_power,
                'segment_count': len(current_step_segments)
            })
            
            current_step_segments = [segments[i]]
    
    # Don't forget last step
    if current_step_segments:
        step_duration = current_step_segments[-1]['end'] - current_step_segments[0]['start']
        step_power = np.mean([s['avg_power'] for s in current_step_segments])
        steps_raw.append({
            'start': current_step_segments[0]['start'],
            'end': current_step_segments[-1]['end'],
            'duration': step_duration,
            'avg_power': step_power,
            'segment_count': len(current_step_segments)
        })
    
    # Step 3: Filter steps by duration (keep only valid 2-4 min steps)
    valid_steps = [s for s in steps_raw if min_step_duration <= s['duration'] <= max_step_duration + 60]
    
    if len(valid_steps) < min_steps:
        return StepTestRange(
            start_time=min_time, end_time=max_time,
            steps=[], min_power=0, max_power=0,
            is_valid=False,
            notes=[f"Only {len(valid_steps)} valid duration steps (need {min_steps}). Raw steps: {len(steps_raw)}"]
        )
    
    # Step 4: Find sequence of steps with consistent power increments
    best_sequence = []
    
    for start_idx in range(len(valid_steps)):
        sequence = [valid_steps[start_idx]]
        
        for i in range(start_idx + 1, len(valid_steps)):
            prev_power = sequence[-1]['avg_power']
            curr_power = valid_steps[i]['avg_power']
            power_diff = curr_power - prev_power
            
            # Check if power increment is within range (increasing)
            if min_power_increment <= power_diff <= max_power_increment:
                sequence.append(valid_steps[i])
            elif power_diff < -20:
                # Significant power decrease - end of test
                break
            # Skip steps with too large increments (might be a gap/warmup)
        
        if len(sequence) > len(best_sequence):
            best_sequence = sequence
    
    if len(best_sequence) < min_steps:
        increments = []
        for i in range(1, len(valid_steps)):
            increments.append(valid_steps[i]['avg_power'] - valid_steps[i-1]['avg_power'])
        
        # Build debug info strings
        powers_str = ", ".join([f"{s['avg_power']:.0f}W" for s in valid_steps])
        inc_str = ", ".join([f"{inc:+.0f}W" for inc in increments]) if increments else "No increments"
        
        return StepTestRange(
            start_time=min_time, end_time=max_time,
            steps=[], min_power=0, max_power=0,
            is_valid=False,
            notes=[
                f"Best sequence has {len(best_sequence)} steps (need {min_steps})",
                f"Valid steps powers: [{powers_str}]",
                f"Power increments: [{inc_str}]"
            ]
        )
    
    # Step 5: Detect end of test (power drop after last step)
    last_step_end = best_sequence[-1]['end']
    max_power_in_test = best_sequence[-1]['avg_power']
    
    # Look for power drop after last step
    mask_after = df[time_column] > last_step_end
    if mask_after.any():
        power_after = df.loc[mask_after, power_column].rolling(window=10, min_periods=3).mean()
        
        test_end_time = df[time_column].max()
        for idx in power_after.dropna().index:
            if power_after[idx] < max_power_in_test * end_power_drop_threshold:
                test_end_time = df.loc[idx, time_column]
                break
    else:
        test_end_time = last_step_end
    
    # Build result
    detected_steps = []
    for i, p in enumerate(best_sequence):
        power_diff = 0 if i == 0 else p['avg_power'] - best_sequence[i-1]['avg_power']
        detected_steps.append(DetectedStep(
            step_number=i + 1,
            start_time=p['start'],
            end_time=p['end'],
            duration_sec=p['duration'],
            avg_power=p['avg_power'],
            power_diff_from_prev=power_diff
        ))
    
    return StepTestRange(
        start_time=best_sequence[0]['start'],
        end_time=test_end_time,
        steps=detected_steps,
        min_power=best_sequence[0]['avg_power'],
        max_power=best_sequence[-1]['avg_power'],
        is_valid=True,
        notes=[
            f"Detected {len(detected_steps)} steps",
            f"Power range: {best_sequence[0]['avg_power']:.0f}W - {best_sequence[-1]['avg_power']:.0f}W",
            f"Avg step duration: {np.mean([s['duration'] for s in best_sequence]):.0f}s",
            f"Avg increment: {np.mean([s.power_diff_from_prev for s in detected_steps[1:]]):.0f}W" if len(detected_steps) > 1 else ""
        ]
    )


@dataclass
class StepVTResult:
    """Result of step-by-step VT detection - returns exact values, not ranges."""
    vt1_watts: Optional[float] = None
    vt1_hr: Optional[float] = None
    vt1_ve: Optional[float] = None  # New: Ventilation at threshold
    vt1_br: Optional[float] = None  # New: Breath Rate at threshold
    vt1_step_number: Optional[int] = None
    vt1_ve_slope: Optional[float] = None
    
    vt2_watts: Optional[float] = None
    vt2_hr: Optional[float] = None
    vt2_ve: Optional[float] = None  # New
    vt2_br: Optional[float] = None  # New
    vt2_step_number: Optional[int] = None
    vt2_ve_slope: Optional[float] = None
    
    step_analysis: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def calculate_slope(time_series: pd.Series, value_series: pd.Series) -> Tuple[float, float, float]:
    """
    Calculate linear regression slope and standard error.
    """
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


def detect_vt_from_steps(
    df: pd.DataFrame,
    step_range: 'StepTestRange',
    ve_column: str = 'tymeventilation',
    power_column: str = 'watts',
    hr_column: str = 'hr',
    time_column: str = 'time',
    vt1_slope_threshold: float = 0.05,  # User requested > 0.05
    vt2_slope_threshold: float = 0.05   # Re-evaluation uses same criteria
) -> StepVTResult:
    """
    Detect VT1 and VT2 using Recursive Window Scan algorithm:
    1. Divide each step into 2 halves (Stages).
    2. Search for slope > 0.05 using increasing window sizes (1 stage, 2 stages, etc).
    3. If found, mark VT1, then restart search for VT2.
    """
    result = StepVTResult()
    
    if not step_range or not step_range.is_valid or len(step_range.steps) < 2:
        result.notes.append("Insufficient steps for VT detection")
        return result
    
    if ve_column not in df.columns:
        result.notes.append(f"Missing VE column: {ve_column}")
        return result
        
    has_hr = hr_column in df.columns
    # Try to find Breath Rate column
    br_column = None
    for cand in ['tymebreathrate', 'br', 'rr', 'breath_rate']:
        if cand in df.columns:
            br_column = cand
            break

    # 1. Prepare Stages (Half-Steps)
    # Filter out first step (User: "z wyłączeniem pierwszego")
    stages = []
    
    for step in step_range.steps[1:]: # Skip Step 1 entirely
        step_mask = (df[time_column] >= step.start_time) & (df[time_column] < step.end_time)
        full_step_data = df[step_mask]
        
        if len(full_step_data) < 4: continue # Too short
            
        mid_time = step.start_time + (step.end_time - step.start_time) / 2
        
        # Split into 2 halves
        halves = [
            (full_step_data[full_step_data[time_column] < mid_time], 1),
            (full_step_data[full_step_data[time_column] >= mid_time], 2)
        ]
        
        for part_data, part_num in halves:
            if len(part_data) < 2: continue
            
            avg_power = part_data[power_column].mean()
            avg_hr = part_data[hr_column].mean() if has_hr else None
            avg_ve = part_data[ve_column].mean()
            avg_br = part_data[br_column].mean() if br_column else None
            
            stages.append({
                'data': part_data,
                'avg_power': avg_power,
                'avg_hr': avg_hr,
                'avg_ve': avg_ve,
                'avg_br': avg_br,
                'step_number': step.step_number,
                'sub_step': part_num,
                'start_time': part_data[time_column].min(),
                'end_time': part_data[time_column].max()
            })

    if not stages:
        result.notes.append("No valid stages generated for analysis")
        return result

    # 2. Recursive Search Function
    def search_for_threshold(start_idx, threshold_value):
        """
        Searches for a block of stages where slope > threshold_value.
        Tries joining 2 stages, then 3, etc.
        Returns (found_index, slope, details_dict) or (None, None, None)
        """
        n_stages = len(stages)
        if start_idx >= n_stages:
            return None, None, None
            
        # Iterate through detected stages
        for i in range(start_idx, n_stages):
            # Try increasing window sizes
            # Max window size? Let's say up to 4 stages (2 full steps) to keep it local
            max_window = min(6, n_stages - i) 
            
            for w in range(1, max_window + 1):
                # combine data from stages i to i+w
                combined_df = pd.concat([s['data'] for s in stages[i : i+w]])
                
                if len(combined_df) < 5: continue
                
                slope, _, _ = calculate_slope(combined_df[time_column], combined_df[ve_column])
                
                if slope > threshold_value:
                    # Found it!
                    # Return the LAST stage of this window as the threshold point
                    # or the AVERAGE of the window? Usually threshold is "at this point it breaks"
                    # User: "podziel... szukaj w każdym etapie... jeśli nie znajdziesz połącz".
                    # This implies the *whole block* has the trend. 
                    # We will attribute the threshold metrics to the *end* of this block (or average of block).
                    # Let's use the LAST stage in the block for Power/HR reference (conservative) 
                    # or the FIRST stage (sensitive)?
                    # "Trend > 0.05". If the trend exists in the block [A, B], the rise is happening there.
                    # We usually pick the stage where the high slope *starts* or is confirmed.
                    # Let's pick the last stage of the window as the "confirmed" point.
                    target_stage = stages[i + w - 1] 
                    return i + w, slope, target_stage # Return next search start index
        
        return None, None, None

    # 3. Find VT1
    vt1_next_idx, vt1_slope, vt1_stage = search_for_threshold(0, vt1_slope_threshold)
    
    if vt1_stage:
        result.vt1_watts = round(vt1_stage['avg_power'], 0)
        result.vt1_hr = round(vt1_stage['avg_hr'], 0) if vt1_stage['avg_hr'] else None
        result.vt1_ve = round(vt1_stage['avg_ve'], 1)
        result.vt1_br = round(vt1_stage['avg_br'], 0) if vt1_stage['avg_br'] else None
        result.vt1_ve_slope = round(vt1_slope, 4)
        result.vt1_step_number = vt1_stage['step_number']
        
        result.notes.append(f"VT1 found at Step {vt1_stage['step_number']}.{vt1_stage['sub_step']} (Slope: {vt1_slope:.4f})")
        
        # 4. Find VT2 (Start searching AFTER VT1)
        # Note: VT2 usually requires a *steeper* slope or a second breakaway.
        # However, user instruction says: "gdy znajdziesz VT1 zacznij szukać zależności od nowa dla poszukiwań VT2"
        # And "Trend VE > 0.05". It implies looking for *another* block with this trend?
        # If the slope stays > 0.05, we might just find the immediate next block.
        # Usually VT2 is > 0.15 or similar. 
        # But I must follow "start searching anew". 
        # I will enforce a small gap or ensure we are looking for a *new* rise.
        # Actually, if the slope continues to be high, it might just trigger immediately.
        # Let's trust the user algo: Search again with same criteria from next point.
        # To avoid re-detecting the same "hill", maybe we search for a HIGHER trend? 
        # Or just the next occurrence? 
        # "kontynuuj tak dopóki nie znajdziesz VT1... zacznij szukać zależności od nowa dla VT2"
        # This implies the same dependency (slope > 0.05).
        # Realistically, for VT2 we often look for slope > 0.10 or RCP.
        # I will stick to the user's explicit request (same check) but I will add 
        # a "buffer" or expect it to be distinct? 
        # No, I will just continue the search from `vt1_next_idx`.
        
        vt2_next_idx, vt2_slope, vt2_stage = search_for_threshold(vt1_next_idx, vt1_slope_threshold) # Same 0.05
        
        if vt2_stage:
             # Basic sanity check: VT2 power should be > VT1 power
             if vt2_stage['avg_power'] > result.vt1_watts:
                result.vt2_watts = round(vt2_stage['avg_power'], 0)
                result.vt2_hr = round(vt2_stage['avg_hr'], 0) if vt2_stage['avg_hr'] else None
                result.vt2_ve = round(vt2_stage['avg_ve'], 1)
                result.vt2_br = round(vt2_stage['avg_br'], 0) if vt2_stage['avg_br'] else None
                result.vt2_ve_slope = round(vt2_slope, 4)
                result.vt2_step_number = vt2_stage['step_number']
                result.notes.append(f"VT2 found at Step {vt2_stage['step_number']}.{vt2_stage['sub_step']} (Slope: {vt2_slope:.4f})")
             else:
                result.notes.append(f"Ignored VT2 candidate at {vt2_stage['avg_power']}W (<= VT1)")
        else:
            result.notes.append("VT2 not found (no subsequent slope > 0.05)")
            
    else:
        result.notes.append("VT1 not found (no slope > 0.05)")

    # Fill step_analysis for UI debug
    result.step_analysis = [
        {
            'step_number': s['step_number'], 
            'sub_step': s['sub_step'],
            'avg_power': s['avg_power'],
            'avg_ve': s['avg_ve'],
            'avg_br': s['avg_br']
        }
        for s in stages
    ]
    
    return result


@dataclass
class StepSmO2Result:
    """Result of step-by-step SmO2 detection."""
    smo2_1_watts: Optional[float] = None
    smo2_1_hr: Optional[float] = None
    smo2_1_step_number: Optional[int] = None
    smo2_1_slope: Optional[float] = None
    
    smo2_2_watts: Optional[float] = None
    smo2_2_hr: Optional[float] = None
    smo2_2_step_number: Optional[int] = None
    smo2_2_slope: Optional[float] = None
    
    step_analysis: List[dict] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def detect_smo2_from_steps(
    df: pd.DataFrame,
    step_range: 'StepTestRange', 
    smo2_column: str = 'smo2',
    power_column: str = 'watts',
    hr_column: str = 'hr',
    time_column: str = 'time',
    smo2_t1_slope_threshold: float = -0.005, # User defined boundary
    smo2_t2_slope_threshold: float = -0.005  # Same boundary, different side
) -> StepSmO2Result:
    """
    Detect SmO2 thresholds:
    - Skip first step
    - Split steps into halves
    - T2 (AnT): First point where slope < -0.005
    - T1 (AeT): The point immediately preceding T2 (Last point where slope > -0.005)
    """
    result = StepSmO2Result()
    
    if not step_range or not step_range.is_valid or len(step_range.steps) < 2:
        result.notes.append("Insufficient steps for SmO2 detection")
        return result
    
    if smo2_column not in df.columns:
        result.notes.append(f"Missing SmO2 column: {smo2_column}")
        return result
    
    has_hr = hr_column in df.columns
    
    # 1. Collect analysis for all sub-steps
    all_substeps = []
    
    for step in step_range.steps[1:]: # Skip Step 1
        step_mask = (df[time_column] >= step.start_time) & (df[time_column] < step.end_time)
        full_step_data = df[step_mask]
        
        if len(full_step_data) < 10:
            continue
            
        mid_time = step.start_time + (step.end_time - step.start_time) / 2
        halves = [
            (full_step_data[full_step_data[time_column] < mid_time], 1),
            (full_step_data[full_step_data[time_column] >= mid_time], 2)
        ]
        
        for part_data, part_num in halves:
            if len(part_data) < 5: continue
            
            slope, _, _ = calculate_slope(part_data[time_column], part_data[smo2_column])
            
            avg_power = part_data[power_column].mean()
            avg_hr = part_data[hr_column].mean() if has_hr else None
            avg_smo2 = part_data[smo2_column].mean()
            
            info = {
                'step_number': step.step_number,
                'sub_step': part_num,
                'start_time': part_data[time_column].min(),
                'end_time': part_data[time_column].max(),
                'avg_power': round(avg_power, 0),
                'avg_hr': round(avg_hr, 0) if avg_hr else None,
                'avg_smo2': round(avg_smo2, 1),
                'slope': round(slope, 5),
                'is_t1': False,
                'is_t2': False
            }
            all_substeps.append(info)
            
    # 2. Identify Thresholds sequentially
    t2_index = -1
    
    # Find T2: First item with slope < -0.005
    for i, item in enumerate(all_substeps):
        if item['slope'] < -0.005:
            t2_index = i
            break
            
    if t2_index != -1:
        # Found T2
        t2_item = all_substeps[t2_index]
        t2_item['is_t2'] = True
        result.smo2_2_watts = t2_item['avg_power']
        result.smo2_2_hr = t2_item['avg_hr']
        result.smo2_2_step_number = t2_item['step_number']
        result.smo2_2_slope = t2_item['slope']
        
        # T1 is the item immediately before T2 (if exists)
        if t2_index > 0:
            t1_item = all_substeps[t2_index - 1]
            # Verify T1 condition? User said "LT1 ... > -0.005".
            # If the previous step satisfies this (which it must, otherwise it would have been picked as T2 earlier), mark it.
            # Exception: What if checking i=0 and it fails? Then T2 is first. No T1.
            t1_item['is_t1'] = True
            result.smo2_1_watts = t1_item['avg_power']
            result.smo2_1_hr = t1_item['avg_hr']
            result.smo2_1_step_number = t1_item['step_number']
            result.smo2_1_slope = t1_item['slope']
    else:
        # No T2 found (never dropped below -0.005)
        # Maybe whole test is T1? Or T1 is the last step?
        # Let's mark the last step as T1 if it meets condition
        if all_substeps:
             last_item = all_substeps[-1]
             if last_item['slope'] > -0.005: 
                 last_item['is_t1'] = True # End of aerobic test
                 result.smo2_1_watts = last_item['avg_power']
                 result.smo2_1_hr = last_item['avg_hr']

    result.step_analysis = all_substeps
    return result


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
    Includes Step Detection, Sliding Window VT, Hysteresis, and Sensitivity Analysis.
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
    
    # 0. NEW: Auto-detect step test range
    step_range = None
    df_test = df  # Default to full data
    
    if has_power:
        step_range = detect_step_test_range(
            df, 
            power_column=power_column, 
            time_column=time_column,
            min_step_duration=120,  # 2 min
            max_step_duration=240,  # 4 min
            min_power_increment=18,  # Slightly lower tolerance (20W - 10%)
            max_power_increment=35,  # Slightly higher tolerance (30W + 17%)
            min_steps=4
        )
        
        if step_range and step_range.is_valid:
            # Filter data to detected test range
            mask = (df[time_column] >= step_range.start_time) & (df[time_column] <= step_range.end_time)
            df_test = df[mask].copy()
            
            result.steps_analyzed = len(step_range.steps)
            result.analysis_notes.append(f"✅ Wykryto test schodkowy: {len(step_range.steps)} stopni")
            for note in step_range.notes:
                result.analysis_notes.append(f"  • {note}")
            
            # NEW: Use step-based VT detection (primary method)
            if has_ve:
                vt_result = detect_vt_from_steps(
                    df, 
                    step_range,
                    ve_column=ve_column,
                    power_column=power_column,
                    hr_column=hr_column,
                    time_column=time_column,
                    vt1_slope_threshold=0.13,
                    vt2_slope_threshold=0.10
                )
                
                # Set results from step-based detection
                result.vt1_watts = vt_result.vt1_watts
                result.vt1_hr = vt_result.vt1_hr
                result.vt2_watts = vt_result.vt2_watts
                result.vt2_watts = vt_result.vt2_watts
                result.vt2_hr = vt_result.vt2_hr
                
                # Assign new VE/BR metrics
                result.vt1_ve = vt_result.vt1_ve
                result.vt1_br = vt_result.vt1_br
                result.vt2_ve = vt_result.vt2_ve
                result.vt2_br = vt_result.vt2_br
                
                for note in vt_result.notes:
                    result.analysis_notes.append(note)
                
                # Store step analysis for UI display
                result.step_ve_analysis = vt_result.step_analysis
            
            # NEW: Step-based SmO2 detection
            if has_smo2:
                smo2_res = detect_smo2_from_steps(
                    df, step_range,
                    smo2_column=smo2_column,
                    power_column=power_column,
                    hr_column=hr_column,
                    time_column=time_column
                )
                result.smo2_1_watts = smo2_res.smo2_1_watts
                result.smo2_1_hr = smo2_res.smo2_1_hr
                result.smo2_2_watts = smo2_res.smo2_2_watts
                result.smo2_2_hr = smo2_res.smo2_2_hr
                result.step_smo2_analysis = smo2_res.step_analysis
                # We reuse analysis_notes, maybe prefix them? Or just append.
                # result.analysis_notes.extend([f"SmO2: {n}" for n in smo2_res.notes])
        else:
            # No valid step test detected - fall back to legacy peak-based segmentation
            notes = step_range.notes if step_range else ["Nie znaleziono prawidłowego testu schodkowego"]
            for note in notes:
                result.analysis_notes.append(f"⚠️ {note}")
            result.analysis_notes.append("Używanie detekcji opartej na szczycie mocy (legacy)")
        
    # Fallback: Sliding Window Analysis for VT (only if step detection didn't work)
    if has_ve and has_power and result.vt1_watts is None:
        # Segment data (use detected test range or full data)
        df_inc, df_dec = segment_load_phases(df_test, power_col=power_column, time_col=time_column)
        
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
            result.analysis_notes.append(f"VT1 (Legacy): {vt1_inc.range_watts[0]:.0f}-{vt1_inc.range_watts[1]:.0f}W")
        
        if vt2_inc:
            result.vt2_watts = (vt2_inc.range_watts[0] + vt2_inc.range_watts[1]) / 2
            if vt2_inc.range_hr:
                result.vt2_hr = (vt2_inc.range_hr[0] + vt2_inc.range_hr[1]) / 2
            result.analysis_notes.append(f"VT2 (Legacy): {vt2_inc.range_watts[0]:.0f}-{vt2_inc.range_watts[1]:.0f}W")

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
