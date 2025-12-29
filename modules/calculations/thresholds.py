"""
Threshold Detection Functions for Step Tests.

Pure functions for detecting VT1/VT2 (ventilatory) and LT1/LT2 (metabolic) thresholds.
Extracted from UI modules for reuse in MCP Server.
"""
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ThresholdResult:
    """Result of threshold detection."""
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
    vt1_hr: Optional[float] = None
    vt2_hr: Optional[float] = None
    steps_analyzed: int = 0
    analysis_notes: List[str] = None
    
    def __post_init__(self):
        if self.analysis_notes is None:
            self.analysis_notes = []


def detect_vent_zone(slope_ve: float) -> ThresholdResult:
    """
    Detect ventilatory zone based on VE (minute ventilation) slope.
    
    Thresholds based on research:
    - < 0.02: Very low intensity (recovery)
    - 0.02-0.05: Below VT1 (aerobic zone)
    - 0.05-0.15: VT1-VT2 (threshold zone)
    - > 0.15: Above VT2 (hyperventilation)
    
    Args:
        slope_ve: Slope of VE over time (L/min per second)
    
    Returns:
        ThresholdResult with zone classification
    """
    if slope_ve < 0.02:
        return ThresholdResult(
            zone_name="Poniżej VT1 (Regeneracja)",
            zone_type="info",
            description="Wentylacja stabilna. Strefa regeneracji.",
            slope_value=slope_ve
        )
    elif slope_ve <= 0.05:
        return ThresholdResult(
            zone_name="Poniżej VT1 (Strefa tlenowa)",
            zone_type="success",
            description="Liniowy wzrost VE. Komfortowa intensywność tlenowa.",
            slope_value=slope_ve
        )
    elif slope_ve <= 0.15:
        return ThresholdResult(
            zone_name="VT1-VT2 (Strefa progowa)",
            zone_type="warning",
            description="Pierwsze przełamanie wentylacyjne. Buforowanie kwasu mlekowego.",
            slope_value=slope_ve
        )
    else:
        return ThresholdResult(
            zone_name="Powyżej VT2 (Hiperwentylacja)",
            zone_type="error",
            description="Wykładniczy wzrost VE. Organizm nie nadąża z usuwaniem CO2.",
            slope_value=slope_ve
        )


def detect_smo2_zone(slope_smo2: float) -> ThresholdResult:
    """
    Detect metabolic zone based on SmO2 (muscle oxygen saturation) slope.
    
    Thresholds:
    - > 0.005: Below LT1 (oxygen surplus)
    - -0.005 to 0.005: Steady state (LT1-LT2)
    - -0.01 to -0.005: Approaching LT2
    - < -0.01: Above LT2 (anaerobic)
    
    Args:
        slope_smo2: Slope of SmO2 over time (% per second)
    
    Returns:
        ThresholdResult with zone classification
    """
    if slope_smo2 > 0.005:
        return ThresholdResult(
            zone_name="Poniżej LT1",
            zone_type="success",
            description="Podaż tlenu przewyższa zużycie. Strefa regeneracji.",
            slope_value=slope_smo2
        )
    elif slope_smo2 >= -0.005:
        return ThresholdResult(
            zone_name="Steady State (LT1-LT2)",
            zone_type="warning",
            description="Równowaga tlenowa. Możliwa długa intensywność.",
            slope_value=slope_smo2
        )
    elif slope_smo2 >= -0.01:
        return ThresholdResult(
            zone_name="Zbliżasz się do LT2",
            zone_type="warning",
            description="Powolna desaturacja. Blisko progu beztlenowego.",
            slope_value=slope_smo2
        )
    else:
        return ThresholdResult(
            zone_name="Powyżej LT2 (Anaerobic)",
            zone_type="error",
            description="Dług tlenowy. Intensywność nie do utrzymania.",
            slope_value=slope_smo2
        )


def calculate_slope(time_series: pd.Series, value_series: pd.Series) -> Tuple[float, float]:
    """
    Calculate linear regression slope for a time series.
    
    Args:
        time_series: Time values
        value_series: Measurement values
    
    Returns:
        Tuple of (slope, intercept)
    """
    if len(time_series) < 2:
        return 0.0, 0.0
    
    # Remove NaN values
    mask = ~(time_series.isna() | value_series.isna())
    if mask.sum() < 2:
        return 0.0, 0.0
    
    slope, intercept, _, _, _ = stats.linregress(
        time_series[mask], 
        value_series[mask]
    )
    return slope, intercept


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
    
    Detects VT1/VT2 from ventilation data and LT1/LT2 from SmO2 data.
    Uses classic 3-minute step test protocol by default.
    
    Args:
        df: DataFrame with training data
        step_duration_sec: Duration of each step in seconds (default: 180 = 3min)
        power_column: Column name for power data
        ve_column: Column name for ventilation data
        smo2_column: Column name for SmO2 data
        hr_column: Column name for heart rate data
        time_column: Column name for time data
    
    Returns:
        StepTestResult with detected thresholds
    """
    result = StepTestResult()
    
    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Check available data
    has_ve = ve_column in df.columns
    has_smo2 = smo2_column in df.columns
    has_power = power_column in df.columns
    has_hr = hr_column in df.columns
    has_time = time_column in df.columns
    
    if not has_time:
        result.analysis_notes.append("Brak kolumny czasu - nie można przeprowadzić analizy")
        return result
    
    if not has_power:
        result.analysis_notes.append("Brak danych mocy - ograniczona analiza")
    
    # Determine test duration and number of steps
    total_duration = df[time_column].max() - df[time_column].min()
    num_steps = int(total_duration / step_duration_sec)
    result.steps_analyzed = num_steps
    
    if num_steps < 3:
        result.analysis_notes.append(f"Za mało stopni ({num_steps}). Potrzeba minimum 3.")
        return result
    
    # Analyze each step
    step_data = []
    start_time = df[time_column].min()
    
    for step in range(num_steps):
        step_start = start_time + step * step_duration_sec
        step_end = step_start + step_duration_sec
        
        # Get data for this step (last 60s to avoid transitions)
        stable_start = step_end - 60
        mask = (df[time_column] >= stable_start) & (df[time_column] < step_end)
        step_df = df[mask]
        
        if len(step_df) < 10:
            continue
        
        step_info = {
            'step': step + 1,
            'start': step_start,
            'end': step_end
        }
        
        if has_power:
            step_info['avg_power'] = step_df[power_column].mean()
        
        if has_hr:
            step_info['avg_hr'] = step_df[hr_column].mean()
        
        # Calculate VE slope within step
        if has_ve:
            ve_slope, _ = calculate_slope(step_df[time_column], step_df[ve_column])
            step_info['ve_slope'] = ve_slope
            step_info['avg_ve'] = step_df[ve_column].mean()
        
        # Calculate SmO2 slope within step
        if has_smo2:
            smo2_slope, _ = calculate_slope(step_df[time_column], step_df[smo2_column])
            step_info['smo2_slope'] = smo2_slope
            step_info['avg_smo2'] = step_df[smo2_column].mean()
        
        step_data.append(step_info)
    
    if not step_data:
        result.analysis_notes.append("Nie udało się wyodrębnić żadnych stopni")
        return result
    
    # Detect VT1/VT2 from ventilation data
    if has_ve and len(step_data) >= 3:
        ve_thresholds = _detect_vent_thresholds(step_data)
        if ve_thresholds:
            result.vt1_watts = ve_thresholds.get('vt1_power')
            result.vt2_watts = ve_thresholds.get('vt2_power')
            result.vt1_hr = ve_thresholds.get('vt1_hr')
            result.vt2_hr = ve_thresholds.get('vt2_hr')
            result.analysis_notes.append(
                f"VT1/VT2 wykryte z wentylacji na stopniach {ve_thresholds.get('vt1_step')}/{ve_thresholds.get('vt2_step')}"
            )
    
    # Detect LT1/LT2 from SmO2 data
    if has_smo2 and len(step_data) >= 3:
        smo2_thresholds = _detect_smo2_thresholds(step_data)
        if smo2_thresholds:
            result.lt1_watts = smo2_thresholds.get('lt1_power')
            result.lt2_watts = smo2_thresholds.get('lt2_power')
            result.analysis_notes.append(
                f"LT1/LT2 wykryte z SmO2 na stopniach {smo2_thresholds.get('lt1_step')}/{smo2_thresholds.get('lt2_step')}"
            )
    
    return result


def _detect_vent_thresholds(step_data: List[dict]) -> Optional[dict]:
    """
    Detect VT1 and VT2 from step test ventilation data.
    
    VT1: First inflection point where VE slope increases disproportionally
    VT2: Second inflection point (respiratory compensation point)
    """
    if len(step_data) < 3:
        return None
    
    # Filter steps with VE data
    ve_steps = [s for s in step_data if 've_slope' in s and 'avg_power' in s]
    if len(ve_steps) < 3:
        return None
    
    slopes = [s['ve_slope'] for s in ve_steps]
    
    # Find VT1: transition from low slope to medium slope (>0.05)
    vt1_idx = None
    for i, slope in enumerate(slopes):
        if slope > 0.05:  # Threshold zone
            vt1_idx = max(0, i - 1)  # Step before threshold
            break
    
    # Find VT2: transition to high slope (>0.15)
    vt2_idx = None
    for i, slope in enumerate(slopes):
        if slope > 0.15:  # Above VT2
            vt2_idx = max(0, i - 1)  # Step before hyperventilation
            break
    
    result = {}
    
    if vt1_idx is not None:
        result['vt1_power'] = ve_steps[vt1_idx].get('avg_power')
        result['vt1_hr'] = ve_steps[vt1_idx].get('avg_hr')
        result['vt1_step'] = ve_steps[vt1_idx].get('step')
    
    if vt2_idx is not None:
        result['vt2_power'] = ve_steps[vt2_idx].get('avg_power')
        result['vt2_hr'] = ve_steps[vt2_idx].get('avg_hr')
        result['vt2_step'] = ve_steps[vt2_idx].get('step')
    
    return result if result else None


def _detect_smo2_thresholds(step_data: List[dict]) -> Optional[dict]:
    """
    Detect LT1 and LT2 from step test SmO2 data.
    
    LT1: Point where SmO2 slope transitions from positive to near-zero
    LT2: Point where SmO2 slope becomes significantly negative
    """
    if len(step_data) < 3:
        return None
    
    # Filter steps with SmO2 data
    smo2_steps = [s for s in step_data if 'smo2_slope' in s and 'avg_power' in s]
    if len(smo2_steps) < 3:
        return None
    
    slopes = [s['smo2_slope'] for s in smo2_steps]
    
    # Find LT1: transition from positive to near-zero slope
    lt1_idx = None
    for i, slope in enumerate(slopes):
        if slope <= 0.005:  # Steady state or below
            lt1_idx = max(0, i - 1)  # Step before steady state
            break
    
    # Find LT2: transition to significantly negative slope
    lt2_idx = None
    for i, slope in enumerate(slopes):
        if slope < -0.01:  # Anaerobic zone
            lt2_idx = max(0, i - 1)  # Step before anaerobic
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
    cp: Optional[int] = None,
    max_hr: int = 185
) -> dict:
    """
    Calculate personalized training zones based on detected thresholds.
    
    Uses VT1/VT2 from step test to create 6-zone power model.
    
    Args:
        vt1_watts: Ventilatory Threshold 1 (aerobic threshold) in watts
        vt2_watts: Ventilatory Threshold 2 (anaerobic threshold) in watts
        cp: Critical Power in watts (defaults to VT2 if not provided)
        max_hr: Maximum heart rate
    
    Returns:
        Dictionary with power and HR zones
    """
    if cp is None:
        cp = vt2_watts
    
    zones = {
        "power_zones": {
            "Z1_Recovery": (0, int(vt1_watts * 0.75)),
            "Z2_Endurance": (int(vt1_watts * 0.75), vt1_watts),
            "Z3_Tempo": (vt1_watts, int((vt1_watts + vt2_watts) / 2)),
            "Z4_Threshold": (int((vt1_watts + vt2_watts) / 2), vt2_watts),
            "Z5_VO2max": (vt2_watts, int(cp * 1.2)),
            "Z6_Anaerobic": (int(cp * 1.2), int(cp * 1.5))
        },
        "hr_zones": {
            "Z1_Recovery": (0, int(max_hr * 0.6)),
            "Z2_Endurance": (int(max_hr * 0.6), int(max_hr * 0.7)),
            "Z3_Tempo": (int(max_hr * 0.7), int(max_hr * 0.8)),
