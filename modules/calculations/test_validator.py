"""
Test Validator Module.

Validates ramp test data quality before running physiological analysis.
Provides quality score and recommendations for test improvement.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .threshold_types import TestValidityReport


VALIDATION_CRITERIA = {
    "duration": {
        "min": 300,
        "max": 3600,
        "weight": 0.15,
        "description": "Czas trwania testu"
    },
    "steps_count": {
        "min": 3,
        "max": 20,
        "weight": 0.20,
        "description": "Liczba stopni testu"
    },
    "monotonicity": {
        "min": 0.70,
        "max": 1.0,
        "weight": 0.25,
        "description": "Monotoniczność wzrostu mocy"
    },
    "data_gaps": {
        "min": 0,
        "max": 5,
        "weight": 0.15,
        "description": "Maksymalna przerwa w danych"
    },
    "cadence": {
        "min": 60,
        "max": 120,
        "weight": 0.10,
        "description": "Kadencja w dozwolonym zakresie"
    },
    "power_stability": {
        "min": 0.0,
        "max": 0.15,
        "weight": 0.15,
        "description": "Stabilność mocy na stopniu"
    }
}


def validate_ramp_test(  # noqa: C901
    df: pd.DataFrame,
    power_column: str = "watts",
    time_column: str = "time",
    cadence_column: Optional[str] = "cadence"
) -> TestValidityReport:
    """
    Validates ramp test data quality.
    
    Args:
        df: DataFrame with test data (1Hz assumed)
        power_column: Name of power column
        time_column: Name of time column
        cadence_column: Name of cadence column (optional)
        
    Returns:
        TestValidityReport with status, criteria, and recommendations
    """
    criteria: Dict[str, bool] = {}
    criteria_details: Dict[str, Dict[str, Any]] = {}
    recommendations: List[str] = []
    warnings: List[str] = []
    
    df.columns = df.columns.str.lower().str.strip()
    power_column = power_column.lower()
    time_column = time_column.lower()
    cadence_column = cadence_column.lower() if cadence_column else None
    
    has_power = power_column in df.columns
    has_time = time_column in df.columns
    has_cadence = cadence_column and cadence_column in df.columns
    
    duration_sec = len(df)
    duration_min = duration_sec / 60
    
    min_dur = VALIDATION_CRITERIA["duration"]["min"]
    max_dur = VALIDATION_CRITERIA["duration"]["max"]
    criteria["duration"] = min_dur <= duration_sec <= max_dur
    criteria_details["duration"] = {
        "value": duration_sec,
        "value_display": f"{duration_min:.1f} min",
        "expected": f"{min_dur/60:.0f}-{max_dur/60:.0f} min"
    }
    
    if duration_sec < min_dur:
        recommendations.append(f"Test zbyt krótki ({duration_min:.1f} min). Minimum: 5 minut.")
    elif duration_sec > max_dur:
        warnings.append(f"Test bardzo długi ({duration_min:.1f} min). Sprawdź czy to ramp test.")
    
    if has_power:
        steps = _detect_power_steps(df[power_column].values)
        steps_count = len(steps)
        
        min_steps = VALIDATION_CRITERIA["steps_count"]["min"]
        max_steps = VALIDATION_CRITERIA["steps_count"]["max"]
        criteria["steps_count"] = min_steps <= steps_count <= max_steps
        criteria_details["steps_count"] = {
            "value": steps_count,
            "value_display": f"{steps_count} stopni",
            "expected": f"{min_steps}-{max_steps} stopni",
            "steps": [{"avg_power": s["avg_power"], "duration": s["duration"]} for s in steps]
        }
        
        if steps_count < min_steps:
            recommendations.append(f"Wykryto tylko {steps_count} stopni. Minimum: 3 dla wiarygodnej analizy.")
        
        monotonicity = _calculate_monotonicity(df[power_column].values)
        min_mono = VALIDATION_CRITERIA["monotonicity"]["min"]
        criteria["monotonicity"] = monotonicity >= min_mono
        criteria_details["monotonicity"] = {
            "value": monotonicity,
            "value_display": f"{monotonicity*100:.0f}%",
            "expected": f"≥{min_mono*100:.0f}%"
        }
        
        if monotonicity < min_mono:
            recommendations.append(
                f"Niska monotoniczność ({monotonicity*100:.0f}%). "
                "Test powinien mieć stopniowo rosnącą moc."
            )
        
        power_cv = _calculate_power_stability(df[power_column].values, steps)
        max_cv = VALIDATION_CRITERIA["power_stability"]["max"]
        criteria["power_stability"] = power_cv <= max_cv
        criteria_details["power_stability"] = {
            "value": power_cv,
            "value_display": f"{power_cv*100:.1f}% CV",
            "expected": f"≤{max_cv*100:.0f}%"
        }
        
        if power_cv > max_cv:
            warnings.append(f"Wysoka zmienność mocy na stopniu ({power_cv*100:.1f}%). Może wpłynąć na dokładność.")
    else:
        criteria["steps_count"] = False
        criteria["monotonicity"] = False
        criteria["power_stability"] = False
        criteria_details["steps_count"] = {"value": 0, "error": "Brak kolumny mocy"}
        recommendations.append("Brak danych mocy - analiza nie będzie możliwa.")
    
    if has_time:
        max_gap = _detect_max_gap(df[time_column].values)
        max_gap_allowed = VALIDATION_CRITERIA["data_gaps"]["max"]
        criteria["data_gaps"] = max_gap <= max_gap_allowed
        criteria_details["data_gaps"] = {
            "value": max_gap,
            "value_display": f"{max_gap}s",
            "expected": f"≤{max_gap_allowed}s"
        }
        
        if max_gap > max_gap_allowed:
            warnings.append(f"Przerwa w danych: {max_gap}s. Może wpłynąć na dokładność.")
    else:
        criteria["data_gaps"] = True
        criteria_details["data_gaps"] = {"value": 0, "note": "Brak kolumny czasu"}
    
    if has_cadence:
        cadence = df[cadence_column].dropna()
        if len(cadence) > 0:
            min_cad = VALIDATION_CRITERIA["cadence"]["min"]
            max_cad = VALIDATION_CRITERIA["cadence"]["max"]
            cadence_in_range = ((cadence >= min_cad) & (cadence <= max_cad)).mean()
            criteria["cadence"] = cadence_in_range >= 0.8
            criteria_details["cadence"] = {
                "value": cadence_in_range,
                "value_display": f"{cadence_in_range*100:.0f}% w zakresie",
                "expected": f"{min_cad}-{max_cad} rpm (≥80%)"
            }
            
            if cadence_in_range < 0.8:
                warnings.append("Kadencja poza optymalnym zakresem (60-120 rpm).")
        else:
            criteria["cadence"] = True
            criteria_details["cadence"] = {"value": None, "note": "Brak danych kadencji"}
    else:
        criteria["cadence"] = True
        criteria_details["cadence"] = {"value": None, "note": "Brak danych kadencji"}
    
    quality_score = _calculate_quality_score(criteria, criteria_details)
    
    if quality_score >= 80 and all(criteria.values()):
        status = "valid"
    elif quality_score >= 50:
        status = "conditional"
    else:
        status = "invalid"
    
    return TestValidityReport(
        status=status,
        criteria=criteria,
        criteria_details=criteria_details,
        quality_score=quality_score,
        recommendations=recommendations,
        warnings=warnings
    )


def _detect_power_steps(power: np.ndarray, min_step_duration: int = 30) -> List[dict]:
    if len(power) < min_step_duration:
        return []
    
    window = min(30, len(power) // 10)
    if window < 5:
        window = 5
    
    smoothed = pd.Series(power).rolling(window=window, center=True).mean().bfill().ffill().values
    gradient = np.gradient(smoothed)
    
    step_threshold = 0.5
    step_starts = [0]
    
    in_transition = False
    for i in range(1, len(gradient)):
        if abs(gradient[i]) > step_threshold and not in_transition:
            in_transition = True
        elif abs(gradient[i]) <= step_threshold and in_transition:
            in_transition = False
            if i - step_starts[-1] >= min_step_duration:
                step_starts.append(i)
    
    steps = []
    for i in range(len(step_starts)):
        start = step_starts[i]
        end = step_starts[i + 1] if i + 1 < len(step_starts) else len(power)
        duration = end - start
        
        if duration >= min_step_duration:
            avg_power = np.mean(power[start:end])
            steps.append({
                "start": start,
                "end": end,
                "duration": duration,
                "avg_power": avg_power
            })
    
    return steps


def _calculate_monotonicity(power: np.ndarray, window: int = 30) -> float:
    if len(power) < window:
        return 0.0
    
    smoothed = pd.Series(power).rolling(window=window, center=True).mean().bfill().ffill().values
    
    increases = 0
    total_transitions = 0
    
    for i in range(1, len(smoothed)):
        diff = smoothed[i] - smoothed[i-1]
        if abs(diff) > 1:
            total_transitions += 1
            if diff > 0:
                increases += 1
    
    if total_transitions == 0:
        return 1.0
    
    return increases / total_transitions


def _detect_max_gap(time: np.ndarray) -> int:
    if len(time) < 2:
        return 0
    
    diffs = np.diff(time)
    expected_diff = 1.0
    
    gaps = diffs - expected_diff
    max_gap = int(np.max(gaps)) if len(gaps) > 0 else 0
    
    return max(0, max_gap)


def _calculate_power_stability(power: np.ndarray, steps: List[dict]) -> float:
    if not steps:
        return 1.0
    
    cvs = []
    for step in steps:
        start, end = step["start"], step["end"]
        step_power = power[start:end]
        
        if len(step_power) > 1:
            mean_p = np.mean(step_power)
            std_p = np.std(step_power)
            if mean_p > 0:
                cvs.append(std_p / mean_p)
    
    if not cvs:
        return 0.0
    
    return float(np.mean(cvs))


def _calculate_quality_score(criteria: Dict[str, bool], details: Dict[str, Dict]) -> float:
    if not criteria:
        return 0.0
    
    total_weight = 0.0
    weighted_score = 0.0
    
    for criterion, passed in criteria.items():
        weight = VALIDATION_CRITERIA.get(criterion, {}).get("weight", 0.1)
        total_weight += weight
        if passed:
            weighted_score += weight
    
    if total_weight == 0:
        return 0.0
    
    return round((weighted_score / total_weight) * 100, 1)
