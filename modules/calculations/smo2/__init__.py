"""
SmO2 Module - Refactored

Clean architecture for SmO2 analysis with separation of concerns.

Usage:
    from modules.calculations.smo2 import analyze_smo2_advanced
    
    metrics = analyze_smo2_advanced(df, smo2_col="smo2", power_col="watts")
    print(metrics.limiter_type)  # 'local', 'central', 'metabolic', 'unknown'
"""

from .types import SmO2AdvancedMetrics, SmO2ThresholdResult
from .constants import LIMITER_THRESHOLDS, RECOMMENDATIONS
from .calculator import SmO2MetricsCalculator
from .classifier import SmO2LimiterClassifier

# Backwards compatibility imports
from ..smo2_advanced import detect_smo2_thresholds_moxy

__all__ = [
    "SmO2AdvancedMetrics",
    "SmO2ThresholdResult", 
    "SmO2MetricsCalculator",
    "SmO2LimiterClassifier",
    "LIMITER_THRESHOLDS",
    "RECOMMENDATIONS",
    "analyze_smo2_advanced",
    "detect_smo2_thresholds_moxy",
]


def analyze_smo2_advanced(
    df,
    smo2_col: str = "SmO2",
    power_col: str = "watts",
    hr_col: str = "hr",
    time_col: str = "seconds"
) -> SmO2AdvancedMetrics:
    """
    Perform complete advanced SmO2 analysis.
    
    Args:
        df: DataFrame with SmO2, power, HR, time columns
        smo2_col: Name of SmO2 column
        power_col: Name of power column
        hr_col: Name of HR column
        time_col: Name of time column
        
    Returns:
        SmO2AdvancedMetrics with all calculated values
    """
    calculator = SmO2MetricsCalculator()
    
    # Check data quality
    if smo2_col not in df.columns:
        return SmO2AdvancedMetrics(
            data_quality="no_smo2",
            interpretation="Brak danych SmO₂."
        )
    
    if power_col not in df.columns:
        return SmO2AdvancedMetrics(
            data_quality="no_power",
            interpretation="Brak danych mocy."
        )
    
    # Calculate metrics
    slope, r2 = calculator.calculate_smo2_slope(df, smo2_col, power_col)
    halftime = calculator.calculate_halftime_reoxygenation(df, smo2_col, time_col, power_col)
    coupling = calculator.calculate_hr_coupling_index(df, smo2_col, hr_col)
    drift = calculator.calculate_smo2_drift(df, smo2_col, power_col)
    
    # Classify limiter
    metrics = SmO2AdvancedMetrics(
        slope_per_100w=slope,
        halftime_reoxy_sec=halftime,
        hr_coupling_r=coupling,
        drift_pct=drift,
        slope_r2=r2,
        data_quality="good" if r2 > 0.3 else "low"
    )
    
    limiter_type, confidence, interpretation = SmO2LimiterClassifier.classify(metrics)
    recommendations = SmO2LimiterClassifier.get_recommendations(limiter_type)
    
    return SmO2AdvancedMetrics(
        slope_per_100w=slope,
        halftime_reoxy_sec=halftime,
        hr_coupling_r=coupling,
        drift_pct=drift,
        slope_r2=r2,
        limiter_type=limiter_type,
        limiter_confidence=confidence,
        interpretation=interpretation,
        recommendations=recommendations,
        data_quality="good" if r2 > 0.3 else "low"
    )
