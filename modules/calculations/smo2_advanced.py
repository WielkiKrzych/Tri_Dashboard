"""
Advanced SmO2 Metrics Module — facade.

Re-exports all public API from focused sub-modules so that existing
callers require no import changes:

- smo2_analysis   : SmO2AdvancedMetrics, analyze_smo2_advanced,
                    calculate_smo2_slope, calculate_halftime_reoxygenation,
                    calculate_hr_coupling_index, calculate_smo2_drift,
                    classify_smo2_limiter, get_recommendations_for_limiter,
                    format_smo2_metrics_for_report
- smo2_thresholds : SmO2ThresholdResult, detect_smo2_thresholds_moxy
"""

from .smo2_analysis import (
    SmO2AdvancedMetrics,
    analyze_smo2_advanced,
    calculate_smo2_slope,
    calculate_halftime_reoxygenation,
    calculate_hr_coupling_index,
    calculate_smo2_drift,
    classify_smo2_limiter,
    get_recommendations_for_limiter,
    format_smo2_metrics_for_report,
)
from .smo2_thresholds import SmO2ThresholdResult, detect_smo2_thresholds_moxy

__all__ = [
    "SmO2AdvancedMetrics",
    "analyze_smo2_advanced",
    "calculate_smo2_slope",
    "calculate_halftime_reoxygenation",
    "calculate_hr_coupling_index",
    "calculate_smo2_drift",
    "classify_smo2_limiter",
    "get_recommendations_for_limiter",
    "format_smo2_metrics_for_report",
    "SmO2ThresholdResult",
    "detect_smo2_thresholds_moxy",
]
