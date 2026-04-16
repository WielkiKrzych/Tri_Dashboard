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


# --- Ported from Analiza Kolarska: Numba JIT acceleration + aggregation ---

import numpy as np

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _fast_gradient(smo2_vals, power_vals):
        """Fast gradient calculation using Numba JIT."""
        n = len(smo2_vals)
        grad = np.zeros(n)
        for i in range(1, n - 1):
            dp = power_vals[i + 1] - power_vals[i - 1]
            if dp != 0:
                grad[i] = (smo2_vals[i + 1] - smo2_vals[i - 1]) / dp
        grad[0] = grad[1] if n > 1 else 0
        grad[-1] = grad[-2] if n > 1 else 0
        return grad

    @jit(nopython=True, cache=True)
    def _fast_curvature(smo2_vals, power_vals):
        """Fast curvature calculation using Numba JIT."""
        n = len(smo2_vals)
        grad = _fast_gradient(smo2_vals, power_vals)
        curv = np.zeros(n)
        for i in range(1, n - 1):
            dp = power_vals[i + 1] - power_vals[i - 1]
            if dp != 0:
                curv[i] = (grad[i + 1] - grad[i - 1]) / dp
        curv[0] = curv[1] if n > 1 else 0
        curv[-1] = curv[-2] if n > 1 else 0
        return curv
else:
    def _fast_gradient(smo2_vals, power_vals):
        """Fallback gradient calculation (no Numba)."""
        return np.gradient(smo2_vals, power_vals)

    def _fast_curvature(smo2_vals, power_vals):
        """Fallback curvature calculation (no Numba)."""
        grad = np.gradient(smo2_vals, power_vals)
        return np.gradient(grad, power_vals)


def aggregate_step(df, smo2_col="smo2", power_col="watts", hr_col="hr",
                   time_col="time", step_duration=180):
    """
    Aggregate SmO2 data by step (time window) for step-test analysis.

    Groups data into fixed-duration windows and calculates mean SmO2,
    power, HR, and SmO2 slope per step.

    Args:
        df: DataFrame with time-series data
        smo2_col: SmO2 column name
        power_col: Power column name
        hr_col: HR column name
        time_col: Time column name (seconds)
        step_duration: Duration of each step in seconds

    Returns:
        List of dicts with step aggregates
    """
    import pandas as pd
    if time_col not in df.columns or smo2_col not in df.columns:
        return []

    max_time = df[time_col].max()
    steps = []
    step_num = 0

    while step_num * step_duration < max_time:
        start = step_num * step_duration
        end = (step_num + 1) * step_duration
        mask = (df[time_col] >= start) & (df[time_col] < end)
        step_data = df[mask]

        if len(step_data) < 10:
            step_num += 1
            continue

        result = {
            "step": step_num + 1,
            "start_time": start,
            "end_time": end,
            "avg_smo2": float(step_data[smo2_col].mean()) if smo2_col in step_data.columns else None,
            "avg_power": float(step_data[power_col].mean()) if power_col in step_data.columns else None,
            "avg_hr": float(step_data[hr_col].mean()) if hr_col in step_data.columns else None,
            "n_points": len(step_data),
        }
        steps.append(result)
        step_num += 1

    return steps
